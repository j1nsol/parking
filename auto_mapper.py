"""
auto_mapper.py — Automatically discovers parking slot layout from video.

Each slot is represented as a QUADRILATERAL (4 corner points) instead of
an axis-aligned rectangle. This fits angled/perspective parking slots
correctly and enables point-in-polygon occupancy detection.

Slot data schema:
    {
        "coords": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],  # clockwise from top-left
        "row":    "A" | "B" | "C",
        "source": "detected" | "inferred" | "manual",
    }

Strategy:
  1. Track vehicles across frames — only accumulate stationary ones.
  2. DBSCAN-cluster stationary detections → one cluster per slot.
  3. Fit cv2.minAreaRect() to each cluster → rotated quad that matches
     the actual parking angle.
  4. Group slots along a configurable axis (horizontal rows, vertical
     columns, grid, or auto-detect) and fill single/double gaps.
  5. Save to slot_config.json.

Layout modes:
  - "horizontal": rows of slots parallel to the X axis (group by Y)
  - "vertical":   columns of slots parallel to the Y axis (group by X)
  - "grid":       both horizontal rows AND vertical columns; results merged
  - "auto":       pick per-group orientation from cluster variance (PCA-lite)
"""

import cv2
import json
import os
import numpy as np
import logging
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)

STATIONARY_THRESHOLD_PX = 12   # was 20 — lowered so slightly wobbling cars (e.g. dark/toy cars)
                                # still count as stationary and accumulate cluster detections.
                                # Raise back to 20 if legitimate parked cars are being skipped.
MIN_STATIONARY_FRAMES   = 5
ACCUM_EVERY             = 10

VALID_LAYOUT_MODES = ("horizontal", "vertical", "grid", "auto")
MIN_GROUP_SIZE_FOR_INFER = 3   # a "row" or "column" must have ≥3 detected
                                # slots before we trust it enough to fill gaps.
                                # Prevents two unrelated slots that happen to
                                # share a Y or X band from spawning road-slots.


def rect_to_quad(rect) -> list:
    """
    Convert cv2.minAreaRect result to 4 corner points sorted clockwise
    from top-left. Returns [[x,y], [x,y], [x,y], [x,y]].
    """
    box = cv2.boxPoints(rect)           # shape (4,2), float32
    box = np.intp(box)
    # Sort: top-left, top-right, bottom-right, bottom-left
    # First sort by Y (top two vs bottom two)
    box = sorted(box, key=lambda p: p[1])
    top    = sorted(box[:2], key=lambda p: p[0])
    bottom = sorted(box[2:], key=lambda p: p[0], reverse=True)
    tl, tr = top
    br, bl = bottom
    return [[int(tl[0]), int(tl[1])],
            [int(tr[0]), int(tr[1])],
            [int(br[0]), int(br[1])],
            [int(bl[0]), int(bl[1])]]


def quad_center(quad: list) -> tuple:
    """Return (cx, cy) of a quad."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return sum(xs)/4, sum(ys)/4


def quad_from_bbox(x1, y1, x2, y2) -> list:
    """Fallback: convert axis-aligned bbox to quad."""
    return [[int(x1), int(y1)], [int(x2), int(y1)],
            [int(x2), int(y2)], [int(x1), int(y2)]]


class AutoMapper:
    def __init__(
        self,
        slot_config_path: str = "slot_config.json",
        min_frames_to_map: int = 150,   # was min_detections_to_map=50; callers pass min_frames_to_map=150
        eps_pixels: int = 60,
        min_samples: int = 5,
        infer_empty_slots: bool = True,
        group_merge_tolerance: float = 0.6,   # renamed from row_merge_tolerance — applies to
                                               # either axis depending on layout_mode
        gap_tolerance: float = 0.22,    # NOTE: gap-fill assumes equal slot spacing;
        max_gap_multiplier: float = 2.3, # unequally-spaced lots may miss some inferred slots.
        layout_mode: str = "horizontal",  # "horizontal" | "vertical" | "grid" | "auto"
    ):
        self.config_path    = slot_config_path
        self.min_detections = min_frames_to_map
        self.eps            = eps_pixels
        self.min_samples    = min_samples
        self.infer_empty    = infer_empty_slots
        self.group_merge_tol = group_merge_tolerance
        self.gap_tolerance  = gap_tolerance
        self.max_gap_mult   = max_gap_multiplier

        if layout_mode not in VALID_LAYOUT_MODES:
            log.warning(f"Invalid layout_mode '{layout_mode}' — defaulting to 'horizontal'.")
            layout_mode = "horizontal"
        self.layout_mode = layout_mode
        log.info(f"AutoMapper layout_mode = '{self.layout_mode}'")

        self._detections: list = []   # (cx,cy,x1,y1,x2,y2)
        self._tracked: dict    = {}   # int id → {cx,cy,x1,y1,x2,y2,still_count}
        self._next_id: int     = 0
        self._frame_count      = 0
        self._slots: dict      = {}

        if os.path.exists(slot_config_path):
            self._load_config()
            log.info(f"Loaded existing slot config: {len(self._slots)} slots")

    # ------------------------------------------------------------------
    # Public: layout mode control
    # ------------------------------------------------------------------
    def set_layout_mode(self, mode: str):
        """Change layout mode (used when admin triggers a remap with a new mode)."""
        if mode not in VALID_LAYOUT_MODES:
            log.warning(f"set_layout_mode: invalid mode '{mode}' — ignored.")
            return
        self.layout_mode = mode
        log.info(f"AutoMapper layout_mode changed to '{mode}'")

    # ------------------------------------------------------------------
    def feed_frame(self, vehicle_boxes: list, frame_shape: tuple):
        self._frame_count += 1

        current = []
        for box in vehicle_boxes:
            x1, y1, x2, y2 = box
            current.append(((x1+x2)/2, (y1+y2)/2, x1, y1, x2, y2))

        new_tracked = {}
        used_ids    = set()

        for cx, cy, x1, y1, x2, y2 in current:
            best_id, best_dist = None, STATIONARY_THRESHOLD_PX * 3
            for tid, prev in self._tracked.items():
                if tid in used_ids:
                    continue
                d = np.hypot(cx - prev["cx"], cy - prev["cy"])
                if d < best_dist:
                    best_dist, best_id = d, tid

            if best_id is not None:
                used_ids.add(best_id)
                prev  = self._tracked[best_id]
                moved = np.hypot(cx - prev["cx"], cy - prev["cy"])
                still = prev["still_count"] + 1 if moved < STATIONARY_THRESHOLD_PX else 0
                new_tracked[best_id] = dict(cx=cx, cy=cy, x1=x1, y1=y1,
                                            x2=x2, y2=y2, still_count=still)
                if still >= MIN_STATIONARY_FRAMES and still % ACCUM_EVERY == 0:
                    self._detections.append((cx, cy, x1, y1, x2, y2))
                    log.debug(f"[MAPPER] still id={best_id} ({round(cx)},{round(cy)}) "
                              f"still={still} total={len(self._detections)}")
            else:
                new_tracked[self._next_id] = dict(cx=cx, cy=cy, x1=x1, y1=y1,
                                                  x2=x2, y2=y2, still_count=0)
                self._next_id += 1

        self._tracked = new_tracked

        if len(self._detections) >= self.min_detections and not self._slots:
            log.info(f"[MAPPER] Clustering — {len(self._detections)} detections "
                     f"over {self._frame_count} frames.")
            self._run_clustering(frame_shape)

    # ------------------------------------------------------------------
    def _run_clustering(self, frame_shape: tuple):
        if len(self._detections) < self.min_samples:
            log.warning("Not enough detections to build slot map.")
            return

        centers = np.array([[d[0], d[1]] for d in self._detections])
        labels  = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(centers)
        cluster_ids = sorted(set(labels) - {-1})
        log.info(f"DBSCAN: {len(cluster_ids)} clusters from {len(self._detections)} detections.")

        slots        = {}
        cluster_data = []   # (cx, cy, quad) per slot

        for i, cid in enumerate(cluster_ids):
            mask  = labels == cid
            # All raw bboxes in this cluster
            raw_boxes = np.array([
                [d[2], d[3], d[4], d[5]]
                for d, m in zip(self._detections, mask) if m
            ])
            # Collect all corner points of all bboxes → fit minAreaRect
            corners = []
            for b in raw_boxes:
                x1, y1, x2, y2 = b
                corners.extend([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            pts  = np.array(corners, dtype=np.float32)
            rect = cv2.minAreaRect(pts)
            quad = rect_to_quad(rect)
            cx, cy = quad_center(quad)

            sid = f"S{i+1:02d}"
            slots[sid] = {
                "coords": quad,
                "row":    self._infer_row(cy, frame_shape[0]),
                "source": "detected",
            }
            cluster_data.append((cx, cy, quad))
            log.info(f"  Slot {sid}: center=({round(cx)},{round(cy)}) "
                     f"angle={rect[2]:.1f}° size={round(rect[1][0])}×{round(rect[1][1])}")

        if self.infer_empty and len(cluster_data) >= 2:
            inferred = self._infer_gaps(cluster_data, frame_shape, set(slots.keys()))
            slots.update(inferred)
            if inferred:
                log.info(f"Gap inference added {len(inferred)} empty slot(s) "
                         f"using mode='{self.layout_mode}'.")

        self._slots = slots
        self._save_config()

    # ------------------------------------------------------------------
    # Gap inference — dispatches to the right strategy based on layout_mode
    # ------------------------------------------------------------------
    def _infer_gaps(self, cluster_data, frame_shape, existing_ids: set) -> dict:
        """
        Dispatch gap inference based on layout_mode.
          - horizontal: group by Y, fill gaps along X within each row
          - vertical:   group by X, fill gaps along Y within each column
          - grid:       run both, deduplicate overlapping inferrals
          - auto:       decide per-group using variance along each axis
        """
        # Collision-safe ID generator — shared across all inference passes
        used_ids = set(existing_ids)
        counter  = [1]
        def next_id():
            while True:
                candidate = f"S{counter[0]:02d}"
                counter[0] += 1
                if candidate not in used_ids:
                    used_ids.add(candidate)
                    return candidate

        mode = self.layout_mode

        if mode == "horizontal":
            return self._infer_along_axis(cluster_data, frame_shape,
                                           group_axis=1, fill_axis=0, next_id=next_id)

        if mode == "vertical":
            return self._infer_along_axis(cluster_data, frame_shape,
                                           group_axis=0, fill_axis=1, next_id=next_id)

        if mode == "grid":
            horiz = self._infer_along_axis(cluster_data, frame_shape,
                                            group_axis=1, fill_axis=0, next_id=next_id)
            # When running the second pass, treat horiz-inferred centers as
            # "seen" so we don't double-infer the same spot.
            combined_seen = [(d[0], d[1]) for d in cluster_data]
            for slot in horiz.values():
                cx, cy = quad_center(slot["coords"])
                combined_seen.append((cx, cy))
            vert = self._infer_along_axis(cluster_data, frame_shape,
                                           group_axis=0, fill_axis=1, next_id=next_id,
                                           extra_seen=combined_seen)
            merged = dict(horiz)
            merged.update(vert)
            return merged

        if mode == "auto":
            return self._infer_auto(cluster_data, frame_shape, next_id)

        # Unknown mode — defensive fallback
        log.warning(f"_infer_gaps: unknown mode '{mode}' — no inference done.")
        return {}

    # ------------------------------------------------------------------
    # Core inference along a single axis
    # ------------------------------------------------------------------
    def _infer_along_axis(
        self,
        cluster_data,
        frame_shape,
        group_axis: int,     # 0=group by X (columns), 1=group by Y (rows)
        fill_axis: int,      # 0=fill along X,        1=fill along Y
        next_id,
        extra_seen: list = None,
    ) -> dict:
        """
        Generic gap-filler. Groups clusters by proximity on `group_axis`, then
        fills single/double gaps along `fill_axis` within each group.

        For horizontal rows:  group_axis=1 (Y), fill_axis=0 (X)
        For vertical columns: group_axis=0 (X), fill_axis=1 (Y)
        """
        if len(cluster_data) < 2:
            return {}

        # Median slot diagonal and perpendicular dimension — used as scale references
        diagonals = []
        perp_sizes = []   # size of slot along group_axis (perpendicular to fill direction)
        for _, _, quad in cluster_data:
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            diagonals.append(np.hypot(max(xs)-min(xs), max(ys)-min(ys)))
            # Perpendicular size: if we group by Y, we care about slot height
            if group_axis == 1:
                perp_sizes.append(max(ys) - min(ys))
            else:
                perp_sizes.append(max(xs) - min(xs))
        med_diag = float(np.median(diagonals))
        med_perp = float(np.median(perp_sizes))

        group_thr = med_perp * self.group_merge_tol
        max_dist  = med_diag * self.max_gap_mult

        # Sort by group-axis coordinate, then assign to groups by proximity
        # cluster_data tuples are (cx, cy, quad); index 0 = cx, index 1 = cy
        points = sorted(cluster_data, key=lambda d: d[group_axis])
        groups: list = []
        for pt in points:
            placed = False
            for grp in groups:
                mean_coord = np.mean([g[group_axis] for g in grp])
                if abs(pt[group_axis] - mean_coord) < group_thr:
                    grp.append(pt)
                    placed = True
                    break
            if not placed:
                groups.append([pt])

        axis_name = "row" if group_axis == 1 else "column"
        log.info(f"[{axis_name.upper()}] grouping: {len(cluster_data)} clusters → {len(groups)} {axis_name}(s)")

        inferred = {}
        seen = [(d[0], d[1]) for d in cluster_data]
        if extra_seen:
            seen.extend(extra_seen)

        for grp_idx, grp in enumerate(groups):
            # Require ≥3 detected slots in a group before trusting it enough
            # to infer road-adjacent neighbors. A 2-slot group is too weak:
            # those slots might not actually share a row/column at all.
            if len(grp) < MIN_GROUP_SIZE_FOR_INFER:
                log.debug(f"[{axis_name.upper()} {grp_idx}] only {len(grp)} slot(s) "
                          f"— below threshold {MIN_GROUP_SIZE_FOR_INFER}, skipping inference.")
                continue

            # Sort along fill axis so adjacent pairs are actual neighbors
            grp.sort(key=lambda p: p[fill_axis])

            for k in range(len(grp) - 1):
                a = grp[k]
                b = grp[k + 1]
                cx_a, cy_a, quad_a = a
                cx_b, cy_b, quad_b = b
                dist = np.hypot(cx_b - cx_a, cy_b - cy_a)

                if dist > max_dist:
                    log.debug(f"[{axis_name.upper()} {grp_idx}]: gap {dist:.0f}px > {max_dist:.0f}px — lane")
                    continue

                expected = med_diag * 1.1
                tol      = expected * self.gap_tolerance

                # Detect both single-slot gaps (2× slot width) and
                # double-slot gaps (3× slot width) between detected clusters.
                gap_slots = None
                if abs(dist - expected * 2) < tol * 2:
                    gap_slots = 1
                elif abs(dist - expected * 3) < tol * 3:
                    gap_slots = 2

                if gap_slots is None:
                    continue

                for gap_i in range(1, gap_slots + 1):
                    frac   = gap_i / (gap_slots + 1)
                    mid_cx = cx_a + frac * (cx_b - cx_a)
                    mid_cy = cy_a + frac * (cy_b - cy_a)

                    if any(np.hypot(mid_cx-sx, mid_cy-sy) < med_diag*0.4 for sx,sy in seen):
                        continue

                    # Interpolate quad between the two neighbors
                    avg_quad = []
                    for pi in range(4):
                        ax = quad_a[pi][0] + frac * (quad_b[pi][0] - quad_a[pi][0])
                        ay = quad_a[pi][1] + frac * (quad_b[pi][1] - quad_a[pi][1])
                        avg_quad.append([ax, ay])

                    # Translate so quad center sits exactly at mid_cx, mid_cy
                    avg_cx, avg_cy = quad_center(avg_quad)
                    dx, dy = mid_cx - avg_cx, mid_cy - avg_cy
                    final_quad = [[round(p[0]+dx), round(p[1]+dy)] for p in avg_quad]

                    sid = next_id()
                    inferred[sid] = {
                        "coords": final_quad,
                        "row":    self._infer_row(mid_cy, frame_shape[0]),
                        "source": "inferred",
                    }
                    seen.append((mid_cx, mid_cy))
                    log.info(f"  → Inferred {sid} at ({round(mid_cx)},{round(mid_cy)}) "
                             f"axis={axis_name} gap={gap_slots} frac={frac:.2f}")

        return inferred

    # ------------------------------------------------------------------
    # Auto mode — pick orientation per-group using variance
    # ------------------------------------------------------------------
    def _infer_auto(self, cluster_data, frame_shape, next_id) -> dict:
        """
        Auto mode: try grouping by both axes, keep whichever produces more
        usable groups (≥ MIN_GROUP_SIZE_FOR_INFER). If both produce usable
        groups, run them both (effectively grid mode for this map).
        If neither does, return empty.

        Rationale: for a vertical column layout, grouping by Y produces many
        one-element "rows" (nothing to infer), while grouping by X produces
        one big column. The grouping that yields more members-per-group wins.
        """
        # Trial run: how many usable groups does each axis produce?
        usable_horiz = self._count_usable_groups(cluster_data, group_axis=1)
        usable_vert  = self._count_usable_groups(cluster_data, group_axis=0)

        log.info(f"[AUTO] Usable groups — horizontal: {usable_horiz}, vertical: {usable_vert}")

        if usable_horiz == 0 and usable_vert == 0:
            log.info("[AUTO] No axis produced a usable group — no inference.")
            return {}

        inferred: dict = {}
        seen_accum = [(d[0], d[1]) for d in cluster_data]

        if usable_horiz > 0:
            horiz = self._infer_along_axis(
                cluster_data, frame_shape,
                group_axis=1, fill_axis=0, next_id=next_id,
            )
            for slot in horiz.values():
                cx, cy = quad_center(slot["coords"])
                seen_accum.append((cx, cy))
            inferred.update(horiz)

        if usable_vert > 0:
            vert = self._infer_along_axis(
                cluster_data, frame_shape,
                group_axis=0, fill_axis=1, next_id=next_id,
                extra_seen=seen_accum,
            )
            inferred.update(vert)

        return inferred

    def _count_usable_groups(self, cluster_data, group_axis: int) -> int:
        """Count how many groups of ≥MIN_GROUP_SIZE_FOR_INFER the given axis produces."""
        if len(cluster_data) < MIN_GROUP_SIZE_FOR_INFER:
            return 0

        perp_sizes = []
        for _, _, quad in cluster_data:
            if group_axis == 1:
                ys = [p[1] for p in quad]
                perp_sizes.append(max(ys) - min(ys))
            else:
                xs = [p[0] for p in quad]
                perp_sizes.append(max(xs) - min(xs))
        med_perp = float(np.median(perp_sizes))
        group_thr = med_perp * self.group_merge_tol

        points = sorted(cluster_data, key=lambda d: d[group_axis])
        groups: list = []
        for pt in points:
            placed = False
            for grp in groups:
                mean_coord = np.mean([g[group_axis] for g in grp])
                if abs(pt[group_axis] - mean_coord) < group_thr:
                    grp.append(pt); placed = True; break
            if not placed:
                groups.append([pt])

        return sum(1 for g in groups if len(g) >= MIN_GROUP_SIZE_FOR_INFER)

    # ------------------------------------------------------------------
    def _infer_row(self, y_center, frame_height):
        r = y_center / frame_height
        return "A" if r < 0.33 else ("B" if r < 0.66 else "C")

    def _save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self._slots, f, indent=2)
        log.info(f"Slot config saved → {self.config_path} ({len(self._slots)} slots)")

    def _load_config(self):
        with open(self.config_path) as f:
            self._slots = json.load(f)

    def is_mapping_complete(self):
        return bool(self._slots)

    def get_slots(self):
        return self._slots

    def add_slot_manual(self, slot_id, quad):
        self._slots[slot_id] = {"coords": quad, "row": "M", "source": "manual"}
        self._save_config()

    def remove_slot(self, slot_id):
        self._slots.pop(slot_id, None)
        self._save_config()