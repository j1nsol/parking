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
  4. Row-based inference fills single-slot gaps within rows.
  5. Save to slot_config.json.
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
        row_merge_tolerance: float = 0.6,
        gap_tolerance: float = 0.22,    # NOTE: gap-fill assumes equal slot spacing;
        max_gap_multiplier: float = 2.3, # unequally-spaced lots may miss some inferred slots.
    ):
        self.config_path    = slot_config_path
        self.min_detections = min_frames_to_map
        self.eps            = eps_pixels
        self.min_samples    = min_samples
        self.infer_empty    = infer_empty_slots
        self.row_merge_tol  = row_merge_tolerance
        self.gap_tolerance  = gap_tolerance
        self.max_gap_mult   = max_gap_multiplier

        self._detections: list = []   # (cx,cy,x1,y1,x2,y2)
        self._tracked: dict    = {}   # int id → {cx,cy,x1,y1,x2,y2,still_count}
        self._next_id: int     = 0
        self._frame_count      = 0
        self._slots: dict      = {}

        if os.path.exists(slot_config_path):
            self._load_config()
            log.info(f"Loaded existing slot config: {len(self._slots)} slots")

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
            inferred = self._infer_by_row(cluster_data, frame_shape, len(slots))
            slots.update(inferred)
            if inferred:
                log.info(f"Row inference added {len(inferred)} empty slot(s).")

        self._slots = slots
        self._save_config()

    # ------------------------------------------------------------------
    def _infer_by_row(self, cluster_data, frame_shape, next_index):
        """
        Group quads into rows by Y proximity, then fill single-slot gaps
        within each row. Inferred slots inherit the average quad shape
        of their two neighbors, rotated to match.
        """
        # Estimate median slot diagonal for scale reference
        diagonals = []
        for _, _, quad in cluster_data:
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            diagonals.append(np.hypot(max(xs)-min(xs), max(ys)-min(ys)))
        med_diag = float(np.median(diagonals))
        med_h    = float(np.median([max(p[1] for p in q) - min(p[1] for p in q)
                                    for _,_,q in cluster_data]))

        y_thr    = med_h * self.row_merge_tol
        max_dist = med_diag * self.max_gap_mult

        points = sorted(cluster_data, key=lambda d: d[1])
        rows: list = []
        for pt in points:
            placed = False
            for row in rows:
                if abs(pt[1] - np.mean([r[1] for r in row])) < y_thr:
                    row.append(pt); placed = True; break
            if not placed:
                rows.append([pt])

        log.info(f"Row grouping: {len(cluster_data)} clusters → {len(rows)} rows")

        inferred = {}
        seen = [(d[0], d[1]) for d in cluster_data]

        for row_idx, row in enumerate(rows):
            if len(row) < 2:
                continue
            row.sort(key=lambda p: p[0])

            for k in range(len(row) - 1):
                cx_a, cy_a, quad_a = row[k]
                cx_b, cy_b, quad_b = row[k+1]
                dist = np.hypot(cx_b - cx_a, cy_b - cy_a)

                if dist > max_dist:
                    log.debug(f"Row {row_idx}: gap {dist:.0f}px > {max_dist:.0f}px — lane")
                    continue

                expected = med_diag * 1.1
                tol      = expected * self.gap_tolerance
                if abs(dist - expected * 2) >= tol * 2:
                    continue

                mid_cx = (cx_a + cx_b) / 2
                mid_cy = (cy_a + cy_b) / 2

                if any(np.hypot(mid_cx-sx, mid_cy-sy) < med_diag*0.4 for sx,sy in seen):
                    continue

                # Build inferred quad by averaging the two neighbor quads
                # and translating to the midpoint
                avg_quad = []
                for pi in range(4):
                    ax = (quad_a[pi][0] + quad_b[pi][0]) / 2
                    ay = (quad_a[pi][1] + quad_b[pi][1]) / 2
                    avg_quad.append([ax, ay])

                # Translate averaged quad so its center sits at mid_cx, mid_cy
                avg_cx, avg_cy = quad_center(avg_quad)
                dx, dy = mid_cx - avg_cx, mid_cy - avg_cy
                final_quad = [[round(p[0]+dx), round(p[1]+dy)] for p in avg_quad]

                sid = f"S{next_index+1:02d}"
                next_index += 1
                inferred[sid] = {
                    "coords": final_quad,
                    "row":    self._infer_row(mid_cy, frame_shape[0]),
                    "source": "inferred",
                }
                seen.append((mid_cx, mid_cy))
                log.info(f"  → Inferred {sid} at ({round(mid_cx)},{round(mid_cy)}) row={row_idx}")

        return inferred

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