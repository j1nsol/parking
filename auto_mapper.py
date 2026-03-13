"""
auto_mapper.py — Automatically discovers parking slot layout from video.

Strategy:
  1. Track each detected vehicle across consecutive frames.
  2. Only accumulate a detection if the vehicle has been stationary for
     at least MIN_STATIONARY_FRAMES consecutive frames. This filters out
     cars driving through lanes, which move significantly between frames.
  3. DBSCAN-cluster the stationary detections → each cluster = one slot.
  4. Row-based empty slot inference fills gaps within rows only.
  5. Save to slot_config.json.

The stationarity filter is the critical addition over v1/v2:
  - Parked car:       moves 0–8px between frames   → accumulated
  - Drive-through car: moves 50–300px between frames → ignored
"""

import json
import os
import numpy as np
import logging
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)

# A detection is considered stationary if its center moves less than this
# many pixels between consecutive frames.
STATIONARY_THRESHOLD_PX = 20   # tune up if jitter causes misses, down if lane cars sneak in

# How many consecutive stationary frames before we trust a detection
MIN_STATIONARY_FRAMES = 5


class AutoMapper:
    def __init__(
        self,
        slot_config_path: str = "slot_config.json",
        min_detections_to_map: int = 50,   # minimum accumulated stationary detections
        eps_pixels: int = 60,
        min_samples: int = 5,              # minimum stationary hits to form a slot cluster
        infer_empty_slots: bool = True,
        row_merge_tolerance: float = 0.6,
        gap_tolerance: float = 0.22,
        max_gap_multiplier: float = 2.3,
    ):
        """
        Args:
            slot_config_path:      Where to save the discovered slot layout.
            min_detections_to_map: Minimum stationary detections before clustering.
                                   Unlike frame count, this scales with how many
                                   parked cars are seen — sparse lots take longer,
                                   full lots map quickly.
            eps_pixels:            DBSCAN neighbourhood radius (pixels).
            min_samples:           Minimum stationary hits per cluster.
                                   Higher = less sensitive to briefly-stopped cars.
            infer_empty_slots:     Whether to infer empty slots from row gaps.
            row_merge_tolerance:   Y-proximity threshold for row grouping
                                   (fraction of slot height).
            gap_tolerance:         How precisely a gap must match slot spacing.
            max_gap_multiplier:    Gaps wider than this × slot size = drive lane.
        """
        self.config_path      = slot_config_path
        self.min_detections   = min_detections_to_map
        self.eps              = eps_pixels
        self.min_samples      = min_samples
        self.infer_empty      = infer_empty_slots
        self.row_merge_tol    = row_merge_tolerance
        self.gap_tolerance    = gap_tolerance
        self.max_gap_mult     = max_gap_multiplier

        # Accumulated stationary detections: list of (cx, cy, x1, y1, x2, y2)
        self._detections: list = []

        # Per-vehicle tracking for stationarity check.
        # Key: approximate center tuple, Value: {cx, cy, x1,y1,x2,y2, still_count}
        self._tracked: dict    = {}

        self._frame_count      = 0
        self._slots: dict      = {}

        if os.path.exists(slot_config_path):
            self._load_config()
            log.info(f"Loaded existing slot config: {len(self._slots)} slots")

    # ------------------------------------------------------------------
    def feed_frame(self, vehicle_boxes: list, frame_shape: tuple):
        """
        Process one frame. Only stationary vehicles contribute to slot mapping.

        Args:
            vehicle_boxes: List of [x1, y1, x2, y2] from YOLO.
            frame_shape:   (height, width, channels).
        """
        self._frame_count += 1

        current_centers = []
        for box in vehicle_boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            current_centers.append((cx, cy, x1, y1, x2, y2))

        # ── Match current detections to tracked vehicles ──────────────────────
        new_tracked = {}
        for cx, cy, x1, y1, x2, y2 in current_centers:
            # Find the closest previously tracked vehicle
            matched_key = None
            best_dist   = STATIONARY_THRESHOLD_PX * 3  # search radius

            for key, prev in self._tracked.items():
                dist = np.hypot(cx - prev["cx"], cy - prev["cy"])
                if dist < best_dist:
                    best_dist   = dist
                    matched_key = key

            if matched_key is not None:
                prev = self._tracked[matched_key]
                movement = np.hypot(cx - prev["cx"], cy - prev["cy"])

                if movement < STATIONARY_THRESHOLD_PX:
                    # Vehicle hasn't moved — increment stationary counter
                    still_count = prev["still_count"] + 1
                else:
                    # Vehicle moved — reset counter (driving through)
                    still_count = 0

                new_tracked[(round(cx), round(cy))] = {
                    "cx": cx, "cy": cy,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "still_count": still_count,
                }

                # Only accumulate once the vehicle has been still long enough
                if still_count == MIN_STATIONARY_FRAMES:
                    self._detections.append((cx, cy, x1, y1, x2, y2))
                    log.debug(
                        f"[MAPPER] Stationary vehicle confirmed at "
                        f"({round(cx)}, {round(cy)}) — accumulated "
                        f"(total: {len(self._detections)})"
                    )
            else:
                # New vehicle — start tracking, not yet stationary
                new_tracked[(round(cx), round(cy))] = {
                    "cx": cx, "cy": cy,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "still_count": 0,
                }

        self._tracked = new_tracked

        # ── Trigger clustering once enough stationary detections accumulated ──
        if len(self._detections) >= self.min_detections and not self._slots:
            log.info(
                f"[MAPPER] {len(self._detections)} stationary detections over "
                f"{self._frame_count} frames — running clustering."
            )
            self._run_clustering(frame_shape)

    # ------------------------------------------------------------------
    def _run_clustering(self, frame_shape: tuple):
        if len(self._detections) < self.min_samples:
            log.warning("Not enough stationary detections to build slot map.")
            return

        centers = np.array([[d[0], d[1]] for d in self._detections])
        labels  = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(centers)

        cluster_ids = sorted(set(labels) - {-1})
        log.info(
            f"DBSCAN: {len(cluster_ids)} slot clusters from "
            f"{len(self._detections)} stationary detections."
        )

        slots        = {}
        cluster_data = []

        for i, cid in enumerate(cluster_ids):
            mask = labels == cid
            boxes = np.array([
                [d[2], d[3], d[4], d[5]]
                for d, m in zip(self._detections, mask) if m
            ])
            avg_box = boxes.mean(axis=0).tolist()
            cx = (avg_box[0] + avg_box[2]) / 2
            cy = (avg_box[1] + avg_box[3]) / 2
            slot_id = f"S{i+1:02d}"
            slots[slot_id] = {
                "coords": [round(v) for v in avg_box],
                "row":    self._infer_row(cy, frame_shape[0]),
                "source": "detected",
            }
            cluster_data.append((cx, cy, avg_box))

        if self.infer_empty and len(cluster_data) >= 2:
            inferred = self._infer_by_row(cluster_data, frame_shape, len(slots))
            slots.update(inferred)
            if inferred:
                log.info(f"Row inference added {len(inferred)} empty slot(s).")

        self._slots = slots
        self._save_config()

    # ------------------------------------------------------------------
    def _infer_by_row(
        self, cluster_data: list, frame_shape: tuple, next_index: int
    ) -> dict:
        """
        Group detected slots into rows by Y proximity, then fill
        exactly-one-slot-sized gaps within each row only.
        """
        boxes   = np.array([d[2] for d in cluster_data])
        widths  = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        med_h   = float(np.median(heights))

        y_threshold = med_h * self.row_merge_tol
        points = [(d[0], d[1], d[2]) for d in cluster_data]
        points.sort(key=lambda p: p[1])

        rows: list[list] = []
        for pt in points:
            placed = False
            for row in rows:
                row_cy = np.mean([r[1] for r in row])
                if abs(pt[1] - row_cy) < y_threshold:
                    row.append(pt)
                    placed = True
                    break
            if not placed:
                rows.append([pt])

        log.info(
            f"Row grouping: {len(cluster_data)} clusters → {len(rows)} rows "
            f"(y_threshold={y_threshold:.0f}px)"
        )

        inferred  = {}
        seen_cx   = set(round(d[0]) for d in cluster_data)
        seen_cy   = set(round(d[1]) for d in cluster_data)
        max_dist  = float(np.median(widths + heights)) * self.max_gap_mult

        for row_idx, row in enumerate(rows):
            if len(row) < 2:
                continue
            row.sort(key=lambda p: p[0])

            for k in range(len(row) - 1):
                cx_a, cy_a, box_a = row[k]
                cx_b, cy_b, box_b = row[k + 1]
                dist = np.hypot(cx_b - cx_a, cy_b - cy_a)

                if dist > max_dist:
                    log.debug(f"  Row {row_idx}: gap {dist:.0f}px > max {max_dist:.0f}px — drive lane")
                    continue

                w_avg = ((box_a[2]-box_a[0]) + (box_b[2]-box_b[0])) / 2
                h_avg = ((box_a[3]-box_a[1]) + (box_b[3]-box_b[1])) / 2
                expected = np.hypot(w_avg, h_avg) * 1.1
                tol      = expected * self.gap_tolerance

                if abs(dist - expected * 2) >= tol * 2:
                    continue

                mid_cx = (cx_a + cx_b) / 2
                mid_cy = (cy_a + cy_b) / 2

                too_close = any(
                    abs(mid_cx - sx) < w_avg * 0.35 and abs(mid_cy - sy) < h_avg * 0.35
                    for sx, sy in zip(seen_cx, seen_cy)
                )
                if too_close:
                    continue

                slot_id = f"S{next_index + 1:02d}"
                next_index += 1
                inferred[slot_id] = {
                    "coords": [
                        round(mid_cx - w_avg/2), round(mid_cy - h_avg/2),
                        round(mid_cx + w_avg/2), round(mid_cy + h_avg/2),
                    ],
                    "row":    self._infer_row(mid_cy, frame_shape[0]),
                    "source": "inferred",
                }
                seen_cx.add(round(mid_cx))
                seen_cy.add(round(mid_cy))
                log.info(
                    f"  → Inferred {slot_id} at ({round(mid_cx)}, {round(mid_cy)}) "
                    f"row={row_idx} gap={dist:.0f}px"
                )

        return inferred

    # ------------------------------------------------------------------
    def _infer_row(self, y_center: float, frame_height: int) -> str:
        ratio = y_center / frame_height
        if ratio < 0.33:  return "A"
        elif ratio < 0.66: return "B"
        return "C"

    def _save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self._slots, f, indent=2)
        log.info(f"Slot config saved → {self.config_path} ({len(self._slots)} slots)")

    def _load_config(self):
        with open(self.config_path) as f:
            self._slots = json.load(f)

    def is_mapping_complete(self) -> bool:
        return bool(self._slots)

    def get_slots(self) -> dict:
        return self._slots

    def add_slot_manual(self, slot_id: str, coords: list):
        self._slots[slot_id] = {"coords": coords, "row": "M", "source": "manual"}
        self._save_config()

    def remove_slot(self, slot_id: str):
        self._slots.pop(slot_id, None)
        self._save_config()