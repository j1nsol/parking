"""
auto_mapper.py — Automatically discovers parking slot layout from video.

How it works:
  1. Collect bounding boxes of detected vehicles over many frames.
  2. Cluster box centers with DBSCAN to find occupied parking positions.
  3. Average the boxes in each cluster → slot boundary.
  4. Infer likely empty slots from spatial gaps in the cluster grid.
  5. Save results to slot_config.json.

Key improvements over v1:
  - Clustering triggers at >= min_frames (not == min_frames) so a reset
    mid-count still works correctly.
  - Empty slot inference: after finding occupied clusters, the mapper
    analyses the spatial grid and fills in slot-sized gaps. This means
    a slot that was never occupied during mapping is still discovered,
    which is essential for demos where not all spots are filled.
"""

import json
import os
import numpy as np
import logging
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)


class AutoMapper:
    def __init__(
        self,
        slot_config_path: str = "slot_config.json",
        min_frames_to_map: int = 150,
        eps_pixels: int = 60,
        min_samples: int = 3,
        infer_empty_slots: bool = True,
        gap_tolerance: float = 0.20,   # tighter: gap must be within 20% of expected distance
        max_gap_multiplier: float = 2.4,  # gaps wider than 2.4× slot size = drive lane, skip
    ):
        """
        Args:
            slot_config_path:    Where to save the discovered slot layout.
            min_frames_to_map:   Minimum frames before mapping is attempted.
            eps_pixels:          DBSCAN neighbourhood radius.
            min_samples:         Minimum detections per cluster to count as a slot.
            infer_empty_slots:   If True, infer slot-sized gaps as empty slots.
            gap_tolerance:       Max fractional error allowed when matching gap to
                                 expected slot spacing. 0.20 = must be within 20%.
                                 Tighter = less false positives in drive lanes.
            max_gap_multiplier:  Hard cap — pairs further apart than this multiple
                                 of slot size are treated as separated by a drive
                                 lane and are never used for inference.
        """
        self.config_path       = slot_config_path
        self.min_frames        = min_frames_to_map
        self.eps               = eps_pixels
        self.min_samples       = min_samples
        self.infer_empty       = infer_empty_slots
        self.gap_tolerance     = gap_tolerance
        self.max_gap_mult      = max_gap_multiplier

        self._detections: list = []
        self._frame_count      = 0
        self._slots: dict      = {}

        if os.path.exists(slot_config_path):
            self._load_config()
            log.info(f"Loaded existing slot config: {len(self._slots)} slots")

    # ------------------------------------------------------------------
    def feed_frame(self, vehicle_boxes: list, frame_shape: tuple):
        """
        Accumulate vehicle detections. Clustering runs automatically once
        min_frames have been collected.

        Args:
            vehicle_boxes: List of [x1, y1, x2, y2] boxes from YOLO.
            frame_shape:   (height, width, channels) of the frame.
        """
        self._frame_count += 1
        for box in vehicle_boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self._detections.append((cx, cy, x1, y1, x2, y2))

        # Use >= so this still fires even if frame_count overshoots min_frames
        if self._frame_count >= self.min_frames and not self._slots:
            self._run_clustering(frame_shape)

    # ------------------------------------------------------------------
    def _run_clustering(self, frame_shape: tuple):
        if len(self._detections) < self.min_samples:
            log.warning("Not enough detections to build slot map.")
            return

        centers = np.array([[d[0], d[1]] for d in self._detections])
        labels  = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(centers)

        cluster_ids = sorted(set(labels) - {-1})
        log.info(f"DBSCAN found {len(cluster_ids)} occupied slot clusters from "
                 f"{len(self._detections)} detections over {self._frame_count} frames.")

        slots = {}
        cluster_boxes = []  # store for gap inference

        for i, cid in enumerate(cluster_ids):
            mask = labels == cid
            boxes = np.array([
                [d[2], d[3], d[4], d[5]]
                for d, m in zip(self._detections, mask) if m
            ])
            avg_box = boxes.mean(axis=0).tolist()
            slot_id = f"S{i+1:02d}"
            slots[slot_id] = {
                "coords": [round(v) for v in avg_box],
                "row":    self._infer_row(avg_box[1], frame_shape[0]),
                "source": "detected",
            }
            cluster_boxes.append(avg_box)

        # ── Infer empty slots from spatial gaps ───────────────────────────────
        if self.infer_empty and len(cluster_boxes) >= 2:
            inferred = self._infer_empty_slots(cluster_boxes, frame_shape, len(slots))
            for sid, sdata in inferred.items():
                slots[sid] = sdata
            if inferred:
                log.info(f"Inferred {len(inferred)} additional empty slot(s) from spatial gaps.")

        self._slots = slots
        self._save_config()

    # ------------------------------------------------------------------
    def _infer_empty_slots(
        self, cluster_boxes: list, frame_shape: tuple, next_index: int
    ) -> dict:
        """
        Look for slot-sized gaps between discovered clusters and add them
        as empty slots. Works by:

        1. Computing median slot width (W) and height (H) from existing clusters.
        2. For each pair of nearby clusters, checking if the gap between their
           centers is close to W or H (i.e. one slot-width apart with nothing
           detected in between).
        3. Placing a new slot at the midpoint of the gap with the same W/H.

        This is intentionally conservative — it only infers gaps that look
        exactly like "one slot is missing" between two detected slots.
        """
        boxes  = np.array(cluster_boxes)   # shape (N, 4): x1 y1 x2 y2
        widths  = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        med_w  = float(np.median(widths))
        med_h  = float(np.median(heights))

        centers = np.array([
            [(b[0]+b[2])/2, (b[1]+b[3])/2] for b in cluster_boxes
        ])

        inferred = {}
        seen_positions = set(map(tuple, centers.round(0).tolist()))

        # Hard cap — any pair further apart than this is separated by a lane, not a slot
        max_allowed_dist = max(med_w, med_h) * self.max_gap_mult

        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dx = centers[j][0] - centers[i][0]
                dy = centers[j][1] - centers[i][1]
                dist = np.hypot(dx, dy)

                # Reject immediately if too far apart — it's a drive lane gap
                if dist > max_allowed_dist:
                    continue

                # Check if gap ≈ 2× slot width (H) or 2× slot height (V)
                # meaning exactly one slot is missing between them.
                # gap_tolerance=0.20 means dist must be within 20% of expected.
                for expected, dim_w, dim_h, direction in [
                    (med_w * 2, med_w, med_h, "H"),
                    (med_h * 2, med_w, med_h, "V"),
                ]:
                    tol = expected * self.gap_tolerance
                    if abs(dist - expected) < tol:
                        # Gap midpoint = where the missing slot center should be
                        mid_cx = (centers[i][0] + centers[j][0]) / 2
                        mid_cy = (centers[i][1] + centers[j][1]) / 2
                        mid_key = (round(mid_cx), round(mid_cy))

                        # Skip if a cluster already exists near this midpoint
                        already_exists = any(
                            abs(mid_cx - p[0]) < med_w * 0.4 and
                            abs(mid_cy - p[1]) < med_h * 0.4
                            for p in seen_positions
                        )
                        if already_exists or mid_key in seen_positions:
                            continue

                        seen_positions.add(mid_key)
                        x1 = mid_cx - dim_w / 2
                        y1 = mid_cy - dim_h / 2
                        x2 = mid_cx + dim_w / 2
                        y2 = mid_cy + dim_h / 2

                        slot_id = f"S{next_index+1:02d}"
                        next_index += 1
                        inferred[slot_id] = {
                            "coords": [round(x1), round(y1), round(x2), round(y2)],
                            "row":    self._infer_row(mid_cy, frame_shape[0]),
                            "source": "inferred",
                        }
                        log.info(
                            f"  → Inferred {slot_id} at ({round(mid_cx)}, {round(mid_cy)}) "
                            f"direction={direction} gap={round(dist)}px expected={round(expected)}px"
                        )
                        break   # only infer one slot per pair

        return inferred

    # ------------------------------------------------------------------
    def _infer_row(self, y_center: float, frame_height: int) -> str:
        ratio = y_center / frame_height
        if ratio < 0.33:
            return "A"
        elif ratio < 0.66:
            return "B"
        return "C"

    # ------------------------------------------------------------------
    def _save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self._slots, f, indent=2)
        log.info(f"Slot config saved → {self.config_path} ({len(self._slots)} slots)")

    def _load_config(self):
        with open(self.config_path) as f:
            self._slots = json.load(f)

    # ------------------------------------------------------------------
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