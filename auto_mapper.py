"""
auto_mapper.py — Automatically discovers parking slot layout from video.

How it works:
  1. Collect bounding boxes of stationary vehicles over many frames.
  2. Cluster box centers with DBSCAN to find "parking positions."
  3. Average the boxes in each cluster to get a slot boundary.
  4. Save results to slot_config.json for the detector to use.
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
        min_frames_to_map: int = 300,
        eps_pixels: int = 40,
        min_samples: int = 10,
    ):
        """
        Args:
            slot_config_path: Where to save/load the discovered slot layout.
            min_frames_to_map: Minimum frames before declaring mapping complete.
            eps_pixels: DBSCAN neighbourhood radius (pixels).
            min_samples: Minimum detections to form a valid parking slot cluster.
        """
        self.config_path = slot_config_path
        self.min_frames = min_frames_to_map
        self.eps = eps_pixels
        self.min_samples = min_samples

        # Accumulated detections: list of (cx, cy, x1, y1, x2, y2)
        self._detections: list[tuple] = []
        self._frame_count = 0
        self._slots: dict = {}

        # Load existing config if present
        if os.path.exists(slot_config_path):
            self._load_config()
            log.info(f"Loaded existing slot config: {len(self._slots)} slots")

    # ------------------------------------------------------------------
    # Feed frames during mapping phase
    # ------------------------------------------------------------------
    def feed_frame(self, vehicle_boxes: list[list[float]], frame_shape: tuple):
        """
        Accumulate vehicle detections for later clustering.

        Args:
            vehicle_boxes: Output of ParkingDetector.detect_vehicles().
            frame_shape: (height, width, channels) of the frame.
        """
        self._frame_count += 1
        for box in vehicle_boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self._detections.append((cx, cy, x1, y1, x2, y2))

        # Run clustering once we have enough data
        if self._frame_count == self.min_frames:
            self._run_clustering(frame_shape)

    # ------------------------------------------------------------------
    # DBSCAN clustering
    # ------------------------------------------------------------------
    def _run_clustering(self, frame_shape: tuple):
        if len(self._detections) < self.min_samples:
            log.warning("Not enough detections to build slot map.")
            return

        centers = np.array([[d[0], d[1]] for d in self._detections])
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(centers)

        cluster_ids = set(labels) - {-1}
        log.info(f"DBSCAN found {len(cluster_ids)} parking slot clusters.")

        slots = {}
        for i, cid in enumerate(sorted(cluster_ids)):
            mask = labels == cid
            cluster_boxes = np.array([
                [d[2], d[3], d[4], d[5]]
                for d, m in zip(self._detections, mask) if m
            ])
            # Average bounding box for the cluster
            avg_box = cluster_boxes.mean(axis=0).tolist()
            slot_id = f"S{i+1:02d}"
            slots[slot_id] = {
                "coords": [round(v) for v in avg_box],
                "row": self._infer_row(avg_box[1], frame_shape[0]),
            }

        self._slots = slots
        self._save_config()

    def _infer_row(self, y_center: float, frame_height: int) -> str:
        """Assign row label A/B/C based on vertical position in frame."""
        ratio = y_center / frame_height
        if ratio < 0.33:
            return "A"
        elif ratio < 0.66:
            return "B"
        return "C"

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------
    def _save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self._slots, f, indent=2)
        log.info(f"Slot config saved to {self.config_path} ({len(self._slots)} slots)")

    def _load_config(self):
        with open(self.config_path) as f:
            self._slots = json.load(f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_mapping_complete(self) -> bool:
        return bool(self._slots)

    def get_slots(self) -> dict:
        return self._slots

    def add_slot_manual(self, slot_id: str, coords: list[int]):
        """Allow admin to manually add/adjust a slot."""
        self._slots[slot_id] = {"coords": coords, "row": "M"}
        self._save_config()

    def remove_slot(self, slot_id: str):
        """Remove a slot (admin correction)."""
        self._slots.pop(slot_id, None)
        self._save_config()
