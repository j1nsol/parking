"""
auto_mapper.py — Automatically discovers parking slot layout from video.

Strategy:
  1. Track each detected vehicle across consecutive frames using stable integer IDs.
  2. Only accumulate a detection if the vehicle has been stationary for
     at least MIN_STATIONARY_FRAMES consecutive frames. This filters out
     cars driving through lanes, which move significantly between frames.
  3. Continue accumulating every ACCUM_EVERY frames while a car stays parked,
     so long-parked cars build stronger DBSCAN clusters than briefly-stopped ones.
  4. DBSCAN-cluster the stationary detections → each cluster = one slot.
  5. Row-based empty slot inference fills gaps within rows only.
  6. Save to slot_config.json.

Key fixes over previous versions:
  - Tracking keys are stable integers, NOT position tuples. Position-based keys
    change every frame due to detection jitter, breaking the still_count counter.
  - Accumulation uses >= + modulo (ACCUM_EVERY) instead of ==, so a car parked
    all day contributes many detections, not just one at frame 5.
  - used_keys set prevents two detections claiming the same tracked vehicle.
"""

import json
import os
import numpy as np
import logging
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)

# Center movement below this = same stationary vehicle (pixels)
STATIONARY_THRESHOLD_PX = 20

# Consecutive stationary frames required before first accumulation
MIN_STATIONARY_FRAMES = 5

# After confirmed stationary, accumulate one detection every N frames.
# Lower = denser clusters, faster mapping. Higher = less data but still correct.
ACCUM_EVERY = 10


class AutoMapper:
    def __init__(
        self,
        slot_config_path: str = "slot_config.json",
        min_detections_to_map: int = 50,
        eps_pixels: int = 60,
        min_samples: int = 5,
        infer_empty_slots: bool = True,
        row_merge_tolerance: float = 0.6,
        gap_tolerance: float = 0.22,
        max_gap_multiplier: float = 2.3,
    ):
        self.config_path    = slot_config_path
        self.min_detections = min_detections_to_map
        self.eps            = eps_pixels
        self.min_samples    = min_samples
        self.infer_empty    = infer_empty_slots
        self.row_merge_tol  = row_merge_tolerance
        self.gap_tolerance  = gap_tolerance
        self.max_gap_mult   = max_gap_multiplier

        self._detections: list = []
        self._tracked: dict    = {}   # int ID → {cx,cy,x1,y1,x2,y2,still_count}
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

        # ── Match detections to tracked vehicles by nearest centre ────────────
        new_tracked = {}
        used_ids    = set()

        for cx, cy, x1, y1, x2, y2 in current:
            best_id   = None
            best_dist = STATIONARY_THRESHOLD_PX * 3   # max match radius

            for tid, prev in self._tracked.items():
                if tid in used_ids:
                    continue
                d = np.hypot(cx - prev["cx"], cy - prev["cy"])
                if d < best_dist:
                    best_dist = d
                    best_id   = tid

            if best_id is not None:
                used_ids.add(best_id)
                prev     = self._tracked[best_id]
                movement = np.hypot(cx - prev["cx"], cy - prev["cy"])
                still    = prev["still_count"] + 1 if movement < STATIONARY_THRESHOLD_PX else 0

                new_tracked[best_id] = {
                    "cx": cx, "cy": cy,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "still_count": still,
                }

                # Accumulate once confirmed stationary, then every ACCUM_EVERY frames
                if still >= MIN_STATIONARY_FRAMES and still % ACCUM_EVERY == 0:
                    self._detections.append((cx, cy, x1, y1, x2, y2))
                    log.debug(
                        f"[MAPPER] Stationary hit id={best_id} "
                        f"({round(cx)},{round(cy)}) still={still} "
                        f"total={len(self._detections)}"
                    )
            else:
                # New vehicle — assign a fresh stable integer ID
                new_tracked[self._next_id] = {
                    "cx": cx, "cy": cy,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "still_count": 0,
                }
                self._next_id += 1

        self._tracked = new_tracked

        # ── Trigger clustering once enough stationary detections ──────────────
        if len(self._detections) >= self.min_detections and not self._slots:
            log.info(
                f"[MAPPER] Clustering triggered — {len(self._detections)} "
                f"stationary detections over {self._frame_count} frames."
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
        log.info(f"DBSCAN: {len(cluster_ids)} clusters from {len(self._detections)} detections.")

        slots        = {}
        cluster_data = []

        for i, cid in enumerate(cluster_ids):
            mask  = labels == cid
            boxes = np.array([
                [d[2], d[3], d[4], d[5]]
                for d, m in zip(self._detections, mask) if m
            ])
            avg   = boxes.mean(axis=0).tolist()
            cx    = (avg[0] + avg[2]) / 2
            cy    = (avg[1] + avg[3]) / 2
            sid   = f"S{i+1:02d}"
            slots[sid] = {
                "coords": [round(v) for v in avg],
                "row":    self._infer_row(cy, frame_shape[0]),
                "source": "detected",
            }
            cluster_data.append((cx, cy, avg))

        if self.infer_empty and len(cluster_data) >= 2:
            inferred = self._infer_by_row(cluster_data, frame_shape, len(slots))
            slots.update(inferred)
            if inferred:
                log.info(f"Row inference added {len(inferred)} empty slot(s).")

        self._slots = slots
        self._save_config()

    # ------------------------------------------------------------------
    def _infer_by_row(self, cluster_data, frame_shape, next_index):
        boxes   = np.array([d[2] for d in cluster_data])
        widths  = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        med_h   = float(np.median(heights))
        max_dist = float(np.median(widths + heights)) * self.max_gap_mult

        # Group into rows by Y proximity
        y_thr  = med_h * self.row_merge_tol
        points = sorted([(d[0], d[1], d[2]) for d in cluster_data], key=lambda p: p[1])
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
        seen_cx  = set(round(d[0]) for d in cluster_data)
        seen_cy  = set(round(d[1]) for d in cluster_data)

        for row_idx, row in enumerate(rows):
            if len(row) < 2:
                continue
            row.sort(key=lambda p: p[0])

            for k in range(len(row) - 1):
                cx_a, cy_a, box_a = row[k]
                cx_b, cy_b, box_b = row[k+1]
                dist = np.hypot(cx_b - cx_a, cy_b - cy_a)

                if dist > max_dist:
                    log.debug(f"Row {row_idx}: gap {dist:.0f}px > max {max_dist:.0f}px — lane")
                    continue

                w_avg    = ((box_a[2]-box_a[0]) + (box_b[2]-box_b[0])) / 2
                h_avg    = ((box_a[3]-box_a[1]) + (box_b[3]-box_b[1])) / 2
                expected = np.hypot(w_avg, h_avg) * 1.1
                tol      = expected * self.gap_tolerance

                if abs(dist - expected * 2) >= tol * 2:
                    continue

                mid_cx = (cx_a + cx_b) / 2
                mid_cy = (cy_a + cy_b) / 2

                if any(abs(mid_cx-sx) < w_avg*0.35 and abs(mid_cy-sy) < h_avg*0.35
                       for sx, sy in zip(seen_cx, seen_cy)):
                    continue

                sid = f"S{next_index+1:02d}"
                next_index += 1
                inferred[sid] = {
                    "coords": [
                        round(mid_cx - w_avg/2), round(mid_cy - h_avg/2),
                        round(mid_cx + w_avg/2), round(mid_cy + h_avg/2),
                    ],
                    "row":    self._infer_row(mid_cy, frame_shape[0]),
                    "source": "inferred",
                }
                seen_cx.add(round(mid_cx))
                seen_cy.add(round(mid_cy))
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

    def add_slot_manual(self, slot_id, coords):
        self._slots[slot_id] = {"coords": coords, "row": "M", "source": "manual"}
        self._save_config()

    def remove_slot(self, slot_id):
        self._slots.pop(slot_id, None)
        self._save_config()