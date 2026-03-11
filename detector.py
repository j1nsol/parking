"""
detector.py — YOLO vehicle detection + occupancy logic
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import logging
import os

from undistort import FisheyeUndistorter

log = logging.getLogger(__name__)

RTSP_URL = "rtsp://admin:Skibidi1@192.168.1.142:554/Streaming/Channels/101"

# Force TCP transport BEFORE any VideoCapture call
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class ParkingDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        rtsp_url: str = RTSP_URL,
        confidence: float = 0.20,
        target_classes: list = None,
        smoothing_window: int = 5,
        undistort: bool = False,
        fov_degrees: float = 185.0,
        zoom: float = 0.7,
    ):
        """
        Args:
            model_path:       YOLO model file (.pt).
            rtsp_url:         RTSP stream URL for the IP camera.
            confidence:       Minimum detection confidence threshold.
            target_classes:   COCO class IDs to detect (2=car, 5=bus, 7=truck).
            smoothing_window: Frames to average for temporal smoothing.
            undistort:        Enable fisheye undistortion before YOLO.
            fov_degrees:      Estimated camera FOV. Tune if result looks wrong.
            zoom:             Zoom level for undistortion (0.5-1.0).
        """
        self.conf = confidence
        self.target_classes = target_classes or [2, 5, 7]
        self.smoothing_window = smoothing_window
        self.rtsp_url = rtsp_url

        # Load YOLO model
        log.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        # Open RTSP stream via TCP
        log.info(f"Connecting to RTSP stream (TCP): {rtsp_url}")
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {rtsp_url}")
        log.info("RTSP stream opened successfully.")

        # Fisheye undistorter
        if undistort:
            self.undistorter = FisheyeUndistorter(fov_degrees=fov_degrees, zoom=zoom)
            log.info("Fisheye undistortion enabled.")
        else:
            self.undistorter = None
            log.info("Fisheye undistortion disabled.")

        # Smoothing history: slot_id -> deque of recent statuses
        self._history: dict[str, deque] = {}

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------
    def capture_frame(self) -> np.ndarray | None:
        """Capture one frame from the RTSP stream, reconnecting if dropped."""
        ret, frame = self.cap.read()
        if not ret:
            log.warning("RTSP stream lost — reconnecting (TCP)...")
            self.cap.release()
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = self.cap.read()
        return frame if ret else None

    # ------------------------------------------------------------------
    # Undistort (fisheye correction)
    # ------------------------------------------------------------------
    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply fisheye correction if enabled, otherwise pass through."""
        if self.undistorter is not None and frame is not None:
            return self.undistorter.process(frame)
        return frame

    def save_undistort_sample(self, frame: np.ndarray, path: str = "undistort_sample.jpg"):
        """Save a before/after comparison image for tuning."""
        if self.undistorter:
            self.undistorter.save_sample(frame, path)

    # ------------------------------------------------------------------
    # Vehicle detection
    # ------------------------------------------------------------------
    def detect_vehicles(self, frame: np.ndarray) -> list[list[float]]:
        """
        Undistort frame then run YOLO to return bounding boxes for vehicles.

        Returns:
            List of [x1, y1, x2, y2] boxes (pixel coordinates).
        """
        frame = self.undistort_frame(frame)

        results = self.model(frame, conf=self.conf, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append([x1, y1, x2, y2])
        return boxes

    # ------------------------------------------------------------------
    # Occupancy logic
    # ------------------------------------------------------------------
    def compute_occupancy(
        self,
        vehicle_boxes: list[list[float]],
        slots: dict,
        iou_threshold: float = 0.35,
    ) -> dict[str, str]:
        """
        Compare detected vehicles against known parking slots.

        Args:
            vehicle_boxes: Output of detect_vehicles().
            slots: Dict mapping slot_id -> {"coords": [x1,y1,x2,y2]}.
            iou_threshold: Overlap fraction to consider a slot occupied.

        Returns:
            Dict mapping slot_id -> "Occupied" | "Vacant"
        """
        statuses = {}
        for slot_id, slot_data in slots.items():
            sx1, sy1, sx2, sy2 = slot_data["coords"]
            slot_area = max(1, (sx2 - sx1) * (sy2 - sy1))
            occupied = False

            for vbox in vehicle_boxes:
                vx1, vy1, vx2, vy2 = vbox
                ix1 = max(sx1, vx1)
                iy1 = max(sy1, vy1)
                ix2 = min(sx2, vx2)
                iy2 = min(sy2, vy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter / slot_area >= iou_threshold:
                    occupied = True
                    break

            statuses[slot_id] = "Occupied" if occupied else "Vacant"
        return statuses

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------
    def apply_smoothing(self, statuses: dict[str, str]) -> None:
        """
        Mutates `statuses` in place using majority-vote over recent frames.
        Prevents flickering when a car is momentarily missed.
        """
        for slot_id, status in statuses.items():
            if slot_id not in self._history:
                self._history[slot_id] = deque(maxlen=self.smoothing_window)
            self._history[slot_id].append(1 if status == "Occupied" else 0)
            majority = sum(self._history[slot_id]) > (self.smoothing_window // 2)
            statuses[slot_id] = "Occupied" if majority else "Vacant"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def release(self):
        """Release the RTSP stream."""
        self.cap.release()
        log.info("RTSP stream released.")