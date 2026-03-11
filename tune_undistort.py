"""
tune_undistort.py — Preview fisheye undistortion on a single frame.

Run this BEFORE main.py to visually check the undistortion result
and tune fov_degrees / zoom until cars look like normal top-down rectangles.

Usage:
    python3 tune_undistort.py

Output:
    undistort_sample.jpg  — side-by-side original vs undistorted
    undistort_result.jpg  — undistorted frame only (what YOLO will see)
"""

import cv2
import os
import logging
from undistort import FisheyeUndistorter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RTSP_URL = "rtsp://admin:Skibidi1@192.168.1.142:554/Streaming/Channels/101"

# ── Tune these two values until cars look like flat rectangles ────────────────
FOV_DEGREES = 185.0   # Try: 160, 175, 185, 200 — lower = less correction
ZOOM        = 0.7     # Try: 0.5, 0.6, 0.7, 0.8 — lower = more zoomed in
# ─────────────────────────────────────────────────────────────────────────────

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def main():
    log.info(f"Connecting to RTSP stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        log.error("Could not open RTSP stream. Check camera IP and credentials.")
        return

    log.info("Capturing frame...")
    # Grab a few frames to let the stream stabilise
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        log.error("Failed to capture frame.")
        return

    log.info(f"Frame captured: {frame.shape[1]}x{frame.shape[0]}")

    # Apply undistortion
    undistorter = FisheyeUndistorter(fov_degrees=FOV_DEGREES, zoom=ZOOM)

    # Save side-by-side comparison
    undistorter.save_sample(frame, "undistort_sample.jpg")
    log.info("Saved: undistort_sample.jpg  (original LEFT | undistorted RIGHT)")

    # Save undistorted-only result
    result = undistorter.process(frame)
    cv2.imwrite("undistort_result.jpg", result)
    log.info("Saved: undistort_result.jpg  (what YOLO will see)")

    log.info("---")
    log.info("If cars still look curved, try lowering FOV_DEGREES (e.g. 160)")
    log.info("If too much is cropped, try raising ZOOM (e.g. 0.8)")
    log.info("Edit FOV_DEGREES and ZOOM at the top of this file and re-run.")
    log.info("Once happy, update the same values in main.py ParkingDetector(...)")


if __name__ == "__main__":
    main()
