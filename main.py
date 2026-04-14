"""
main.py — AI-Powered Smart Parking System
Main entry point — runs on Raspberry Pi 5

⚠️  DEPRECATED — Fix #5: flask_api.py is the unified entry point and includes
the full detection loop, MJPEG stream, slot editor API, distortion panel, and
Firebase sync. You should run:
    python3 flask_api.py
instead of this file.

This file is kept only as a reference / fallback for headless (no-API) use.
"""

import time
import logging
from detector import ParkingDetector
from firebase_sync import FirebaseSync
from auto_mapper import AutoMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

RTSP_URL     = "rtsp://admin:Skibidi1@192.168.1.142:554/Streaming/Channels/101"
FIREBASE_URL = "https://automapping-parking-slot-default-rtdb.asia-southeast1.firebasedatabase.app"
CREDENTIALS  = "serviceAccountKey.json"

# ── Undistortion tuning — run tune_undistort.py first to find good values ────
FOV_DEGREES = 185.0   # Adjust until cars look like flat rectangles
ZOOM        = 0.7     # Adjust to control how much of the frame is kept
# ─────────────────────────────────────────────────────────────────────────────


def main():
    log.info("=== AI Smart Parking System Starting ===")

    # 1. Connect to Firebase
    firebase = FirebaseSync(
        credentials_path=CREDENTIALS,
        database_url=FIREBASE_URL,
    )

    # 2. Initialize auto-mapper
    mapper = AutoMapper(
        slot_config_path="slot_config.json",
        min_frames_to_map=150,   # Fix #1/#5: parameter was misnamed min_frames_to_map before
        min_samples=3,
    )

    # 3. Initialize YOLO detector with fisheye undistortion
    detector = ParkingDetector(
        model_path="yolov8n.pt",
        rtsp_url=RTSP_URL,
        confidence=0.20,
        target_classes=[2, 5, 7],   # COCO: car, bus, truck
        undistort=False,              # Enable fisheye correction
        fov_degrees=FOV_DEGREES,
        zoom=ZOOM,
    )

    log.info("All modules initialized. Starting detection loop...")

    frame_count      = 0
    mapping_complete = mapper.is_mapping_complete()

    if mapping_complete:
        log.info(f"Existing slot map loaded — {len(mapper.get_slots())} slots. Skipping mapping phase.")
    else:
        log.info("No slot map found — starting auto-mapping phase (~150 frames)...")

    # Save an undistortion preview on the very first frame
    sample_saved = False

    while True:
        try:
            frame = detector.capture_frame()
            if frame is None:
                log.warning("No frame captured — retrying in 1s")
                time.sleep(1)
                continue

            # Save undistort sample once on startup for visual inspection
            if not sample_saved:
                detector.save_undistort_sample(frame, "undistort_sample.jpg")
                log.info("Saved undistort_sample.jpg — check this to verify undistortion looks correct.")
                sample_saved = True

            vehicle_boxes = detector.detect_vehicles(frame)

            # ── Phase 1: Auto-Mapping ─────────────────────────────────────
            if not mapping_complete:
                mapper.feed_frame(vehicle_boxes, frame.shape)
                mapping_complete = mapper.is_mapping_complete()

                if frame_count % 10 == 0:
                    log.info(f"Mapping... frame {frame_count}/150 | vehicles seen: {len(vehicle_boxes)}")

                if mapping_complete:
                    log.info(f"Auto-mapping complete! {len(mapper.get_slots())} slots discovered.")
                    firebase.push_slot_layout(mapper.get_slots())

                frame_count += 1
                time.sleep(1)
                continue

            # ── Phase 2: Occupancy Detection ──────────────────────────────
            slots    = mapper.get_slots()
            statuses = detector.compute_occupancy(vehicle_boxes, slots)
            detector.apply_smoothing(statuses)

            occupied = sum(1 for s in statuses.values() if s == "Occupied")
            vacant   = len(statuses) - occupied

            if frame_count % 2 == 0:
                firebase.push_occupancy(statuses)
                log.info(
                    f"Frame {frame_count} | "
                    f"Occupied: {occupied}/{len(statuses)} | "
                    f"Vacant: {vacant} | "
                    f"Vehicles detected: {len(vehicle_boxes)}"
                )

            frame_count += 1
            time.sleep(1)

        except KeyboardInterrupt:
            log.info("Shutdown requested by user.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)
            time.sleep(2)

    detector.release()
    log.info("System stopped.")


if __name__ == "__main__":
    main()
