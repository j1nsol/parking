"""
AI-Powered Smart Parking System
Main entry point — runs on Raspberry Pi 5
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


def main():
    log.info("=== AI Smart Parking System Starting ===")

    # 1. Connect to Firebase
    firebase = FirebaseSync(
        credentials_path="serviceAccountKey.json",
        database_url="https://automapping-parking-slot-default-rtdb.asia-southeast1.firebasedatabase.app"
    )

    # 2. Initialize auto-mapper (discovers slot layout from video)
    mapper = AutoMapper(
        slot_config_path="slot_config.json",
        min_frames_to_map=300     # ~5 minutes at 1 FPS
    )

    # 3. Initialize YOLO detector
    detector = ParkingDetector(
        model_path="yolov5n.pt",   # downloads automatically on first run
        camera_index=0,            # USB camera
        confidence=0.45,
        target_classes=[2, 5, 7],  # COCO: car, bus, truck
    )

    log.info("All modules initialized. Starting detection loop...")

    frame_count = 0
    mapping_complete = mapper.is_mapping_complete()

    while True:
        try:
            frame = detector.capture_frame()
            if frame is None:
                log.warning("No frame captured — retrying in 1s")
                time.sleep(1)
                continue

            # --- Phase 1: Auto-Mapping (first ~5 min) ---
            if not mapping_complete:
                vehicle_boxes = detector.detect_vehicles(frame)
                mapper.feed_frame(vehicle_boxes, frame.shape)
                mapping_complete = mapper.is_mapping_complete()
                if mapping_complete:
                    log.info("Auto-mapping complete! Slot layout saved.")
                    firebase.push_slot_layout(mapper.get_slots())
                frame_count += 1
                continue

            # --- Phase 2: Occupancy Detection ---
            vehicle_boxes = detector.detect_vehicles(frame)
            slots = mapper.get_slots()
            statuses = detector.compute_occupancy(vehicle_boxes, slots)

            # Temporal smoothing — avoids flickering
            detector.apply_smoothing(statuses)

            # Push to Firebase every 2 seconds
            if frame_count % 2 == 0:
                firebase.push_occupancy(statuses)
                log.info(
                    f"Frame {frame_count} | "
                    f"Occupied: {sum(1 for s in statuses.values() if s == 'Occupied')} / {len(statuses)}"
                )

            frame_count += 1
            time.sleep(1)   # 1 FPS is enough for parking detection

        except KeyboardInterrupt:
            log.info("Shutdown requested.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)
            time.sleep(2)

    detector.release()
    log.info("System stopped.")


if __name__ == "__main__":
    main()
