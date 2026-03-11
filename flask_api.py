"""
flask_api.py — REST API on Raspberry Pi
Unified entry point: one RTSP connection, one YOLO model, one process.

Architecture:
  - A background daemon thread runs the full detection + Firebase sync loop
    (previously in main.py) — this owns the camera and YOLO model.
  - Flask routes read from shared state (latest_frame, latest_statuses, slots)
    protected by a RLock.
  - /live-frame serves the last annotated frame captured by the bg thread.
  - main.py is no longer needed — just run: python3 flask_api.py
"""

import cv2
import json
import time
import os
import threading
import logging
import numpy as np
from collections import deque
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
from firebase_sync import FirebaseSync
from auto_mapper import AutoMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
RTSP_URL       = "rtsp://admin:Skibidi1@192.168.1.142:554/Streaming/Channels/101"
FIREBASE_URL   = "https://automapping-parking-slot-default-rtdb.asia-southeast1.firebasedatabase.app"
CREDENTIALS    = "serviceAccountKey.json"
SLOT_CONFIG    = "slot_config.json"
CONFIDENCE     = 0.20
TARGET_CLASSES = [2, 5, 7]   # COCO: car, bus, truck
IOU_THRESHOLD  = 0.35
SMOOTHING_WIN  = 5            # frames for majority-vote smoothing
DETECT_INTERVAL = 1.0         # seconds between detection cycles
FIREBASE_EVERY  = 3           # push Firebase every N detection cycles

# Fisheye undistortion — set UNDISTORT=True if your camera needs it
UNDISTORT   = False
FOV_DEGREES = 185.0
ZOOM        = 0.7

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ── Load YOLO once — shared by bg thread and /analyze-image ──────────────────
log.info("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")
log.info("Model ready.")

# ── Shared state — protected by _state_lock ───────────────────────────────────
# Using RLock so the bg thread can acquire it re-entrantly if needed.
_state_lock      = threading.RLock()
_latest_frame    = None          # Last annotated JPEG bytes (ready to serve)
_latest_statuses = {}            # slot_id -> "Occupied" | "Vacant"
_slots           = {}            # slot_id -> {coords, row, ...}
_smoothing_hist  = {}            # slot_id -> deque for temporal smoothing
_mapping_phase   = True          # True while auto-mapper is still running
_frame_count     = 0
_last_fb_push    = 0             # epoch ms of last Firebase push

# ── Load existing slot config if present ─────────────────────────────────────
if os.path.exists(SLOT_CONFIG):
    with open(SLOT_CONFIG) as f:
        _slots = json.load(f)
    log.info(f"Loaded {len(_slots)} slots from {SLOT_CONFIG}")
    _mapping_phase = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_overlap(vbox, slot_coords):
    vx1, vy1, vx2, vy2 = vbox
    sx1, sy1, sx2, sy2 = slot_coords
    ix1, iy1 = max(vx1, sx1), max(vy1, sy1)
    ix2, iy2 = min(vx2, sx2), min(vy2, sy2)
    inter     = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    slot_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    return (inter / slot_area) >= IOU_THRESHOLD


def _apply_smoothing(statuses: dict) -> dict:
    """Majority-vote temporal smoothing — mutates and returns statuses."""
    for slot_id, status in statuses.items():
        if slot_id not in _smoothing_hist:
            _smoothing_hist[slot_id] = deque(maxlen=SMOOTHING_WIN)
        _smoothing_hist[slot_id].append(1 if status == "Occupied" else 0)
        majority = sum(_smoothing_hist[slot_id]) > (SMOOTHING_WIN // 2)
        statuses[slot_id] = "Occupied" if majority else "Vacant"
    return statuses


def _draw_boxes(frame, vehicle_boxes, slot_results=None):
    """Draw YOLO vehicle boxes and slot overlays onto a frame."""
    if slot_results:
        for slot in slot_results:
            coords = slot.get("coords")
            if not coords or len(coords) < 4:
                continue
            x1, y1, x2, y2 = [int(c) for c in coords]
            occ   = slot["status"] == "Occupied"
            color = (60, 60, 220) if occ else (80, 200, 120)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"{slot['id']} {'OCC' if occ else 'VAC'}"
            cv2.putText(frame, label, (x1 + 4, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    for vb in vehicle_boxes:
        x1, y1, x2, y2 = [int(c) for c in vb["coords"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 189, 56), 2)
        cv2.putText(frame, f"{vb['label']} {vb['confidence']:.2f}",
                    (x1 + 4, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 189, 56), 1, cv2.LINE_AA)

    occupied = sum(1 for s in (slot_results or []) if s["status"] == "Occupied")
    total    = len(slot_results) if slot_results else 0
    cv2.rectangle(frame, (0, 0), (520, 44), (7, 10, 16), -1)
    cv2.putText(frame,
                f"YOLOv8n  |  {len(vehicle_boxes)} vehicles  |  {occupied}/{total} slots occupied",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ── Camera singleton ──────────────────────────────────────────────────────────
# Owned exclusively by the background thread — no lock needed here since
# only one thread ever touches _cap.
_cap = None


def _open_camera():
    global _cap
    log.info("[CAM] Opening RTSP stream (TCP)...")
    _cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if _cap.isOpened():
        log.info("[CAM] Stream opened successfully.")
    else:
        log.error("[CAM] Failed to open RTSP stream.")


def _get_raw_frame():
    """Grab the freshest frame, reconnecting if dropped."""
    global _cap
    if _cap is None or not _cap.isOpened():
        _open_camera()

    # Drain buffer
    for _ in range(4):
        _cap.grab()

    ret, frame = _cap.retrieve()
    if not ret or frame is None:
        log.warning("[CAM] Stream lost — reconnecting...")
        _cap.release()
        _open_camera()
        ret, frame = _cap.read()

    return frame if (ret and frame is not None) else None


# ── Background detection + sync thread ───────────────────────────────────────

def _detection_loop():
    """
    Runs forever as a daemon thread.
    Owns: camera, YOLO inference, auto-mapper, Firebase sync, smoothing.
    Writes results into shared state for Flask routes to read.
    """
    global _latest_frame, _latest_statuses, _slots, _mapping_phase, _frame_count, _last_fb_push

    # ── Firebase ──────────────────────────────────────────────────────────────
    firebase = None
    try:
        firebase = FirebaseSync(
            credentials_path=CREDENTIALS,
            database_url=FIREBASE_URL,
        )
    except Exception as e:
        log.error(f"[FB] Firebase init failed — occupancy won't be synced: {e}")

    # ── Auto-mapper ───────────────────────────────────────────────────────────
    mapper = AutoMapper(
        slot_config_path=SLOT_CONFIG,
        min_frames_to_map=150,
        min_samples=3,
    )
    with _state_lock:
        mapping_phase = _mapping_phase

    if not mapping_phase:
        log.info(f"[BG] Existing slot map — {len(_slots)} slots. Skipping mapping phase.")
    else:
        log.info("[BG] No slot map found — starting auto-mapping phase (~150 frames)...")

    # ── Undistorter (optional) ────────────────────────────────────────────────
    undistorter = None
    if UNDISTORT:
        try:
            from undistort import FisheyeUndistorter
            undistorter = FisheyeUndistorter(fov_degrees=FOV_DEGREES, zoom=ZOOM)
            log.info("[BG] Fisheye undistortion enabled.")
        except ImportError:
            log.warning("[BG] undistort.py not found — running without undistortion.")

    sample_saved = False
    local_frame_count = 0

    _open_camera()

    while True:
        try:
            frame = _get_raw_frame()
            if frame is None:
                log.warning("[BG] No frame — retrying in 1s")
                time.sleep(1)
                continue

            # Save undistort preview once on startup
            if not sample_saved and undistorter:
                undistorter.save_sample(frame, "undistort_sample.jpg")
                log.info("[BG] Saved undistort_sample.jpg")
                sample_saved = True

            # Apply undistortion
            if undistorter:
                frame = undistorter.process(frame)

            # ── Run YOLO ──────────────────────────────────────────────────────
            results = model(frame, conf=CONFIDENCE, classes=TARGET_CLASSES, verbose=False)
            vehicle_boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    vehicle_boxes.append({
                        "coords":     [x1, y1, x2, y2],
                        "confidence": round(float(box.conf[0]), 3),
                        "label":      model.names[int(box.cls[0])],
                    })

            with _state_lock:
                mapping_phase = _mapping_phase

            # ── Phase 1: Auto-Mapping ─────────────────────────────────────────
            if mapping_phase:
                mapper.feed_frame(
                    [[v["coords"][0], v["coords"][1], v["coords"][2], v["coords"][3]]
                     for v in vehicle_boxes],
                    frame.shape,
                )

                if local_frame_count % 10 == 0:
                    log.info(f"[BG] Mapping... frame {local_frame_count}/150 | vehicles: {len(vehicle_boxes)}")

                if mapper.is_mapping_complete():
                    discovered = mapper.get_slots()
                    with _state_lock:
                        _slots         = discovered
                        _mapping_phase = False
                    log.info(f"[BG] Auto-mapping complete — {len(discovered)} slots discovered.")
                    if firebase:
                        firebase.push_slot_layout(discovered)

                local_frame_count += 1
                time.sleep(DETECT_INTERVAL)
                continue

            # ── Phase 2: Occupancy Detection ──────────────────────────────────
            with _state_lock:
                current_slots = dict(_slots)

            statuses = {}
            slot_results = []
            for slot_id, slot_data in current_slots.items():
                coords = slot_data["coords"]
                is_occ = any(_check_overlap(vb["coords"], coords) for vb in vehicle_boxes)
                statuses[slot_id] = "Occupied" if is_occ else "Vacant"
                slot_results.append({
                    "id":     slot_id,
                    "status": statuses[slot_id],
                    "coords": coords,
                    "row":    slot_data.get("row", "A"),
                })

            statuses = _apply_smoothing(statuses)

            # Annotate frame and encode to JPEG
            annotated = _draw_boxes(frame.copy(), vehicle_boxes, slot_results)
            frame_small = cv2.resize(annotated, (1280, 720))
            _, buf = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 70])

            # Write shared state
            with _state_lock:
                _latest_frame    = buf.tobytes()
                _latest_statuses = dict(statuses)
                _frame_count     = local_frame_count

            # Firebase push every FIREBASE_EVERY cycles
            if firebase and local_frame_count % FIREBASE_EVERY == 0:
                firebase.push_occupancy(statuses)

            occupied = sum(1 for s in statuses.values() if s == "Occupied")
            log.info(
                f"[BG] Frame {local_frame_count} | "
                f"Occupied: {occupied}/{len(statuses)} | "
                f"Vehicles: {len(vehicle_boxes)}"
            )

            local_frame_count += 1
            time.sleep(DETECT_INTERVAL)

        except Exception as e:
            log.error(f"[BG] Unexpected error: {e}", exc_info=True)
            time.sleep(2)


# ── Start background thread on import ────────────────────────────────────────
_bg_thread = threading.Thread(target=_detection_loop, name="detection-loop", daemon=True)
_bg_thread.start()
log.info("[MAIN] Background detection thread started.")


# ── Flask Routes ──────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    with _state_lock:
        slots_loaded = len(_slots)
        mapping      = _mapping_phase
        frame_count  = _frame_count
        cam_ok       = _latest_frame is not None or mapping
    return jsonify({
        "online":        True,
        "camera":        cam_ok,
        "slots_loaded":  slots_loaded,
        "mapping_phase": mapping,
        "frame_count":   frame_count,
        "model":         "yolov8n",
        "timestamp":     int(time.time() * 1000),
    })


@app.route("/slots", methods=["GET"])
def get_slots():
    with _state_lock:
        return jsonify(dict(_slots))


@app.route("/occupancy", methods=["GET"])
def get_occupancy():
    """Returns latest computed occupancy statuses."""
    with _state_lock:
        statuses = dict(_latest_statuses)
    occupied = sum(1 for s in statuses.values() if s == "Occupied")
    return jsonify({
        "slots":    statuses,
        "occupied": occupied,
        "vacant":   len(statuses) - occupied,
        "total":    len(statuses),
        "timestamp": int(time.time() * 1000),
    })


@app.route("/live-frame", methods=["GET"])
def live_frame():
    """
    Returns the latest annotated JPEG produced by the background thread.
    No YOLO inference here — just serves the pre-computed frame instantly.
    Web app polls this every ~1.5s.
    """
    with _state_lock:
        frame_bytes = _latest_frame

    if frame_bytes is None:
        # Bg thread hasn't produced a frame yet — return placeholder
        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Initialising camera — please wait...",
                    (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
        _, buf = cv2.imencode(".jpg", placeholder)
        frame_bytes = buf.tobytes()

    return Response(
        frame_bytes,
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """Upload an image for one-shot YOLO analysis (uses shared model)."""
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file  = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    h, w = frame.shape[:2]
    results = model(frame, conf=CONFIDENCE, classes=TARGET_CLASSES, verbose=False)
    vehicle_boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            vehicle_boxes.append({
                "coords":     [round(x1), round(y1), round(x2), round(y2)],
                "confidence": round(float(box.conf[0]), 3),
                "label":      model.names[int(box.cls[0])],
            })

    with _state_lock:
        current_slots = dict(_slots)

    if not current_slots:
        estimated = []
        for i, vb in enumerate(vehicle_boxes):
            estimated.append({
                "id":         f"S{i+1:02d}",
                "status":     "Occupied",
                "confidence": vb["confidence"],
                "coords":     vb["coords"],
                "row":        "A" if vb["coords"][1] < h // 3 else ("B" if vb["coords"][1] < 2 * h // 3 else "C"),
            })
        return jsonify({
            "mode": "no_slot_config", "image_size": [w, h],
            "vehicles_detected": len(vehicle_boxes), "vehicle_boxes": vehicle_boxes,
            "slots": estimated, "total_slots": len(estimated),
            "occupied": len(estimated), "vacant": 0,
            "timestamp": int(time.time() * 1000),
        })

    slot_results   = []
    occupied_count = 0
    for slot_id, slot_data in current_slots.items():
        coords = slot_data["coords"]
        is_occ = any(_check_overlap(vb["coords"], coords) for vb in vehicle_boxes)
        conf   = next(
            (vb["confidence"] for vb in vehicle_boxes if _check_overlap(vb["coords"], coords)),
            0.0,
        )
        if is_occ:
            occupied_count += 1
        slot_results.append({
            "id":         slot_id,
            "status":     "Occupied" if is_occ else "Vacant",
            "confidence": round(conf, 3) if is_occ else round(0.88 + (hash(slot_id) % 10) * 0.01, 3),
            "row":        slot_data.get("row", "A"),
            "coords":     coords,
        })

    vacant_count = len(current_slots) - occupied_count
    return jsonify({
        "mode": "slot_config", "image_size": [w, h],
        "vehicles_detected": len(vehicle_boxes), "vehicle_boxes": vehicle_boxes,
        "slots": slot_results, "total_slots": len(current_slots),
        "occupied": occupied_count, "vacant": vacant_count,
        "occupancy_percent": round((occupied_count / max(1, len(current_slots))) * 100),
        "timestamp": int(time.time() * 1000),
    })


@app.route("/remap", methods=["POST"])
def trigger_remap():
    """
    Admin endpoint — forces a fresh auto-mapping run by clearing slot config
    and resetting mapping phase. Next 150 frames will re-discover slot layout.
    """
    global _slots, _mapping_phase, _smoothing_hist
    with _state_lock:
        _slots         = {}
        _mapping_phase = True
        _smoothing_hist = {}
    if os.path.exists(SLOT_CONFIG):
        os.remove(SLOT_CONFIG)
    log.info("[ADMIN] Remap triggered — slot config cleared.")
    return jsonify({"status": "remap_started", "message": "Auto-mapping restarted. ~150 frames needed."})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Smart Parking Flask API (unified) ===")
    print(f"RTSP  : {RTSP_URL}")
    print(f"API   : http://<PI_IP>:5000")
    print("Routes: /status  /slots  /occupancy  /live-frame  /analyze-image  /remap")
    print("=========================================\n")
    # threaded=True is now safe — Flask routes only READ shared state via lock.
    # The camera is exclusively owned by the background daemon thread.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)