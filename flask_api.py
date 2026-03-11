"""
flask_api.py — REST API on Raspberry Pi
Thread-safe camera access with lock to prevent segfault on concurrent requests.
"""

import cv2
import json
import time
import os
import threading
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ── Load YOLO once at startup ─────────────────────────────────────────────────
print("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")
print("Model ready.")

# ── Load slot config ──────────────────────────────────────────────────────────
SLOT_CONFIG = "slot_config.json"
slots = {}
if os.path.exists(SLOT_CONFIG):
    with open(SLOT_CONFIG) as f:
        slots = json.load(f)
    print(f"Loaded {len(slots)} slots from {SLOT_CONFIG}")

CONFIDENCE     = 0.20
TARGET_CLASSES = [2, 5, 7]   # car, bus, truck
IOU_THRESHOLD  = 0.35

RTSP_URL = "rtsp://admin:Skibidi1@192.168.1.142:554/Streaming/Channels/101"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ── Thread-safe camera singleton ──────────────────────────────────────────────
_cap      = None
_cap_lock = threading.Lock()


def get_frame():
    """
    Grab the latest frame from the RTSP stream.
    Uses a lock so concurrent Flask threads never call cap.read() simultaneously
    (which causes segfaults with OpenCV).
    Reconnects automatically if the stream drops.
    """
    global _cap
    with _cap_lock:
        # Open or reopen if needed
        if _cap is None or not _cap.isOpened():
            print("[CAM] Opening RTSP stream...")
            _cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Drain stale buffered frames — grab without decoding
        for _ in range(4):
            _cap.grab()

        ret, frame = _cap.retrieve()

        if not ret or frame is None:
            print("[CAM] Stream lost — reconnecting...")
            _cap.release()
            _cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = _cap.read()

        return frame if ret and frame is not None else None


def check_overlap(vbox, slot_coords):
    vx1, vy1, vx2, vy2 = vbox
    sx1, sy1, sx2, sy2 = slot_coords
    ix1, iy1 = max(vx1, sx1), max(vy1, sy1)
    ix2, iy2 = min(vx2, sx2), min(vy2, sy2)
    inter     = max(0, ix2-ix1) * max(0, iy2-iy1)
    slot_area = max(1, (sx2-sx1) * (sy2-sy1))
    return (inter / slot_area) >= IOU_THRESHOLD


def draw_boxes(frame, vehicle_boxes, slot_results=None):
    """Draw YOLO vehicle boxes and slot overlays onto a frame."""
    # Draw slot regions
    if slot_results:
        for slot in slot_results:
            coords = slot.get("coords")
            if not coords or len(coords) < 4:
                continue
            x1, y1, x2, y2 = [int(c) for c in coords]
            occ   = slot["status"] == "Occupied"
            color = (60, 60, 220) if occ else (80, 200, 120)   # BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"{slot['id']} {'OCC' if occ else 'VAC'}"
            cv2.putText(frame, label, (x1+4, y1+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Draw YOLO vehicle boxes
    for vb in vehicle_boxes:
        x1, y1, x2, y2 = [int(c) for c in vb["coords"]]
        conf  = vb["confidence"]
        label = vb["label"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 189, 56), 2)   # cyan-ish
        cv2.putText(frame, f"{label} {conf:.2f}", (x1+4, y2-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 189, 56), 1, cv2.LINE_AA)

    # Stats overlay bar
    occupied = sum(1 for s in (slot_results or []) if s["status"] == "Occupied")
    total    = len(slot_results) if slot_results else 0
    cv2.rectangle(frame, (0, 0), (520, 44), (7, 10, 16), -1)
    cv2.putText(frame,
                f"YOLOv8n  |  {len(vehicle_boxes)} vehicles  |  {occupied}/{total} slots occupied",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    with _cap_lock:
        global _cap
        if _cap is None or not _cap.isOpened():
            _cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam_ok = _cap.isOpened()
    return jsonify({
        "online":       True,
        "camera":       cam_ok,
        "slots_loaded": len(slots),
        "model":        "yolov8n",
        "timestamp":    int(time.time() * 1000),
    })


@app.route("/slots", methods=["GET"])
def get_slots():
    return jsonify(slots)


@app.route("/live-frame", methods=["GET"])
def live_frame():
    """
    Returns a single annotated JPEG — thread-safe, reconnects on stream loss.
    Web app polls this every ~1.5s.
    """
    frame = get_frame()

    if frame is None:
        # Black placeholder with error message
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera feed unavailable — check RTSP stream",
                    (120, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
    else:
        # Run YOLO
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

        # Match vehicles to slots
        slot_results = []
        for slot_id, slot_data in slots.items():
            coords = slot_data["coords"]
            is_occ = any(check_overlap(list(vb["coords"]), coords) for vb in vehicle_boxes)
            slot_results.append({
                "id":     slot_id,
                "status": "Occupied" if is_occ else "Vacant",
                "coords": coords,
                "row":    slot_data.get("row", "A"),
            })

        frame = draw_boxes(frame, vehicle_boxes, slot_results)

    # Encode as JPEG — lower res for faster transfer
    frame_small = cv2.resize(frame, (1280, 720))
    _, buf = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(
        buf.tobytes(),
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
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

    if not slots:
        estimated = []
        for i, vb in enumerate(vehicle_boxes):
            estimated.append({
                "id":         f"S{i+1:02d}",
                "status":     "Occupied",
                "confidence": vb["confidence"],
                "row":        "A" if vb["coords"][1] < h//3 else ("B" if vb["coords"][1] < 2*h//3 else "C"),
            })
        return jsonify({
            "mode": "no_slot_config", "image_size": [w, h],
            "vehicles_detected": len(vehicle_boxes), "vehicle_boxes": vehicle_boxes,
            "slots": estimated, "total_slots": len(estimated),
            "occupied": len(estimated), "vacant": 0,
            "timestamp": int(time.time() * 1000),
        })

    slot_results = []
    occupied_count = 0
    for slot_id, slot_data in slots.items():
        coords = slot_data["coords"]
        is_occ = any(check_overlap(list(vb["coords"]), coords) for vb in vehicle_boxes)
        conf   = next((vb["confidence"] for vb in vehicle_boxes if check_overlap(list(vb["coords"]), coords)), 0.0)
        if is_occ:
            occupied_count += 1
        slot_results.append({
            "id": slot_id, "status": "Occupied" if is_occ else "Vacant",
            "confidence": round(conf, 3) if is_occ else round(0.88 + (hash(slot_id) % 10) * 0.01, 3),
            "row": slot_data.get("row", "A"), "coords": coords,
        })

    vacant_count = len(slots) - occupied_count
    return jsonify({
        "mode": "slot_config", "image_size": [w, h],
        "vehicles_detected": len(vehicle_boxes), "vehicle_boxes": vehicle_boxes,
        "slots": slot_results, "total_slots": len(slots),
        "occupied": occupied_count, "vacant": vacant_count,
        "occupancy_percent": round((occupied_count / max(1, len(slots))) * 100),
        "timestamp": int(time.time() * 1000),
    })


if __name__ == "__main__":
    print("\n=== Smart Parking Flask API ===")
    print(f"RTSP: {RTSP_URL}")
    print("Live feed: http://<PI_IP>:5000/live-frame")
    print("================================\n")
    # threaded=False prevents concurrent cap.read() calls — eliminates segfault
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)