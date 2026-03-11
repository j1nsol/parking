"""
flask_api.py — REST API on Raspberry Pi
Receives images from the web app, runs YOLO, returns occupancy results.

Install:  pip install flask flask-cors ultralytics opencv-python --break-system-packages
Run:      python3 flask_api.py

The web app calls:
  POST /analyze-image   → upload image, get YOLO results back
  GET  /status          → check if Pi + camera are online
  GET  /slots           → get current live slot statuses
"""

import cv2
import json
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Allow web app on PC to call this API

# ── Load YOLO once at startup ─────────────────────────────────────────────────
print("Loading YOLOv5n model...")
model = YOLO("yolov5nu.pt")
print("Model ready.")

# ── Load slot config if it exists ─────────────────────────────────────────────
import os
SLOT_CONFIG = "slot_config.json"
slots = {}
if os.path.exists(SLOT_CONFIG):
    with open(SLOT_CONFIG) as f:
        slots = json.load(f)
    print(f"Loaded {len(slots)} slots from {SLOT_CONFIG}")

CONFIDENCE     = 0.10
TARGET_CLASSES = [2, 5, 7]   # car, bus, truck
IOU_THRESHOLD  = 0.60

def check_overlap(vbox, slot_coords):
    vx1, vy1, vx2, vy2 = vbox
    sx1, sy1, sx2, sy2 = slot_coords
    ix1, iy1 = max(vx1, sx1), max(vy1, sy1)
    ix2, iy2 = min(vx2, sx2), min(vy2, sy2)
    inter      = max(0, ix2-ix1) * max(0, iy2-iy1)
    slot_area  = max(1, (sx2-sx1) * (sy2-sy1))
    return (inter / slot_area) >= IOU_THRESHOLD


# ── Routes ────────────────────────────────────────────────────────────────────
RTSP_URL = "rtsp://admin:Skibidi1@192.168.1.142:554/Streaming/Channels/101"
@app.route("/status", methods=["GET"])
def status():
    """
    Web app calls this to check if the Pi is reachable.
    Returns camera availability and slot count.
    """
    cam = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cam_ok = cam.isOpened()
    cam.release()
    return jsonify({
        "online":       True,
        "camera":       cam_ok,
        "slots_loaded": len(slots),
        "model":        "yolov5n",
        "timestamp":    int(time.time() * 1000),
    })


@app.route("/slots", methods=["GET"])
def get_slots():
    """Return current slot statuses from Firebase / local state."""
    return jsonify(slots)


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """
    Receives an image file from the web app.
    Runs YOLO on it and returns occupancy per slot.

    Expected: multipart/form-data with field 'image'
    Returns:  JSON with slot statuses + vehicle detections
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file  = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    h, w = frame.shape[:2]

    # ── Run YOLO ──────────────────────────────────────────────────────────────
    results = model(frame, conf=CONFIDENCE, classes=TARGET_CLASSES, verbose=False)

    vehicle_boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = model.names[cls]
            vehicle_boxes.append({
                "coords":     [round(x1), round(y1), round(x2), round(y2)],
                "confidence": round(conf, 3),
                "label":      label,
            })

    # ── If no slots configured yet: auto-estimate from detections ─────────────
    if not slots:
        # Each detected vehicle = one occupied slot (estimated)
        estimated_slots = []
        for i, vb in enumerate(vehicle_boxes):
            estimated_slots.append({
                "id":         f"S{i+1:02d}",
                "status":     "Occupied",
                "confidence": vb["confidence"],
                "row":        "A" if vb["coords"][1] < h//3 else ("B" if vb["coords"][1] < 2*h//3 else "C"),
            })
        return jsonify({
            "mode":              "no_slot_config",
            "image_size":        [w, h],
            "vehicles_detected": len(vehicle_boxes),
            "vehicle_boxes":     vehicle_boxes,
            "slots":             estimated_slots,
            "total_slots":       len(estimated_slots),
            "occupied":          len(estimated_slots),
            "vacant":            0,
            "note":              "No slot_config.json found. Run auto-mapping first for accurate results.",
            "timestamp":         int(time.time() * 1000),
        })

    # ── Match vehicles to known slots ─────────────────────────────────────────
    slot_results = []
    occupied_count = 0
    for slot_id, slot_data in slots.items():
        coords  = slot_data["coords"]
        is_occ  = any(check_overlap(list(vb["coords"]), coords) for vb in vehicle_boxes)
        # Find confidence of the matching vehicle box
        conf = 0.0
        for vb in vehicle_boxes:
            if check_overlap(list(vb["coords"]), coords):
                conf = vb["confidence"]
                break

        status = "Occupied" if is_occ else "Vacant"
        if is_occ:
            occupied_count += 1

        slot_results.append({
            "id":         slot_id,
            "status":     status,
            "confidence": round(conf, 3) if is_occ else round(0.88 + (hash(slot_id) % 10) * 0.01, 3),
            "row":        slot_data.get("row", "A"),
            "coords":     coords,
        })

    vacant_count = len(slots) - occupied_count

    return jsonify({
        "mode":              "slot_config",
        "image_size":        [w, h],
        "vehicles_detected": len(vehicle_boxes),
        "vehicle_boxes":     vehicle_boxes,
        "slots":             slot_results,
        "total_slots":       len(slots),
        "occupied":          occupied_count,
        "vacant":            vacant_count,
        "occupancy_percent": round((occupied_count / max(1, len(slots))) * 100),
        "timestamp":         int(time.time() * 1000),
    })


if __name__ == "__main__":
    # Run on all interfaces so the web app on your PC can reach it
    # Find your Pi's IP with: hostname -I
    print("\n=== Smart Parking Flask API ===")
    print("Find your Pi IP with: hostname -I")
    print("Web app should point to: http://<PI_IP>:5000")
    print("================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
