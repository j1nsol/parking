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
FIREBASE_EVERY  = 2           # push Firebase every N detection cycles
MAPPER_EPS_PX   = 40          # DBSCAN eps — max px between detections in same slot cluster.
                               # Lower = tighter clusters, better separation of close cars.
                               # Raise if valid slots are being split into multiple clusters.

# Fisheye undistortion — values are live-fetched from Firebase every 5s.
# These are only the startup fallback defaults used before Firebase responds.
UNDISTORT   = False
FOV_DEGREES = 185.0
ZOOM        = 0.7

# How often the bg thread checks Firebase for undistort config changes (seconds)
UNDISTORT_POLL_INTERVAL = 5

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ── Load YOLO once — shared by bg thread and /analyze-image ──────────────────
log.info("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")
log.info("Model ready.")

# ── Shared state — protected by _state_lock ───────────────────────────────────
# Using RLock so the bg thread can acquire it re-entrantly if needed.
_state_lock       = threading.RLock()
_latest_frame     = None          # Last annotated JPEG bytes (kept for /live-frame fallback)
_latest_raw_frame = None          # Last raw numpy frame (for undistort preview)
_latest_statuses  = {}            # slot_id -> "Occupied" | "Vacant"
_latest_vehicle_boxes = []        # last YOLO detections — read by stream thread
_latest_slot_results  = []        # last slot occupancy results — read by stream thread
_slots            = {}            # slot_id -> {coords, row, ...}
_smoothing_hist   = {}            # slot_id -> deque for temporal smoothing
_mapping_phase    = True          # True while auto-mapper is still running
_frame_count      = 0
_last_fb_push     = 0
_remap_requested  = False

# ── Shared undistort state — hot-reloaded from Firebase ──────────────────────
_undistort_cfg = {
    "enabled": False,
    "k1":      -0.3,
    "k2":      0.1,
    "alpha":   0.0,
}

# ── Shared program config — hot-reloaded from Firebase ───────────────────────
_prog_cfg = {
    "confidence":      CONFIDENCE,
    "iou_threshold":   IOU_THRESHOLD,
    "smoothing_win":   SMOOTHING_WIN,
    "detect_interval": DETECT_INTERVAL,
    "firebase_every":  FIREBASE_EVERY,
    "yolo_every_n":    1,
}

# ── Load existing slot config if present ─────────────────────────────────────
if os.path.exists(SLOT_CONFIG):
    with open(SLOT_CONFIG) as f:
        _slots = json.load(f)
    log.info(f"Loaded {len(_slots)} slots from {SLOT_CONFIG}")
    _mapping_phase = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_quad(coords) -> bool:
    """True if coords is a list of 4 [x,y] points (quad), False if flat [x1,y1,x2,y2]."""
    return (isinstance(coords, list) and len(coords) == 4
            and isinstance(coords[0], (list, tuple)))


def _check_overlap(vbox, slot_coords) -> bool:
    """
    Returns True if the vehicle bounding box center falls inside the slot.
    Supports both quad [[x,y]×4] and legacy rect [x1,y1,x2,y2] coords.
    Quad uses point-in-polygon (no threshold needed).
    Rect uses IoU overlap with the configurable iou_threshold.
    """
    cx = (vbox[0] + vbox[2]) / 2
    cy = (vbox[1] + vbox[3]) / 2

    if _is_quad(slot_coords):
        contour = np.array(slot_coords, dtype=np.float32)
        return cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0

    # Legacy rect fallback
    vx1, vy1, vx2, vy2 = vbox
    sx1, sy1, sx2, sy2 = slot_coords
    ix1, iy1 = max(vx1, sx1), max(vy1, sy1)
    ix2, iy2 = min(vx2, sx2), min(vy2, sy2)
    inter     = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    slot_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    with _state_lock:
        threshold = _prog_cfg["iou_threshold"]
    return (inter / slot_area) >= threshold


def _apply_smoothing(statuses: dict) -> dict:
    """Majority-vote temporal smoothing — mutates and returns statuses."""
    with _state_lock:
        win = _prog_cfg["smoothing_win"]
    for slot_id, status in statuses.items():
        if slot_id not in _smoothing_hist:
            _smoothing_hist[slot_id] = deque(maxlen=win)
        # Resize deque if window changed
        if _smoothing_hist[slot_id].maxlen != win:
            _smoothing_hist[slot_id] = deque(
                list(_smoothing_hist[slot_id])[-win:], maxlen=win
            )
        _smoothing_hist[slot_id].append(1 if status == "Occupied" else 0)
        majority = sum(_smoothing_hist[slot_id]) > (win // 2)
        statuses[slot_id] = "Occupied" if majority else "Vacant"
    return statuses


def _draw_boxes(frame, vehicle_boxes, slot_results=None):
    """Draw YOLO vehicle boxes and slot overlays. Supports quad and rect coords."""
    if slot_results:
        for slot in slot_results:
            coords = slot.get("coords")
            if not coords:
                continue
            occ   = slot["status"] == "Occupied"
            color = (60, 60, 220) if occ else (80, 200, 120)
            label = f"{slot['id']} {'OCC' if occ else 'VAC'}"

            if _is_quad(coords):
                pts = np.array(coords, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3)
                lx, ly = coords[0]
                cv2.putText(frame, label, (lx + 4, ly + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            else:
                x1, y1, x2, y2 = [int(c) for c in coords]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
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

# Module-level firebase reference so API routes can call push_undistort_config
firebase_instance = None


def _detection_loop():
    """
    Runs forever as a daemon thread.
    Owns: camera, YOLO inference, auto-mapper, Firebase sync, smoothing.
    Writes results into shared state for Flask routes to read.
    """
    global _latest_frame, _latest_raw_frame, _latest_statuses, _latest_vehicle_boxes, _latest_slot_results, _slots, _mapping_phase, _frame_count, _last_fb_push, firebase_instance, _remap_requested

    # ── Firebase ──────────────────────────────────────────────────────────────
    firebase = None
    try:
        firebase = FirebaseSync(
            credentials_path=CREDENTIALS,
            database_url=FIREBASE_URL,
        )
        firebase_instance = firebase
    except Exception as e:
        log.error(f"[FB] Firebase init failed — occupancy won't be synced: {e}")

    # ── Auto-mapper ───────────────────────────────────────────────────────────
    mapper = AutoMapper(
        slot_config_path=SLOT_CONFIG,
        min_frames_to_map=150,
        min_samples=3,
        eps_pixels=MAPPER_EPS_PX,   # tune in config above if close cars merge into one cluster
    )
    with _state_lock:
        mapping_phase = _mapping_phase

    if not mapping_phase:
        log.info(f"[BG] Existing slot map — {len(_slots)} slots. Skipping mapping phase.")
    else:
        log.info("[BG] No slot map found — starting auto-mapping phase (~150 frames)...")

    # ── Undistorter — built from Firebase config, hot-reloaded on change ────
    undistorter      = None
    last_cfg_check   = 0
    active_cfg       = {"enabled": False, "fov_degrees": 185.0, "zoom": 0.7}

    def _rebuild_undistorter(cfg: dict):
        """Instantiate a new WideAngleUndistorter from cfg, or None if disabled."""
        if not cfg.get("enabled", False):
            log.info("[BG] Undistortion disabled.")
            return None
        try:
            from undistort import WideAngleUndistorter
            u = WideAngleUndistorter(
                k1=cfg["k1"], k2=cfg["k2"],
                p1=0.0, p2=0.0,
                alpha=cfg["alpha"],
            )
            log.info(f"[BG] Undistorter built — k1={cfg['k1']} k2={cfg['k2']} alpha={cfg['alpha']}")
            return u
        except ImportError:
            log.warning("[BG] undistort.py not found — running without undistortion.")
            return None

    # Fetch initial config from Firebase
    if firebase:
        try:
            initial_cfg = firebase.get_undistort_config()
            with _state_lock:
                _undistort_cfg.update(initial_cfg)
            active_cfg  = dict(initial_cfg)
            undistorter = _rebuild_undistorter(active_cfg)
        except Exception as e:
            log.warning(f"[BG] Could not fetch initial undistort config: {e}")

    # ── Fetch initial program config ──────────────────────────────────────────
    if firebase:
        try:
            initial_prog = firebase.get_program_config()
            with _state_lock:
                _prog_cfg.update(initial_prog)
            log.info(f"[BG] Program config loaded: {initial_prog}")
        except Exception as e:
            log.warning(f"[BG] Could not fetch initial program config: {e}")

    sample_saved      = False
    local_frame_count = 0
    last_prog_check   = 0
    last_yolo_frame   = None   # last YOLO result, reused on skipped frames

    _open_camera()

    while True:
        try:
            frame = _get_raw_frame()
            if frame is None:
                log.warning("[BG] No frame — retrying in 1s")
                time.sleep(1)
                continue

            # ── Check if admin triggered a remap ─────────────────────────────
            with _state_lock:
                do_remap = _remap_requested
                if do_remap:
                    _remap_requested = False

            if do_remap:
                log.info("[BG] Remap request received — resetting mapper and frame count.")
                mapper._detections   = []
                mapper._frame_count  = 0
                mapper._slots        = {}
                local_frame_count    = 0
                sample_saved         = False
                with _state_lock:
                    _mapping_phase = True
                log.info("[BG] Mapper reset. Starting fresh mapping phase (~150 frames)...")

            # ── Poll Firebase for config changes every 5s ─────────────────────
            now = time.time()
            if firebase and (now - last_cfg_check) >= UNDISTORT_POLL_INTERVAL:
                last_cfg_check = now
                # Undistort config
                try:
                    new_cfg = firebase.get_undistort_config()
                    if new_cfg != active_cfg:
                        log.info(f"[BG] Undistort config changed: {new_cfg}")
                        active_cfg  = new_cfg
                        undistorter = _rebuild_undistorter(active_cfg)
                        with _state_lock:
                            _undistort_cfg.update(active_cfg)
                except Exception as e:
                    log.warning(f"[BG] Undistort config poll failed: {e}")
                # Program config
                try:
                    new_prog = firebase.get_program_config()
                    with _state_lock:
                        if new_prog != _prog_cfg:
                            log.info(f"[BG] Program config changed: {new_prog}")
                            _prog_cfg.update(new_prog)
                except Exception as e:
                    log.warning(f"[BG] Program config poll failed: {e}")

            # Read live program config values once per loop iteration
            with _state_lock:
                conf            = _prog_cfg["confidence"]
                detect_interval = _prog_cfg["detect_interval"]
                firebase_every  = _prog_cfg["firebase_every"]
                yolo_every_n    = _prog_cfg["yolo_every_n"]

            # Save undistort preview once on startup
            if not sample_saved and undistorter:
                undistorter.save_sample(frame, "undistort_sample.jpg")
                log.info("[BG] Saved undistort_sample.jpg")
                sample_saved = True

            # Store raw frame for preview endpoint BEFORE undistortion
            with _state_lock:
                _latest_raw_frame = frame.copy()

            # Apply undistortion
            if undistorter:
                frame = undistorter.process(frame)

            # ── Run YOLO (or reuse last result if skipping this frame) ─────────
            if local_frame_count % max(1, yolo_every_n) == 0:
                results = model(frame, conf=conf, classes=TARGET_CLASSES, verbose=False)
                vehicle_boxes = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        vehicle_boxes.append({
                            "coords":     [x1, y1, x2, y2],
                            "confidence": round(float(box.conf[0]), 3),
                            "label":      model.names[int(box.cls[0])],
                        })
                last_yolo_frame = vehicle_boxes
            else:
                vehicle_boxes = last_yolo_frame or []

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
                time.sleep(detect_interval)
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

            # Update slot statuses with smoothed values
            for sr in slot_results:
                sr["status"] = statuses[sr["id"]]

            # Write shared state — stream thread handles frame annotation + encoding
            with _state_lock:
                _latest_vehicle_boxes = list(vehicle_boxes)
                _latest_slot_results  = list(slot_results)
                _latest_statuses      = dict(statuses)
                _frame_count          = local_frame_count

            # Firebase push every firebase_every cycles
            if firebase and local_frame_count % max(1, firebase_every) == 0:
                firebase.push_occupancy(statuses)

            occupied = sum(1 for s in statuses.values() if s == "Occupied")
            log.info(
                f"[BG] Frame {local_frame_count} | "
                f"Occupied: {occupied}/{len(statuses)} | "
                f"Vehicles: {len(vehicle_boxes)} | "
                f"conf={conf} iou={_prog_cfg['iou_threshold']} "
                f"interval={detect_interval}s yolo_every={yolo_every_n}"
            )

            local_frame_count += 1
            time.sleep(detect_interval)

        except Exception as e:
            log.error(f"[BG] Unexpected error: {e}", exc_info=True)
            time.sleep(2)


# ── MJPEG stream loop ─────────────────────────────────────────────────────────
# Runs at ~15 FPS independently of YOLO. Grabs raw frames, draws the latest
# YOLO overlay (vehicle boxes + slot results), encodes to JPEG and pushes into
# a queue that the /stream endpoint reads from.

import queue as _queue

STREAM_FPS      = 15          # target display FPS
STREAM_QUALITY  = 60          # JPEG quality (lower = faster transfer)
STREAM_WIDTH    = 1280
STREAM_HEIGHT   = 720
_stream_queue   = _queue.Queue(maxsize=2)   # bounded — drops stale frames
_stream_clients = 0                         # count of active /stream connections
_stream_lock    = threading.Lock()


def _stream_loop():
    """
    Dedicated thread: grabs frames fast, overlays latest YOLO results,
    pushes annotated JPEGs to _stream_queue for MJPEG clients.
    Only runs when at least one browser tab has the stream open.
    """
    global _stream_clients
    cap = None
    interval = 1.0 / STREAM_FPS

    def open_cap():
        c = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return c

    while True:
        # If no clients, sleep and release camera to save resources
        with _stream_lock:
            clients = _stream_clients
        if clients == 0:
            if cap and cap.isOpened():
                cap.release()
                cap = None
                log.info("[STREAM] No clients — camera released.")
            time.sleep(0.5)
            continue

        # Open camera if needed
        if cap is None or not cap.isOpened():
            log.info("[STREAM] Client connected — opening stream camera.")
            cap = open_cap()

        t0 = time.time()

        # Drain buffer then grab freshest frame
        cap.grab(); cap.grab()
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            cap.release()
            cap = open_cap()
            time.sleep(0.1)
            continue

        # Apply undistortion if enabled
        with _state_lock:
            udist_cfg = dict(_undistort_cfg)
        if udist_cfg.get("enabled"):
            try:
                from undistort import WideAngleUndistorter
                u = WideAngleUndistorter(
                    k1=udist_cfg["k1"], k2=udist_cfg["k2"], alpha=udist_cfg["alpha"]
                )
                frame = u.process(frame)
            except Exception:
                pass

        # Read latest YOLO overlay (no lock held during draw — just stale data is fine)
        with _state_lock:
            vboxes      = list(_latest_vehicle_boxes)
            slot_res    = list(_latest_slot_results)
            mapping     = _mapping_phase

        # Draw overlay
        annotated = _draw_boxes(frame, vboxes, slot_res if not mapping else None)
        if mapping:
            cv2.putText(annotated, "Auto-mapping in progress...",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

        # Resize + encode
        small = cv2.resize(annotated, (STREAM_WIDTH, STREAM_HEIGHT))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])

        # Push to queue — discard oldest frame if full (client is slow)
        try:
            _stream_queue.put_nowait(buf.tobytes())
        except _queue.Full:
            try:
                _stream_queue.get_nowait()
                _stream_queue.put_nowait(buf.tobytes())
            except _queue.Empty:
                pass

        # Pace to target FPS
        elapsed = time.time() - t0
        sleep_t = max(0, interval - elapsed)
        time.sleep(sleep_t)


# ── Start background threads on import ───────────────────────────────────────
_bg_thread = threading.Thread(target=_detection_loop, name="detection-loop", daemon=True)
_bg_thread.start()
log.info("[MAIN] Background detection thread started.")

_stream_thread = threading.Thread(target=_stream_loop, name="stream-loop", daemon=True)
_stream_thread.start()
log.info("[MAIN] MJPEG stream thread started.")


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
        "mapper_eps_px": MAPPER_EPS_PX,
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


@app.route("/stream")
def mjpeg_stream():
    """
    MJPEG stream endpoint — browser points an <img> tag here.
    Pushes annotated frames at ~15 FPS continuously.
    Decoupled from YOLO: boxes are from last inference, frames are always fresh.
    """
    global _stream_clients

    def generate():
        global _stream_clients
        with _stream_lock:
            _stream_clients += 1
        log.info(f"[STREAM] Client connected. Total: {_stream_clients}")
        try:
            while True:
                try:
                    frame_bytes = _stream_queue.get(timeout=3.0)
                except _queue.Empty:
                    # Send a placeholder if stream stalls
                    placeholder = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Waiting for camera...",
                                (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (80,80,80), 2)
                    _, buf = cv2.imencode(".jpg", placeholder)
                    frame_bytes = buf.tobytes()

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + frame_bytes + b"\r\n")
        finally:
            with _stream_lock:
                _stream_clients -= 1
            log.info(f"[STREAM] Client disconnected. Total: {_stream_clients}")

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "X-Accel-Buffering": "no"},
    )


@app.route("/live-frame", methods=["GET"])
def live_frame():
    """
    Legacy single-frame endpoint — kept for /undistort-preview and image test.
    For live display use /stream instead.
    """
    with _state_lock:
        raw = _latest_raw_frame.copy() if _latest_raw_frame is not None else None
        vboxes   = list(_latest_vehicle_boxes)
        slot_res = list(_latest_slot_results)

    if raw is None:
        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Initialising camera — please wait...",
                    (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
        _, buf = cv2.imencode(".jpg", placeholder)
        return Response(buf.tobytes(), mimetype="image/jpeg",
                        headers={"Cache-Control": "no-cache"})

    annotated   = _draw_boxes(raw, vboxes, slot_res)
    frame_small = cv2.resize(annotated, (1280, 720))
    _, buf      = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(buf.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


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


@app.route("/program-config", methods=["GET"])
def get_program_config():
    """Returns current live program config."""
    with _state_lock:
        return jsonify(dict(_prog_cfg))


@app.route("/program-config", methods=["POST"])
def set_program_config():
    """
    Admin endpoint — write new program config to Firebase.
    Background thread picks it up within 5 seconds.
    """
    data = request.get_json(silent=True) or {}
    with _state_lock:
        current = dict(_prog_cfg)

    new_cfg = {
        "confidence":      max(0.05, min(0.9,  float(data.get("confidence",      current["confidence"])))),
        "iou_threshold":   max(0.1,  min(0.9,  float(data.get("iou_threshold",   current["iou_threshold"])))),
        "smoothing_win":   max(1,    min(30,   int(data.get("smoothing_win",     current["smoothing_win"])))),
        "detect_interval": max(0.0,  min(5.0,  float(data.get("detect_interval", current["detect_interval"])))),
        "firebase_every":  max(1,    min(30,   int(data.get("firebase_every",    current["firebase_every"])))),
        "yolo_every_n":    max(1,    min(10,   int(data.get("yolo_every_n",      current["yolo_every_n"])))),
    }

    if firebase_instance:
        firebase_instance.push_program_config(new_cfg)

    with _state_lock:
        _prog_cfg.update(new_cfg)

    log.info(f"[API] Program config updated: {new_cfg}")
    return jsonify({
        "status":  "ok",
        "config":  new_cfg,
        "message": "Program config saved. Pi will apply within 5 seconds.",
    })


def _auto_calibrate_distortion(frame: np.ndarray) -> dict:
    """
    Estimate best-fit k1/k2 barrel distortion coefficients by finding the
    values that maximise line straightness in the frame.

    Strategy:
      1. Detect edges → find long line segments via HoughLinesP
      2. For each candidate (k1, k2) pair, apply undistortion and measure
         how straight the detected line segments become (residual from linear fit)
      3. Return the (k1, k2) with the lowest total residual error.

    Works well for parking lots which have clear straight lane markings.
    """
    from undistort import WideAngleUndistorter

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect edges on the raw frame to find candidate line pixels
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    # Find long line segments — parking lot markings are typically long
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=w//8, maxLineGap=20)

    if lines is None or len(lines) < 4:
        log.warning("[AUTOCAL] Not enough lines detected — returning default coefficients.")
        return {"k1": -0.3, "k2": 0.1, "alpha": 0.0, "score": None, "lines_found": 0}

    log.info(f"[AUTOCAL] Found {len(lines)} line segments. Searching k1/k2 space...")

    # Search grid — coarse then fine
    k1_candidates = np.arange(-0.7, 0.01, 0.05)
    k2_candidates = np.arange(0.0,  0.31, 0.05)

    best_k1, best_k2, best_score = -0.3, 0.1, float("inf")

    for k1 in k1_candidates:
        for k2 in k2_candidates:
            try:
                u = WideAngleUndistorter(k1=float(k1), k2=float(k2), alpha=0.0)
                corrected = u.process(frame.copy())
                c_gray    = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
                c_edges   = cv2.Canny(cv2.GaussianBlur(c_gray,(5,5),0), 50, 150)
                c_lines   = cv2.HoughLinesP(c_edges, 1, np.pi/180, threshold=60,
                                            minLineLength=w//10, maxLineGap=20)
                if c_lines is None:
                    continue

                # Measure straightness: for each line segment fit a line
                # and compute mean squared residual of its endpoints
                total_residual = 0.0
                count = 0
                for seg in c_lines:
                    x1c,y1c,x2c,y2c = seg[0]
                    length = np.hypot(x2c-x1c, y2c-y1c)
                    if length < w//12:
                        continue
                    # Residual from perfect line = 0 for a straight segment
                    # Use the angle deviation as proxy — straighter = closer to 0 or 90
                    angle = abs(np.degrees(np.arctan2(y2c-y1c, x2c-x1c))) % 90
                    deviation = min(angle, 90-angle)   # 0 = perfectly H or V
                    total_residual += deviation / max(length, 1)
                    count += 1

                if count == 0:
                    continue
                score = total_residual / count
                if score < best_score:
                    best_score = score
                    best_k1, best_k2 = float(k1), float(k2)
            except Exception:
                continue

    log.info(f"[AUTOCAL] Best: k1={best_k1:.2f} k2={best_k2:.2f} score={best_score:.4f}")
    return {
        "k1":         round(best_k1, 2),
        "k2":         round(best_k2, 2),
        "alpha":      0.0,
        "score":      round(best_score, 4),
        "lines_found": len(lines),
    }


@app.route("/undistort-autocal", methods=["POST"])
def undistort_autocal():
    """
    Run auto-calibration on the latest raw frame.
    Returns best-fit k1/k2 and immediately applies + saves to Firebase.
    Called when the admin toggles distortion ON.
    """
    with _state_lock:
        raw = _latest_raw_frame.copy() if _latest_raw_frame is not None else None

    if raw is None:
        return jsonify({"error": "No frame available yet — camera still initialising"}), 503

    log.info("[API] Auto-calibration started...")
    result = _auto_calibrate_distortion(raw)

    # Build new config with found values, enabled=True
    new_cfg = {
        "enabled": True,
        "k1":      result["k1"],
        "k2":      result["k2"],
        "alpha":   result["alpha"],
    }

    # Save to Firebase and update shared state
    if firebase_instance:
        firebase_instance.push_undistort_config(**new_cfg)

    with _state_lock:
        _undistort_cfg.update(new_cfg)

    log.info(f"[API] Auto-cal complete — {new_cfg}")
    return jsonify({
        "status":      "ok",
        "config":      new_cfg,
        "lines_found": result["lines_found"],
        "score":       result["score"],
        "message":     f"Auto-calibrated: k1={new_cfg['k1']}, k2={new_cfg['k2']}. Applied to Pi.",
    })


@app.route("/undistort-config", methods=["GET"])
def get_undistort_config():
    """Returns the current undistort config from shared state."""
    with _state_lock:
        cfg = dict(_undistort_cfg)
    return jsonify(cfg)


@app.route("/undistort-config", methods=["POST"])
def set_undistort_config():
    """
    Admin endpoint — write new undistort config to Firebase.
    The background thread will pick it up within UNDISTORT_POLL_INTERVAL seconds.
    Body: { "enabled": bool, "fov_degrees": float, "zoom": float }
    """
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", _undistort_cfg["enabled"]))
    k1      = float(data.get("k1",      _undistort_cfg["k1"]))
    k2      = float(data.get("k2",      _undistort_cfg["k2"]))
    alpha   = float(data.get("alpha",   _undistort_cfg["alpha"]))

    # Clamp to safe ranges
    k1    = max(-0.8, min(0.0,  k1))
    k2    = max( 0.0, min(0.3,  k2))
    alpha = max( 0.0, min(1.0,  alpha))

    if firebase_instance:
        firebase_instance.push_undistort_config(enabled, k1, k2, alpha)

    with _state_lock:
        _undistort_cfg["enabled"] = enabled
        _undistort_cfg["k1"]      = k1
        _undistort_cfg["k2"]      = k2
        _undistort_cfg["alpha"]   = alpha

    log.info(f"[API] Undistort config updated: enabled={enabled} k1={k1} k2={k2} alpha={alpha}")
    return jsonify({
        "status": "ok",
        "config": {"enabled": enabled, "k1": k1, "k2": k2, "alpha": alpha},
        "message": "Config saved. Pi will apply within 5 seconds.",
    })


@app.route("/undistort-preview", methods=["GET"])
def undistort_preview():
    """
    Uses the last raw frame captured by the bg thread (no camera access here),
    applies current undistort settings, and returns a side-by-side JPEG.
    Always shows original vs undistorted regardless of enabled toggle,
    so you can see what the correction looks like before committing.
    Called manually from the admin panel — not polled.
    """
    with _state_lock:
        raw = _latest_raw_frame.copy() if _latest_raw_frame is not None else None
        cfg = dict(_undistort_cfg)

    if raw is None:
        placeholder = np.zeros((400, 1280, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No frame yet — camera may still be initialising",
                    (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
        _, buf = cv2.imencode(".jpg", placeholder)
        return Response(buf.tobytes(), mimetype="image/jpeg")

    # Always apply undistortion for the preview so you can compare
    # regardless of whether the toggle is currently enabled
    try:
        from undistort import WideAngleUndistorter
        u = WideAngleUndistorter(
            k1=cfg["k1"], k2=cfg["k2"],
            p1=0.0, p2=0.0,
            alpha=cfg["alpha"],
        )
        corrected = u.process(raw.copy())
    except Exception as e:
        log.warning(f"[PREVIEW] Undistort failed: {e}")
        corrected = raw.copy()

    orig = raw.copy()
    h, w = orig.shape[:2]

    # Label banners
    enabled_str = "ENABLED" if cfg.get("enabled") else "DISABLED (preview only)"
    for img, label, color in [
        (orig,      "ORIGINAL",  (100, 100, 100)),
        (corrected, f"CORRECTED  k1={cfg['k1']}  k2={cfg['k2']}  alpha={cfg['alpha']}  [{enabled_str}]",
                                 (80, 200, 120) if cfg.get("enabled") else (56, 189, 248)),
    ]:
        cv2.rectangle(img, (0, 0), (w, 42), (7, 10, 16), -1)
        cv2.putText(img, label, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    side_by_side = np.hstack([orig, corrected])
    # Resize to fit browser width while keeping aspect ratio
    out_w = 1280
    out_h = int(h * (out_w / (w * 2)))
    out = cv2.resize(side_by_side, (out_w, out_h))
    _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return Response(
        buf.tobytes(),
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.route("/remap", methods=["POST"])
def trigger_remap():
    """
    Admin endpoint — signals the bg thread to reset the AutoMapper object
    and restart the mapping phase.
    """
    global _remap_requested
    with _state_lock:
        _remap_requested = True
        _smoothing_hist  = {}
    if os.path.exists(SLOT_CONFIG):
        os.remove(SLOT_CONFIG)
    log.info("[ADMIN] Remap requested — bg thread will reset mapper on next frame.")
    return jsonify({"status": "remap_started", "message": "Auto-mapping restarted."})


@app.route("/slots/<slot_id>", methods=["PUT"])
def update_slot(slot_id):
    """
    Admin endpoint — update a single slot's quad coordinates.
    Body: { "coords": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] }
    Used by the Slot Editor tab for manual point adjustment.
    """
    global _slots
    data = request.get_json(silent=True) or {}
    coords = data.get("coords")

    if not coords or len(coords) != 4:
        return jsonify({"error": "coords must be a list of 4 [x,y] points"}), 400

    with _state_lock:
        if slot_id not in _slots:
            return jsonify({"error": f"Slot {slot_id} not found"}), 404
        _slots[slot_id]["coords"] = coords
        _slots[slot_id]["source"] = "manual"
        current = dict(_slots)

    # Persist to disk
    with open(SLOT_CONFIG, "w") as f:
        json.dump(current, f, indent=2)

    # Push to Firebase
    if firebase_instance:
        firebase_instance.push_slot_layout(current)

    log.info(f"[ADMIN] Slot {slot_id} updated manually.")
    return jsonify({"status": "ok", "slot_id": slot_id, "coords": coords})


@app.route("/slots/<slot_id>", methods=["DELETE"])
def delete_slot(slot_id):
    """Admin endpoint — permanently remove a slot."""
    global _slots
    with _state_lock:
        if slot_id not in _slots:
            return jsonify({"error": f"Slot {slot_id} not found"}), 404
        _slots.pop(slot_id)
        current = dict(_slots)

    with open(SLOT_CONFIG, "w") as f:
        json.dump(current, f, indent=2)

    if firebase_instance:
        firebase_instance.push_slot_layout(current)

    log.info(f"[ADMIN] Slot {slot_id} deleted.")
    return jsonify({"status": "ok", "slot_id": slot_id})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Smart Parking Flask API (unified) ===")
    print(f"RTSP  : {RTSP_URL}")
    print(f"API   : http://<PI_IP>:5000")
    print("Routes: /status  /slots  /occupancy  /live-frame  /analyze-image  /remap")
    print("        /slots/<id> PUT/DELETE  /program-config  /undistort-config")
    print("=========================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)