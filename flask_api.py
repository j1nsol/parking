"""
flask_api.py — REST API on Raspberry Pi
Unified entry point: one camera source, one YOLO model, one process.

Architecture:
  - A background daemon thread runs the full detection + Firebase sync loop
    (previously in main.py) — this owns the camera and YOLO model.
  - Flask routes read from shared state (latest_frame, latest_statuses, slots)
    protected by a RLock.
  - /live-frame serves the last annotated frame captured by the bg thread.
  - main.py is no longer needed — just run: python3 flask_api.py
"""

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import base64
import cv2
import json
import math
import time
import os
import tempfile
import threading
import logging
import numpy as np
from collections import deque
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
from firebase_admin import db
from firebase_sync import FirebaseSync
from auto_mapper import AutoMapper, renumber_slots_by_position
from ai_slot_gen import generate_filled_lot, generate_filled_lot_nb, extract_slots_from_ai_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
# RTSP_URL — live IP camera stream.
# VIDEO_SOURCE is set to RTSP_URL by default so the grabber connects on startup.
# To use a video file instead, upload via POST /video/load then POST /video/start.
# When the video file is stopped or unloaded, the grabber falls back to RTSP.
RTSP_URL       = "rtsp://admin:Skibidi1@192.168.1.142:554/Streaming/Channels/101"
VIDEO_SOURCE   = RTSP_URL

# Firebase is optional. Set FIREBASE_ENABLED = True and supply valid
# CREDENTIALS + FIREBASE_URL to push occupancy data to the cloud.
FIREBASE_ENABLED        = True
FIREBASE_URL            = "https://automapping-parking-slot-default-rtdb.asia-southeast1.firebasedatabase.app"
CREDENTIALS             = "serviceAccountKey.json"
# Firebase Storage bucket — required only for AI slot generation via Nano Banana Pro.
# Find this in Firebase Console → Storage → bucket name (e.g. "your-project-id.appspot.com").
FIREBASE_STORAGE_BUCKET = "automapping-parking-slot.firebasestorage.app"   # ← fill in your bucket name to enable Nano Banana

# Pi identity — must be unique per physical Pi and URL-safe.
# Must match an entry in /map_pins/ created via the web admin Pins tab.
LOCAL_PIN_CODE = "GLEPARK"
FLASK_PORT     = 5000
SLOT_CONFIG    = f"{LOCAL_PIN_CODE}_slot_config.json"
GUIDE_CONFIG   = f"{LOCAL_PIN_CODE}_row_guides.json"
CONFIDENCE     = 0.20
TARGET_CLASSES = [0,1,3]            # custom model: 0=car, 1=cone, 3=reserve (2=pwd skipped)
IOU_THRESHOLD  = 0.50
SMOOTHING_WIN  = 5            # frames for majority-vote smoothing
DETECT_INTERVAL = 1.0         # seconds between detection cycles
FIREBASE_EVERY  = 2           # push Firebase every N detection cycles
MAPPER_EPS_PX   = 40          # DBSCAN eps pixels — lower separates close-parked cars better

# Fisheye undistortion — values are live-fetched from Firebase every 5s.
# These are only the startup fallback defaults used before Firebase responds.
UNDISTORT   = False
FOV_DEGREES = 185.0
ZOOM        = 0.7

# How often the bg thread checks Firebase for undistort config changes (seconds)
UNDISTORT_POLL_INTERVAL = 5

# Force TCP transport for RTSP only — not needed for webcam/file sources.
# Must be set before any VideoCapture call.
_using_rtsp = isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE.lower().startswith("rtsp://")
if _using_rtsp:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp"
        "|stimeout;5000000"           # 5 s TCP socket timeout — keeps retries short
        "|fflags;nobuffer"
        "|flags;low_delay"
        "|max_delay;0"
        "|reorder_queue_size;0"
        "|analyzeduration;1000000"
        "|probesize;1000000"
    )
os.environ["OPENCV_LOG_LEVEL"]        = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"]  = "8"   # AV_LOG_FATAL only

# ── Load YOLO once — shared by bg thread and /analyze-image ──────────────────
# Fix #1: YOLO inference is NOT thread-safe. Guard every model() call with
# this lock so the bg detection thread and /analyze-image never run together.
log.info("Loading custom model...")
model      = YOLO("best.pt")
_yolo_lock = threading.Lock()
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
_overridden_slots: set = set()   # slot IDs currently in manual override — not overwritten by detection
_frame_count      = 0
_last_fb_push     = 0
_remap_requested  = False
_remap_layout_mode = "auto"       # mode requested by the last /remap call —
                                   # applied by the bg thread when it resets the mapper
_row_guides: list = []             # [{y, x1, x2}] dicts (camera coords) set by /remap; empty = auto

# Reserve-spot and parked-car constants
RESERVE_GUIDE_TOL_PX   = 80    # class 1/3 must be within this many px of a row guide to get a slot
PARK_FRAMES_THRESHOLD  = 240   # frames (~4 min at 1 FPS) a class-0 car must be still to auto-create slot
RESERVE_OVERRIDE_FRAMES = 180  # frames (~3 min at 1 FPS): car parked over a reserve slot rewrites its bbox
RESERVE_STABLE_SECS    = 240   # wall-clock seconds a class 1/3 must be stationary before creating a slot
MOVING_CAR_COOLDOWN_SECS = 4   # seconds before a drive-lane car becomes an active emoji on the driver UI
MOVING_CAR_MATCH_PX      = 80  # px radius to match a detection to the same car across frames
_parked_watcher:  dict = {}    # grid-key -> {cx, cy, box, still_count}
_reserve_watcher: dict = {}    # grid-key -> {cx, cy, box, still_count, cls_id}
_moving_watcher:  dict = {}    # grid-key -> {cx, cy, first_seen, last_seen} (drive-lane cars)

# ── AI-gen slot discovery state ───────────────────────────────────────────────
_ai_phase          = "idle"    # "idle" | "generating" | "review" | "error"
_ai_proposed_slots = {}        # proposed slots pending admin review
_ai_generated_image: bytes | None = None   # JPEG bytes of the AI-generated frame
_ai_error_msg      = ""

# ── Auto-mapper instance — set by the background thread, read by route handlers ─
mapper = None

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
    "conf_cls0":       0.20,   # per-class confidence floor — cars
    "conf_cls1":       0.60,   # per-class confidence floor — cones
    "conf_cls3":       0.60,   # per-class confidence floor — reserve (class 2=pwd not detected)
    "iou_threshold":   IOU_THRESHOLD,
    "smoothing_win":   SMOOTHING_WIN,
    "detect_interval": DETECT_INTERVAL,
    "firebase_every":  FIREBASE_EVERY,
    "yolo_every_n":    1,
    "max_reserve_box_area":  0,   # 0 = disabled; set e.g. 8000 to reject person-sized boxes for class 1/3
    "nano_banana_api_key":   os.getenv("NANO_BANANA_API_KEY", ""),  # primary AI provider
    "gemini_api_key":        os.getenv("GEMINI_API_KEY", ""),       # fallback provider
    "ai_prompt":             "",  # custom prompt for either provider (empty = use default)
}

# ── Load existing slot config if present ─────────────────────────────────────
# One-time migration: rename legacy slot_config.json to the pincode-scoped name.
_legacy_config = "slot_config.json"
if not os.path.exists(SLOT_CONFIG) and os.path.exists(_legacy_config):
    os.rename(_legacy_config, SLOT_CONFIG)
    log.info(f"Migrated {_legacy_config} → {SLOT_CONFIG}")

if os.path.exists(SLOT_CONFIG):
    with open(SLOT_CONFIG) as f:
        _raw = f.read().strip()
    if _raw:
        _slots = json.loads(_raw)
        log.info(f"Loaded {len(_slots)} slots from {SLOT_CONFIG}")
        _mapping_phase = False
    else:
        _slots = {}
        with open(SLOT_CONFIG, "w") as f:
            json.dump({}, f)
        log.info(f"Slot config {SLOT_CONFIG} was empty — initialized to {{}}")
else:
    with open(SLOT_CONFIG, "w") as f:
        json.dump({}, f)
    log.info(f"Created empty slot config {SLOT_CONFIG}")

if os.path.exists(GUIDE_CONFIG):
    try:
        with open(GUIDE_CONFIG) as f:
            _loaded_guides = json.load(f)
        if isinstance(_loaded_guides, list):
            _row_guides = _loaded_guides
            log.info(f"Loaded {len(_row_guides)} row guides from {GUIDE_CONFIG}")
    except Exception as _e:
        log.warning(f"Could not load {GUIDE_CONFIG}: {_e}")


# ── Pi identity + Firebase registration ──────────────────────────────────────

def get_zerotier_ip() -> str | None:
    """
    Return the IPv4 address of the active ZeroTier network, or None.
    Tries the ZeroTier local API first (works on Windows/Mac/Linux),
    then falls back to scanning for a zt* interface (Linux/Pi only).
    """
    import os, urllib.request, json as _json

    # ZeroTier local API — cross-platform, most reliable.
    # Works on Windows where ZeroTier adapters are GUID-named (not zt*).
    token_paths = [
        r"C:\ProgramData\ZeroTier\One\authtoken.secret",          # Windows
        "/var/lib/zerotier-one/authtoken.secret",                  # Linux / Pi
        os.path.expanduser("~/Library/Application Support/"
                           "ZeroTier/One/authtoken.secret"),       # macOS
    ]
    for p in token_paths:
        try:
            with open(p) as f:
                token = f.read().strip()
            req = urllib.request.Request(
                "http://localhost:9993/network",
                headers={"X-ZT1-Auth": token},
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                networks = _json.loads(resp.read())
            for net in networks:
                for addr in net.get("assignedAddresses", []):
                    ip = addr.split("/")[0]
                    if ":" not in ip:   # IPv4 only
                        return ip
        except OSError:
            continue   # token file not found on this OS — try next path
        except Exception:
            break       # found token but API call failed

    # zt* interface scan — Linux / Pi fallback if local API unavailable
    try:
        import netifaces
        for iface in netifaces.interfaces():
            if iface.startswith("zt"):
                addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])
                if addrs:
                    return addrs[0]["addr"]
    except ImportError:
        log.warning("netifaces not installed — run: pip install netifaces")

    return None


def resolve_pin_code() -> str:
    """
    Check /pi_config/active_pin in Firebase for a manual override.
    Falls back to LOCAL_PIN_CODE if none is set.
    """
    try:
        manual = db.reference("/pi_config/active_pin").get()
        if manual and isinstance(manual, str):
            log.info(f"Manual pin override active: {manual}")
            return manual
    except Exception as e:
        log.warning(f"Could not read pi_config/active_pin: {e}")
    return LOCAL_PIN_CODE


def register_pi() -> None:
    """Update /pi_config/active_pins heartbeat and /pi_registry metadata in Firebase."""
    # Always update active_pins regardless of ZeroTier connectivity
    try:
        active_pin = resolve_pin_code()
        db.reference(f"/pi_config/active_pins/{active_pin}").set(int(time.time() * 1000))
    except Exception as e:
        log.warning(f"Failed to update active_pins heartbeat: {e}")
        active_pin = LOCAL_PIN_CODE

    zt_ip = get_zerotier_ip()
    if not zt_ip:
        log.warning("ZeroTier not connected — skipping Pi registry update")
        return
    try:
        db.reference(f"/pi_registry/{LOCAL_PIN_CODE}").update({
            "pinCode":       LOCAL_PIN_CODE,
            "activePinCode": active_pin,
            "apiUrl":        f"http://{zt_ip}:{FLASK_PORT}",
            "ztIp":          zt_ip,
            "lastSeen":      int(time.time() * 1000),
        })
        log.debug(f"Registered as {LOCAL_PIN_CODE} → {zt_ip}:{FLASK_PORT} (active pin: {active_pin})")
    except Exception as e:
        log.warning(f"Pi registration failed: {e}")


# Set to the resolved pin code on startup; used by _deregister_pi for cleanup.
_current_active_pin = None


def _heartbeat_loop() -> None:
    """Update active_pins heartbeat and ZeroTier registry in Firebase every 15 seconds."""
    while True:
        time.sleep(15)
        try:
            register_pi()
        except Exception as e:
            log.warning(f"Heartbeat failed: {e}")


def _deregister_pi() -> None:
    """Remove this Pi from /pi_config/active_pins on clean shutdown."""
    if not FIREBASE_ENABLED or not _current_active_pin:
        return
    try:
        db.reference(f"/pi_config/active_pins/{_current_active_pin}").delete()
        log.info(f"[shutdown] Removed {_current_active_pin} from active_pins")
    except Exception:
        pass  # best-effort cleanup
    try:
        if firebase_instance:
            firebase_instance.clear_moving_cars()
    except Exception:
        pass


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
    cy = vbox[1] + (vbox[3] - vbox[1]) * 0.75  # bottom-biased: tires are at base of bbox

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


def _find_overlapping_reserve_slot(vbox):
    """Return (slot_id, slot) if vbox center falls inside a 'reserve' source slot, else None."""
    with _state_lock:
        snapshot = dict(_slots)
    for sid, slot in snapshot.items():
        if slot.get("source") != "reserve":
            continue
        if _check_overlap(vbox, slot["coords"]):
            return sid, slot
    return None


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
        _STATUS_NUM = {"Occupied": 1, "Reserved": 2, "Vacant": 0}
        _smoothing_hist[slot_id].append(_STATUS_NUM.get(status, 0))
        hist = list(_smoothing_hist[slot_id])
        counts = [hist.count(0), hist.count(1), hist.count(2)]
        winner = counts.index(max(counts))
        statuses[slot_id] = ["Vacant", "Occupied", "Reserved"][winner]
    return statuses


def _draw_boxes(frame, vehicle_boxes, slot_results=None):
    """Draw YOLO vehicle boxes and slot overlays. Supports quad and rect coords."""
    if slot_results:
        for slot in slot_results:
            coords = slot.get("coords")
            if not coords:
                continue
            occ = slot["status"] == "Occupied"
            res = slot["status"] == "Reserved"
            # BGR: occupied=red, reserved=orange(22,115,249), vacant=green
            color = (60, 60, 220) if occ else ((22, 115, 249) if res else (80, 200, 120))
            label = f"{slot['id']} {'OCC' if occ else ('RES' if res else 'VAC')}"

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

    occupied     = sum(1 for s in (slot_results or []) if s["status"] == "Occupied")
    reserved     = sum(1 for s in (slot_results or []) if s["status"] == "Reserved")
    total        = len(slot_results) if slot_results else 0
    car_count    = sum(1 for vb in vehicle_boxes if vb.get("cls_id", 0) == 0)
    marker_count = len(vehicle_boxes) - car_count
    cv2.rectangle(frame, (0, 0), (700, 44), (7, 10, 16), -1)
    cv2.putText(frame,
                f"YOLOv8n  |  {car_count} cars  {marker_count} markers  |  {occupied} occ  {reserved} res  / {total} slots",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _perp_dist_to_segment(px: float, py: float,
                           x1: float, y1: float, x2: float, y2: float):
    """
    Perpendicular distance from (px, py) to segment (x1,y1)-(x2,y2).
    Returns (distance, t) where t ∈ [0,1] is the parametric position on the
    segment at which the perpendicular foot lands.
    """
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx*dx + dy*dy
    if length_sq == 0:
        return math.hypot(px - x1, py - y1), 0.5
    t = max(0.0, min(1.0, ((px - x1)*dx + (py - y1)*dy) / length_sq))
    return math.hypot(px - (x1 + t*dx), py - (y1 + t*dy)), t


def _guide_mid_y(g) -> float:
    """Midpoint Y of a guide — used for top-to-bottom ordering (Row A, B, C …)."""
    if isinstance(g, dict):
        if "y1" in g:
            return (g["y1"] + g["y2"]) / 2
        return g.get("y", 0)
    return float(g)


def _point_within_guide_extent(px: float, py: float, g: dict) -> bool:
    """Return True if (px,py) projects within the extent of guide g (raw t ∈ [0,1])."""
    if not (isinstance(g, dict) and "y1" in g):
        return True  # legacy horizontal guides — no extent check
    dx, dy = g["x2"] - g["x1"], g["y2"] - g["y1"]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return True
    t = ((px - g["x1"]) * dx + (py - g["y1"]) * dy) / length_sq
    return 0.0 <= t <= 1.0


# ── Row / occupancy helpers ───────────────────────────────────────────────────

def _infer_row_from_y(cy: float, frame_h: float) -> str:
    """Return row letter A/B/C based on vertical thirds of the frame."""
    r = cy / max(frame_h, 1)
    return "A" if r < 0.33 else ("B" if r < 0.66 else "C")


def _row_label_for_point(cx: float, cy: float, guides: list) -> str:
    """
    Return a row letter for a detected object given the current row guides.
    Uses perpendicular distance to each guide segment for angled-guide support.
    Guides are dicts {x1,y1,x2,y2} (new) or {y,x1,x2} / plain ints (legacy).
    """
    if not guides:
        return "A"
    sorted_guides = sorted(guides, key=_guide_mid_y)
    best_idx, best_dist = 0, float("inf")
    for i, g in enumerate(sorted_guides):
        if isinstance(g, dict) and "y1" in g:
            d, _ = _perp_dist_to_segment(cx, cy, g["x1"], g["y1"], g["x2"], g["y2"])
        else:
            gy = g.get("y", 0) if isinstance(g, dict) else float(g)
            d = abs(cy - gy)
        if d < best_dist:
            best_dist, best_idx = d, i
    return chr(ord("A") + best_idx)


# ── Slot renumber helper ──────────────────────────────────────────────────────

def _renumber_and_persist():
    """Renumber _slots by position, save to disk, and push layout to Firebase."""
    global _slots
    _slots = renumber_slots_by_position(_slots)
    with open(SLOT_CONFIG, "w") as f:
        json.dump(dict(_slots), f, indent=2)
    if firebase_instance:
        firebase_instance.push_slot_layout(dict(_slots))
        firebase_instance.reset_slots({sid: "Vacant" for sid in _slots})


# ── Camera grabber — single shared RTSP session ──────────────────────────────
# Previously the bg detection thread and the MJPEG stream thread each opened
# their own VideoCapture against the same RTSP URL. That caused two problems:
#   1. Two RTSP sessions to one Hikvision compete for camera CPU and bandwidth,
#      producing decode failures even when the picture is visually clean.
#   2. The bg thread sleeps detect_interval (1s) between reads, which is longer
#      than the H.264 GOP. After waking, the decoder often can't recover because
#      its reference frames have been evicted — hence sporadic decode failures
#      during mapping/detection that don't correlate with any visible glitch.
#
# Fix: ONE grabber thread continuously decodes at native FPS and publishes the
# latest frame to shared state. Both the detection loop and the stream loop
# read from that shared state instead of opening their own captures.

# Max consecutive decode failures before forcing a full reconnect
_MAX_DECODE_FAILS = 8

# Max age (seconds) a published frame may have before consumers treat it as
# missing. Should be > one grabber cycle but small enough to detect a hung
# camera quickly.
_FRAME_STALE_AFTER = 2.0

# Shared frame state — read by detection loop and stream loop
_grabber_lock      = threading.Lock()
_grabber_frame     = None   # latest decoded numpy frame, or None
_grabber_frame_ts  = 0.0    # time.monotonic() of last successful decode
_grabber_alive     = False  # True once we have produced at least one frame
_grabber_stop      = False  # set True to ask the grabber to exit (not used yet)

# ── Video file playback state ─────────────────────────────────────────────────
_vid_lock   = threading.Lock()
_vid_path   = None       # path to loaded temp file, or None
_vid_state  = "stopped"  # "stopped" | "playing" | "paused"
_vid_frame  = 0          # current frame index
_vid_total  = 0          # total frames in loaded video
_vid_fps    = 25.0       # native FPS of loaded video


def _open_capture(retries: int = 5, delay: float = 3.0):
    """
    Open a VideoCapture for any source: local webcam, video file, or RTSP stream.
    Returns the open VideoCapture, or None after exhausting retries.
    Returns None early if a video file becomes ready to play during RTSP retries.
    """
    src_label = (
        f"webcam {VIDEO_SOURCE}" if isinstance(VIDEO_SOURCE, int)
        else ("RTSP stream" if _using_rtsp else f"file {VIDEO_SOURCE}")
    )
    for attempt in range(1, retries + 1):
        # If a video file was loaded (regardless of play state), bail so the
        # grabber can switch to video mode without waiting for this RTSP attempt.
        with _vid_lock:
            if _vid_path is not None:
                log.info("[CAM] Video file loaded — aborting RTSP retry.")
                return None
        log.info(f"[CAM] Opening {src_label} (attempt {attempt}/{retries})...")
        if _using_rtsp:
            cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC,     cv2.VideoWriter_fourcc(*"H264"))
            cap.set(cv2.CAP_PROP_FPS,        15)
        else:
            cap = cv2.VideoCapture(VIDEO_SOURCE)
        if cap.isOpened():
            if _using_rtsp:
                for _ in range(4):
                    cap.grab()
            log.info(f"[CAM] {src_label} opened successfully.")
            return cap
        cap.release()
        log.warning(f"[CAM] Attempt {attempt} failed — retrying in {delay}s...")
        # Sleep in short increments so a video-load request is noticed quickly.
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            with _vid_lock:
                if _vid_path is not None:
                    log.info("[CAM] Video file loaded — aborting RTSP retry.")
                    return None
            time.sleep(0.1)
    log.error(f"[CAM] Could not open {src_label} after {retries} attempts.")
    return None


def _grabber_loop():
    """
    Sole owner of all VideoCapture instances. Two modes:
      - Video file: reads the loaded file at its native FPS; supports
        play / pause / stop via _vid_* globals.
      - Live source (webcam / RTSP): continuously decodes so the decoder
        stays warm. Falls back here whenever no video file is playing.
    Both modes publish to the same _grabber_frame shared state.
    """
    global _grabber_frame, _grabber_frame_ts, _grabber_alive
    global _vid_frame, _vid_state

    live_cap              = None
    vid_cap               = None
    decode_fail_count     = 0
    decode_streak_logged  = False
    decode_streak_start   = 0.0

    while not _grabber_stop:
        # ── Video file mode ──────────────────────────────────────────────────
        with _vid_lock:
            vid_path  = _vid_path
            vid_state = _vid_state
            vid_fps   = _vid_fps
            vid_total = _vid_total

        # Desktop/webcam mode: idle until user explicitly loads a video file.
        # Without this guard the grabber opens webcam 0, feeds blank frames to
        # the detection loop, and the mapper runs forever with 0 vehicles.
        if vid_path is None and isinstance(VIDEO_SOURCE, int) and not _using_rtsp:
            time.sleep(0.1)
            continue

        if vid_path is not None and vid_state == "playing":
            if live_cap is not None:
                try: live_cap.release()
                except Exception: pass
                live_cap = None
                decode_fail_count    = 0
                decode_streak_logged = False

            if vid_cap is None or not vid_cap.isOpened():
                vid_cap = cv2.VideoCapture(vid_path)

            ret, frame = vid_cap.read()
            if ret and frame is not None:
                with _vid_lock:
                    _vid_frame = min(_vid_frame + 1, vid_total)
                with _grabber_lock:
                    _grabber_frame    = frame
                    _grabber_frame_ts = time.monotonic()
                    _grabber_alive    = True
                time.sleep(1.0 / max(vid_fps, 1))
            else:
                log.info("[VID] Playback finished.")
                with _vid_lock:
                    _vid_state = "stopped"
                    _vid_frame = 0
                if vid_cap is not None:
                    vid_cap.release()
                    vid_cap = None
            continue

        if vid_path is not None and vid_state == "paused":
            if live_cap is not None:
                try: live_cap.release()
                except Exception: pass
                live_cap = None
            time.sleep(0.05)
            continue

        if vid_path is not None and vid_state == "stopped":
            if vid_cap is not None:
                try: vid_cap.release()
                except Exception: pass
                vid_cap = None
            if not _using_rtsp and isinstance(VIDEO_SOURCE, int):
                time.sleep(0.1)
                continue

        # ── Live source mode (webcam / RTSP) ─────────────────────────────────
        if vid_cap is not None:
            try: vid_cap.release()
            except Exception: pass
            vid_cap = None

        if live_cap is None or not live_cap.isOpened():
            # Skip RTSP entirely if a video file is already loaded.
            with _vid_lock:
                if _vid_path is not None:
                    time.sleep(0.1)
                    continue
            live_cap = _open_capture()
            if live_cap is None:
                time.sleep(2.0)
                continue

        ret, frame = live_cap.read()

        if ret and frame is not None:
            if decode_streak_logged and decode_fail_count > 0:
                duration = time.time() - decode_streak_start
                log.info(f"[CAM] Recovered after {decode_fail_count} dropped frame(s) "
                         f"over {duration:.1f}s")
            decode_fail_count    = 0
            decode_streak_logged = False
            with _grabber_lock:
                _grabber_frame    = frame
                _grabber_frame_ts = time.monotonic()
                _grabber_alive    = True
            continue

        decode_fail_count += 1
        if not decode_streak_logged:
            decode_streak_logged = True
            decode_streak_start  = time.time()
            log.info("[CAM] Frame decode failure — suppressing further messages until recovery.")

        if decode_fail_count >= _MAX_DECODE_FAILS:
            duration = time.time() - decode_streak_start
            log.warning(f"[CAM] {decode_fail_count} consecutive decode failures "
                        f"over {duration:.1f}s — forcing reconnect.")
            decode_fail_count    = 0
            decode_streak_logged = False
            try:
                live_cap.release()
            except Exception:
                pass
            live_cap = None
            time.sleep(2.0)

        else:
            time.sleep(0.05)


def _get_raw_frame():
    """
    Non-blocking accessor — returns a copy of the latest grabber frame, or
    None if the grabber hasn't produced a frame yet or its frame is stale.

    Returning a copy means consumers can safely undistort/annotate without
    racing the grabber overwriting _grabber_frame mid-read.
    """
    with _grabber_lock:
        if _grabber_frame is None:
            return None
        age = time.monotonic() - _grabber_frame_ts
        if age > _FRAME_STALE_AFTER:
            return None
        return _grabber_frame.copy()


def _frame_age() -> float:
    """How old (seconds) the latest published frame is. inf if none yet."""
    with _grabber_lock:
        if _grabber_frame is None:
            return float("inf")
        return time.monotonic() - _grabber_frame_ts


# ── Background detection + sync thread ───────────────────────────────────────

# Module-level firebase reference so API routes can call push_undistort_config
firebase_instance = None


def _detection_loop():
    """
    Runs forever as a daemon thread.
    Owns: camera, YOLO inference, auto-mapper, Firebase sync, smoothing.
    Writes results into shared state for Flask routes to read.
    """
    global _latest_frame, _latest_raw_frame, _latest_statuses, _latest_vehicle_boxes, _latest_slot_results, _slots, _mapping_phase, _frame_count, _last_fb_push, firebase_instance, _remap_requested, _remap_layout_mode, mapper, _overridden_slots, _moving_watcher

    # ── Firebase ──────────────────────────────────────────────────────────────
    firebase = None
    if FIREBASE_ENABLED:
        if firebase_instance is not None:
            firebase = firebase_instance
            log.info("[FB] Using pre-initialized Firebase instance.")
        else:
            try:
                firebase = FirebaseSync(
                    credentials_path=CREDENTIALS,
                    database_url=FIREBASE_URL,
                    pin_code=LOCAL_PIN_CODE,
                    storage_bucket=FIREBASE_STORAGE_BUCKET,
                )
                firebase_instance = firebase
            except Exception as e:
                log.error(f"[FB] Firebase init failed — occupancy won't be synced: {e}")
    else:
        log.info("[FB] Firebase disabled — running in local-only mode.")

    # ── Auto-mapper ───────────────────────────────────────────────────────────
    with _state_lock:
        initial_mode = _remap_layout_mode
    with _state_lock:
        initial_guides = list(_row_guides)
    mapper = AutoMapper(
        slot_config_path=SLOT_CONFIG,
        min_frames_to_map=150,
        min_samples=3,
        eps_pixels=MAPPER_EPS_PX,
        layout_mode=initial_mode,
        row_guides=initial_guides,
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
    no_frame_streak   = 0      # consecutive iterations with no frame — used
                                # to throttle the "[BG] No frame" log line so
                                # camera glitches don't flood the CLI.

    # Camera is owned by the grabber thread (started below). Wait briefly for
    # the first frame to arrive before entering the main loop, just so we
    # don't immediately log a "no frame" warning at startup.
    log.info("[BG] Waiting for first frame from grabber...")
    for _ in range(50):   # up to ~5 seconds
        if _get_raw_frame() is not None:
            break
        time.sleep(0.1)
    log.info("[BG] Detection loop running.")

    while True:
        try:
            frame = _get_raw_frame()
            if frame is None:
                no_frame_streak += 1
                # Log the first miss, then again every 30s of sustained outage.
                # Avoids one warning per second when the camera is briefly down.
                if no_frame_streak == 1 or no_frame_streak % 30 == 0:
                    log.warning(f"[BG] No frame for {no_frame_streak}s — "
                                f"camera offline or stream stalled.")
                time.sleep(1)
                continue

            # Frame recovered
            if no_frame_streak > 0:
                if no_frame_streak >= 3:   # only summarize meaningful outages
                    log.info(f"[BG] Frames flowing again after {no_frame_streak}s outage.")
                no_frame_streak = 0

            # ── Check if admin triggered a remap ─────────────────────────────
            with _state_lock:
                do_remap = _remap_requested
                requested_mode = _remap_layout_mode
                if do_remap:
                    _remap_requested = False

            if do_remap:
                log.info(f"[BG] Remap request received (mode={requested_mode}) — "
                         f"resetting mapper and frame count.")
                mapper._detections   = []
                mapper._frame_count  = 0
                mapper._slots        = {}
                mapper.set_layout_mode(requested_mode)
                with _state_lock:
                    mapper.row_guides = list(_row_guides)
                local_frame_count    = 0
                sample_saved         = False
                with _state_lock:
                    _slots         = {}   # ensure stale slots don't leak into occupancy push
                    _mapping_phase = True
                log.info(f"[BG] Mapper reset in '{requested_mode}' mode. "
                         f"Starting fresh mapping phase (~150 frames)...")

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
                # Program config — compare outside lock, only acquire to write
                try:
                    new_prog = firebase.get_program_config()
                    with _state_lock:
                        current_prog = dict(_prog_cfg)
                    if new_prog != current_prog:   # Fix #4: comparison now outside lock
                        log.info(f"[BG] Program config changed: {new_prog}")
                        with _state_lock:
                            _prog_cfg.update(new_prog)
                except Exception as e:
                    log.warning(f"[BG] Program config poll failed: {e}")
                # Overridden slots — respect web-app manual overrides
                try:
                    new_overridden = firebase.get_overridden_slots()
                    with _state_lock:
                        _overridden_slots = new_overridden
                except Exception as e:
                    log.warning(f"[BG] Overridden slots poll failed: {e}")

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
                with _yolo_lock:   # Fix #1: prevent race with /analyze-image
                    results = model(frame, conf=conf, classes=TARGET_CLASSES, verbose=False)
                vehicle_boxes = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        vehicle_boxes.append({
                            "coords":     [x1, y1, x2, y2],
                            "confidence": round(float(box.conf[0]), 3),
                            "label":      model.names[int(box.cls[0])],
                            "cls_id":     int(box.cls[0]),
                        })
                # Apply per-class confidence thresholds (filter after global floor)
                _cls_confs = {
                    0: _prog_cfg.get("conf_cls0", 0.20),
                    1: _prog_cfg.get("conf_cls1", 0.45),
                    3: _prog_cfg.get("conf_cls3", 0.60),
                }
                vehicle_boxes = [
                    vb for vb in vehicle_boxes
                    if vb["confidence"] >= _cls_confs.get(vb["cls_id"], _prog_cfg.get("confidence", 0.20))
                ]
                last_yolo_frame = vehicle_boxes
            else:
                vehicle_boxes = last_yolo_frame or []

            with _state_lock:
                mapping_phase = _mapping_phase

            # ── Phase 1: Auto-Mapping ─────────────────────────────────────────
            if mapping_phase:
                # Only cluster class-0 cars; cones/stands are handled separately
                car_boxes_for_mapper = [
                    [v["coords"][0], v["coords"][1], v["coords"][2], v["coords"][3]]
                    for v in vehicle_boxes if v.get("cls_id", 0) == 0
                ]
                car_confs = [v["confidence"] for v in vehicle_boxes if v.get("cls_id", 0) == 0]
                _, _frame_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                mapper.feed_frame(car_boxes_for_mapper, frame.shape,
                                  frame_bytes=bytes(_frame_jpeg), confidences=car_confs)

                if local_frame_count % 30 == 0:
                    car_count = sum(1 for vb in vehicle_boxes if vb.get("cls_id", 0) == 0)
                    log.info(f"[BG] Mapping... frame {local_frame_count}/150 | cars: {car_count}")

                if mapper.is_mapping_complete():
                    discovered = mapper.get_slots()
                    with _state_lock:
                        _slots         = discovered
                        _mapping_phase = False
                    log.info(f"[BG] Auto-mapping complete — {len(discovered)} slots discovered.")
                    if firebase:
                        firebase.push_slot_layout(discovered)
                        firebase.reset_slots({sid: "Vacant" for sid in discovered})

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
                occ_vb = next((vb for vb in vehicle_boxes if _check_overlap(vb["coords"], coords)), None)
                if occ_vb is None:
                    statuses[slot_id] = "Vacant"
                elif occ_vb.get("cls_id", 0) in (1, 3):
                    statuses[slot_id] = "Reserved"
                else:
                    statuses[slot_id] = "Occupied"
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

            # ── Reserve-object stability watcher (class 1 & 3 near row guides) ────
            # Uses wall-clock time so fast detection loops / yolo_every_n reuse
            # cannot inflate the count. Walking person resets first_seen every
            # ~20 px of movement; a stationary cone accumulates real seconds.
            now = time.time()
            with _state_lock:
                guides         = list(_row_guides)
                cur_slots_snap = dict(_slots)
            frame_h = frame.shape[0]
            new_reserve_watcher: dict = {}
            for vb in vehicle_boxes:
                if vb.get("cls_id", 0) not in (1, 3):
                    continue
                with _state_lock:
                    max_area = _prog_cfg.get("max_reserve_box_area", 0)
                if max_area > 0:
                    _x1, _y1, _x2, _y2 = vb["coords"]
                    if (_x2 - _x1) * (_y2 - _y1) > max_area:
                        continue
                cx = (vb["coords"][0] + vb["coords"][2]) / 2
                cy = (vb["coords"][1] + vb["coords"][3]) / 2
                near_guide = (not guides) or any(
                    (
                        _perp_dist_to_segment(cx, cy, gy["x1"], gy["y1"], gy["x2"], gy["y2"])[0] < RESERVE_GUIDE_TOL_PX
                        and _point_within_guide_extent(cx, cy, gy)
                    )
                    if (isinstance(gy, dict) and "y1" in gy)
                    else abs(cy - (gy.get("y", 0) if isinstance(gy, dict) else gy)) < RESERVE_GUIDE_TOL_PX
                    for gy in guides
                )
                if not near_guide:
                    continue
                key  = f"{int(cx // 20)}_{int(cy // 20)}"
                prev = _reserve_watcher.get(key)
                if prev and np.hypot(cx - prev["cx"], cy - prev["cy"]) < 20:
                    first_seen = prev["first_seen"]   # still in same spot — keep original timestamp
                else:
                    first_seen = now                  # moved or brand-new — reset clock
                new_reserve_watcher[key] = {
                    "cx": cx, "cy": cy, "box": vb["coords"],
                    "cls_id": vb["cls_id"], "first_seen": first_seen,
                }
                if now - first_seen < RESERVE_STABLE_SECS:
                    continue
                already = any(_check_overlap(vb["coords"], s["coords"]) for s in cur_slots_snap.values())
                if already:
                    continue
                x1r, y1r, x2r, y2r = [int(c) for c in vb["coords"]]
                with _state_lock:
                    tmp_id = f"_tmp_{int(cx)}_{int(cy)}"
                    _slots[tmp_id] = {
                        "coords": [[x1r,y1r],[x2r,y1r],[x2r,y2r],[x1r,y2r]],
                        "row":    _row_label_for_point(cx, cy, guides) if guides else _infer_row_from_y(cy, frame_h),
                        "source": "reserve",
                    }
                    _renumber_and_persist()
                    cur_slots_snap = dict(_slots)
                log.info(f"[BG] Reserve slot created at ({int(cx)},{int(cy)}) after {now-first_seen:.1f}s stationary.")
            _reserve_watcher.clear()
            _reserve_watcher.update(new_reserve_watcher)

            # ── Parked-car watcher (class 0 stationary ≥ PARK_FRAMES_THRESHOLD) ─
            new_watcher: dict = {}
            for vb in vehicle_boxes:
                if vb.get("cls_id", 0) != 0:
                    continue
                cx = (vb["coords"][0] + vb["coords"][2]) / 2
                cy = (vb["coords"][1] + vb["coords"][3]) / 2
                key  = f"{int(cx // 20)}_{int(cy // 20)}"
                prev = _parked_watcher.get(key)
                if prev and np.hypot(cx - prev["cx"], cy - prev["cy"]) < 20:
                    still = prev["still_count"] + 1
                else:
                    still = 1
                new_watcher[key] = {"cx": cx, "cy": cy, "box": vb["coords"], "still_count": still}
                if still >= PARK_FRAMES_THRESHOLD:
                    with _state_lock:
                        already = any(_check_overlap(vb["coords"], s["coords"]) for s in _slots.values())
                    if not already:
                        x1p, y1p, x2p, y2p = [int(c) for c in vb["coords"]]
                        with _state_lock:
                            tmp_id = f"_tmp_{int(cx)}_{int(cy)}"
                            _slots[tmp_id] = {
                                "coords": [[x1p,y1p],[x2p,y1p],[x2p,y2p],[x1p,y2p]],
                                "row":    _row_label_for_point(cx, cy, guides) if guides else _infer_row_from_y(cy, frame_h),
                                "source": "parked",
                            }
                            _renumber_and_persist()
                        log.info(f"[BG] Auto-created parked-car slot at ({int(cx)},{int(cy)}) after {still}s still.")
                elif still >= RESERVE_OVERRIDE_FRAMES:
                    hit = _find_overlapping_reserve_slot(vb["coords"])
                    if hit:
                        sid, _ = hit
                        x1p, y1p, x2p, y2p = [int(c) for c in vb["coords"]]
                        with _state_lock:
                            _slots[sid]["coords"] = [[x1p,y1p],[x2p,y1p],[x2p,y2p],[x1p,y2p]]
                            _slots[sid]["source"] = "parked"
                            _renumber_and_persist()
                        log.info(f"[BG] Reserve slot {sid} bbox rewritten to car at ({int(cx)},{int(cy)}) after {still} frames still.")
            _parked_watcher.clear()
            _parked_watcher.update(new_watcher)

            # ── Moving-car watcher (class-0 cars in drive lanes, not near any row guide) ─
            # Requires row guides to be set — disabled when _row_guides is empty.
            if guides:
                new_moving: dict = {}
                mono_now = time.monotonic()
                for vb in vehicle_boxes:
                    if vb.get("cls_id", 0) != 0:
                        continue
                    cx = (vb["coords"][0] + vb["coords"][2]) / 2
                    cy = (vb["coords"][1] + vb["coords"][3]) / 2
                    if any(
                        (
                            _perp_dist_to_segment(cx, cy, gy["x1"], gy["y1"], gy["x2"], gy["y2"])[0] < RESERVE_GUIDE_TOL_PX
                            and _point_within_guide_extent(cx, cy, gy)
                        )
                        if (isinstance(gy, dict) and "y1" in gy)
                        else abs(cy - (gy.get("y", 0) if isinstance(gy, dict) else gy)) < RESERVE_GUIDE_TOL_PX
                        for gy in guides
                    ):
                        continue  # near a parking row — handled by _parked_watcher
                    matched_key = next(
                        (k for k, v in _moving_watcher.items()
                         if abs(cx - v["cx"]) < MOVING_CAR_MATCH_PX
                         and abs(cy - v["cy"]) < MOVING_CAR_MATCH_PX),
                        None,
                    )
                    first_seen = _moving_watcher[matched_key]["first_seen"] if matched_key else mono_now
                    new_key = f"car_{int(cx // 20)}_{int(cy // 20)}"
                    new_moving[new_key] = {
                        "cx": int(cx), "cy": int(cy),
                        "first_seen": first_seen, "last_seen": mono_now,
                    }
                _moving_watcher.clear()
                _moving_watcher.update(new_moving)
                active_movers = {
                    k: {"cx": v["cx"], "cy": v["cy"]}
                    for k, v in _moving_watcher.items()
                    if (mono_now - v["first_seen"]) >= MOVING_CAR_COOLDOWN_SECS
                }
                if firebase:
                    firebase.push_moving_cars(active_movers)
            else:
                if _moving_watcher:
                    _moving_watcher.clear()
                    if firebase:
                        firebase.push_moving_cars({})

            # Write shared state — stream thread handles frame annotation + encoding
            with _state_lock:
                _latest_vehicle_boxes = list(vehicle_boxes)
                _latest_slot_results  = list(slot_results)
                _latest_statuses      = dict(statuses)
                _frame_count          = local_frame_count

            # Firebase push every firebase_every cycles — skip manually overridden slots
            if firebase and local_frame_count % max(1, firebase_every) == 0:
                with _state_lock:
                    skip = set(_overridden_slots)
                firebase.push_occupancy(statuses, skip_slots=skip)

            occupied     = sum(1 for s in statuses.values() if s == "Occupied")
            reserved     = sum(1 for s in statuses.values() if s == "Reserved")
            car_count    = sum(1 for vb in vehicle_boxes if vb.get("cls_id", 0) == 0)
            marker_count = len(vehicle_boxes) - car_count
            if local_frame_count % 30 == 0:
                log.info(
                    f"[BG] Frame {local_frame_count} | "
                    f"Occ: {occupied}  Res: {reserved}  / {len(statuses)} | "
                    f"Cars: {car_count}  Markers: {marker_count} | "
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
    Dedicated thread: pulls the latest frame from the shared grabber, overlays
    YOLO results, and pushes annotated JPEGs to _stream_queue for MJPEG clients.
    Only runs when at least one browser tab has the stream open.

    Architecture change: previously this thread held its own VideoCapture
    against the same RTSP URL as the bg detection thread, causing camera-side
    contention. Now both threads share the grabber's _grabber_frame, so there
    is exactly one RTSP session open at a time.
    """
    global _stream_clients
    interval               = 1.0 / STREAM_FPS
    no_frame_streak        = 0       # consecutive iterations with no frame from grabber
    no_frame_streak_logged = False

    # Cache the undistorter — only rebuild when config actually changes
    _cached_udist     = None
    _cached_udist_cfg = {}

    while True:
        # If no clients, idle. We don't need to "release" anything because we
        # don't own a capture — the grabber stays connected regardless.
        with _stream_lock:
            clients = _stream_clients
        if clients == 0:
            time.sleep(0.5)
            no_frame_streak        = 0
            no_frame_streak_logged = False
            continue

        t0 = time.time()

        frame = _get_raw_frame()

        if frame is None:
            no_frame_streak += 1
            # Log only the first miss of a streak; grabber already logs its
            # own decode failures, so this is just extra context for the
            # stream-side perspective.
            if not no_frame_streak_logged and no_frame_streak >= 5:
                no_frame_streak_logged = True
                log.info("[STREAM] No fresh frames from grabber — waiting...")
            time.sleep(interval)
            continue

        # Recovery
        if no_frame_streak_logged:
            log.info(f"[STREAM] Frames flowing again after {no_frame_streak} miss(es).")
        no_frame_streak        = 0
        no_frame_streak_logged = False

        # Apply undistortion if enabled
        with _state_lock:
            udist_cfg = dict(_undistort_cfg)
        if udist_cfg.get("enabled"):
            try:
                # Rebuild only when config actually changes
                if udist_cfg != _cached_udist_cfg:
                    from undistort import WideAngleUndistorter
                    _cached_udist = WideAngleUndistorter(
                        k1=udist_cfg["k1"], k2=udist_cfg["k2"], alpha=udist_cfg["alpha"]
                    )
                    _cached_udist_cfg = dict(udist_cfg)
                if _cached_udist is not None:
                    frame = _cached_udist.process(frame)
            except Exception:
                pass

        # Read latest YOLO overlay (no lock held during draw — stale data is fine)
        with _state_lock:
            vboxes   = list(_latest_vehicle_boxes)
            slot_res = list(_latest_slot_results)
            mapping  = _mapping_phase

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
# Order matters: grabber must start first so consumers have frames to read.
_grabber_thread = threading.Thread(target=_grabber_loop, name="frame-grabber", daemon=True)
_grabber_thread.start()
log.info("[MAIN] Frame grabber thread started.")

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
        cam_ok       = _latest_raw_frame is not None   # Fix #3: was always True during mapping
    return jsonify({
        "online":        True,
        "camera":        cam_ok,   # Fix #3: now True only when a real frame exists
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


@app.route("/debug-frame", methods=["GET"])
def debug_frame():
    """Full-resolution high-quality snapshot for inspecting YOLO detection boxes."""
    with _state_lock:
        raw      = _latest_raw_frame.copy() if _latest_raw_frame is not None else None
        vboxes   = list(_latest_vehicle_boxes)
        slot_res = list(_latest_slot_results)

    if raw is None:
        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No frame yet", (200, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
        _, buf = cv2.imencode(".jpg", placeholder)
        return Response(buf.tobytes(), mimetype="image/jpeg",
                        headers={"Cache-Control": "no-cache"})

    annotated = _draw_boxes(raw, vboxes, slot_res)
    _, buf    = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
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
    with _state_lock:
        live_conf  = _prog_cfg["confidence"]
        _cls_confs = {
            0: _prog_cfg.get("conf_cls0", 0.20),
            1: _prog_cfg.get("conf_cls1", 0.45),
            3: _prog_cfg.get("conf_cls3", 0.60),
        }
    with _yolo_lock:
        results = model(frame, conf=live_conf, classes=TARGET_CLASSES, verbose=False)

    # Build vehicle_boxes identically to the live detection loop (includes cls_id)
    vehicle_boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            vehicle_boxes.append({
                "coords":     [round(x1), round(y1), round(x2), round(y2)],
                "confidence": round(float(box.conf[0]), 3),
                "label":      model.names[int(box.cls[0])],
                "cls_id":     int(box.cls[0]),
            })
    # Apply per-class confidence floors (same as live detection)
    vehicle_boxes = [
        vb for vb in vehicle_boxes
        if vb["confidence"] >= _cls_confs.get(vb["cls_id"], live_conf)
    ]

    with _state_lock:
        current_slots = dict(_slots)

    def _annotated_b64(frm, vboxes, slot_res):
        annotated = _draw_boxes(frm.copy(), vboxes, slot_res)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    if not current_slots:
        estimated = []
        for i, vb in enumerate(vehicle_boxes):
            # class 1/3 (traffic cone / reserve) → Reserved
            status = "Reserved" if vb["cls_id"] in (1, 3) else "Occupied"
            estimated.append({
                "id":         f"S{i+1:02d}",
                "status":     status,
                "confidence": vb["confidence"],
                "coords":     vb["coords"],
                "row":        "A" if vb["coords"][1] < h // 3 else ("B" if vb["coords"][1] < 2 * h // 3 else "C"),
            })
        occupied = sum(1 for s in estimated if s["status"] == "Occupied")
        reserved = sum(1 for s in estimated if s["status"] == "Reserved")
        return jsonify({
            "mode": "no_slot_config", "image_size": [w, h],
            "vehicles_detected": len(vehicle_boxes), "vehicle_boxes": vehicle_boxes,
            "slots": estimated, "total_slots": len(estimated),
            "occupied": occupied, "reserved": reserved, "vacant": 0,
            "annotated_image": _annotated_b64(frame, vehicle_boxes, estimated),
            "timestamp": int(time.time() * 1000),
        })

    slot_results   = []
    occupied_count = 0
    reserved_count = 0
    for slot_id, slot_data in current_slots.items():
        coords     = slot_data["coords"]
        matched_vb = next(
            (vb for vb in vehicle_boxes if _check_overlap(vb["coords"], coords)),
            None,
        )
        if matched_vb is None:
            status = "Vacant"
        elif matched_vb["cls_id"] in (1, 3):
            status = "Reserved"
            reserved_count += 1
        else:
            status = "Occupied"
            occupied_count += 1
        conf = matched_vb["confidence"] if matched_vb else round(0.88 + (hash(slot_id) % 10) * 0.01, 3)
        slot_results.append({
            "id":         slot_id,
            "status":     status,
            "confidence": round(conf, 3),
            "row":        slot_data.get("row", "A"),
            "coords":     coords,
        })

    vacant_count = len(current_slots) - occupied_count - reserved_count
    return jsonify({
        "mode": "slot_config", "image_size": [w, h],
        "vehicles_detected": len(vehicle_boxes), "vehicle_boxes": vehicle_boxes,
        "slots": slot_results, "total_slots": len(current_slots),
        "occupied": occupied_count, "reserved": reserved_count, "vacant": vacant_count,
        "occupancy_percent": round((occupied_count / max(1, len(current_slots))) * 100),
        "annotated_image": _annotated_b64(frame, vehicle_boxes, slot_results),
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
        "confidence":      max(0.05, min(0.95, float(data.get("confidence",      current["confidence"])))),
        "conf_cls0":       max(0.05, min(0.95, float(data.get("conf_cls0",       current.get("conf_cls0", 0.20))))),
        "conf_cls1":       max(0.05, min(0.95, float(data.get("conf_cls1",       current.get("conf_cls1", 0.45))))),
        "conf_cls3":       max(0.05, min(0.95, float(data.get("conf_cls3",       current.get("conf_cls3", 0.60))))),
        "iou_threshold":   max(0.1,  min(0.9,  float(data.get("iou_threshold",   current["iou_threshold"])))),
        "smoothing_win":   max(1,    min(30,   int(data.get("smoothing_win",     current["smoothing_win"])))),
        "detect_interval": max(0.0,  min(5.0,  float(data.get("detect_interval", current["detect_interval"])))),
        "firebase_every":  max(1,    min(30,   int(data.get("firebase_every",    current["firebase_every"])))),
        "yolo_every_n":    max(1,    min(10,   int(data.get("yolo_every_n",      current["yolo_every_n"])))),
        "max_reserve_box_area": max(0, int(data.get("max_reserve_box_area", current.get("max_reserve_box_area", 0)))),
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

    Body (optional JSON): {
      "layout_mode": "horizontal" | "vertical" | "grid" | "auto",
      "row_guides":  [y1, y2, ...]   # Y pixel values in camera coords; omit for auto
    }
    """
    global _remap_requested, _smoothing_hist, _remap_layout_mode, _row_guides, _parked_watcher, _slots

    data = request.get_json(silent=True) or {}
    mode = str(data.get("layout_mode", "auto")).lower().strip()
    valid_modes = {"horizontal", "vertical", "grid", "auto"}
    if mode not in valid_modes:
        log.warning(f"[ADMIN] /remap received invalid layout_mode='{mode}' — defaulting to 'auto'.")
        mode = "auto"

    raw_guides = data.get("row_guides", [])
    guides = []
    if isinstance(raw_guides, list):
        for g in raw_guides:
            if isinstance(g, dict):
                if "y1" in g:           # new angled format {x1,y1,x2,y2}
                    guides.append({"x1": int(g["x1"]), "y1": int(g["y1"]),
                                   "x2": int(g["x2"]), "y2": int(g["y2"])})
                else:                   # previous {y, x1, x2} horizontal → convert
                    y = int(g.get("y", 0))
                    guides.append({"x1": int(g.get("x1", 0)), "y1": y,
                                   "x2": int(g.get("x2", 9999)), "y2": y})
            elif isinstance(g, (int, float)):   # legacy plain int Y value
                guides.append({"x1": 0, "y1": int(g), "x2": 9999, "y2": int(g)})

    with _state_lock:
        _remap_requested   = True
        _remap_layout_mode = mode
        _smoothing_hist.clear()
        _row_guides        = guides
        _slots             = {}   # wipe in-memory slots so no stale push fires before BG thread resets
    _parked_watcher.clear()
    _reserve_watcher.clear()
    if guides:
        try:
            with open(GUIDE_CONFIG, "w") as _gf:
                json.dump(guides, _gf)
        except Exception as _e:
            log.warning(f"Could not save guides: {_e}")
    elif os.path.exists(GUIDE_CONFIG):
        os.remove(GUIDE_CONFIG)
    if os.path.exists(SLOT_CONFIG):
        os.remove(SLOT_CONFIG)
    # Clear Firebase immediately so the map goes blank during remapping
    if firebase_instance:
        firebase_instance.clear_slots_and_layout()
    log.info(f"[ADMIN] Remap requested (mode={mode}, {len(guides)} guides) — bg thread will reset mapper on next frame.")
    return jsonify({
        "status":      "remap_started",
        "layout_mode": mode,
        "row_guides":  guides,
        "message":     f"Auto-mapping restarted in '{mode}' mode with {len(guides)} row guide(s).",
    })


@app.route("/remap/guides", methods=["GET"])
def get_remap_guides():
    """Return the current row guide configuration and tolerance band."""
    with _state_lock:
        guides = list(_row_guides)
    return jsonify({"guides": guides, "tolerance_px": RESERVE_GUIDE_TOL_PX})


@app.route("/remap/suggest-guides", methods=["GET"])
def suggest_row_guides():
    """
    Suggest row guide lines.
    Strategy:
      1. Try gap-clustering by X and by Y (threshold 150 px).
      2. Pick whichever axis produces more multi-car (≥2) groups.
      3. For each group, fit a PCA line through that group's points.
    Returns [{x1, y1, x2, y2, label}].
    """
    import numpy as np

    with _state_lock:
        boxes = list(_latest_vehicle_boxes)

    car_boxes = [vb for vb in boxes if vb.get("cls_id", 0) == 0]
    if not car_boxes:
        return jsonify({"guides": [], "note": "No vehicles detected in current frame."})

    centers = []
    for vb in car_boxes:
        c = vb["coords"]
        centers.append(((c[0] + c[2]) / 2, (c[1] + c[3]) / 2))

    if len(centers) < 2:
        return jsonify({"guides": [], "note": "Need at least 2 vehicles to suggest guides."})

    GAP = 150

    def _gap_cluster(pts, axis):
        s = sorted(pts, key=lambda p: p[axis])
        clusters, cur = [], [s[0]]
        for p in s[1:]:
            if p[axis] - cur[-1][axis] > GAP:
                clusters.append(cur); cur = []
            cur.append(p)
        clusters.append(cur)
        return clusters

    y_cls = _gap_cluster(centers, 1)
    x_cls = _gap_cluster(centers, 0)

    def _usable(cls): return sum(1 for c in cls if len(c) >= 2)
    # Prefer the axis that produces more multi-car groups; tie goes to Y (horizontal rows)
    clusters = y_cls if _usable(y_cls) >= _usable(x_cls) else x_cls
    clusters = [c for c in clusters if len(c) >= 2][:6]

    if not clusters:
        return jsonify({"guides": [], "note": "Could not find multi-car rows in current frame."})

    PADDING = 80

    def _fit_line(pts):
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        cx, cy = xs.mean(), ys.mean()
        stacked = np.column_stack([xs - cx, ys - cy])
        _, _, vt = np.linalg.svd(stacked, full_matrices=False)
        d = vt[0]
        projs = stacked @ d
        p_min, p_max = projs.min() - PADDING, projs.max() + PADDING
        return (int(cx + p_min * d[0]), int(cy + p_min * d[1]),
                int(cx + p_max * d[0]), int(cy + p_max * d[1]))

    # Sort clusters by centroid Y (top→bottom = A, B, C…)
    clusters.sort(key=lambda cl: sum(p[1] for p in cl) / len(cl))

    guides = []
    for idx, cl in enumerate(clusters):
        x1, y1, x2, y2 = _fit_line(cl)
        guides.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                       "label": chr(ord("A") + idx)})

    return jsonify({"guides": guides})


# ── AI-Gen Slot Discovery endpoints ──────────────────────────────────────────

@app.route("/ai-mapping/best-frame", methods=["GET"])
def ai_best_frame():
    """Return the highest-confidence frame captured during the last mapping session."""
    if mapper is None:
        return jsonify({"error": "Mapper not ready — camera thread has not started."}), 503
    frame_bytes = mapper.get_best_frame()
    if frame_bytes is None:
        return jsonify({"error": "No best frame available — run a mapping session first."}), 404
    return frame_bytes, 200, {"Content-Type": "image/jpeg"}


@app.route("/ai-mapping/status", methods=["GET"])
def ai_mapping_status():
    global _ai_phase, _ai_proposed_slots, _ai_error_msg
    with _state_lock:
        return jsonify({
            "phase":               _ai_phase,
            "proposed_slot_count": len(_ai_proposed_slots),
            "error":               _ai_error_msg,
        })


@app.route("/ai-mapping/generate", methods=["POST"])
def ai_mapping_generate():
    """
    Trigger AI slot generation:
      1. Grab the best frame from the mapper.
      2. Send it to the AI provider (Nano Banana Pro preferred, Gemini fallback)
         to generate a filled-parking-lot image.
      3. Run YOLO on the result to extract slot quads.
    Body (optional JSON): { "nb_api_key": "...", "gemini_api_key": "...", "prompt": "..." }
    Keys fall back to _prog_cfg values if omitted.
    """
    global _ai_phase, _ai_proposed_slots, _ai_generated_image, _ai_error_msg

    if mapper is None:
        return jsonify({"error": "Mapper not ready — camera thread has not started."}), 503
    frame_bytes = mapper.get_best_frame()
    if frame_bytes is None:
        return jsonify({"error": "No best frame captured yet — complete a mapping session first."}), 400

    data           = request.get_json(silent=True) or {}
    nb_key         = data.get("nb_api_key")    or _prog_cfg.get("nano_banana_api_key", "")
    gemini_key     = data.get("gemini_api_key") or _prog_cfg.get("gemini_api_key", "")
    prompt         = data.get("prompt")         or _prog_cfg.get("ai_prompt", "")
    cone_boxes_snap = list(_latest_vehicle_boxes)  # snapshot for pre-processing

    if not nb_key and not gemini_key:
        return jsonify({"error": "No AI key set — add nano_banana_api_key or gemini_api_key in Firebase config."}), 400

    with _state_lock:
        if _ai_phase == "generating":
            return jsonify({"status": "generating", "message": "Already generating — please wait."})
        _ai_phase     = "generating"
        _ai_error_msg = ""

    def _generate_thread():
        global _ai_phase, _ai_proposed_slots, _ai_generated_image, _ai_error_msg
        blob_name = None
        try:
            cone_boxes = [
                list(map(int, vb["coords"]))
                for vb in cone_boxes_snap if vb.get("cls_id", 0) in (1, 3)
            ]
            if nb_key:
                # Nano Banana Pro — needs a public URL; upload frame to Firebase Storage first
                frame_url, blob_name = firebase_instance.upload_temp_frame(frame_bytes)
                try:
                    generated = generate_filled_lot_nb(frame_url, nb_key, prompt or None)
                except Exception as nb_exc:
                    log.warning(f"[AI/NB] NB Pro failed ({nb_exc}); falling back to Gemini.")
                    if not gemini_key:
                        raise
                    generated = generate_filled_lot(frame_bytes, gemini_key, prompt or None,
                                                    cone_boxes=cone_boxes)
            else:
                # Gemini only
                generated = generate_filled_lot(frame_bytes, gemini_key, prompt or None,
                                                cone_boxes=cone_boxes)

            with _state_lock:
                current_slot_ids = set(_slots.keys())
                live_conf        = _prog_cfg["confidence"]
                row_guides_snap  = list(_row_guides)

            with _yolo_lock:  # Fix #1: YOLO not thread-safe — guard shared model
                proposed = extract_slots_from_ai_frame(
                    generated, model, live_conf, current_slot_ids,
                    cam_frame_bytes=frame_bytes,
                    row_guides=row_guides_snap,
                    row_tol=RESERVE_GUIDE_TOL_PX,
                )
            log.info(f"[AI] Extraction complete — {len(proposed)} proposed slots")

            with _state_lock:
                _ai_generated_image = generated
                _ai_proposed_slots  = proposed
                _ai_phase           = "review"
        except Exception as exc:
            log.error(f"[AI] Generation failed: {exc}")
            with _state_lock:
                _ai_phase     = "error"
                _ai_error_msg = str(exc)
        finally:
            if blob_name and firebase_instance:
                firebase_instance.delete_temp_frame(blob_name)

    threading.Thread(target=_generate_thread, daemon=True).start()
    return jsonify({"status": "generating"})


@app.route("/ai-mapping/generated-image", methods=["GET"])
def ai_generated_image_endpoint():
    """Return the AI-generated filled-parking-lot image for UI preview."""
    with _state_lock:
        img = _ai_generated_image
    if img is None:
        return jsonify({"error": "No generated image available."}), 404
    return img, 200, {"Content-Type": "image/jpeg"}


@app.route("/ai-mapping/proposed-slots", methods=["GET"])
def ai_proposed_slots_endpoint():
    """Return the AI-proposed slots (pending admin review)."""
    with _state_lock:
        return jsonify(_ai_proposed_slots)


@app.route("/ai-mapping/confirm", methods=["POST"])
def ai_mapping_confirm():
    """
    Merge AI-proposed slots into the active slot set, persist to disk and Firebase.
    Clears the AI generation state and switches to live occupancy mode.
    Optional body: {"exclude": ["S01", "S03"]} — slot IDs to drop before confirming.
    """
    global _slots, _mapping_phase, _ai_phase, _ai_proposed_slots, _ai_generated_image

    data    = request.get_json(silent=True) or {}
    exclude = set(data.get("exclude", []))

    with _state_lock:
        if not _ai_proposed_slots:
            return jsonify({"error": "No proposed slots to confirm."}), 400
        filtered            = {k: v for k, v in _ai_proposed_slots.items() if k not in exclude}
        _slots              = renumber_slots_by_position(filtered)
        _mapping_phase      = False
        _ai_phase           = "idle"
        _ai_proposed_slots  = {}
        _ai_generated_image = None
        merged = dict(_slots)

    with open(SLOT_CONFIG, "w") as f:
        json.dump(merged, f, indent=2)
    if firebase_instance:
        firebase_instance.push_slot_layout(merged)
        firebase_instance.reset_slots({sid: "Vacant" for sid in merged})

    saved = len(filtered)
    log.info(f"[AI] Confirmed {saved} AI-generated slots (excluded {len(exclude)}) — replaced DBSCAN slots.")
    return jsonify({"saved": saved, "total": len(merged)})


@app.route("/ai-mapping/reject", methods=["POST"])
def ai_mapping_reject():
    """Discard AI-proposed slots and return to idle (regular mapping slots are kept)."""
    global _ai_phase, _ai_proposed_slots, _ai_generated_image
    with _state_lock:
        _ai_phase           = "idle"
        _ai_proposed_slots  = {}
        _ai_generated_image = None
    log.info("[AI] Admin rejected AI-proposed slots.")
    return jsonify({"status": "rejected"})


@app.route("/slots", methods=["POST"])
def add_slot():
    """
    Admin endpoint — add a brand-new slot with a given quad.
    Fix #7: previously there was no way to add a slot the automapper missed
    without hand-editing <PIN>_slot_config.json on the Pi.
    Body: { "slot_id": "S99", "coords": [[x,y],[x,y],[x,y],[x,y]], "row": "A" }
    """
    global _slots
    data     = request.get_json(silent=True) or {}
    slot_id  = data.get("slot_id", "").strip()
    coords   = data.get("coords")
    row      = data.get("row", "A")

    if not slot_id:
        return jsonify({"error": "slot_id is required"}), 400
    if not coords or len(coords) != 4:
        return jsonify({"error": "coords must be a list of 4 [x,y] points"}), 400

    with _state_lock:
        if slot_id in _slots:
            return jsonify({"error": f"Slot {slot_id} already exists — use PUT to update"}), 409
        _slots[slot_id] = {"coords": coords, "row": row, "source": "manual"}
        _renumber_and_persist()
        assigned_id = next((k for k, v in _slots.items() if v.get("coords") == coords), slot_id)

    log.info(f"[ADMIN] Slot {assigned_id} added manually at row={row}.")
    return jsonify({"status": "ok", "slot_id": assigned_id, "coords": coords, "row": row}), 201


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


# ── Video file playback endpoints ─────────────────────────────────────────────
@app.route("/video/load", methods=["POST"])
def video_load():
    """Accept an uploaded video file and prepare it for playback."""
    global _vid_path, _vid_state, _vid_frame, _vid_total, _vid_fps
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "no file uploaded"}), 400
    ext = os.path.splitext(f.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.save(tmp.name)
    tmp.close()
    cap   = cv2.VideoCapture(tmp.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    with _vid_lock:
        if _vid_path and os.path.exists(_vid_path):
            try: os.unlink(_vid_path)
            except Exception: pass
        _vid_path  = tmp.name
        _vid_state = "stopped"
        _vid_frame = 0
        _vid_total = total
        _vid_fps   = fps
    log.info(f"[VID] Loaded {f.filename} — {total} frames @ {fps:.1f} FPS")
    return jsonify({"status": "loaded", "total": total, "fps": round(fps, 1)})


@app.route("/video/start", methods=["POST"])
def video_start():
    """Start or resume video playback."""
    global _vid_state
    with _vid_lock:
        if not _vid_path:
            return jsonify({"error": "no video loaded"}), 400
        _vid_state = "playing"
    log.info("[VID] Playback started.")
    return jsonify({"status": "playing"})


@app.route("/video/pause", methods=["POST"])
def video_pause():
    """Pause video playback."""
    global _vid_state
    with _vid_lock:
        _vid_state = "paused"
    log.info("[VID] Playback paused.")
    return jsonify({"status": "paused"})


@app.route("/video/stop", methods=["POST"])
def video_stop():
    """Stop video playback and reset to beginning. Grabber falls back to RTSP."""
    global _vid_state, _vid_frame
    with _vid_lock:
        _vid_state = "stopped"
        _vid_frame = 0
    log.info("[VID] Playback stopped — grabber will revert to RTSP stream.")
    return jsonify({"status": "stopped", "source": "rtsp"})


@app.route("/video/unload", methods=["POST"])
def video_unload():
    """Unload the video file and return to live RTSP stream."""
    global _vid_path, _vid_state, _vid_frame
    with _vid_lock:
        if _vid_path and os.path.exists(_vid_path):
            try:
                os.unlink(_vid_path)
            except Exception:
                pass
        _vid_path  = None
        _vid_state = "stopped"
        _vid_frame = 0
    log.info("[VID] Video unloaded — reverting to RTSP stream.")
    return jsonify({"status": "unloaded", "source": "rtsp", "rtsp_url": RTSP_URL})


@app.route("/video/status", methods=["GET"])
def video_status():
    """Return current video playback state and active source."""
    with _vid_lock:
        path  = _vid_path
        state = _vid_state
        frame = _vid_frame
        total = _vid_total
        fps   = _vid_fps
    active_source = "file" if path is not None else "rtsp"
    return jsonify({
        "state":        state,
        "frame":        frame,
        "total":        total,
        "fps":          round(fps, 1),
        "source":       active_source,
        "rtsp_url":     RTSP_URL,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Smart Parking Flask API (localhost) ===")
    print(f"Pi ID       : {LOCAL_PIN_CODE}")
    print(f"Live source : {RTSP_URL}")
    print(f"Video file  : POST /video/load  →  POST /video/start")
    print(f"Switch back : POST /video/stop  or  POST /video/unload")
    print(f"API         : http://localhost:{FLASK_PORT}")
    print("Routes: /status  /slots  /occupancy  /live-frame  /analyze-image  /remap")
    print("        /slots/<id> PUT/DELETE  /program-config  /undistort-config")
    print("        /video/load  /video/start  /video/pause  /video/stop  /video/unload  /video/status")
    print("=========================================\n")

    if FIREBASE_ENABLED:
        import firebase_admin
        from firebase_admin import credentials as _fb_creds
        try:
            _cred = _fb_creds.Certificate(CREDENTIALS)
            _fb_options = {"databaseURL": FIREBASE_URL}
            if FIREBASE_STORAGE_BUCKET:
                _fb_options["storageBucket"] = FIREBASE_STORAGE_BUCKET
            firebase_admin.initialize_app(_cred, _fb_options)
        except ValueError:
            pass  # Already initialized by the background detection thread

        active_pin = resolve_pin_code()
        firebase_instance = FirebaseSync(
            credentials_path=CREDENTIALS,
            database_url=FIREBASE_URL,
            pin_code=active_pin,
            storage_bucket=FIREBASE_STORAGE_BUCKET,
        )

        # Store resolved pin so _deregister_pi knows which entry to remove
        _current_active_pin = active_pin

        register_pi()
        threading.Thread(target=_heartbeat_loop, daemon=True).start()

        # Register shutdown handlers so the pin is removed from active_pins
        # when this process exits cleanly (Ctrl+C, SIGTERM, service stop, etc.)
        import atexit, signal as _signal, sys as _sys
        atexit.register(_deregister_pi)
        try:
            _signal.signal(_signal.SIGTERM, lambda *_: _sys.exit(0))
        except (OSError, ValueError):
            pass  # SIGTERM not available on all platforms

    try:
        app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, threaded=True)
    finally:
        _deregister_pi()  # covers KeyboardInterrupt / unexpected Flask exit