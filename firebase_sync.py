"""
firebase_sync.py — Pushes occupancy data to Firebase Realtime Database.
"""

import time
import logging
import firebase_admin
from firebase_admin import credentials, db, storage

log = logging.getLogger(__name__)

# Default undistort config — written to Firebase on first run if not present
DEFAULT_UNDISTORT_CONFIG = {
    "enabled": False,
    "k1":      -0.3,
    "k2":      0.1,
    "alpha":   0.0,
}

DEFAULT_PROGRAM_CONFIG = {
    "confidence":       0.20,
    "conf_cls0":        0.20,
    "conf_cls1":        0.45,
    "conf_cls2":        0.45,
    "iou_threshold":    0.35,
    "smoothing_win":    5,
    "detect_interval":  1.0,
    "firebase_every":   2,
    "yolo_every_n":     1,
}


class FirebaseSync:
    def __init__(self, credentials_path: str, database_url: str, pin_code: str = "default",
                 storage_bucket: str = ""):
        """
        Args:
            credentials_path: Path to Firebase service account JSON key.
            database_url:     Firebase Realtime Database URL.
            pin_code:         Resolved pin code — sets base path to /locations/{pin_code}.
            storage_bucket:   Firebase Storage bucket name (e.g. "project-id.appspot.com").
                              Required only for AI slot generation via Nano Banana.
        """
        self._base           = f"locations/{pin_code.strip('/')}"
        self._storage_bucket = storage_bucket
        options = {"databaseURL": database_url}
        if storage_bucket:
            options["storageBucket"] = storage_bucket
        try:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred, options)
            log.info(f"Firebase connected successfully (path=/{self._base}).")
        except ValueError:
            log.info(f"Firebase already initialized (path=/{self._base}).")
        except Exception as e:
            log.error(f"Firebase init failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Push occupancy status
    # ------------------------------------------------------------------
    def push_occupancy(self, statuses: dict):
        """Write {slotId: status_string} to /locations/{pinCode}/slots."""
        payload = {
            slot_id: {
                "status":    status,
                "updatedAt": int(time.time() * 1000),
            }
            for slot_id, status in statuses.items()
        }
        try:
            db.reference(f"/{self._base}/slots").update(payload)
        except Exception as e:
            log.warning(f"Firebase occupancy push failed: {e}")

    # ------------------------------------------------------------------
    # Push slot layout (after auto-mapping)
    # ------------------------------------------------------------------
    def push_slot_layout(self, slots: dict):
        """Write full slot layout to /locations/{pinCode}/layout."""
        from collections import defaultdict

        def _cx(coords):
            if not coords:
                return 0
            if isinstance(coords[0], (list, tuple)):
                return sum(p[0] for p in coords) / len(coords)
            return (coords[0] + coords[2]) / 2  # legacy [x1,y1,x2,y2]

        rows = defaultdict(list)
        for slot_id, s in slots.items():
            rows[s.get("row", "?")].append((slot_id, _cx(s.get("coords"))))

        col_map = {}
        for row_slots in rows.values():
            for idx, (slot_id, _) in enumerate(sorted(row_slots, key=lambda t: t[1]), start=1):
                col_map[slot_id] = idx

        layout = {
            slot_id: {
                "coords": s.get("coords"),
                "row":    s.get("row"),
                "col":    col_map.get(slot_id),
            }
            for slot_id, s in slots.items()
        }
        try:
            db.reference(f"/{self._base}/layout").set(layout)
            log.info(f"Slot layout pushed to /{self._base}/layout.")
        except Exception as e:
            log.warning(f"Firebase layout push failed: {e}")

    # ------------------------------------------------------------------
    # Clear slots + layout at remap start
    # ------------------------------------------------------------------
    def clear_slots_and_layout(self):
        """Wipe both /slots and /layout immediately when a remap begins."""
        try:
            db.reference(f"/{self._base}/slots").set({})
            db.reference(f"/{self._base}/layout").set({})
            log.info(f"Cleared slots and layout in /{self._base}.")
        except Exception as e:
            log.warning(f"Firebase clear failed: {e}")

    # ------------------------------------------------------------------
    # Reset slots (used after a remap to clear stale entries)
    # ------------------------------------------------------------------
    def reset_slots(self, initial_statuses: dict):
        """Replace /slots entirely — clears stale entries from prior sessions."""
        payload = {
            slot_id: {
                "status":    status,
                "updatedAt": int(time.time() * 1000),
            }
            for slot_id, status in initial_statuses.items()
        }
        try:
            db.reference(f"/{self._base}/slots").set(payload)
            log.info(f"Slots reset to {len(payload)} entries in /{self._base}/slots.")
        except Exception as e:
            log.warning(f"Firebase slots reset failed: {e}")

    # ------------------------------------------------------------------
    # Push notification trigger
    # ------------------------------------------------------------------
    def push_notification(self, message: str, slot_id: str = None):
        """Write a notification record that the web app listens for."""
        payload = {
            "message":   message,
            "slot_id":   slot_id,
            "timestamp": int(time.time() * 1000),
        }
        try:
            db.reference("/notifications").push(payload)
        except Exception as e:
            log.warning(f"Notification push failed: {e}")

    # ------------------------------------------------------------------
    # Undistort config — read / write / init
    # ------------------------------------------------------------------
    def get_undistort_config(self) -> dict:
        """
        Read undistort config from Firebase.
        Returns the stored config, or DEFAULT_UNDISTORT_CONFIG if not set.
        Also initialises the key with defaults if it doesn't exist yet.
        """
        try:
            ref = db.reference("/undistort_config")
            val = ref.get()
            if val is None:
                # First run — seed with defaults
                ref.set(DEFAULT_UNDISTORT_CONFIG)
                log.info("[FB] Undistort config initialised with defaults.")
                return dict(DEFAULT_UNDISTORT_CONFIG)
            return {
                "enabled": bool(val.get("enabled", DEFAULT_UNDISTORT_CONFIG["enabled"])),
                "k1":      float(val.get("k1",      DEFAULT_UNDISTORT_CONFIG["k1"])),
                "k2":      float(val.get("k2",      DEFAULT_UNDISTORT_CONFIG["k2"])),
                "alpha":   float(val.get("alpha",   DEFAULT_UNDISTORT_CONFIG["alpha"])),
            }
        except Exception as e:
            log.warning(f"Failed to read undistort config — using defaults: {e}")
            return dict(DEFAULT_UNDISTORT_CONFIG)

    def push_undistort_config(self, enabled: bool, k1: float, k2: float, alpha: float):
        """Write undistort config to Firebase."""
        payload = {
            "enabled": enabled,
            "k1":      round(float(k1),    2),
            "k2":      round(float(k2),    3),
            "alpha":   round(float(alpha), 1),
        }
        try:
            db.reference("/undistort_config").set(payload)
            log.info(f"[FB] Undistort config pushed: {payload}")
        except Exception as e:
            log.warning(f"Failed to push undistort config: {e}")

    # ------------------------------------------------------------------
    # Program config — read / write / init
    # ------------------------------------------------------------------
    def get_program_config(self) -> dict:
        """
        Read program config from Firebase.
        Returns stored config, or DEFAULT_PROGRAM_CONFIG if not set yet.
        Seeds defaults on first run.
        """
        try:
            ref = db.reference("/program_config")
            val = ref.get()
            if val is None:
                ref.set(DEFAULT_PROGRAM_CONFIG)
                log.info("[FB] Program config initialised with defaults.")
                return dict(DEFAULT_PROGRAM_CONFIG)
            d = DEFAULT_PROGRAM_CONFIG
            return {
                "confidence":      float(val.get("confidence",      d["confidence"])),
                "conf_cls0":       float(val.get("conf_cls0",       d["conf_cls0"])),
                "conf_cls1":       float(val.get("conf_cls1",       d["conf_cls1"])),
                "conf_cls2":       float(val.get("conf_cls2",       d["conf_cls2"])),
                "iou_threshold":   float(val.get("iou_threshold",   d["iou_threshold"])),
                "smoothing_win":   int(val.get("smoothing_win",     d["smoothing_win"])),
                "detect_interval": float(val.get("detect_interval", d["detect_interval"])),
                "firebase_every":  int(val.get("firebase_every",    d["firebase_every"])),
                "yolo_every_n":    int(val.get("yolo_every_n",      d["yolo_every_n"])),
            }
        except Exception as e:
            log.warning(f"Failed to read program config — using defaults: {e}")
            return dict(DEFAULT_PROGRAM_CONFIG)

    def push_program_config(self, cfg: dict):
        """Write program config to Firebase."""
        payload = {
            "confidence":      round(float(cfg["confidence"]),      2),
            "conf_cls0":       round(float(cfg.get("conf_cls0", 0.20)), 2),
            "conf_cls1":       round(float(cfg.get("conf_cls1", 0.45)), 2),
            "conf_cls2":       round(float(cfg.get("conf_cls2", 0.45)), 2),
            "iou_threshold":   round(float(cfg["iou_threshold"]),   2),
            "smoothing_win":   int(cfg["smoothing_win"]),
            "detect_interval": round(float(cfg["detect_interval"]), 1),
            "firebase_every":  int(cfg["firebase_every"]),
            "yolo_every_n":    int(cfg["yolo_every_n"]),
        }
        try:
            db.reference("/program_config").set(payload)
            log.info(f"[FB] Program config pushed: {payload}")
        except Exception as e:
            log.warning(f"Failed to push program config: {e}")

    # ------------------------------------------------------------------
    # Firebase Storage — temp frame hosting for AI slot generation
    # ------------------------------------------------------------------
    def upload_temp_frame(self, jpeg_bytes: bytes) -> tuple[str, str]:
        """
        Upload a JPEG frame to Firebase Storage as a public temp file.

        Returns (public_url, blob_name) so the caller can delete it after use.
        Raises RuntimeError if storage_bucket was not configured.
        """
        if not self._storage_bucket:
            raise RuntimeError(
                "Firebase Storage not configured — set FIREBASE_STORAGE_BUCKET in flask_api.py."
            )
        blob_name = f"ai_temp/frame_{int(time.time() * 1000)}.jpg"
        bucket = storage.bucket(self._storage_bucket)
        blob   = bucket.blob(blob_name)
        blob.upload_from_string(jpeg_bytes, content_type="image/jpeg")
        blob.make_public()
        url = blob.public_url
        log.info(f"[FB] Temp frame uploaded → {url}")
        return url, blob_name

    def delete_temp_frame(self, blob_name: str):
        """Delete a previously uploaded temp frame from Firebase Storage."""
        if not self._storage_bucket:
            return
        try:
            storage.bucket(self._storage_bucket).blob(blob_name).delete()
            log.info(f"[FB] Temp frame deleted: {blob_name}")
        except Exception as e:
            log.warning(f"[FB] Failed to delete temp frame '{blob_name}': {e}")
