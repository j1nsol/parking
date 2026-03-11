"""
firebase_sync.py — Pushes occupancy data to Firebase Realtime Database.
"""

import time
import logging
import firebase_admin
from firebase_admin import credentials, db

log = logging.getLogger(__name__)

# Default undistort config — written to Firebase on first run if not present
DEFAULT_UNDISTORT_CONFIG = {
    "enabled":     False,
    "fov_degrees": 185.0,
    "zoom":        0.7,
}


class FirebaseSync:
    def __init__(self, credentials_path: str, database_url: str):
        """
        Args:
            credentials_path: Path to Firebase service account JSON key.
            database_url:     Firebase Realtime Database URL.
        """
        try:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred, {"databaseURL": database_url})
            log.info("Firebase connected successfully.")
        except Exception as e:
            log.error(f"Firebase init failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Push occupancy status
    # ------------------------------------------------------------------
    def push_occupancy(self, statuses: dict[str, str]):
        """
        Push slot statuses + summary stats to Firebase.

        Args:
            statuses: Dict of slot_id -> "Occupied" | "Vacant"
        """
        total    = len(statuses)
        occupied = sum(1 for s in statuses.values() if s == "Occupied")
        vacant   = total - occupied

        payload = {
            "slots": statuses,
            "summary": {
                "total":        total,
                "occupied":     occupied,
                "vacant":       vacant,
                "last_updated": int(time.time() * 1000),
            },
        }

        try:
            db.reference("/parking").set(payload)
        except Exception as e:
            log.warning(f"Firebase push failed (will retry next cycle): {e}")

    # ------------------------------------------------------------------
    # Push slot layout (after auto-mapping)
    # ------------------------------------------------------------------
    def push_slot_layout(self, slots: dict):
        """
        Push the discovered slot coordinate layout so the web app
        can render the parking map overlay.
        """
        try:
            db.reference("/slot_layout").set(slots)
            log.info("Slot layout pushed to Firebase.")
        except Exception as e:
            log.warning(f"Failed to push slot layout: {e}")

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
                "enabled":     bool(val.get("enabled",     DEFAULT_UNDISTORT_CONFIG["enabled"])),
                "fov_degrees": float(val.get("fov_degrees", DEFAULT_UNDISTORT_CONFIG["fov_degrees"])),
                "zoom":        float(val.get("zoom",        DEFAULT_UNDISTORT_CONFIG["zoom"])),
            }
        except Exception as e:
            log.warning(f"Failed to read undistort config — using defaults: {e}")
            return dict(DEFAULT_UNDISTORT_CONFIG)

    def push_undistort_config(self, enabled: bool, fov_degrees: float, zoom: float):
        """
        Write undistort config to Firebase.
        Called by the web admin panel when the user applies new settings.
        """
        payload = {
            "enabled":     enabled,
            "fov_degrees": round(float(fov_degrees), 1),
            "zoom":        round(float(zoom), 2),
        }
        try:
            db.reference("/undistort_config").set(payload)
            log.info(f"[FB] Undistort config pushed: {payload}")
        except Exception as e:
            log.warning(f"Failed to push undistort config: {e}")