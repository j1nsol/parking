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
    "enabled": False,
    "k1":      -0.3,
    "k2":      0.1,
    "alpha":   0.0,
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