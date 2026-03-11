"""
firebase_sync.py — Pushes occupancy data to Firebase Realtime Database.
"""

import time
import logging
import firebase_admin
from firebase_admin import credentials, db

log = logging.getLogger(__name__)


class FirebaseSync:
    def __init__(self, credentials_path: str, database_url: str):
        """
        Args:
            firebase = FirebaseSync(credentials_path="serviceAccountKey.json",
	database_url="https://automapping-parking-slot-default-rtdb.asia-southeast1.firebasedatabase.app"
)
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
        total = len(statuses)
        occupied = sum(1 for s in statuses.values() if s == "Occupied")
        vacant = total - occupied

        payload = {
            "slots": statuses,
            "summary": {
                "total": total,
                "occupied": occupied,
                "vacant": vacant,
                "last_updated": int(time.time() * 1000),   # ms epoch
            }
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
        """
        Write a notification record that the web app listens for.
        """
        payload = {
            "message": message,
            "slot_id": slot_id,
            "timestamp": int(time.time() * 1000),
        }
        try:
            db.reference("/notifications").push(payload)
        except Exception as e:
            log.warning(f"Notification push failed: {e}")
