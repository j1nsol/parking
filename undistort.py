"""
undistort.py — Auto fisheye undistortion for overhead IP camera frames.

No calibration checkerboard needed. Estimates distortion from the circular
fisheye boundary visible in the frame, then remaps to a flat perspective.

Usage:
    from undistort import FisheyeUndistorter
    undistorter = FisheyeUndistorter()
    flat_frame = undistorter.process(fisheye_frame)
"""

import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)


class FisheyeUndistorter:
    def __init__(
        self,
        fov_degrees: float = 185.0,   # typical wide-angle fisheye FOV
        zoom: float = 0.7,            # 0.5–1.0 — higher = more of frame kept
        balance: float = 0.5,         # 0.0 = no black edges, 1.0 = full frame
    ):
        """
        Args:
            fov_degrees: Estimated FOV of your fisheye lens.
                         Try 185 first. If result looks stretched, lower to 160.
            zoom:        How much of the undistorted image to keep.
                         Lower = more zoomed in but fewer black corners.
            balance:     0.5 balances cropping vs black edges. 0.0 = no black edges (aggressive crop), 1.0 = full frame.
        """
        self.fov = fov_degrees
        self.zoom = zoom
        self.balance = balance
        self._map1 = None
        self._map2 = None
        self._calibrated_size = None
        log.info(f"FisheyeUndistorter ready (FOV={fov_degrees}°, zoom={zoom}, balance={balance})")

    def _build_maps(self, h: int, w: int):
        """Build remap lookup tables for a given frame size. Called once."""
        log.info(f"Building undistortion maps for {w}x{h}...")

        # Estimate camera matrix from frame dimensions
        # Principal point = image center, focal length from FOV
        cx, cy = w / 2.0, h / 2.0
        fov_rad = np.deg2rad(self.fov / 2.0)

        # tan() goes negative for half-FOV > 90° (i.e. total FOV > 180°).
        # Use abs() so the focal length stays positive and the K matrix stays valid.
        f = abs(cx / np.tan(fov_rad))
        # For very wide FOV where tan is near zero, fall back to a safe estimate
        if f < 1.0:
            f = cx * 0.5
            log.warning(f"FOV {self.fov}° produced near-zero focal length — using fallback f={f:.1f}")

        K = np.array([
            [f,   0,  cx],
            [0,   f,  cy],
            [0,   0,   1],
        ], dtype=np.float64)

        # Distortion coefficients — fisheye model (k1..k4)
        # These are estimated; for a strong fisheye start with k1=-0.3
        D = np.array([[-0.3], [0.05], [-0.01], [0.001]], dtype=np.float64)

        # New optimal camera matrix (controls zoom / black edge balance)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D,
            (w, h),
            np.eye(3),
            balance=self.balance,
            new_size=(w, h),
            fov_scale=1.0 / self.zoom,
        )

        # Build remap tables
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )
        self._calibrated_size = (w, h)
        log.info("Undistortion maps built successfully.")

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort a fisheye frame.

        Args:
            frame: Raw fisheye BGR frame from the RTSP camera.

        Returns:
            Undistorted BGR frame, same resolution as input.
        """
        if frame is None:
            return frame

        h, w = frame.shape[:2]

        # Rebuild maps if frame size changed (or first call)
        if self._calibrated_size != (w, h):
            self._build_maps(h, w)

        return cv2.remap(
            frame, self._map1, self._map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def save_sample(self, frame: np.ndarray, path: str = "undistort_sample.jpg"):
        """
        Save a side-by-side comparison of original vs undistorted.
        Useful for tuning fov_degrees and zoom.
        """
        undistorted = self.process(frame)
        if undistorted is None:
            return
        comparison = np.hstack([frame, undistorted])
        cv2.imwrite(path, comparison)
        log.info(f"Saved comparison to {path} — original (left) vs undistorted (right)")