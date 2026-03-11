"""
undistort.py — Barrel distortion correction for standard wide-angle IP cameras.

Designed for: Hikvision HiWatch HWI-B140H and similar wide-angle lenses.
              Horizontal FOV ~98°, NOT a fisheye lens.

Uses OpenCV's standard lens model (cv2.undistort), NOT cv2.fisheye,
which is only appropriate for lenses with FOV > 160°.

The camera matrix K is estimated from the known FOV spec of the lens.
Distortion coefficients (k1, k2) are tunable — run tune_undistort.py
to find values that make straight lines look straight in your frame.

Usage:
    from undistort import WideAngleUndistorter
    undistorter = WideAngleUndistorter()
    flat_frame  = undistorter.process(frame)
"""

import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)

# ── HWI-B140H lens specs ──────────────────────────────────────────────────────
# Horizontal FOV: 98°   Vertical FOV: 53.1°   Diagonal FOV: 114.7°
HFOV_DEGREES = 98.0
VFOV_DEGREES = 53.1


class WideAngleUndistorter:
    def __init__(
        self,
        k1: float = -0.3,    # Radial distortion — negative = barrel (most wide-angle lenses)
        k2: float = 0.1,     # Radial distortion 2nd order — usually small
        p1: float = 0.0,     # Tangential distortion — near 0 for fixed-mount cameras
        p2: float = 0.0,     # Tangential distortion — near 0 for fixed-mount cameras
        alpha: float = 0.0,  # 0.0 = crop black edges, 1.0 = keep full frame
    ):
        """
        Args:
            k1:    Primary radial distortion. Negative = barrel distortion
                   (typical for wide-angle). Start at -0.3 and tune.
            k2:    Secondary radial distortion. Usually 0.0 to 0.15.
            p1/p2: Tangential distortion. Leave at 0.0 for fixed cameras.
            alpha: 0.0 = crop all black edges (recommended).
                   1.0 = keep full remapped frame with black corners.
        """
        self.k1    = k1
        self.k2    = k2
        self.p1    = p1
        self.p2    = p2
        self.alpha = alpha

        self._K          = None
        self._dist       = None
        self._new_K      = None
        self._map1       = None
        self._map2       = None
        self._calibrated = None

        log.info(
            f"WideAngleUndistorter ready "
            f"(k1={k1}, k2={k2}, p1={p1}, p2={p2}, alpha={alpha})"
        )

    def _build_maps(self, h: int, w: int):
        """Build remap lookup tables from frame dimensions and FOV specs."""
        log.info(f"Building undistortion maps for {w}x{h}...")

        fx = (w / 2.0) / np.tan(np.deg2rad(HFOV_DEGREES / 2.0))
        fy = (h / 2.0) / np.tan(np.deg2rad(VFOV_DEGREES / 2.0))
        cx = w / 2.0
        cy = h / 2.0

        self._K = np.array([
            [fx,  0,  cx],
            [ 0, fy,  cy],
            [ 0,  0,   1],
        ], dtype=np.float64)

        self._dist = np.array(
            [self.k1, self.k2, self.p1, self.p2],
            dtype=np.float64,
        )

        self._new_K, _ = cv2.getOptimalNewCameraMatrix(
            self._K, self._dist, (w, h),
            alpha=self.alpha,
            newImgSize=(w, h),
        )

        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._K, self._dist, None, self._new_K,
            (w, h), cv2.CV_16SC2,
        )

        self._calibrated = (w, h)
        log.info(
            f"Maps built — fx={fx:.1f} fy={fy:.1f} | "
            f"dist=[{self.k1}, {self.k2}, {self.p1}, {self.p2}]"
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Correct barrel distortion. Returns undistorted frame."""
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        if self._calibrated != (w, h):
            self._build_maps(h, w)
        return cv2.remap(
            frame, self._map1, self._map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def save_sample(self, frame: np.ndarray, path: str = "undistort_sample.jpg"):
        """Save side-by-side before/after comparison for tuning."""
        corrected = self.process(frame)
        if corrected is None:
            return
        cv2.imwrite(path, np.hstack([frame, corrected]))
        log.info(f"Saved comparison → {path}  (original left | corrected right)")


# Backward-compatible alias — flask_api.py imports FisheyeUndistorter by name
FisheyeUndistorter = WideAngleUndistorter