"""Screen capture module using mss.

Captures frames from the game window (or primary monitor) and returns them
as NumPy arrays (BGR, compatible with OpenCV).
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import mss
import mss.tools
import numpy as np


class ScreenCapture:
    """Captures the game screen using the *mss* library.

    Parameters
    ----------
    monitor_index:
        Index of the monitor to capture (1 = primary).
    resolution:
        Optional ``(width, height)`` to resize the captured frame.  If
        ``None`` the frame is returned at native resolution.
    fps:
        Target capture rate.  :meth:`capture` will sleep the appropriate
        amount to stay on schedule.
    """

    def __init__(
        self,
        monitor_index: int = 1,
        resolution: Optional[Tuple[int, int]] = None,
        fps: float = 5.0,
    ) -> None:
        self._monitor_index = monitor_index
        self._resolution = resolution
        self._frame_duration = 1.0 / fps if fps > 0 else 0.0
        self._last_capture_time: float = 0.0
        self._sct: Optional[mss.base.MssBase] = None

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------

    def __enter__(self) -> "ScreenCapture":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the underlying mss context."""
        if self._sct is None:
            self._sct = mss.mss()

    def close(self) -> None:
        """Release resources held by the mss context."""
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def get_monitor_info(self) -> Dict:
        """Return the mss monitor dict for the configured monitor index."""
        if self._sct is None:
            raise RuntimeError("Call open() before get_monitor_info().")
        monitors = self._sct.monitors
        if self._monitor_index >= len(monitors):
            raise ValueError(
                f"Monitor index {self._monitor_index} out of range "
                f"(available monitors: {len(monitors) - 1})."
            )
        return monitors[self._monitor_index]

    def capture(self) -> np.ndarray:
        """Capture one frame and return it as a BGR NumPy array.

        If a target ``fps`` was specified the method will block until the
        next frame is due, preventing the caller from polling faster than
        the configured rate.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(H, W, 3)`` with dtype ``uint8`` (BGR order).
        """
        if self._sct is None:
            raise RuntimeError("Call open() (or use as context manager) first.")

        # Rate-limiting
        elapsed = time.monotonic() - self._last_capture_time
        wait = self._frame_duration - elapsed
        if wait > 0:
            time.sleep(wait)

        monitor = self.get_monitor_info()
        screenshot = self._sct.grab(monitor)
        self._last_capture_time = time.monotonic()

        # mss returns BGRA; drop the alpha channel to get BGR
        frame_bgra = np.frombuffer(screenshot.raw, dtype=np.uint8)
        frame_bgra = frame_bgra.reshape((screenshot.height, screenshot.width, 4))
        frame_bgr = frame_bgra[:, :, :3]

        if self._resolution is not None:
            import cv2  # lazy import – only needed when resizing

            frame_bgr = cv2.resize(
                frame_bgr,
                self._resolution,
                interpolation=cv2.INTER_LINEAR,
            )

        return frame_bgr

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Capture a specific region of the screen.

        Parameters
        ----------
        x, y:
            Top-left corner of the region in screen coordinates.
        width, height:
            Dimensions of the region in pixels.

        Returns
        -------
        numpy.ndarray
            Cropped BGR frame of shape ``(height, width, 3)``.
        """
        if self._sct is None:
            raise RuntimeError("Call open() (or use as context manager) first.")

        region = {"left": x, "top": y, "width": width, "height": height}
        screenshot = self._sct.grab(region)

        frame_bgra = np.frombuffer(screenshot.raw, dtype=np.uint8)
        frame_bgra = frame_bgra.reshape((screenshot.height, screenshot.width, 4))
        return frame_bgra[:, :, :3]
