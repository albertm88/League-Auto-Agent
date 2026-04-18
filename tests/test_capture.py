"""Unit tests for the screen capture module.

These tests mock out the *mss* library so they can run without a display.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_fake_screenshot(width: int = 100, height: int = 60) -> MagicMock:
    """Return a mock mss screenshot with BGRA raw data."""
    raw = np.zeros((height, width, 4), dtype=np.uint8)
    raw[:, :, 2] = 128  # R channel value for a sanity check
    mock_shot = MagicMock()
    mock_shot.width = width
    mock_shot.height = height
    mock_shot.raw = raw.tobytes()
    return mock_shot


def _make_mock_sct(width: int = 100, height: int = 60) -> MagicMock:
    """Return a mock mss context."""
    mock_sct = MagicMock()
    mock_sct.monitors = [
        {},                             # index 0 (all monitors combined)
        {"left": 0, "top": 0, "width": width, "height": height},  # primary
    ]
    mock_sct.grab.return_value = _make_fake_screenshot(width, height)
    return mock_sct


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScreenCapture:
    def test_capture_returns_bgr_array(self):
        from src.capture.screen_capture import ScreenCapture

        mock_sct = _make_mock_sct(320, 180)
        with patch("mss.mss", return_value=mock_sct):
            with ScreenCapture(monitor_index=1, fps=0) as sc:
                frame = sc.capture()

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # BGR – no alpha

    def test_capture_shape_matches_monitor(self):
        from src.capture.screen_capture import ScreenCapture

        w, h = 320, 180
        mock_sct = _make_mock_sct(w, h)
        with patch("mss.mss", return_value=mock_sct):
            with ScreenCapture(monitor_index=1, fps=0) as sc:
                frame = sc.capture()

        assert frame.shape == (h, w, 3)

    def test_capture_with_resize(self):
        from src.capture.screen_capture import ScreenCapture

        mock_sct = _make_mock_sct(320, 180)
        target_res = (160, 90)
        with patch("mss.mss", return_value=mock_sct):
            with ScreenCapture(monitor_index=1, resolution=target_res, fps=0) as sc:
                frame = sc.capture()

        assert frame.shape == (90, 160, 3)

    def test_open_close_lifecycle(self):
        from src.capture.screen_capture import ScreenCapture

        mock_sct = _make_mock_sct()
        with patch("mss.mss", return_value=mock_sct):
            sc = ScreenCapture()
            sc.open()
            assert sc._sct is not None
            sc.close()
            assert sc._sct is None

    def test_capture_without_open_raises(self):
        from src.capture.screen_capture import ScreenCapture

        sc = ScreenCapture()
        with pytest.raises(RuntimeError, match="open()"):
            sc.capture()

    def test_invalid_monitor_index_raises(self):
        from src.capture.screen_capture import ScreenCapture

        mock_sct = _make_mock_sct()
        with patch("mss.mss", return_value=mock_sct):
            with ScreenCapture(monitor_index=99) as sc:
                with pytest.raises(ValueError, match="Monitor index"):
                    sc.get_monitor_info()

    def test_capture_region(self):
        from src.capture.screen_capture import ScreenCapture

        mock_sct = _make_mock_sct(320, 180)
        with patch("mss.mss", return_value=mock_sct):
            with ScreenCapture(monitor_index=1, fps=0) as sc:
                region = sc.capture_region(0, 0, 100, 60)

        assert region.ndim == 3
        assert region.shape[2] == 3
