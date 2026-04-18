"""Game vision module using OpenCV.

Processes raw BGR frames captured from the League of Legends screen and
extracts structured game-state features:

* Resized observation frame (for the PPO network)
* Player health ratio  [0, 1]
* Player mana ratio    [0, 1]
* Minimap as a small RGB image
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GameState:
    """Structured game-state extracted from a single captured frame.

    Attributes
    ----------
    obs_frame:
        Resized BGR observation frame fed into the PPO network.
        Shape: ``(obs_height, obs_width, 3)``.
    health_ratio:
        Player health as a fraction in ``[0, 1]``.
    mana_ratio:
        Player mana as a fraction in ``[0, 1]``.
    minimap:
        Small BGR minimap crop.  Shape: ``(minimap_h, minimap_w, 3)``.
    raw_frame:
        Original, unprocessed BGR frame (before any resizing).
    """

    obs_frame: np.ndarray
    health_ratio: float = 1.0
    mana_ratio: float = 1.0
    minimap: Optional[np.ndarray] = None
    raw_frame: Optional[np.ndarray] = None


class GameVision:
    """Extracts game-state features from a raw screen frame.

    Parameters
    ----------
    cfg:
        The ``vision`` section of the YAML configuration dictionary.
    """

    def __init__(self, cfg: Dict) -> None:
        self._cfg = cfg
        self._obs_w: int = cfg.get("obs_width", 160)
        self._obs_h: int = cfg.get("obs_height", 90)

        # HSV color thresholds for bar detection
        self._health_lower = np.array(cfg.get("health_color_lower", [40, 100, 100]), dtype=np.uint8)
        self._health_upper = np.array(cfg.get("health_color_upper", [80, 255, 255]), dtype=np.uint8)
        self._mana_lower = np.array(cfg.get("mana_color_lower", [100, 100, 100]), dtype=np.uint8)
        self._mana_upper = np.array(cfg.get("mana_color_upper", [140, 255, 255]), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> GameState:
        """Process a raw BGR frame and return a :class:`GameState`.

        Parameters
        ----------
        frame:
            Raw BGR frame from :class:`~src.capture.screen_capture.ScreenCapture`.

        Returns
        -------
        GameState
        """
        h, w = frame.shape[:2]

        obs_frame = self._resize_obs(frame)
        health_ratio = self._extract_bar_ratio(frame, h, w, bar="health")
        mana_ratio = self._extract_bar_ratio(frame, h, w, bar="mana")
        minimap = self._extract_minimap(frame, h, w)

        return GameState(
            obs_frame=obs_frame,
            health_ratio=health_ratio,
            mana_ratio=mana_ratio,
            minimap=minimap,
            raw_frame=frame,
        )

    def describe(self, state: GameState) -> str:
        """Return a short text description of the game state for LLM input.

        Parameters
        ----------
        state:
            A :class:`GameState` produced by :meth:`process`.

        Returns
        -------
        str
        """
        health_pct = int(state.health_ratio * 100)
        mana_pct = int(state.mana_ratio * 100)
        health_status = self._level_label(state.health_ratio)
        mana_status = self._level_label(state.mana_ratio)

        return (
            f"Player health: {health_pct}% ({health_status}). "
            f"Player mana: {mana_pct}% ({mana_status})."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resize_obs(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to the configured observation resolution."""
        return cv2.resize(
            frame,
            (self._obs_w, self._obs_h),
            interpolation=cv2.INTER_LINEAR,
        )

    def _get_region_pixels(
        self,
        frame: np.ndarray,
        h: int,
        w: int,
        key: str,
    ) -> Optional[np.ndarray]:
        """Crop a relative region from the frame.

        The region coordinates are specified as fractions of the frame
        dimensions in the config dict under *key* (``x_frac``, ``y_frac``,
        ``w_frac``, ``h_frac``).
        """
        region_cfg = self._cfg.get(key)
        if region_cfg is None:
            return None

        x = int(region_cfg["x_frac"] * w)
        y = int(region_cfg["y_frac"] * h)
        rw = max(1, int(region_cfg["w_frac"] * w))
        rh = max(1, int(region_cfg["h_frac"] * h))

        # Guard against out-of-bounds
        x = min(x, w - 1)
        y = min(y, h - 1)
        rw = min(rw, w - x)
        rh = min(rh, h - y)

        return frame[y : y + rh, x : x + rw]

    def _extract_bar_ratio(
        self,
        frame: np.ndarray,
        h: int,
        w: int,
        bar: str,
    ) -> float:
        """Estimate the fill ratio of a health or mana bar.

        Strategy: crop the bar region, convert to HSV, threshold for the
        bar colour, and compute the ratio of coloured pixels to total pixels.
        Falls back to 1.0 when the region cannot be found.
        """
        key = "health_bar" if bar == "health" else "mana_bar"
        region = self._get_region_pixels(frame, h, w, key)
        if region is None or region.size == 0:
            return 1.0

        lower = self._health_lower if bar == "health" else self._mana_lower
        upper = self._health_upper if bar == "health" else self._mana_upper

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        return min(1.0, max(0.0, ratio))

    def _extract_minimap(
        self,
        frame: np.ndarray,
        h: int,
        w: int,
    ) -> Optional[np.ndarray]:
        """Crop the minimap from the bottom-right corner of the frame."""
        return self._get_region_pixels(frame, h, w, "minimap")

    @staticmethod
    def _level_label(ratio: float) -> str:
        if ratio > 0.66:
            return "high"
        if ratio > 0.33:
            return "medium"
        return "low"
