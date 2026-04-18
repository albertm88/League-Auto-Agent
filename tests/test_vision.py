"""Unit tests for the game vision module."""

from __future__ import annotations

import numpy as np
import pytest


def _make_frame(width: int = 320, height: int = 180) -> np.ndarray:
    """Return a dummy BGR frame filled with mid-grey values."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


# Minimal vision config that mirrors the relevant keys from config.yaml
VISION_CFG = {
    "minimap": {
        "x_frac": 0.79,
        "y_frac": 0.79,
        "w_frac": 0.20,
        "h_frac": 0.20,
    },
    "health_bar": {
        "x_frac": 0.37,
        "y_frac": 0.89,
        "w_frac": 0.10,
        "h_frac": 0.02,
    },
    "mana_bar": {
        "x_frac": 0.37,
        "y_frac": 0.91,
        "w_frac": 0.10,
        "h_frac": 0.02,
    },
    "health_color_lower": [40, 100, 100],
    "health_color_upper": [80, 255, 255],
    "mana_color_lower": [100, 100, 100],
    "mana_color_upper": [140, 255, 255],
    "obs_width": 160,
    "obs_height": 90,
}


class TestGameVision:
    def test_process_returns_game_state(self):
        from src.vision.game_vision import GameVision, GameState

        gv = GameVision(VISION_CFG)
        frame = _make_frame()
        state = gv.process(frame)

        assert isinstance(state, GameState)

    def test_obs_frame_shape(self):
        from src.vision.game_vision import GameVision

        gv = GameVision(VISION_CFG)
        frame = _make_frame(320, 180)
        state = gv.process(frame)

        assert state.obs_frame.shape == (90, 160, 3)

    def test_health_ratio_is_float_in_range(self):
        from src.vision.game_vision import GameVision

        gv = GameVision(VISION_CFG)
        state = gv.process(_make_frame())

        assert isinstance(state.health_ratio, float)
        assert 0.0 <= state.health_ratio <= 1.0

    def test_mana_ratio_is_float_in_range(self):
        from src.vision.game_vision import GameVision

        gv = GameVision(VISION_CFG)
        state = gv.process(_make_frame())

        assert isinstance(state.mana_ratio, float)
        assert 0.0 <= state.mana_ratio <= 1.0

    def test_minimap_is_cropped(self):
        from src.vision.game_vision import GameVision

        gv = GameVision(VISION_CFG)
        state = gv.process(_make_frame(320, 180))

        assert state.minimap is not None
        assert state.minimap.ndim == 3
        assert state.minimap.shape[2] == 3

    def test_raw_frame_preserved(self):
        from src.vision.game_vision import GameVision

        gv = GameVision(VISION_CFG)
        frame = _make_frame()
        state = gv.process(frame)

        assert state.raw_frame is frame

    def test_describe_contains_percentages(self):
        from src.vision.game_vision import GameVision

        gv = GameVision(VISION_CFG)
        state = gv.process(_make_frame())
        description = gv.describe(state)

        assert "health" in description.lower()
        assert "mana" in description.lower()
        assert "%" in description

    def test_describe_level_labels(self):
        from src.vision.game_vision import GameVision

        gv = GameVision(VISION_CFG)
        # Manually tweak ratios
        state = gv.process(_make_frame())
        state.health_ratio = 0.9
        state.mana_ratio = 0.1
        description = gv.describe(state)

        assert "high" in description
        assert "low" in description

    def test_missing_region_config_defaults(self):
        """Vision with no bar config should fall back to ratio=1.0."""
        from src.vision.game_vision import GameVision

        minimal_cfg = {"obs_width": 80, "obs_height": 45}
        gv = GameVision(minimal_cfg)
        state = gv.process(_make_frame())

        assert state.health_ratio == 1.0
        assert state.mana_ratio == 1.0
        assert state.minimap is None
