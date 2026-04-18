"""Unit tests for the action executor module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Stub out pyautogui before action_executor is imported so that these tests
# can run on headless CI machines that have no X11 / DISPLAY available.
# ---------------------------------------------------------------------------
_pyautogui_stub = MagicMock()
_pyautogui_stub.FAILSAFE = False
sys.modules.setdefault("pyautogui", _pyautogui_stub)


ACTIONS_CFG = {
    "ability_keys": ["q", "w", "e", "r"],
    "item_keys": ["1", "2", "3", "4", "5", "6", "7"],
    "recall_key": "b",
    "camera_lock_key": "y",
    "move_grid_cols": 4,
    "move_grid_rows": 4,
    "action_delay": 0.0,  # no sleep during tests
}


class TestActionExecutor:
    def _make_executor(self, screen_w: int = 1920, screen_h: int = 1080):
        from src.actions.action_executor import ActionExecutor
        return ActionExecutor(ACTIONS_CFG, screen_w, screen_h)

    def test_move_grid_size(self):
        ae = self._make_executor()
        targets = ae.get_move_targets()
        expected = ACTIONS_CFG["move_grid_cols"] * ACTIONS_CFG["move_grid_rows"]
        assert len(targets) == expected

    def test_move_targets_are_within_screen(self):
        sw, sh = 1920, 1080
        ae = self._make_executor(sw, sh)
        for x, y in ae.get_move_targets():
            assert 0 <= x < sw
            assert 0 <= y < sh

    @patch("src.actions.action_executor.pyautogui")
    def test_execute_move_action_calls_rightclick(self, mock_pyautogui):
        ae = self._make_executor()
        ae.execute(0)  # first move action

        mock_pyautogui.rightClick.assert_called_once()

    @patch("src.actions.action_executor.pyautogui")
    def test_execute_ability_q(self, mock_pyautogui):
        ae = self._make_executor()
        num_move = len(ae.get_move_targets())
        ae.execute(num_move)  # first ability = Q

        mock_pyautogui.press.assert_called_with("q")

    @patch("src.actions.action_executor.pyautogui")
    def test_execute_ability_w(self, mock_pyautogui):
        ae = self._make_executor()
        num_move = len(ae.get_move_targets())
        ae.execute(num_move + 1)  # W

        mock_pyautogui.press.assert_called_with("w")

    @patch("src.actions.action_executor.pyautogui")
    def test_execute_ability_r(self, mock_pyautogui):
        ae = self._make_executor()
        num_move = len(ae.get_move_targets())
        ae.execute(num_move + 3)  # R

        mock_pyautogui.press.assert_called_with("r")

    @patch("src.actions.action_executor.pyautogui")
    def test_recall(self, mock_pyautogui):
        ae = self._make_executor()
        ae.recall()

        mock_pyautogui.press.assert_called_with("b")

    @patch("src.actions.action_executor.pyautogui")
    def test_use_item_valid_slot(self, mock_pyautogui):
        ae = self._make_executor()
        ae.use_item(0)

        mock_pyautogui.press.assert_called_with("1")

    @patch("src.actions.action_executor.pyautogui")
    def test_use_item_invalid_slot_no_call(self, mock_pyautogui):
        ae = self._make_executor()
        ae.use_item(99)  # out of range

        mock_pyautogui.press.assert_not_called()

    @patch("src.actions.action_executor.pyautogui")
    def test_attack_move(self, mock_pyautogui):
        ae = self._make_executor()
        ae.attack_move(500, 400)

        mock_pyautogui.press.assert_called_with("a")
        mock_pyautogui.rightClick.assert_called_with(500, 400)
