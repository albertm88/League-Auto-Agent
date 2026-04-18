"""Action executor module.

Translates discrete agent actions into League of Legends inputs via
``pyautogui``.

Action Space (20 actions total)
---------------------------------
0  – 15  : Movement grid (4 cols × 4 rows)
             Divides the game viewport into a 4×4 grid; action i moves the
             champion to the centre of grid cell i by right-clicking at the
             corresponding screen coordinate.
16        : Cast ability Q
17        : Cast ability W
18        : Cast ability E
19        : Cast ability R

Additional actions (not in the discrete PPO space, called contextually):
* Recall (B key)
* Attack-move (A + click)
* Item slots 1-7
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import pyautogui


class ActionExecutor:
    """Converts discrete agent actions to keyboard / mouse inputs.

    Parameters
    ----------
    cfg:
        The ``actions`` section of the YAML configuration dictionary.
    screen_width:
        Width of the game screen in pixels.
    screen_height:
        Height of the game screen in pixels.
    """

    ABILITY_KEYS: List[str] = ["q", "w", "e", "r"]

    def __init__(
        self,
        cfg: Dict,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        self._cfg = cfg
        self._screen_w = screen_width
        self._screen_h = screen_height
        self._action_delay: float = cfg.get("action_delay", 0.05)
        # Prevent pyautogui from raising FailSafeException at screen corners
        pyautogui.FAILSAFE = False

        cols: int = cfg.get("move_grid_cols", 4)
        rows: int = cfg.get("move_grid_rows", 4)
        self._move_targets = self._build_move_grid(cols, rows)

        self._ability_keys: List[str] = cfg.get("ability_keys", self.ABILITY_KEYS)
        self._item_keys: List[str] = cfg.get("item_keys", [str(i) for i in range(1, 8)])
        self._recall_key: str = cfg.get("recall_key", "b")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, action_idx: int) -> None:
        """Execute the action corresponding to *action_idx*.

        Parameters
        ----------
        action_idx:
            Integer action produced by the PPO agent.  Valid range is
            ``[0, num_actions)``.
        """
        num_move = len(self._move_targets)
        num_abilities = len(self._ability_keys)

        if action_idx < num_move:
            self._move(action_idx)
        elif action_idx < num_move + num_abilities:
            ability_idx = action_idx - num_move
            self._cast_ability(ability_idx)
        # Additional actions beyond [move + abilities] are no-ops here;
        # subclasses can extend this method as needed.

        time.sleep(self._action_delay)

    def recall(self) -> None:
        """Press the recall key (``B``)."""
        pyautogui.press(self._recall_key)
        time.sleep(self._action_delay)

    def use_item(self, slot: int) -> None:
        """Activate the item in the given slot (0-indexed).

        Parameters
        ----------
        slot:
            Item slot index in ``[0, len(item_keys))``.
        """
        if 0 <= slot < len(self._item_keys):
            pyautogui.press(self._item_keys[slot])
            time.sleep(self._action_delay)

    def attack_move(self, x: int, y: int) -> None:
        """Issue an attack-move command at ``(x, y)``."""
        pyautogui.press("a")
        pyautogui.rightClick(x, y)
        time.sleep(self._action_delay)

    def get_move_targets(self) -> List[Tuple[int, int]]:
        """Return the list of ``(x, y)`` screen coordinates for move actions."""
        return list(self._move_targets)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _move(self, grid_idx: int) -> None:
        """Right-click at the grid-cell centre corresponding to *grid_idx*."""
        x, y = self._move_targets[grid_idx]
        pyautogui.rightClick(x, y)

    def _cast_ability(self, ability_idx: int) -> None:
        """Press the ability key for *ability_idx* (0=Q, 1=W, 2=E, 3=R)."""
        key = self._ability_keys[ability_idx % len(self._ability_keys)]
        pyautogui.press(key)

    def _build_move_grid(
        self,
        cols: int,
        rows: int,
    ) -> List[Tuple[int, int]]:
        """Partition the game viewport into a grid and return cell centres.

        The minimap occupies the bottom-right ~20% of the screen, so we
        only use the left 80% of the width to avoid clicking there.
        """
        usable_w = int(self._screen_w * 0.78)
        cell_w = usable_w // cols
        cell_h = self._screen_h // rows

        targets: List[Tuple[int, int]] = []
        for row in range(rows):
            for col in range(cols):
                cx = cell_w * col + cell_w // 2
                cy = cell_h * row + cell_h // 2
                targets.append((cx, cy))
        return targets
