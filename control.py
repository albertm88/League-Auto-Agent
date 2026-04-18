"""LoL 控制执行器：pydirectinput 游戏输入 + 辅助前台管理。

核心原则：
  1) 输入永不因前台检测失败而阻塞——前台管理仅做辅助提醒
  2) 所有键鼠事件统一走 pydirectinput（SendInput 驱动）
  3) 用户需确保 LoL 在前台（脚本启动时有倒计时切换窗口）
"""

import ctypes
import ctypes.wintypes as _w
import logging
import random
import subprocess
import time

import pydirectinput

log = logging.getLogger(__name__)

pydirectinput.PAUSE = 0.0
pydirectinput.FAILSAFE = False

_LOL_TITLES = ("英雄联盟", "league of legends", "league of legends (tm) client")
SW_RESTORE = 9
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
_KEY = {
    "q": "q", "w": "w", "e": "e", "r": "r",
    "a": "a", "b": "b", "s": "s", "d": "d", "f": "f", "y": "y",
    "p": "p", "escape": "escape",
}


class ControlExecutor:
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self._last_focus_warn_ts = 0.0
        self._focus_lost_count = 0   # 连续失焦帧计数

    # ── 窗口查找 ──────────────────────────────────────────────────────────────

    def _find_lol_hwnd(self):
        """枚举窗口，找到 LoL 游戏主窗口句柄。"""
        candidates = []
        enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, _w.HWND, _w.LPARAM)

        def _cb(hwnd, _):
            if not self.user32.IsWindowVisible(hwnd):
                return True
            buf = ctypes.create_unicode_buffer(256)
            self.user32.GetWindowTextW(hwnd, buf, 256)
            title = buf.value.lower()
            if any(t in title for t in _LOL_TITLES):
                # 偏好更长标题（通常是游戏主窗口，不是空标题子窗）
                candidates.append((len(buf.value), hwnd))
            return True

        self.user32.EnumWindows(enum_proc(_cb), 0)
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _foreground_title(self) -> str:
        hwnd = self.user32.GetForegroundWindow()
        if hwnd == 0:
            return ""
        buf = ctypes.create_unicode_buffer(256)
        self.user32.GetWindowTextW(hwnd, buf, 256)
        return buf.value

    # ── 前台状态检测（纯查询，不改变任何状态）─────────────────────────────

    def is_lol_foreground(self) -> bool:
        """LoL 是否当前在前台（纯检测，不尝试激活）。"""
        title = self._foreground_title().lower()
        return any(t in title for t in _LOL_TITLES)

    # ── 辅助性前台激活（尽力而为，永不阻塞调用者）────────────────────────

    def try_activate(self) -> bool:
        """
        尝试激活 LoL 窗口到前台。
        这是辅助方法——成功与否不影响后续操作执行。
        Windows 限制后台进程抢焦点，因此经常会失败，这是正常的。
        """
        if self.is_lol_foreground():
            self._focus_lost_count = 0
            return True

        hwnd = self._find_lol_hwnd()
        if not hwnd:
            return False

        try:
            self.user32.ShowWindow(hwnd, SW_RESTORE)
            self.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
            self.user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0,
                                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
            self.user32.BringWindowToTop(hwnd)
            fg = self.user32.GetForegroundWindow()
            fg_tid = self.user32.GetWindowThreadProcessId(fg, None)
            cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()
            tgt_tid = self.user32.GetWindowThreadProcessId(hwnd, None)
            self.user32.AttachThreadInput(cur_tid, fg_tid, True)
            self.user32.AttachThreadInput(cur_tid, tgt_tid, True)
            self.user32.SetForegroundWindow(hwnd)
            self.user32.SetFocus(hwnd)
            self.user32.SetActiveWindow(hwnd)
            self.user32.AttachThreadInput(cur_tid, fg_tid, False)
            self.user32.AttachThreadInput(cur_tid, tgt_tid, False)
        except Exception:
            pass

        # PowerShell fallback
        if not self.is_lol_foreground():
            for title in ("英雄联盟", "League of Legends"):
                try:
                    subprocess.run(
                        ["powershell", "-WindowStyle", "Hidden", "-Command",
                         f'(New-Object -ComObject WScript.Shell).AppActivate("{title}")'],
                        capture_output=True, timeout=2,
                    )
                except Exception:
                    pass
            time.sleep(0.15)

        ok = self.is_lol_foreground()
        if ok:
            self._focus_lost_count = 0
        return ok

    def warn_if_not_focused(self):
        """检查焦点状态，失焦时节流输出警告（不阻塞，不激活）。"""
        if self.is_lol_foreground():
            if self._focus_lost_count > 0:
                log.info("✅ LoL 已恢复前台")
            self._focus_lost_count = 0
            return

        self._focus_lost_count += 1
        now = time.time()
        if now - self._last_focus_warn_ts > 3.0:
            self._last_focus_warn_ts = now
            fg = self._foreground_title() or "<none>"
            log.warning(
                f"⚠️  LoL 不在前台 (连续 {self._focus_lost_count} 帧) "
                f"| 当前窗口: {fg}"
            )
            # 每 10 次连续失焦尝试拉一次
            if self._focus_lost_count % 10 == 0:
                self.try_activate()

    # ── 启动倒计时：给用户时间切换到 LoL 窗口 ─────────────────────────────

    def startup_countdown(self, seconds: int = 5):
        """启动前倒计时，给用户切换到 LoL 窗口的时间。"""
        if self.is_lol_foreground():
            print("✅ LoL 已在前台")
            return True

        # 先尝试激活一次
        if self.try_activate():
            time.sleep(0.3)
            if self.is_lol_foreground():
                print("✅ LoL 已自动激活到前台")
                return True

        print(f"\n{'='*50}")
        print(f"  ⚠️  请在 {seconds} 秒内点击英雄联盟游戏窗口！")
        print(f"{'='*50}")
        for i in range(seconds, 0, -1):
            if self.is_lol_foreground():
                print("  ✅ 检测到 LoL 前台！")
                return True
            print(f"  ⏱️  {i}...")
            time.sleep(1)

        ok = self.is_lol_foreground()
        if ok:
            print("  ✅ LoL 在前台，开始运行")
        else:
            print("  ⚠️  LoL 仍不在前台，将继续运行（操作可能无法送达游戏）")
            print("  💡  建议：手动点击一次游戏窗口即可")
        return ok

    # ── 输入方法（无论前台状态均执行，不阻塞）─────────────────────────────

    def get_cursor_pos(self):
        pt = _w.POINT()
        self.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def move_mouse(self, x: int, y: int):
        pydirectinput.moveTo(x, y)
        time.sleep(0.02)

    def left_click(self):
        pydirectinput.click(button="left")
        time.sleep(0.03)

    def right_click(self, x: int, y: int):
        if x is not None and y is not None:
            pydirectinput.click(x, y, button="right")
        else:
            pydirectinput.click(button="right")
        time.sleep(0.03)

    def left_click_at(self, x: int, y: int):
        pydirectinput.click(x, y, button="left")
        time.sleep(0.03)

    def press_key(self, key: str):
        k = _KEY.get(key.lower(), key.lower())
        pydirectinput.keyDown(k)
        time.sleep(random.uniform(0.05, 0.09))
        pydirectinput.keyUp(k)
        time.sleep(0.02)

    def attack_move(self, x: int, y: int):
        pydirectinput.keyDown("a")
        time.sleep(0.05)
        pydirectinput.keyUp("a")
        time.sleep(0.03)
        pydirectinput.click(x, y, button="left")
        time.sleep(0.03)

    def level_up_skill(self, key: str):
        k = _KEY.get(key.lower())
        if k is None:
            return
        log.info(f"⬆️  LEVEL-UP Ctrl+{key.upper()}")
        pydirectinput.keyDown("ctrl")
        time.sleep(0.04)
        pydirectinput.keyDown(k)
        time.sleep(0.06)
        pydirectinput.keyUp(k)
        time.sleep(0.02)
        pydirectinput.keyUp("ctrl")
        time.sleep(0.03)

    def lock_camera(self):
        log.info("📷  LOCK CAMERA (Y)")
        pydirectinput.keyDown("y")
        time.sleep(0.06)
        pydirectinput.keyUp("y")

    # ── 小地图点击导航 ────────────────────────────────────────────────────────
    # 小地图 ROI（与 view.py ScreenReader._MINIMAP 一致）
    _MINIMAP_REL = (0.857, 0.741, 0.998, 0.999)

    def minimap_click(self, norm_x: float, norm_y: float,
                      screen_w: int = 1920, screen_h: int = 1080):
        """
        右键点击小地图上的指定位置，使英雄寻路到该地图坐标。
        norm_x, norm_y ∈ [0, 1]：小地图内归一化坐标
          (0,0) = 小地图左上角（红方基地侧）
          (1,1) = 小地图右下角（蓝方基地侧）
        """
        x1, y1, x2, y2 = self._MINIMAP_REL
        # 小地图在屏幕上的像素范围
        mm_left   = int(x1 * screen_w)
        mm_top    = int(y1 * screen_h)
        mm_right  = int(x2 * screen_w)
        mm_bottom = int(y2 * screen_h)
        # 小地图内坐标 → 屏幕坐标
        px = mm_left + int(norm_x * (mm_right - mm_left))
        py = mm_top  + int(norm_y * (mm_bottom - mm_top))
        # 限制在小地图范围内
        px = max(mm_left, min(mm_right, px))
        py = max(mm_top,  min(mm_bottom, py))
        log.info(f"🗺️  MINIMAP CLICK ({norm_x:.2f},{norm_y:.2f}) → screen({px},{py})")
        pydirectinput.click(px, py, button="right")
        time.sleep(0.05)