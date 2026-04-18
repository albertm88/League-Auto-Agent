"""
agent.py — GameAgent 完整版

相比原版的核心修改：
  1. replay_buffer 存储 (obs, action, reward, logp, done) 五元组
     → 传递 done 信息，让 GAE 在 episode 边界正确截断，不跨 episode 混算
  2. _train_step() 改用 train_episode()，传入 next_obs 做 bootstrap
     → Critic 能正确估计 episode 末尾的 V(s_T)
  3. 真实模式奖励来自 _compute_reward()，去掉 VLM reward shaping 中的
     "和 VLM 意图一致就加分" 逻辑——改为只保留宏观位置评价（更可靠）
  4. episode 结束时主动 flush replay_buffer（不等凑满 32 步），
     防止跨 episode 数据污染
  5. 离线预热同步修改：传 done，单 episode 结束时立即训练
  6. 其余逻辑（三线程、执行器、VLM 循环、闭环修正）不改动
"""

import cv2
import numpy as np
import random
import math
import time
import ctypes
import queue
import threading
import logging

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("⚠️  mss 未安装，真实截图功能不可用")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from env import LoLEnv
from view import VisualPerception, ScreenReader, GameStateDetector
from think import DecisionThinker
from policy import Policy
from rl_model import PPOAgent
from control import ControlExecutor

log = logging.getLogger(__name__)

ACTION_DIM        = 10    # MOVE/ATTACK/Q/W/E/R/SPELL_D/SPELL_F/HOLD/RECALL
STATE_DIM_BOT     = 13    # LoLEnv 13 维（11 原始 + 2 小地图坐标）
STATE_DIM_VLM     = 4     # VLM/CV 语义特征 4 维
STATE_DIM         = STATE_DIM_BOT + STATE_DIM_VLM   # = 17，离线/真实统一维度

LOL_PROCESSES     = {"LeagueClient.exe", "League of Legends.exe", "LeagueClientUx.exe"}
DECISION_INTERVAL = 0.10  # Thread A：100ms/tick
VLM_INTERVAL      = 0.50  # Thread C：500ms/tick

MODEL_PATH        = "./model/ppo_agent.pt"
SAVE_EVERY        = 10
REAL_EP_TIMEOUT   = 3000   # 10Hz × 300s = 5 分钟
EMERGENCY_STOP_VK = 0x79   # F10

SKILL_LEVEL_PRIORITY = ['r', 'q', 'e', 'w']

# ── 训练触发阈值 ──────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE  = 64     # 真实模式：积累 N 步后触发训练（原版 32）
                           # episode 结束时无论多少步都立即训练


# ─────────────────────────────────────────────────────────────────────────────
def _compute_reward(
    prev_obs:     np.ndarray,
    obs:          np.ndarray,
    intent:       str = "",
    vlm_strategy: dict = None,
    macro_cache:  dict = None,
) -> float:
    """
    真实模式奖励函数（辅助信号，主奖励由 env.py 提供）。

    修改：去掉"VLM intent == agent intent 就加分"的 reward shaping。
    原因：VLM（4B 量化本地模型）对 LoL 截图的判断准确率有限，
    用它的 intent 作为 ground truth 会把 VLM 的错误强化进 PPO。
    保留：宏观位置评价（reward_modifier），这个信号更稳定，
    且有 macro_confidence 门控，误判时影响小。
    """
    reward = 0.0

    # 宏观位置评价：VLM 判断当前走位是否合理（有 confidence 门控）
    if macro_cache:
        mc  = float(macro_cache.get("macro_confidence", 0.0))
        rm  = float(macro_cache.get("reward_modifier",  0.0))
        if mc > 0.35:
            # 权重适度，不超过战斗奖励量级
            reward += rm * mc * 1.5

    return float(reward)


def _is_lol_running() -> bool:
    if not PSUTIL_AVAILABLE:
        return True
    return any(p.name() in LOL_PROCESSES for p in psutil.process_iter(["name"]))


def _is_lol_foreground() -> bool:
    _LOL_TITLES = ("league of legends", "英雄联盟", "lol")
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if hwnd == 0:
            return False
        buf = ctypes.create_unicode_buffer(256)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
        title = buf.value.lower()
        return any(kw in title for kw in _LOL_TITLES)
    except Exception:
        return True


# ─────────────────────────────────────────────────────────────────────────────
class GameAgent:
    def __init__(self, real_game_mode: bool = True):
        self.real_game_mode = real_game_mode
        self.state_dim      = STATE_DIM

        self.env          = LoLEnv()
        self.think        = DecisionThinker()
        self.view         = VisualPerception(self.think)
        self.screen_reader = ScreenReader()
        self.game_state   = GameStateDetector()
        self.policy       = Policy(PPOAgent(state_dim=self.state_dim, action_dim=ACTION_DIM))
        self.control      = ControlExecutor()

        self._action_queue: "queue.Queue[int]" = queue.Queue(maxsize=2)

        # ── replay_buffer 改为存 5 元组：(obs, action, reward, logp, done) ──
        self.replay_buffer: list = []

        self.episode       = 0
        self._stop_event   = threading.Event()
        self._hotkey_down  = False
        self._emergency_stopped = False
        self._user32       = ctypes.windll.user32
        self._vlm_tick_count: int = 0
        self._latest_obs: np.ndarray = np.zeros(STATE_DIM, dtype=np.float32)
        self._macro_lock  = threading.Lock()
        self._macro_cache: dict = {
            "macro_goal":       "farm",
            "minimap_target_x": 0.5,
            "minimap_target_y": 0.5,
            "reward_modifier":  0.0,
            "action_weights":   {},
            "macro_confidence": 0.0,
        }

        self.spawn_side: str  = GameStateDetector.SIDE_UNKNOWN
        self._is_dead:   bool = False
        self._last_units: dict = {
            "enemies": [], "enemy_minions": [], "allies": [], "ally_minions": []
        }

        self._skill_levels: dict = {'q': 0, 'w': 0, 'e': 0, 'r': 0}
        self._death_frames: int  = 0
        self._DEATH_CONFIRM      = 3

        self._shop_closed:        bool  = False
        self._last_shop_toggle_time: float = 0.0
        self._shop_toggle_cooldown: float  = 1.2
        self._last_save_time: float = time.time()
        self._prev_minimap_pos: "tuple | None" = None
        self._stuck_frames:  int = 0
        self._escape_wp_idx: int = 0
        self._force_escape_until: float = 0.0
        self._last_levelup_time: float = 0.0
        self._last_focus_block_warn: float = 0.0
        self._last_not_fg_warn: float = 0.0

        # ── 自动加载检查点 ────────────────────────────────────────────────────
        import os
        if os.path.exists(MODEL_PATH):
            try:
                self.policy.ppo.load(MODEL_PATH)
                log.info(f"📂 已加载检查点: {MODEL_PATH}")
            except Exception as e:
                log.warning(f"检查点加载失败，重新训练: {e}")
        else:
            log.info("未找到检查点，从头训练")

        if MSS_AVAILABLE:
            self.sct = mss.mss()
        else:
            self.sct = None

    # ── 截图 ─────────────────────────────────────────────────────────────────

    def capture(self) -> np.ndarray:
        if self.sct is None:
            raise RuntimeError("mss 未安装")
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        img = self.sct.grab(monitor)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

    # ── Thread C：VLM 慢循环（2Hz）───────────────────────────────────────────

    def _vlm_loop(self):
        while not self._stop_event.is_set():
            self._poll_emergency_stop()
            try:
                frame = self.capture()
                cv2.imwrite("tmp_vlm.jpg", frame)
                self.think.vision_parse("tmp_vlm.jpg")

                self._vlm_tick_count += 1
                if self._vlm_tick_count % 2 == 0:
                    try:
                        minimap = self.screen_reader.get_minimap_crop(frame)
                        if minimap.size > 0:
                            cv2.imwrite("tmp_minimap.jpg", minimap)
                            state_13 = self.screen_reader.read_state(frame)
                            my_pos = (
                                float(state_13[11]) if len(state_13) > 11 else 0.5,
                                float(state_13[12]) if len(state_13) > 12 else 0.5,
                            )
                            self.think.vision_parse_minimap(
                                "tmp_minimap.jpg", my_pos, self.spawn_side,
                            )
                    except Exception as mm_err:
                        log.debug(f"Thread-C 小地图分析异常: {mm_err}")
            except Exception as e:
                log.debug(f"Thread-C VLM 异常: {e}")
            time.sleep(VLM_INTERVAL)

    # ── 技能升级 ──────────────────────────────────────────────────────────────

    def _try_level_skills(self, frame) -> bool:
        now = time.time()
        if now - self._last_levelup_time < 2.0:
            return False
        _MAX_LEVEL = {'q': 5, 'w': 5, 'e': 5, 'r': 3}
        try:
            levelup = self.screen_reader.read_levelup(frame)
        except Exception as e:
            log.debug(f"read_levelup 异常: {e}")
            return False
        for sk in SKILL_LEVEL_PRIORITY:
            if levelup.get(sk, False):
                if self._skill_levels[sk] >= _MAX_LEVEL.get(sk, 5):
                    continue
                try:
                    self.control.level_up_skill(sk)
                    self._skill_levels[sk] += 1
                    self._last_levelup_time = time.time()
                    avail     = self.screen_reader.skill_available(
                        self.screen_reader.read_state(frame), self._skill_levels
                    )
                    avail_str = " ".join(f"{k.upper()}{v}" for k, v in self._skill_levels.items())
                    ready_str = "/".join(k.upper() for k, v in avail.items() if v) or "none"
                    log.info(
                        f"⬆️  技能升级: {sk.upper()} → Lv{self._skill_levels[sk]} "
                        f"| 等级[{avail_str}] | 就绪[{ready_str}]"
                    )
                except Exception as e:
                    log.warning(f"技能升级失败 {sk}: {e}")
                return True
        return False

    # ── Thread B：执行队列消费 ────────────────────────────────────────────────

    def _exec_loop(self):
        while not self._stop_event.is_set():
            self._poll_emergency_stop()
            try:
                action_idx = self._action_queue.get(timeout=0.2)
                if self._is_dead:
                    self._action_queue.task_done()
                    continue
                self.control.warn_if_not_focused()
                self._execute_action(action_idx)
                self._action_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log.warning(f"Thread-B 执行异常: {e}")

    # ── 寻路 / 脱困辅助 ──────────────────────────────────────────────────────

    _LANE_WAYPOINTS = {
        GameStateDetector.SIDE_BLUE: [(1200, 300), (1300, 250), (1100, 350)],
        GameStateDetector.SIDE_RED:  [(720,  750), (620,  800), (820,  700)],
    }
    _lane_wp_idx = 0
    _ESCAPE_WAYPOINTS_MM = {
        GameStateDetector.SIDE_BLUE: [(0.76, 0.78), (0.70, 0.70), (0.63, 0.60), (0.56, 0.52)],
        GameStateDetector.SIDE_RED:  [(0.24, 0.22), (0.30, 0.30), (0.37, 0.40), (0.44, 0.48)],
    }

    def _get_lane_target(self) -> tuple:
        wps = self._LANE_WAYPOINTS.get(
            self.spawn_side, self._LANE_WAYPOINTS[GameStateDetector.SIDE_BLUE]
        )
        idx = self._lane_wp_idx % len(wps)
        self._lane_wp_idx += 1
        x, y = wps[idx]
        x += random.randint(-60, 60)
        y += random.randint(-40, 40)
        return max(50, min(1870, x)), max(50, min(1030, y))

    def _is_in_fountain(self, mx: float, my: float) -> bool:
        if self.spawn_side == GameStateDetector.SIDE_RED:
            return mx < 0.22 and my < 0.22
        return mx > 0.78 and my > 0.78

    def _ensure_shop_closed(self, frame) -> bool:
        now = time.time()
        shop_open = self.game_state.detect_shop_open(frame)
        if shop_open and now - self._last_shop_toggle_time > self._shop_toggle_cooldown:
            self.control.press_key('p')
            self._last_shop_toggle_time = now
            self._shop_closed = True
            log.warning("🛠️ 异常修正: 检测到商店打开，已发送 P 关闭")
            return True
        if not shop_open:
            self._shop_closed = True
        return False

    def _update_stuck_state(self, obs: np.ndarray):
        mx, my = float(obs[11]), float(obs[12])
        cur = (mx, my)
        if self._prev_minimap_pos is None:
            self._prev_minimap_pos = cur
            return
        moved = math.hypot(mx - self._prev_minimap_pos[0], my - self._prev_minimap_pos[1])
        self._stuck_frames = self._stuck_frames + 1 if moved < 0.006 else 0
        self._prev_minimap_pos = cur

    def _get_escape_minimap_target(self) -> tuple:
        wps = self._ESCAPE_WAYPOINTS_MM.get(
            self.spawn_side, self._ESCAPE_WAYPOINTS_MM[GameStateDetector.SIDE_BLUE]
        )
        idx = self._escape_wp_idx % len(wps)
        self._escape_wp_idx += 1
        return wps[idx]

    def _is_ghost_enemy_frame(self, obs: np.ndarray, units: dict, in_fountain: bool) -> bool:
        enemies = units.get("enemies", [])
        if not enemies:
            return False
        if in_fountain:
            return True
        minimap_far        = float(obs[3]) >= 0.95
        vlm_enemy_absent   = float(obs[13]) < 0.5
        if minimap_far and vlm_enemy_absent:
            return True
        if all(float(e[1]) < 0.30 for e in enemies):
            return True
        return False

    # ── 动作执行 ──────────────────────────────────────────────────────────────

    def _execute_action(self, action_idx: int):
        enemy   = self._last_units["enemies"][0] if self._last_units["enemies"] else None
        e_minion = self._last_units["enemy_minions"][0] if self._last_units["enemy_minions"] else None

        ex = int(enemy[0] * 1920)   if enemy    else None
        ey = int(enemy[1] * 1080)   if enemy    else None
        farm_x = int(e_minion[0] * 1920) if e_minion else None
        farm_y = int(e_minion[1] * 1080) if e_minion else None
        lane_x, lane_y = self._get_lane_target()

        x = ex if ex is not None else (farm_x if farm_x is not None else lane_x)
        y = ey if ey is not None else (farm_y if farm_y is not None else lane_y)

        _ACT_NAME = ["MOVE","ATTACK","Q","W","E","R","D(闪)","F(点燃)","HOLD","RECALL(B)"]
        tgt_type  = "敌英雄" if ex else ("敌小兵" if farm_x else "线路推进")
        log.info(f"▶ EXEC [{_ACT_NAME[action_idx]}] 目标=({x},{y}) [{tgt_type}]")

        if action_idx == 0:    # MOVE
            with self._macro_lock:
                macro_cache = dict(self._macro_cache)
                latest_obs  = np.array(self._latest_obs, copy=True)
            if time.time() < self._force_escape_until and len(latest_obs) >= 13:
                emx, emy = self._get_escape_minimap_target()
                self.control.minimap_click(emx, emy)
                log.info(f"🧭 MOVE 脱困路径点=({emx:.2f},{emy:.2f})")
                return
            tx = float(macro_cache.get("minimap_target_x", 0.5))
            ty = float(macro_cache.get("minimap_target_y", 0.5))
            mc = float(macro_cache.get("macro_confidence", 0.0))
            if len(latest_obs) >= 13 and mc > 0.3:
                mm_dist = math.hypot(tx - float(latest_obs[11]), ty - float(latest_obs[12]))
                if mm_dist > 0.15:
                    self.control.minimap_click(tx, ty)
                    log.info(f"🗺️  MOVE 小地图导航 conf={mc:.2f} dist={mm_dist:.2f}")
                    return
            self.control.right_click(x, y)  # type: ignore
            log.info("🗺️  MOVE 近距屏幕导航")
        elif action_idx == 1:  # ATTACK
            self.control.attack_move(ex if ex is not None else (farm_x if farm_x is not None else lane_x),
                                     ey if ey is not None else (farm_y if farm_y is not None else lane_y))  # type: ignore
        elif action_idx == 2:  # Q
            self.control.press_key('q')
            if x is not None: self.control.left_click_at(x, y)  # type: ignore
        elif action_idx == 3:  # W
            self.control.press_key('w')
            if ex is not None: self.control.left_click_at(ex, ey)  # type: ignore
        elif action_idx == 4:  # E
            self.control.press_key('e')
            if ex is not None: self.control.left_click_at(ex, ey)  # type: ignore
        elif action_idx == 5:  # R
            self.control.press_key('r')
            if ex is not None: self.control.left_click_at(ex, ey)  # type: ignore
        elif action_idx == 6:  # D 闪现
            self.control.press_key('d')
            if ex is not None: self.control.left_click_at(ex, ey)  # type: ignore
        elif action_idx == 7:  # F 点燃/治疗
            self.control.press_key('f')
            if ex is not None: self.control.left_click_at(ex, ey)  # type: ignore
        elif action_idx == 8:  # HOLD 撤退
            rx = 400  if self.spawn_side == GameStateDetector.SIDE_BLUE else 1520
            ry = 800  if self.spawn_side == GameStateDetector.SIDE_BLUE else 280
            self.control.right_click(rx, ry)
        elif action_idx == 9:  # RECALL B
            self.control.press_key('b')
        log.info(f"✅ EXEC DONE [{_ACT_NAME[action_idx]}]")

    # ── 急停热键 ──────────────────────────────────────────────────────────────

    def _poll_emergency_stop(self):
        pressed = bool(self._user32.GetAsyncKeyState(EMERGENCY_STOP_VK) & 0x8000)
        if pressed and not self._hotkey_down:
            self._hotkey_down = True
            self._stop_event.set()
            while not self._action_queue.empty():
                try: self._action_queue.get_nowait()
                except queue.Empty: break
            if not self._emergency_stopped:
                self._emergency_stopped = True
                log.warning("🛑 F10 急停，正在安全退出...")
        elif not pressed:
            self._hotkey_down = False

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def run(self):
        if self.real_game_mode:
            self._wait_for_game()
            threading.Thread(target=self._exec_loop, daemon=True, name="Thread-B-Exec").start()
            threading.Thread(target=self._vlm_loop,  daemon=True, name="Thread-C-VLM").start()
            log.info(f"🚀 Agent 启动 | 真实模式 | state_dim={self.state_dim}")
            self._real_game_loop()
        else:
            log.info(f"🚀 Agent 启动 | 离线预热 | state_dim={self.state_dim}")
            self._offline_loop()

    def _wait_for_game(self):
        print("⏳ 等待英雄联盟进程...")
        while not _is_lol_running():
            time.sleep(5)
        print("✅ 检测到游戏进程")

        print("⏳ 等待游戏加载完成（检测 HUD 血条）...")
        stable   = 0
        deadline = time.time() + 180
        while time.time() < deadline and not self._stop_event.is_set():
            try:
                frame = self.capture()
                if self.game_state.is_game_loaded(frame, self.screen_reader):
                    stable += 1
                    if stable >= 3:
                        print("✅ 游戏 HUD 已就绪，开始运行")
                        break
                else:
                    stable = 0
            except Exception:
                pass
            time.sleep(1.0)
        else:
            print("⚠️  等待超时，尝试继续运行")

        self.control.startup_countdown(seconds=5)
        try:
            time.sleep(0.3)
            self.control.lock_camera()
            log.info("📷 相机已锁定")
        except Exception as e:
            log.warning(f"相机锁定失败（不影响运行）: {e}")

        try:
            frame = self.capture()
            self.spawn_side = self.game_state.detect_spawn_side(frame)
            side_zh = {"blue": "蓝方（左下出生）", "red": "红方（右上出生）"}.get(
                self.spawn_side, "未知"
            )
            print(f"🗺️  出生方判断: {side_zh}")
        except Exception as e:
            log.warning(f"出生方检测失败: {e}")

        try:
            time.sleep(0.5)
            frame = self.capture()
            if self.game_state.detect_shop_open(frame):
                self.control.press_key('p')
                self._last_shop_toggle_time = time.time()
                self._shop_closed = True
                log.info("🏪 关闭开局商店")
            else:
                self._shop_closed = True
            time.sleep(0.3)
        except Exception:
            pass

    # ── Thread A：真实游戏主循环（10Hz）──────────────────────────────────────

    def _real_game_loop(self):
        prev_obs:   "np.ndarray | None" = None
        step        = 0
        ep_reward   = 0.0
        last_intent = "unknown"

        while not self._stop_event.is_set():
            tick_start = time.perf_counter()
            self._poll_emergency_stop()
            try:
                frame      = self.capture()
                base_state = self.screen_reader.read_state(frame)
                obs        = self.view.get_observation(base_state, frame)

                if len(obs) != self.state_dim:
                    log.warning(f"obs 维度异常: {len(obs)} != {self.state_dim}")
                    time.sleep(0.1)
                    continue

                with self._macro_lock:
                    self._latest_obs = np.array(obs, copy=True)
                self._update_stuck_state(obs)

                hp_now = float(obs[0])

                # ── 死亡检测 ─────────────────────────────────────────────────
                if self.game_state.detect_death(frame, hp_now):
                    self._death_frames += 1
                else:
                    self._death_frames = 0

                if self._death_frames >= self._DEATH_CONFIRM:
                    if not self._is_dead:
                        self._is_dead = True
                        while not self._action_queue.empty():
                            try: self._action_queue.get_nowait()
                            except queue.Empty: break

                        # ── episode 结束：立即用已有数据训练，标记最后一步 done=True ──
                        if self.replay_buffer:
                            # 把缓冲区最后一条的 done 改为 True
                            *rest, last = self.replay_buffer
                            self.replay_buffer = rest + [(*last[:4], True)]
                            self._train_step(next_obs=None)   # done=True → bootstrap=0

                        self.episode += 1
                        log.info(
                            f"📊 [真实] Episode {self.episode:>4d} | "
                            f"步数={step} | 累计奖励={ep_reward:.2f} | "
                            f"意图={last_intent} | HP=0% | 结束原因=阵亡"
                        )
                        if self.episode % SAVE_EVERY == 0:
                            try: self.policy.ppo.save(MODEL_PATH)
                            except Exception as e: log.warning(f"模型保存失败: {e}")
                        prev_obs  = None
                        step      = 0
                        ep_reward = 0.0
                        log.info("💀 英雄阵亡，等待复活...")
                    time.sleep(0.5)
                    continue

                if self._is_dead:
                    log.info("🔄 英雄复活，恢复决策")
                    self._is_dead      = False
                    self._death_frames = 0
                    self._shop_closed  = False
                    time.sleep(0.5)
                    self._ensure_shop_closed(frame)
                    time.sleep(0.3)
                    prev_obs = None

                # ── 异常闭环 ─────────────────────────────────────────────────
                self._ensure_shop_closed(frame)
                self._try_level_skills(frame)

                units       = self.game_state.detect_units(frame)
                in_fountain = self._is_in_fountain(float(obs[11]), float(obs[12]))
                if self._is_ghost_enemy_frame(obs, units, in_fountain):
                    units = {"enemies": [], "enemy_minions": [], "allies": [], "ally_minions": []}
                    log.warning("🛠️ 异常修正: 敌人检测与VLM/小地图冲突，已屏蔽本帧单位结果")
                self._last_units = units

                if in_fountain and self._stuck_frames >= 12:
                    self._force_escape_until = time.time() + 2.5
                    self._stuck_frames = 0
                    log.warning("🛠️ 异常修正: 泉水卡住，启用强制脱困路径")

                # ── 决策 ─────────────────────────────────────────────────────
                intent      = self.think.decide(obs)
                macro_cache = self.think.get_macro_cache()
                with self._macro_lock:
                    self._macro_cache = dict(macro_cache)
                last_intent = intent

                vlm_strat = self.think.get_vlm_strategy()
                reward = (
                    _compute_reward(prev_obs, obs,
                                    intent=intent,
                                    vlm_strategy=vlm_strat,
                                    macro_cache=macro_cache)
                    if prev_obs is not None else 0.0
                )

                action, logp = self.policy.decide(obs, intent, macro_cache=macro_cache)

                # ── 日志 ─────────────────────────────────────────────────────
                _ACT    = ["MOVE","ATK","Q","W","E","R","D","F","HOLD","RCLL"]
                n_en    = len(units.get("enemies", []))
                n_em    = len(units.get("enemy_minions", []))
                en_hp_str = (
                    f"{units['enemies'][0][2]*100:.0f}%"
                    if n_en > 0 else "--"
                )
                log.info(
                    f"[TICK] hp={obs[0]*100:.0f}% mp={obs[1]*100:.0f}% "
                    f"| en_hp={en_hp_str}({n_en}) em={n_em} "
                    f"| dist={obs[3]:.2f} "
                    f"| Q={obs[4]:.2f} W={obs[5]:.2f} E={obs[6]:.2f} R={obs[7]:.2f} "
                    f"| vlm_near={obs[13]:.0f} vlm_danger={obs[15]:.0f} "
                    f"| intent={intent:<7s} act={_ACT[action]} "
                    f"rew={reward:+.2f} death_f={self._death_frames}"
                )

                # ── 写入执行队列 ──────────────────────────────────────────────
                if self._action_queue.full():
                    try: self._action_queue.get_nowait()
                    except queue.Empty: pass
                self._action_queue.put_nowait(action)

                # ── 收集经验（5 元组，含 done=False） ─────────────────────────
                if prev_obs is not None:
                    self.replay_buffer.append((prev_obs, action, reward, logp, False))
                    ep_reward += reward
                    step      += 1

                    # 每积累 TRAIN_BATCH_SIZE 步触发一次中途训练
                    if len(self.replay_buffer) >= TRAIN_BATCH_SIZE:
                        self._train_step(next_obs=obs)   # 未结束：bootstrap V(obs)

                # ── 定期保存 ──────────────────────────────────────────────────
                _now = time.time()
                if _now - self._last_save_time > 300:
                    try:
                        self.policy.ppo.save(MODEL_PATH)
                        log.info("💾 定期自动保存模型 (每5分钟)")
                    except Exception as e:
                        log.warning(f"定期保存失败: {e}")
                    self._last_save_time = _now

                prev_obs = obs

                # ── episode 超时 ──────────────────────────────────────────────
                if step >= REAL_EP_TIMEOUT:
                    if self.replay_buffer:
                        *rest, last = self.replay_buffer
                        self.replay_buffer = rest + [(*last[:4], True)]
                        self._train_step(next_obs=None)
                    self.episode += 1
                    log.info(
                        f"📊 [真实] Episode {self.episode:>4d} | "
                        f"步数={step} | 累计奖励={ep_reward:.2f} | 结束原因=超时"
                    )
                    if self.episode % SAVE_EVERY == 0:
                        try: self.policy.ppo.save(MODEL_PATH)
                        except Exception as e: log.warning(f"模型保存失败: {e}")
                    prev_obs  = None
                    step      = 0
                    ep_reward = 0.0

            except Exception as e:
                log.error(f"Thread-A 异常: {e}")
                time.sleep(0.5)

            elapsed = time.perf_counter() - tick_start
            time.sleep(max(0.0, DECISION_INTERVAL - elapsed))

    # ── 离线预热循环 ──────────────────────────────────────────────────────────

    def _offline_loop(self):
        state     = self.env.reset()
        step      = 0
        ep_reward = 0.0
        log.info("🏋️  离线预热开始 | 按 Ctrl+C 停止")

        while not self._stop_event.is_set():
            self._poll_emergency_stop()
            obs = state

            if len(obs) != STATE_DIM_BOT:
                log.warning(f"env obs 维度异常: {len(obs)} != {STATE_DIM_BOT}")
                state, _, done, _ = self.env.step(8)
                if done: state = self.env.reset()
                continue

            # 模拟 VLM 4 维语义
            vlm_sim = np.array([
                1.0 if obs[3] < 0.5  else 0.0,
                1.0 if obs[2] < 0.30 else 0.0,
                1.0 if obs[0] < 0.30 else 0.0,
                1.0 if obs[2] < 0.15 and obs[3] < 0.4 else 0.0,
            ], dtype=np.float32)
            obs_full = np.concatenate([obs, vlm_sim])

            if len(obs_full) != STATE_DIM:
                log.warning(f"offline obs_full 维度异常: {len(obs_full)} != {STATE_DIM}")
                state, _, done, _ = self.env.step(8)
                if done: state = self.env.reset()
                continue

            intent       = self.think.decide(obs_full)
            action, logp = self.policy.decide(obs_full, intent)
            next_state, reward, done, _ = self.env.step(action)

            vlm_sim_next = np.array([
                1.0 if next_state[3] < 0.5  else 0.0,
                1.0 if next_state[2] < 0.30 else 0.0,
                1.0 if next_state[0] < 0.30 else 0.0,
                1.0 if next_state[2] < 0.15 and next_state[3] < 0.4 else 0.0,
            ], dtype=np.float32)
            next_obs_full = np.concatenate([next_state, vlm_sim_next])

            ep_reward += reward
            step      += 1

            # ── 存 5 元组（含 done）────────────────────────────────────────
            self.replay_buffer.append((obs_full, action, reward, logp, done))

            if done:
                # episode 结束：立即训练，bootstrap = 0（done=True）
                self._train_step(next_obs=None)
                self.episode += 1
                log.info(
                    f"📊 Episode {self.episode:>4d} | "
                    f"步数={step:>3d} | 累计奖励={ep_reward:>7.2f} | "
                    f"意图={intent:<7s} | "
                    f"HP={obs_full[0]*100:.0f}% mana={obs_full[1]*100:.0f}% "
                    f"dist={obs_full[3]*1200:.0f}"
                )
                if self.episode % SAVE_EVERY == 0:
                    try: self.policy.ppo.save(MODEL_PATH)
                    except Exception as e: log.warning(f"模型保存失败: {e}")
                state     = self.env.reset()
                step      = 0
                ep_reward = 0.0
            else:
                # 未结束：积累到 TRAIN_BATCH_SIZE 时中途训练
                if len(self.replay_buffer) >= TRAIN_BATCH_SIZE:
                    self._train_step(next_obs=next_obs_full)
                state = next_state

            time.sleep(0.005)

    # ── PPO 训练（统一入口）──────────────────────────────────────────────────

    def _train_step(self, next_obs: "np.ndarray | None" = None):
        """
        用 replay_buffer 训练 PPO，然后清空 buffer。

        next_obs：buffer 末尾下一帧的 obs（用于 bootstrap V(s_T)）。
          - episode 正常结束 / 死亡：传 None（bootstrap = 0）
          - 中途截断（batch 满）：传当前 obs（bootstrap V(当前状态)）
        """
        if not self.replay_buffer:
            return

        # 把 5 元组解包，构造 train_episode 需要的格式
        rollout = [
            (np.array(obs), int(act), float(rew), float(lp), bool(dn))
            for obs, act, rew, lp, dn in self.replay_buffer
        ]
        self.replay_buffer.clear()

        try:
            loss     = self.policy.ppo.train_episode(rollout, next_obs=next_obs)
            rewards  = [r for _, _, r, _, _ in rollout]
            avg_rew  = sum(rewards) / len(rewards)
            rl_trust = min(1.0, self.policy._total_steps / 10000.0)
            log.info(
                f"🧠 [PPO训练] Episode {self.episode} | "
                f"batch={len(rollout)} | loss={loss:.4f} | "
                f"avg_reward={avg_rew:+.3f} | "
                f"rl_trust={rl_trust:.1%}"
            )
        except Exception as e:
            log.warning(f"训练异常: {e}")

    # ── 停止 ─────────────────────────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()
        # 退出前将剩余 buffer 训练一次
        if self.replay_buffer:
            try:
                self._train_step(next_obs=None)
            except Exception:
                pass
        try:
            self.policy.ppo.save(MODEL_PATH)
            log.info("💾 退出保存完成")
        except Exception as e:
            log.warning(f"退出保存失败: {e}")
        print("🛑 Agent 已停止")
