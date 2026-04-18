import numpy as np
import random

# 归一化上限常量（真实游戏对齐，所有维度 → 0~1）
_MAX_HP   = 150.0
_MAX_MANA = 100.0
_MAX_DIST = 1200.0
_MAX_CD   = 300.0   # 最长 CD = R = 300步（30s×10Hz）
_MAX_GOLD = 10000.0

# ── 技能 CD（步数，1步≈0.1s） ──────────────────────────────────────────────
# Q=5s W=12s E=8s R=30s D(闪现)=120s F(点燃)=30s
_CD_Q  =  50
_CD_W  = 120
_CD_E  =  80
_CD_R  = 300
_CD_D  = 1200   # 闪现 120s
_CD_F  =  300   # 点燃/治疗 30s

# 敌方每步近战伤害（dist < 400 时生效）
_ENEMY_DPS_MIN = 3
_ENEMY_DPS_MAX = 8


def _normalize(state_raw: np.ndarray) -> np.ndarray:
    """将原始 state 归一化到 [0, 1]，与真实游戏 ScreenReader 输出保持一致。
    输出 13 维：11 原始 + 2 小地图坐标 (minimap_x, minimap_y)。
    """
    hp, mana, enemy, dist, q, w, e, r, d_cd, f_cd, gold, mm_x, mm_y = state_raw
    return np.array([
        hp     / _MAX_HP,
        mana   / _MAX_MANA,
        enemy  / _MAX_HP,
        dist   / _MAX_DIST,
        q      / _MAX_CD,
        w      / _MAX_CD,
        e      / _MAX_CD,
        r      / _MAX_CD,
        d_cd   / _MAX_CD,
        f_cd   / _MAX_CD,
        gold   / _MAX_GOLD,
        mm_x,   # 小地图 x 坐标，已经是 [0,1]
        mm_y,   # 小地图 y 坐标，已经是 [0,1]
    ], dtype=np.float32)


class LoLEnv:
    def __init__(self):
        self.max_steps = 80
        self._raw = np.zeros(13, dtype=np.float32)
        self._last_action = -1
        self._repeat_count = 0

    def reset(self):
        self.step_count = 0
        self._last_action = -1
        self._repeat_count = 0
        # 随机化初始状态，覆盖不同场景：低血、蓝不足、距离远近等
        self._raw = np.array([
            random.uniform(50, _MAX_HP),     # 0: health  （50~150）
            random.uniform(40, _MAX_MANA),   # 1: mana    （40~100）
            random.uniform(60, _MAX_HP),     # 2: enemy_hp（60~150）
            random.uniform(300, 900),        # 3: distance（300~900）
            0, 0, 0, 0,                      # 4-7: Q W E R cooldown（已冷却）
            0, 0,                            # 8-9: D F cooldown
            random.uniform(0, 3000),         # 10: gold
            random.uniform(0.3, 0.7),        # 11: minimap_x（模拟位置）
            random.uniform(0.3, 0.7),        # 12: minimap_y（模拟位置）
        ], dtype=np.float32)
        return _normalize(self._raw)

    def step(self, action):
        self.step_count += 1
        health, mana, enemy, dist, q_cd, w_cd, e_cd, r_cd, d_cd, f_cd, gold, mm_x, mm_y = self._raw

        prev_enemy  = enemy
        prev_health = health
        prev_dist   = dist

        # ── 动作效果 ──────────────────────────────────────────────────────────
        if action == 0:    # MOVE：向敌方靠近
            dist = max(0, dist + random.randint(-120, 20))

        elif action == 1:  # ATTACK：普攻（无 CD，需在射程内）
            if dist < 550:
                enemy -= random.randint(5, 15)

        elif action == 2:  # SKILL_Q
            if mana >= 5 and q_cd <= 0:
                mana  -= 5
                enemy -= random.randint(10, 20)
                q_cd   = _CD_Q
            # CD 中视为无效动作，不给惩罚

        elif action == 3:  # SKILL_W（护盾/治疗）
            if mana >= 5 and w_cd <= 0:
                mana   -= 5
                health += random.randint(5, 10)
                w_cd    = _CD_W

        elif action == 4:  # SKILL_E（突进）
            if mana >= 5 and e_cd <= 0:
                mana -= 5
                dist  = max(0, dist - random.randint(100, 200))
                e_cd  = _CD_E

        elif action == 5:  # SKILL_R（大招）
            if mana >= 10 and r_cd <= 0:
                mana  -= 10
                enemy -= random.randint(30, 60)
                r_cd   = _CD_R

        elif action == 6:  # SPELL_D（闪现）
            if d_cd <= 0:
                dist = max(0, dist - random.randint(200, 400))
                d_cd = _CD_D

        elif action == 7:  # SPELL_F（点燃/治疗）
            if f_cd <= 0:
                health += random.randint(0, 30)
                f_cd    = _CD_F

        elif action == 8:  # HOLD（撤退后移）
            dist = min(_MAX_DIST, dist + random.randint(30, 100))

        elif action == 9:  # RECALL（B 键回城）
            # dist < 400 且敌方存活 → 读条被打断；否则成功回城
            if dist < 400 and enemy > 0:
                health -= random.randint(5, 15)   # 读条被打断，挨一下
            else:
                health = _MAX_HP                   # 满血
                mana   = _MAX_MANA                 # 满蓝
                dist   = _MAX_DIST                 # 回到基地

        # ── 敌方反击（dist < 400 时造成伤害）────────────────────────────────
        if dist < 400 and enemy > 0:
            health -= random.randint(_ENEMY_DPS_MIN, _ENEMY_DPS_MAX)

        # ── 每步自然回蓝 ──────────────────────────────────────────────────────
        mana = min(_MAX_MANA, mana + 0.5)

        # ── CD 递减（每步 = 0.1s） ────────────────────────────────────────────
        q_cd  = max(0, q_cd  - 1)
        w_cd  = max(0, w_cd  - 1)
        e_cd  = max(0, e_cd  - 1)
        r_cd  = max(0, r_cd  - 1)
        d_cd  = max(0, d_cd  - 1)
        f_cd  = max(0, f_cd  - 1)

        # ── reward 计算（抑制回城刷分 + 保持原框架） ───────────────────────────────
        dmg_dealt = max(0.0, prev_enemy - enemy)
        dmg_taken = max(0.0, prev_health - health)
        dist_norm = float(max(0.0, min(1.0, dist / _MAX_DIST)))

        # 基础战斗奖励
        reward = dmg_dealt * 0.65 - dmg_taken * 0.22

        # 距离 shaping（只奖励合理区间，几乎取消累积惩罚）
        if 0.25 <= dist_norm <= 0.55:
            reward += 0.55                          # 理想对线距离强正奖励
        elif dist_norm > 0.82:
            reward -= 0.45                          # 极远一次性惩罚
        elif dist_norm < 0.22:
            reward -= 0.20

        # 每步存活奖励（必须够强，鼓励活下去）
        reward += 0.38

        # 血量健康额外奖励
        if health > _MAX_HP * 0.45:
            reward += 0.25

        # 关键事件大 bonus
        if enemy <= 0 and prev_enemy > 0.1:         # 成功击杀
            reward += 160.0
        if health <= 0:                             # 死亡
            reward -= 10.0
        if action == 9 and dist >= _MAX_DIST * 0.75:  # 成功回城
            reward += 12.0

        # MOVE 动作小奖励（鼓励主动靠近）
        if action == 0:
            reward += 0.12
        # 敌方可见且可交战时，鼓励战斗动作，避免“走两步就回城”
        if enemy > 0 and dist < 650 and action in (1, 2, 3, 4, 5):
            reward += 0.18

        # 动作重复惩罚（抑制策略塌缩）
        if action == self._last_action:
            self._repeat_count += 1
        else:
            self._repeat_count = 0
        self._last_action = action
        if self._repeat_count >= 4:
            reward -= min(1.5, 0.25 * (self._repeat_count - 3))

        # 回城刷分惩罚：高血或敌方仍在场时回城，强负反馈
        if action == 9:
            if health > _MAX_HP * 0.35:
                reward -= 2.0
            if enemy > 0 and prev_dist < 700:
                reward -= 1.5

        # ── 小地图坐标模拟 ────────────────────────────────────────────────────
        if action == 0:    # MOVE 向敌方移动 → x/y 随机飘移
            mm_x = np.clip(mm_x + random.uniform(-0.02, 0.02), 0, 1)
            mm_y = np.clip(mm_y + random.uniform(-0.02, 0.02), 0, 1)
        elif action == 9:  # RECALL → 回到泉水
            if dist >= _MAX_DIST * 0.8:
                mm_x, mm_y = 0.85, 0.85

        self._raw = np.array([
            min(_MAX_HP, max(0, health)),
            max(0, mana),
            max(0, enemy),
            max(0, dist),
            q_cd, w_cd, e_cd, r_cd,
            d_cd, f_cd,
            gold,
            mm_x, mm_y,
        ], dtype=np.float32)

        done = (
            self.step_count >= self.max_steps
            or enemy  <= 0
            or health <= 0
        )
        return _normalize(self._raw), reward, done, {}
