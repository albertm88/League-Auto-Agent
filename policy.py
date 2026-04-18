# action_idx → state 中对应 CD 的下标
# state: [hp, mana, enemy_hp, dist, q_cd(4), w_cd(5), e_cd(6), r_cd(7), d_cd(8), f_cd(9), gold, ...]
_SKILL_CD_IDX = {2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9}  # Q/W/E/R/D/F
_CD_READY_THRESH = 0.05  # cd_ratio < 5% 视为就绪（0=满格就绪，1=全CD）

# ── 意图 → 允许的动作集合（白名单制） ──────────────────────────────────
# 不在白名单内的动作不会被硬覆盖，只是降低优先级
_INTENT_ACTIONS = {
    "recall":  {8, 9},       # HOLD, RECALL
    "retreat": {0, 8, 6},    # MOVE, HOLD, FLASH
    "attack":  {1, 2, 3, 4, 5, 6, 7},  # ATTACK + skills + spells
    "farm":    {0, 1, 2, 3, 4},         # MOVE, ATTACK, Q/W/E（补刀用）
    "push":    {0, 1, 2, 3, 4, 5},      # MOVE, ATTACK, Q/W/E/R
}

# 启发式行为表（RL 未训练时的默认合理动作，按意图选择）
_HEURISTIC = {
    "recall":  9,   # B
    "retreat": 0,   # MOVE（走位脱离）
    "attack":  1,   # ATTACK（普攻）
    "farm":    1,   # ATTACK（补刀）
    "push":    0,   # MOVE（推线走位）
}


class Policy:
    def __init__(self, ppo):
        self.ppo = ppo
        self._total_steps = 0   # 累计决策步数，用于判断 RL 成熟度

    def _recall_allowed(self, state) -> bool:
        """回城门控：避免在近战区/高血量时反复回城导致策略塌缩。"""
        hp = float(state[0]) if len(state) > 0 else 1.0
        enemy_hp = float(state[2]) if len(state) > 2 else 0.0
        dist = float(state[3]) if len(state) > 3 else 1.0
        # 低血允许回城；否则要求“敌方不在场 + 距离足够远”
        if hp < 0.18:
            return True
        return enemy_hp <= 0.05 and dist >= 0.72

    def decide(self, state, intent, macro_cache=None):
        self._total_steps += 1
        action, logp = self.ppo.act(state)

        hp = float(state[0]) if len(state) > 0 else 1.0
        enemy_hp = float(state[2]) if len(state) > 2 else 0.0
        dist = float(state[3]) if len(state) > 3 else 1.0

        # ── 紧急硬覆盖：仅在生死攸关时强制（RL 再怎么探索也不该送死）───
        if intent == "recall" and hp < 0.15:
            # 近身时先撤离，避免原地读条被打断
            if enemy_hp > 0.05 and dist < 0.65:
                return 8, logp
            return 9, logp     # 极低血量且较安全 → 回城
        if intent == "retreat" and hp < 0.25:
            return 0, logp     # 低血量 → 必须撤退（MOVE 脱离）

        # ── RL 成熟度混合：前 500 步以启发式为主，之后逐步信任 RL ─────
        # 这样 RL 有时间收集合理经验，而不是一开始就乱按
        import random
        rl_trust = min(1.0, self._total_steps / 10000.0)  # 0→1 线性增长
        use_rl = random.random() < rl_trust

        if not use_rl:
            # 启发式：根据意图选合理动作
            heuristic_action = _HEURISTIC.get(intent, 1)
            # CD 保护
            if heuristic_action in _SKILL_CD_IDX:
                cd_idx = _SKILL_CD_IDX[heuristic_action]
                if cd_idx < len(state) and float(state[cd_idx]) > _CD_READY_THRESH:
                    heuristic_action = 1   # 技能 CD 中 → 普攻
            # 回城门控（启发式阶段同样生效）
            if heuristic_action == 9 and not self._recall_allowed(state):
                heuristic_action = 8 if hp < 0.35 else (1 if enemy_hp > 0.05 and dist < 0.6 else 0)
            return heuristic_action, logp

        # ── RL 探索：如果动作不合理（CD 中），修正为普攻 ────────────────
        if action in _SKILL_CD_IDX:
            cd_idx = _SKILL_CD_IDX[action]
            if cd_idx < len(state) and float(state[cd_idx]) > _CD_READY_THRESH:
                action = 1   # ATTACK（普攻不受 CD 限制）

        # 回城门控（RL 阶段）：不满足回城条件时改为撤退/推进
        if action == 9 and not self._recall_allowed(state):
            action = 8 if hp < 0.35 else (1 if enemy_hp > 0.05 and dist < 0.6 else 0)

        # ── RL 软调制：高置信宏观策略只做概率性偏置，不硬覆盖 ───────────────
        if macro_cache:
            import random
            conf_raw = macro_cache.get("macro_confidence", 0.0)
            conf = float(conf_raw) if isinstance(conf_raw, (int, float)) else 0.0
            weights = macro_cache.get("action_weights", {})
            if conf > 0.25 and isinstance(weights, dict):
                # 软调制触发概率受置信度约束，最多 35%
                if random.random() < min(0.35, conf * 0.35):
                    allowed = _INTENT_ACTIONS.get(intent, set(range(10)))
                    pool, probs = [action], [1.0]
                    for aid_raw, w_raw in weights.items():
                        if not isinstance(aid_raw, (int, str)) or not isinstance(w_raw, (int, float)):
                            continue
                        if isinstance(aid_raw, str) and not aid_raw.lstrip("-").isdigit():
                            continue
                        aid = int(aid_raw)
                        w = float(w_raw)
                        if aid == action or aid not in allowed:
                            continue
                        if aid in _SKILL_CD_IDX:
                            cd_idx = _SKILL_CD_IDX[aid]
                            if cd_idx < len(state) and float(state[cd_idx]) > _CD_READY_THRESH:
                                continue
                        score = 1.0 + (w * conf * 2.0)
                        if score > 0.05:
                            pool.append(aid)
                            probs.append(score)
                    action = random.choices(pool, weights=probs, k=1)[0]
                    if action == 9 and not self._recall_allowed(state):
                        action = 8 if hp < 0.35 else (1 if enemy_hp > 0.05 and dist < 0.6 else 0)

        return action, logp
