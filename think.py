import io
import json
import base64
import logging
import threading
from typing import TYPE_CHECKING, cast

from PIL import Image

if TYPE_CHECKING:
    from llama_cpp import Llama, CreateChatCompletionResponse

log = logging.getLogger(__name__)

# 优先使用 Qwen3VL（真正的视觉语言模型），降级到 Qwen3.5+mmproj
_MODEL_CANDIDATES = [
    # 直接在 model/ 目录（实际文件位置）
    ("./model/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
     "./model/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"),
    # 子目录备用
    ("./model/qwen3vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
     "./model/qwen3vl/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"),
    ("./model/qwen3.5/Qwen3.5-4B-IQ4_XS.gguf",
     "./model/qwen3.5/mmproj-F16.gguf"),
]

# VLM 缓存默认值
_VLM_DEFAULT = {
    "enemy_near":   0.0,
    "enemy_low_hp": 0.0,
    "in_danger":    0.0,
    "can_kill":     0.0,
}

# VLM 策略建议默认值（高层指导，用于驱动 RL 奖励塑形）
_VLM_STRATEGY_DEFAULT = {
    "suggested_intent": "farm",    # recall/retreat/attack/farm/push
    "aggression":       0.5,       # 0=纯防御, 1=全力进攻
    "confidence":       0.0,       # VLM 对自身判断的置信度 0-1
}


class DecisionThinker:
    def __init__(
        self,
        model_path:  "str | None" = None,
        mmproj_path: "str | None" = None,
    ):
        self.llm_available          = False
        self.llm: "Llama | None"    = None

        # Thread-safe VLM 缓存（Thread C 写入，Thread A 读取，互不阻塞）
        self._vlm_lock  = threading.Lock()
        self._vlm_cache = dict(_VLM_DEFAULT)
        self._vlm_strategy = dict(_VLM_STRATEGY_DEFAULT)

        # 自动选择可用模型
        import os
        candidates = []
        if model_path and mmproj_path:
            candidates.append((model_path, mmproj_path))
        candidates.extend(_MODEL_CANDIDATES)

        for mp, mm in candidates:
            if os.path.isfile(mp) and os.path.isfile(mm):
                try:
                    from llama_cpp import Llama
                    self.llm = Llama(
                        model_path   = mp,
                        mmproj_path  = mm,
                        n_gpu_layers = -1,
                        n_ctx        = 4096,
                        chat_format  = "chatml",
                        verbose      = False,
                    )
                    self.llm_available = True
                    log.info(f"✅ VLM 已加载: {mp}")
                    log.info(f"   mmproj: {mm}")
                    break
                except Exception as e:
                    log.warning(f"⚠️  VLM 加载失败 ({mp}): {e}")
        if not self.llm_available:
            log.warning("⚠️  所有 VLM 模型均不可用，视觉语义将保持零向量")

    # ── Thread C 调用（慢，可阻塞，~300-600ms）───────────────────────────────

    # ── VLM 宏观指令缓存（Thread C 写，Thread A 读）─────────────────────────
    _MACRO_DEFAULT: dict = {
        "macro_goal": "farm",           # farm / push / group / recall / invade
        "minimap_target_x": 0.5,        # 小地图目标坐标 x [0,1]
        "minimap_target_y": 0.5,        # 小地图目标坐标 y [0,1]
        "reward_modifier": 0.0,         # [-1, +1] VLM 对当前行为的评价
        "action_weights": {},           # 动作权重调节 {action_id: weight_bonus}
        "macro_confidence": 0.0,        # 宏观判断置信度
    }

    def vision_parse(self, image_path: str) -> dict:
        """
        读取游戏截图，VLM 判断战场语义，结果写入线程安全缓存。
        由 Thread C（慢循环，~2Hz）调用，Thread A 永不直接调用此方法。
        """
        if not self.llm_available:
            return dict(_VLM_DEFAULT)

        try:
            # 缩小到 640×360 再传 VLM，大幅减少 image token 用量（原图约 2000+ tokens）
            img = Image.open(image_path).resize((640, 360), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except FileNotFoundError:
            return dict(_VLM_DEFAULT)

        prompt_text = (
            "你在分析一张英雄联盟截图。\n"
            "请判断以下信息，仅输出JSON，不要有任何其他文字：\n"
            '{\n'
            '  "enemy_near": 0或1,\n'
            '  "enemy_low_hp": 0或1,\n'
            '  "in_danger": 0或1,\n'
            '  "can_kill": 0或1,\n'
            '  "suggested_intent": "farm"或"attack"或"retreat"或"push"或"recall",\n'
            '  "aggression": 0.0到1.0之间的小数,\n'
            '  "confidence": 0.0到1.0之间的小数\n'
            '}\n'
            '说明：\n'
            '- suggested_intent: 根据画面判断当前最佳策略\n'
            '- aggression: 进攻倾向(0=保守防御 1=全力进攻)\n'
            '- confidence: 你对自己判断的信心(0=完全不确定 1=非常确定)'
        )
        # ── 打印发送给 LLM 的上下文 ──────────────────────────────────────────
        log.info(
            f"\n{'─'*60}\n"
            f"[LLM 输入]\n{prompt_text}\n"
            f"[图像] {image_path} (640×360 JPEG)\n"
            f"{'─'*60}"
        )
        try:
            assert self.llm is not None
            res: "CreateChatCompletionResponse" = cast(
                "CreateChatCompletionResponse",
                self.llm.create_chat_completion(
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }],
                    temperature = 0.1,
                    max_tokens  = 60,
                    stream      = False,
                )
            )
            text = res["choices"][0]["message"]["content"]
            assert isinstance(text, str)
            # ── 打印 LLM 原始回复 ─────────────────────────────────────────────
            log.info(f"[LLM 原始回复] {text.strip()}")
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start != -1 and end > 0:
                raw    = json.loads(text[start:end])
                parsed = {k: float(raw.get(k, 0)) for k in _VLM_DEFAULT}
                # 提取策略建议
                strategy = {
                    "suggested_intent": str(raw.get("suggested_intent", "farm")),
                    "aggression":       float(raw.get("aggression", 0.5)),
                    "confidence":       float(raw.get("confidence", 0.0)),
                }
                # 校验 intent 合法值
                valid_intents = {"recall", "retreat", "attack", "farm", "push"}
                if strategy["suggested_intent"] not in valid_intents:
                    strategy["suggested_intent"] = "farm"
                strategy["aggression"] = max(0.0, min(1.0, strategy["aggression"]))
                strategy["confidence"] = max(0.0, min(1.0, strategy["confidence"]))

                with self._vlm_lock:
                    self._vlm_cache = parsed
                    self._vlm_strategy = strategy
                log.info(
                    f"[LLM 解析] "
                    f"enemy_near={parsed['enemy_near']:.0f}  "
                    f"enemy_low_hp={parsed['enemy_low_hp']:.0f}  "
                    f"in_danger={parsed['in_danger']:.0f}  "
                    f"can_kill={parsed['can_kill']:.0f}  "
                    f"| 策略={strategy['suggested_intent']} "
                    f"攻击性={strategy['aggression']:.1f} "
                    f"置信度={strategy['confidence']:.1f}"
                )
                return parsed
            else:
                log.warning(f"[LLM] 回复无法解析为 JSON，原文: {text.strip()}")
        except Exception as e:
            log.debug(f"vision_parse 失败: {e}")

        return dict(_VLM_DEFAULT)

    # ── VLM 小地图分析（宏观决策 + 目标坐标 + 奖励调节 + 动作权重）────────────

    def vision_parse_minimap(self, minimap_path: str, my_pos: tuple,
                             spawn_side: str = "blue") -> dict:
        """
        VLM 分析小地图裁切图，输出宏观策略指令。

        Parameters:
            minimap_path: 小地图截图路径
            my_pos: (x_norm, y_norm) 当前位置
            spawn_side: "blue" 或 "red"

        返回值写入 self._macro_cache，Thread A 通过 get_macro_cache() 读取。
        """
        if not self.llm_available:
            return dict(self._MACRO_DEFAULT)

        try:
            img = Image.open(minimap_path).resize((256, 256), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except FileNotFoundError:
            return dict(self._MACRO_DEFAULT)

        prompt_text = (
            f"你在分析英雄联盟的小地图。我方是{spawn_side}方，"
            f"我当前位置在小地图坐标 ({my_pos[0]:.2f}, {my_pos[1]:.2f})。\n"
            "坐标说明：(0,0)=左上角(红方基地), (1,1)=右下角(蓝方基地)。\n"
            "请根据小地图上敌我双方位置分布，判断宏观策略。\n"
            "仅输出JSON，不要有任何其他文字：\n"
            '{\n'
            '  "macro_goal": "farm"或"push"或"group"或"recall"或"invade",\n'
            '  "minimap_target_x": 0.0到1.0(建议移动的小地图x坐标),\n'
            '  "minimap_target_y": 0.0到1.0(建议移动的小地图y坐标),\n'
            '  "reward_modifier": -1.0到1.0(对当前位置的评价，'
            '正=位置好应奖励,负=位置差应惩罚),\n'
            '  "push_weight": -0.3到0.3(推线动作权重调整),\n'
            '  "retreat_weight": -0.3到0.3(撤退动作权重调整),\n'
            '  "attack_weight": -0.3到0.3(进攻动作权重调整),\n'
            '  "macro_confidence": 0.0到1.0\n'
            '}\n'
            '说明：\n'
            '- macro_goal: 整体策略方向\n'
            '- minimap_target: 建议英雄接下来应前往的地图位置\n'
            '- reward_modifier: 当前走位是否合理(正=好,负=差)\n'
            '- *_weight: 对应动作类型的概率调整(正=鼓励,负=抑制)\n'
            '- macro_confidence: 判断置信度'
        )

        log.info(
            f"\n{'─'*60}\n"
            f"[LLM 小地图输入]\n{prompt_text}\n"
            f"[小地图] {minimap_path} (256×256 JPEG)\n"
            f"{'─'*60}"
        )
        try:
            assert self.llm is not None
            res = cast(
                "CreateChatCompletionResponse",
                self.llm.create_chat_completion(
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }],
                    temperature=0.1,
                    max_tokens=120,
                    stream=False,
                )
            )
            text = res["choices"][0]["message"]["content"]
            assert isinstance(text, str)
            log.info(f"[LLM 小地图回复] {text.strip()}")
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start != -1 and end > 0:
                raw = json.loads(text[start:end])
                macro = {
                    "macro_goal": str(raw.get("macro_goal", "farm")),
                    "minimap_target_x": max(0.0, min(1.0, float(raw.get("minimap_target_x", 0.5)))),
                    "minimap_target_y": max(0.0, min(1.0, float(raw.get("minimap_target_y", 0.5)))),
                    "reward_modifier": max(-1.0, min(1.0, float(raw.get("reward_modifier", 0.0)))),
                    "action_weights": {},
                    "macro_confidence": max(0.0, min(1.0, float(raw.get("macro_confidence", 0.0)))),
                }
                # 解析动作权重调节
                for key, action_ids in [
                    ("push_weight",    [0, 1, 2, 3, 4, 5]),   # MOVE_LANE + skills
                    ("retreat_weight",  [8]),                   # HOLD_RETREAT
                    ("attack_weight",   [1, 2, 3, 4, 5, 6, 7]), # ATTACK + skills
                ]:
                    w = max(-0.3, min(0.3, float(raw.get(key, 0.0))))
                    if abs(w) > 0.05:
                        for aid in action_ids:
                            macro["action_weights"][aid] = \
                                macro["action_weights"].get(aid, 0.0) + w

                valid_goals = {"farm", "push", "group", "recall", "invade"}
                if macro["macro_goal"] not in valid_goals:
                    macro["macro_goal"] = "farm"

                with self._vlm_lock:
                    self._macro_cache = macro
                log.info(
                    f"[LLM 宏观策略] "
                    f"goal={macro['macro_goal']} "
                    f"target=({macro['minimap_target_x']:.2f},{macro['minimap_target_y']:.2f}) "
                    f"reward_mod={macro['reward_modifier']:+.2f} "
                    f"conf={macro['macro_confidence']:.1f} "
                    f"weights={macro['action_weights']}"
                )
                return macro
            else:
                log.warning(f"[LLM 小地图] 无法解析JSON: {text.strip()}")
        except Exception as e:
            log.debug(f"vision_parse_minimap 失败: {e}")

        return dict(self._MACRO_DEFAULT)

    def get_vlm_cache(self) -> dict:
        """Thread A 无阻塞读取最新 VLM 缓存，始终立即返回。"""
        with self._vlm_lock:
            return dict(self._vlm_cache)

    def get_vlm_strategy(self) -> dict:
        """Thread A 无阻塞读取 VLM 策略建议，始终立即返回。"""
        with self._vlm_lock:
            return dict(self._vlm_strategy)

    def get_macro_cache(self) -> dict:
        """Thread A 无阻塞读取最新宏观策略缓存。"""
        with self._vlm_lock:
            return dict(getattr(self, "_macro_cache", self._MACRO_DEFAULT))

    # ── Thread A 调用（纯规则，<1ms，绝不调用 LLM）──────────────────────────

    def decide(self, state) -> str:
        """
        高层意图判断，综合规则 + VLM 语义，保证 <1ms 返回。
        state 布局（17 维推荐；兼容旧 15 维）：
          [0]hp [1]mana [2]enemy_hp [3]dist
          [4]q_cd [5]w_cd [6]e_cd [7]r_cd [8]d_cd [9]f_cd [10]gold
          [11]minimap_x [12]minimap_y
          [13]enemy_near [14]enemy_low_hp [15]in_danger [16]can_kill
        返回: recall / retreat / attack / farm / push

        策略概述：
          级别 1 (生存)： hp极低→先距离安全处再 recall；没攟出嘅先退
          级别 2 (经济)：金币足够且安全→回城买装备
          级别 3 (战斗)：有敌且可击杀→进攻；敌方血量足→等待时机
          级别 4 (发展)：无敌/安全→打小兵/推路
        """
        hp       = float(state[0])
        mana     = float(state[1])
        enemy_hp = float(state[2])
        dist     = float(state[3])
        gold     = float(state[10]) if len(state) > 10 else 0.0

        # OCR 误读防护：金币读值 > 0.90（即 ≥9000 金）视为 OCR 错误，归零
        if gold > 0.90:
            gold = 0.0

        # VLM 语义（17 维布局优先；兼容旧 15 维布局）
        if len(state) >= 17:
            enemy_near = float(state[13])
            in_danger  = float(state[15])
            can_kill   = float(state[16])
        else:
            enemy_near = float(state[11]) if len(state) > 11 else 0.0
            in_danger  = float(state[13]) if len(state) > 13 else 0.0
            can_kill   = float(state[14]) if len(state) > 14 else 0.0

        # ── 级别 1：生存优先 ────────────────────────────────────────────────────────
        # 血量极低：不能就地 recall（读条被打断），必须先拉开安全距离
        if hp < 0.10 or in_danger > 0.7:
            # 敌人还在身边，先退（跳闪现、咕肖跟迪）
            if dist < 0.6 or enemy_near > 0.5:
                return "retreat"
            # 已远离敌人，可以安全回城
            return "recall"

        # 血量低且敌方在辺：优先撃杀敌人或者退
        if hp < 0.25:
            if (can_kill > 0.5 or enemy_hp < 0.15) and dist < 0.35:
                return "attack"   # 血红眼击杀
            return "retreat"

        # ── 级别 2：经济思路 ────────────────────────────────────────────────────────
        # 金币超过 3000（可以买大件）且当前安全时→回城
        # 但满血满蓝时大概率在基地，不应回城（避免无限回城循环）
        in_base = (hp > 0.95 and mana > 0.90 and enemy_hp <= 0.0 and dist >= 0.99)
        if gold > 0.30 and enemy_near < 0.3 and not in_base:
            # 血量不满说明在外面打过架，可以回城
            if hp < 0.85 and dist > 0.7:
                return "recall"

        # ── 级别 3：战斗判断 ────────────────────────────────────────────────────────
        if enemy_hp > 0.0:   # 敌方活着
            # VLM/距离判断可击杀
            can_kill_flag = can_kill > 0.5 or (enemy_hp < 0.15 and dist < 0.4)
            # 敌方血量少且在射程内
            in_range = dist < 0.42   # ~500 履带

            if can_kill_flag and in_range:
                return "attack"
            if enemy_hp < 0.5 and dist < 0.33:   # 敌方半血且距离近
                return "attack"
            # 敌方尚足且她在辺：保持安全距离、会和等待 CD
            if enemy_near > 0.5 and enemy_hp > 0.5:
                return "farm"   # 暂时远离等待时机（farm 小兵来操控距离）

        # ── 级别 4：发展（敌方已死或不在诊） ──────────────────────────────────────
        if enemy_hp <= 0.0:
            return "push"   # 下塔推路
        rule_intent = "farm"       # 打小兵补 CS

        # ── VLM 策略覆盖（高置信度时 VLM 建议优先于规则） ──────────────────
        vlm_strat = self.get_vlm_strategy()
        vlm_intent    = vlm_strat.get("suggested_intent", "farm")
        vlm_conf      = vlm_strat.get("confidence", 0.0)
        vlm_aggro     = vlm_strat.get("aggression", 0.5)

        # VLM 置信度 > 0.6 时，采纳 VLM 的高层建议（覆盖规则层 3-4 的结论）
        if vlm_conf > 0.6 and vlm_intent in ("attack", "retreat", "push", "recall"):
            log.debug(
                f"[VLM覆盖] 规则={rule_intent} → VLM={vlm_intent} "
                f"(conf={vlm_conf:.1f} aggro={vlm_aggro:.1f})"
            )
            return vlm_intent

        return rule_intent
