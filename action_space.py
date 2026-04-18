from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(str, Enum):
    MOVE = "move"
    ATTACK = "attack"
    SKILL_Q = "skill_q"
    SKILL_W = "skill_w"
    SKILL_E = "skill_e"
    SKILL_R = "skill_r"
    SPELL_D = "spell_d"   # 召唤师技能 D（固定：闪现）
    SPELL_F = "spell_f"   # 召唤师技能 F（可配置：治疗 / 点燃 / 护盾 / 净化等）
    HOLD   = "hold"
    RECALL = "recall"     # B 键回城（8s 读条，低血时逃生）


@dataclass
class Action:
    action_type: ActionType
    x: Optional[int] = None
    y: Optional[int] = None

    def to_index(self):
        return list(ActionType).index(self.action_type)

    @staticmethod
    def from_index(idx: int):
        return Action(action_type=list(ActionType)[idx])