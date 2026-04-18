"""行为逻辑测试 — 模拟各种游戏场景，验证 think + policy 输出。"""
import numpy as np
from think import DecisionThinker
from policy import Policy
from rl_model import PPOAgent

# 用 dummy VLM (不加载模型)
think = DecisionThinker.__new__(DecisionThinker)
think.llm_available = False
think.llm = None
think._vlm_lock = __import__('threading').Lock()
think._vlm_cache = {"enemy_near": 0, "enemy_low_hp": 0, "in_danger": 0, "can_kill": 0} # type: ignore
think._vlm_strategy = {"suggested_intent": "farm", "aggression": 0.5, "confidence": 0.0}

policy = Policy(PPOAgent(state_dim=17, action_dim=10))

ACT = ["MOVE","ATK","Q","W","E","R","D","F","HOLD","RCLL"]

def test(name, state_17):
    intent = think.decide(state_17)
    action, logp = policy.decide(state_17, intent)
    print(f"  {name:30s} → intent={intent:<8s} act={ACT[action]:<6s} (#{action})")

print("=" * 70)
print("场景测试：think.decide() + policy.decide()")
print("=" * 70)

# 17维: hp,mana,enemy_hp,dist,q,w,e,r,d,f,gold,minimap_x,minimap_y,near,lowhp,danger,kill
print("\n--- 在基地（满血满蓝无敌人）---")
test("刚出生/在基地，低金币",    np.array([1.0, 1.0, 0.0, 1.0, 0,0,0,0,0,0, 0.05, 0.85,0.85, 0,0,0,0]))
test("在基地，中等金币(2300g)",  np.array([1.0, 1.0, 0.0, 1.0, 0,0,0,0,0,0, 0.23, 0.85,0.85, 0,0,0,0]))
test("在基地，高金币(4000g)",   np.array([1.0, 1.0, 0.0, 1.0, 0,0,0,0,0,0, 0.40, 0.85,0.85, 0,0,0,0]))

print("\n--- 在线上（有敌方小兵，无敌英雄）---")
test("线上补刀，正常血量",       np.array([0.80, 0.70, 0.0, 1.0, 0,0,0,0,0,0, 0.10, 0.62,0.58, 0,0,0,0]))
test("线上推线，满血",          np.array([0.95, 0.85, 0.0, 0.8, 0,0,0,0,0,0, 0.15, 0.66,0.52, 0,0,0,0]))

print("\n--- 对线（敌方英雄可见）---")
test("对线，双方满血距离远",     np.array([0.90, 0.80, 0.90, 0.6, 0,0,0,0,0,0, 0.10, 0.55,0.45, 1,0,0,0]))
test("对线，敌方半血近距离",     np.array([0.85, 0.60, 0.40, 0.3, 0,0,0,0,0,0, 0.10, 0.56,0.46, 1,1,0,0]))
test("对线，敌方残血可击杀",     np.array([0.70, 0.50, 0.10, 0.25, 0,0,0,0,0,0, 0.05, 0.57,0.47, 1,1,0,1]))

print("\n--- 危险状态 ---")
test("血量低，敌人在旁边",       np.array([0.20, 0.30, 0.70, 0.3, 0,0,0,0,0,0, 0.05, 0.58,0.48, 1,0,1,0]))
test("血量极低，敌人远了",       np.array([0.08, 0.20, 0.50, 0.8, 0,0,0,0,0,0, 0.05, 0.30,0.70, 0,0,0,0]))
test("血量极低，敌人近身",       np.array([0.05, 0.10, 0.80, 0.2, 0,0,0,0,0,0, 0.02, 0.58,0.48, 1,0,1,0]))

print("\n--- 回城场景（正确触发）---")
test("残血远离+高金币",         np.array([0.60, 0.40, 0.0, 1.0, 0,0,0,0,0,0, 0.35, 0.35,0.75, 0,0,0,0]))
test("7成血远离+高金币",        np.array([0.75, 0.50, 0.0, 1.0, 0,0,0,0,0,0, 0.40, 0.35,0.75, 0,0,0,0]))

print("\n" + "=" * 70)
print("✅ 如果上面的意图和动作合理，则逻辑修复成功")
print("=" * 70)
