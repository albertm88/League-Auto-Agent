"""Full system integration test - validates all components work together."""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np

# 验证 1: GameAgent 构造
from agent import GameAgent, _compute_reward, ACTION_DIM
agent = GameAgent(real_game_mode=True)
print('✅ GameAgent 构造完成')

# 验证 2: 截图 + 状态读取
frame = agent.capture()
print(f'✅ 截图: {frame.shape}')

state = agent.screen_reader.read_state(frame)
print(f'✅ read_state: {[round(x,3) for x in state]}')

obs = agent.view.get_observation(state, frame)
print(f'✅ get_observation ({len(obs)} dim): {[round(x,3) for x in obs]}')

# 验证 3: think.decide
intent = agent.think.decide(obs)
print(f'✅ think.decide: intent={intent}')

# 验证 4: policy.decide
action, logp = agent.policy.decide(obs, intent)
print(f'✅ policy.decide: action={action}, logp={logp:.3f}')

# 验证 5: 单位检测
units = agent.game_state.detect_units(frame)
ne = len(units.get("enemies", []))
nm = len(units.get("enemy_minions", []))
print(f'✅ detect_units: enemies={ne}, minions={nm}')

# 验证 6: 技能升级检测
lvlup = agent.screen_reader.read_levelup(frame)
print(f'✅ read_levelup: {lvlup}')

# 验证 7: VLM策略
vlm = agent.think.get_vlm_strategy()
print(f'✅ vlm_strategy: {vlm}')

# 验证 8: 奖励计算
r = _compute_reward(np.array(obs), np.array(obs), intent=intent, vlm_strategy=vlm)
print(f'✅ _compute_reward (same-frame): {r:.4f}')

# 验证 9: spawn side & dims
print(f'✅ spawn_side: {agent.spawn_side}')
from agent import ACTION_DIM
print(f'✅ state_dim={agent.state_dim}, ACTION_DIM={ACTION_DIM}')

# 验证 10: lane target
lx, ly = agent._get_lane_target()
print(f'✅ lane_target: ({lx}, {ly})')

print()
print('=' * 50)
print('🎉 所有验证通过！系统可以正常运行。')
print('=' * 50)
