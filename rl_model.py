"""
rl_model.py — PPO 完整实现（最终版）

修复清单（对比原版）：
  1. Critic 真正参与训练：加入 value_loss，actor + critic 共同优化
  2. GAE（Generalized Advantage Estimation）替代原来的单步 advantage
  3. Discounted return 计算，不再使用原始即时 reward
  4. Entropy bonus：防止策略过早坍缩到单一动作
  5. 梯度裁剪：防止大梯度导致训练不稳定
  6. 网络加深：两层隐藏层 + LayerNorm，提升表达能力
  7. train_step 接口兼容 agent.py 现有调用（无需改 agent.py）
  8. 新增 train_episode()：支持完整 episode 训练（离线预热推荐）
  9. save/load 保存完整训练状态（含 episode 计数）
"""

import os
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

log = logging.getLogger(__name__)

# ── 超参数（集中管理，方便调整）─────────────────────────────────────────────
GAMMA       = 0.99    # 折扣因子：越高越重视长期奖励
GAE_LAMBDA  = 0.95    # GAE λ：bias-variance 权衡，0.9-0.97 均合理
CLIP_EPS    = 0.2     # PPO clip 范围：ratio ∈ [1-ε, 1+ε]
VF_COEF     = 0.5     # value loss 权重（相对于 policy loss）
ENT_COEF    = 0.01    # entropy bonus 权重（防止策略过早坍缩）
MAX_GRAD    = 0.5     # 梯度裁剪上限
PPO_EPOCHS  = 4       # 每批数据重复训练次数
HIDDEN_DIM  = 256     # 隐藏层宽度（原版 128 → 256）


# ─────────────────────────────────────────────────────────────────────────────
class PPO(nn.Module):
    """
    Actor-Critic 网络。

    相比原版改进：
      - 两层隐藏层（原版一层）
      - LayerNorm 稳定训练
      - Actor/Critic 共享前两层特征提取，减少参数量
      - 输出层使用正交初始化，加快收敛
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )

        # Actor 头：输出动作概率分布
        self.actor_head = nn.Linear(HIDDEN_DIM, action_dim)

        # Critic 头：输出状态价值 V(s)
        self.critic_head = nn.Linear(HIDDEN_DIM, 1)

        self._init_weights()

    def _init_weights(self):
        """正交初始化加快收敛，输出层用小权重。"""
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat  = self.shared(x)
        probs = torch.softmax(self.actor_head(feat), dim=-1)
        value = self.critic_head(feat)
        return probs, value

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """
        推理时一次前向传播同时返回：
          action（采样或传入）、log_prob、entropy、value
        训练时传入 action 以计算 new_log_prob。
        """
        probs, value = self.forward(x)
        dist  = Categorical(probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ─────────────────────────────────────────────────────────────────────────────
class PPOAgent:
    """
    PPO Agent，兼容 agent.py 现有调用接口。

    主要接口：
      act(state)                          → (action, log_prob)
      train_step(states, actions,         → loss_total（兼容旧调用）
                 rewards, log_probs)
      train_episode(rollout)              → loss_total（推荐，完整 episode）
      save(path) / load(path)
    """

    def __init__(self, state_dim: int = 17, action_dim: int = 10):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.model      = PPO(state_dim=state_dim, action_dim=action_dim)
        self.optimizer  = optim.Adam(self.model.parameters(), lr=3e-4, eps=1e-5)

        # 训练统计（用于日志）
        self._train_calls   = 0
        self._total_updates = 0

    # ── 推理（Thread A，<1ms）────────────────────────────────────────────────

    def act(self, state) -> Tuple[int, float]:
        """
        给定 state 采样动作，返回 (action_idx, log_prob)。
        与原版接口完全相同，agent.py 无需修改。
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state 维度 {state.shape[-1]} ≠ 模型期望 {self.state_dim}"
            )

        with torch.no_grad():
            action, log_prob, _, _ = self.model.get_action_and_value(state)

        return action.item(), log_prob.item()

    # ── GAE 计算 ─────────────────────────────────────────────────────────────

    @staticmethod
    def compute_gae(
        rewards:    List[float],
        values:     torch.Tensor,
        next_value: float,
        dones:      List[bool],
        gamma:      float = GAMMA,
        lam:        float = GAE_LAMBDA,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 GAE advantages 和 discounted returns。

        公式：
          δ_t    = r_t + γ * V(s_{t+1}) * (1-done) - V(s_t)
          A_t    = δ_t + γλ * A_{t+1}
          R_t    = A_t + V(s_t)   （用于 value loss 的 target）

        返回：
          advantages : shape (T,)，标准化后
          returns    : shape (T,)，未标准化（用于 value loss）
        """
        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0.0

        vals_np = values.detach().squeeze(-1).cpu().numpy()

        for t in reversed(range(T)):
            v_next  = next_value if t == T - 1 else float(vals_np[t + 1])
            mask    = 0.0 if dones[t] else 1.0
            delta   = rewards[t] + gamma * v_next * mask - float(vals_np[t])
            gae     = delta + gamma * lam * mask * gae
            advantages[t] = gae

        returns = advantages + values.detach().squeeze(-1).cpu()

        # 标准化 advantage（降低方差，稳定训练）
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    # ── 核心训练（完整 episode，推荐） ──────────────────────────────────────

    def train_episode(
        self,
        rollout: List[Tuple],   # [(obs, action, reward, log_prob, done), ...]
        next_obs: np.ndarray = None,
    ) -> float:
        """
        用完整 episode（或多步 rollout）训练 PPO。

        rollout 格式：[(obs, action, reward, log_prob, done), ...]
          - done=True 表示该步是 episode 结束帧

        next_obs：rollout 结束后的下一帧观测（用于计算 bootstrap value）。
          若为 None 或 done=True 结束，bootstrap value = 0。
        """
        if len(rollout) == 0:
            return 0.0

        # 解包
        has_done = len(rollout[0]) == 5
        if has_done:
            obs_list, act_list, rew_list, lp_list, done_list = zip(*rollout)
        else:
            obs_list, act_list, rew_list, lp_list = zip(*rollout)
            done_list = [False] * (len(rollout) - 1) + [True]

        states_t   = torch.FloatTensor(np.stack(obs_list))
        actions_t  = torch.LongTensor(list(act_list))
        old_lp_t   = torch.FloatTensor(list(lp_list))

        # 一次前向：计算所有 V(s_t)
        with torch.no_grad():
            _, values = self.model(states_t)

            # Bootstrap value for the state after rollout ends
            if next_obs is not None and not done_list[-1]:
                ns_t = torch.FloatTensor(next_obs).unsqueeze(0)
                _, nv = self.model(ns_t)
                next_value = float(nv.item())
            else:
                next_value = 0.0

        advantages, returns = self.compute_gae(
            list(rew_list), values, next_value, list(done_list)
        )
        advantages = advantages.to(states_t.device)
        returns    = returns.to(states_t.device)

        # ── PPO_EPOCHS 次更新 ────────────────────────────────────────────────
        total_loss = 0.0
        for epoch in range(PPO_EPOCHS):
            # 随机打乱 mini-batch（增强样本独立性）
            idx = torch.randperm(len(rollout))
            s_b  = states_t[idx]
            a_b  = actions_t[idx]
            lp_b = old_lp_t[idx]
            adv_b = advantages[idx]
            ret_b = returns[idx]

            _, new_lp, entropy, new_val = self.model.get_action_and_value(s_b, a_b)

            # Policy loss（PPO clip）
            ratio       = (new_lp - lp_b).exp()
            policy_loss = -torch.min(
                ratio * adv_b,
                torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_b,
            ).mean()

            # Value loss（Huber loss 对异常奖励更鲁棒）
            value_loss = nn.functional.huber_loss(
                new_val.squeeze(-1), ret_b, delta=10.0
            )

            # Entropy bonus（防止策略过早收敛）
            entropy_loss = -entropy.mean()

            loss = policy_loss + VF_COEF * value_loss + ENT_COEF * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD)
            self.optimizer.step()

            total_loss += loss.item()
            self._total_updates += 1

        avg_loss = total_loss / PPO_EPOCHS
        self._train_calls += 1

        if self._train_calls % 20 == 0:
            with torch.no_grad():
                _, v_check = self.model(states_t)
            log.info(
                f"🧠 [PPO] call={self._train_calls} "
                f"updates={self._total_updates} "
                f"loss={avg_loss:.4f} "
                f"adv_mean={advantages.mean():.3f} adv_std={advantages.std():.3f} "
                f"V_mean={v_check.mean():.3f} V_std={v_check.std():.3f} "
                f"ret_mean={returns.mean():.3f}"
            )

        return avg_loss

    # ── 兼容旧接口（agent.py 直接调用，无需修改 agent.py）──────────────────

    def train_step(
        self,
        states:    List[np.ndarray],
        actions:   List[int],
        rewards:   List[float],
        log_probs: List[float],
    ) -> float:
        """
        兼容 agent.py 现有调用：
          loss = self.policy.ppo.train_step(states, actions, rewards, log_probs)

        内部转换为 train_episode() 调用，补充完整 GAE 计算。
        注意：这里没有 done 信息，默认最后一步为 episode 结束。
        如果 agent.py 传入的是跨 episode 混合 batch，精度会略低，
        但仍远优于原版（原版完全没有 value loss 和 GAE）。
        """
        rollout = list(zip(states, actions, rewards, log_probs))
        return self.train_episode(rollout, next_obs=None)

    # ── 保存 / 加载 ──────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model":         self.model.state_dict(),
                "optimizer":     self.optimizer.state_dict(),
                "state_dim":     self.state_dim,
                "action_dim":    self.action_dim,
                "train_calls":   self._train_calls,
                "total_updates": self._total_updates,
            },
            path,
        )
        log.info(f"💾 模型已保存: {path}  (updates={self._total_updates})")

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        saved_state = ckpt.get("state_dim", self.state_dim)
        saved_act   = ckpt.get("action_dim", self.action_dim)
        if saved_state != self.state_dim or saved_act != self.action_dim:
            raise ValueError(
                f"检查点维度 state={saved_state}/action={saved_act} "
                f"与当前 state={self.state_dim}/action={self.action_dim} 不匹配"
            )

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._train_calls   = ckpt.get("train_calls", 0)
        self._total_updates = ckpt.get("total_updates", 0)
        log.info(
            f"📂 模型已加载: {path}  "
            f"(train_calls={self._train_calls}, updates={self._total_updates})"
        )
