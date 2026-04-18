"""Unit tests for the PPO components (network, buffer, agent)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ---- Configuration fixtures -------------------------------------------------

OBS_SHAPE = (3, 45, 80)   # Small obs for fast tests
LLM_DIM = 8
NUM_ACTIONS = 20
HIDDEN_DIM = 64

PPO_CFG = {
    "obs_shape": list(OBS_SHAPE),
    "llm_feature_dim": LLM_DIM,
    "num_actions": NUM_ACTIONS,
    "hidden_dim": HIDDEN_DIM,
    "lr": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "update_epochs": 2,
    "batch_size": 16,
    "buffer_capacity": 64,
    "checkpoint_dir": "/tmp/test_checkpoints",
    "checkpoint_interval": 9999,  # disable auto-save during tests
}


# ---- PPONetwork tests -------------------------------------------------------

class TestPPONetwork:
    def _make_network(self):
        from src.ppo.network import PPONetwork
        return PPONetwork(OBS_SHAPE, LLM_DIM, NUM_ACTIONS, HIDDEN_DIM)

    def test_forward_shapes(self):
        net = self._make_network()
        B = 4
        obs = torch.rand(B, *OBS_SHAPE)
        llm = torch.rand(B, LLM_DIM)
        logits, value = net(obs, llm)

        assert logits.shape == (B, NUM_ACTIONS)
        assert value.shape == (B, 1)

    def test_get_action_and_value(self):
        net = self._make_network()
        obs = torch.rand(2, *OBS_SHAPE)
        llm = torch.rand(2, LLM_DIM)
        action, log_prob, entropy, value = net.get_action_and_value(obs, llm)

        assert action.shape == (2,)
        assert log_prob.shape == (2,)
        assert entropy.shape == (2,)
        assert value.shape == (2, 1)

    def test_provided_action_respected(self):
        net = self._make_network()
        obs = torch.rand(3, *OBS_SHAPE)
        llm = torch.rand(3, LLM_DIM)
        fixed_actions = torch.tensor([0, 5, 19])
        action, _, _, _ = net.get_action_and_value(obs, llm, fixed_actions)

        assert torch.all(action == fixed_actions)

    def test_gradients_flow(self):
        net = self._make_network()
        obs = torch.rand(2, *OBS_SHAPE, requires_grad=False)
        llm = torch.rand(2, LLM_DIM)
        _, log_prob, _, value = net.get_action_and_value(obs, llm)
        loss = -log_prob.mean() + value.mean()
        loss.backward()

        for param in net.parameters():
            if param.requires_grad:
                assert param.grad is not None


# ---- RolloutBuffer tests ----------------------------------------------------

class TestRolloutBuffer:
    def _make_buffer(self, capacity: int = 32) -> "RolloutBuffer":
        from src.ppo.replay_buffer import RolloutBuffer
        return RolloutBuffer(
            capacity=capacity,
            obs_shape=OBS_SHAPE,
            llm_feature_dim=LLM_DIM,
            gamma=0.99,
            gae_lambda=0.95,
        )

    def _dummy_transition(self):
        obs = np.random.rand(*OBS_SHAPE).astype(np.float32)
        llm = np.zeros(LLM_DIM, dtype=np.float32)
        llm[0] = 1.0
        return obs, llm

    def test_add_and_size(self):
        buf = self._make_buffer(8)
        obs, llm = self._dummy_transition()
        buf.add(obs, llm, 0, -0.5, 1.0, False, 0.3)

        assert buf._size == 1
        assert not buf.is_full()

    def test_is_full_after_capacity(self):
        cap = 8
        buf = self._make_buffer(cap)
        obs, llm = self._dummy_transition()
        for _ in range(cap):
            buf.add(obs, llm, 0, 0.0, 0.0, False, 0.0)

        assert buf.is_full()

    def test_clear(self):
        buf = self._make_buffer(8)
        obs, llm = self._dummy_transition()
        buf.add(obs, llm, 0, 0.0, 0.0, False, 0.0)
        buf.clear()

        assert buf._size == 0
        assert buf._ptr == 0

    def test_compute_advantages_shape(self):
        cap = 16
        buf = self._make_buffer(cap)
        obs, llm = self._dummy_transition()
        for _ in range(cap):
            buf.add(obs, llm, 0, 0.0, 1.0, False, 0.5)

        adv, ret = buf.compute_advantages(last_value=0.5)
        assert adv.shape == (cap,)
        assert ret.shape == (cap,)

    def test_get_batches_yields_tensors(self):
        cap = 32
        buf = self._make_buffer(cap)
        obs, llm = self._dummy_transition()
        for _ in range(cap):
            buf.add(obs, llm, 2, -0.1, 0.5, False, 0.3)

        batches = list(buf.get_batches(batch_size=16, last_value=0.0))
        assert len(batches) == 2  # 32 / 16 = 2 full batches

        for batch in batches:
            assert len(batch) == 6
            for tensor in batch:
                assert isinstance(tensor, torch.Tensor)


# ---- PPOAgent tests ---------------------------------------------------------

class TestPPOAgent:
    def _make_agent(self) -> "PPOAgent":
        from src.ppo.ppo_agent import PPOAgent
        return PPOAgent(PPO_CFG, device="cpu")

    def test_select_action_returns_tuple(self):
        agent = self._make_agent()
        obs = np.random.rand(*OBS_SHAPE).astype(np.float32)
        llm = np.zeros(LLM_DIM, dtype=np.float32)
        llm[0] = 1.0

        action, log_prob, value = agent.select_action(obs, llm)

        assert isinstance(action, int)
        assert 0 <= action < NUM_ACTIONS
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_store_and_buffer_size(self):
        agent = self._make_agent()
        obs = np.random.rand(*OBS_SHAPE).astype(np.float32)
        llm = np.zeros(LLM_DIM, dtype=np.float32)
        llm[0] = 1.0
        agent.store_transition(obs, llm, 0, -0.5, 1.0, False, 0.3)

        assert agent.buffer._size == 1

    def test_update_returns_stats(self):
        agent = self._make_agent()
        obs = np.random.rand(*OBS_SHAPE).astype(np.float32)
        llm = np.zeros(LLM_DIM, dtype=np.float32)
        llm[0] = 1.0
        cap = PPO_CFG["buffer_capacity"]

        for _ in range(cap):
            agent.store_transition(obs, llm, 0, -0.5, 1.0, False, 0.3)

        stats = agent.update(last_value=0.0)

        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy" in stats

    def test_update_clears_buffer(self):
        agent = self._make_agent()
        obs = np.random.rand(*OBS_SHAPE).astype(np.float32)
        llm = np.zeros(LLM_DIM, dtype=np.float32)
        llm[0] = 1.0
        cap = PPO_CFG["buffer_capacity"]

        for _ in range(cap):
            agent.store_transition(obs, llm, 0, -0.5, 1.0, False, 0.3)

        agent.update(last_value=0.0)
        assert agent.buffer._size == 0

    def test_save_and_load_checkpoint(self, tmp_path):
        agent = self._make_agent()
        ckpt_path = str(tmp_path / "test.pt")
        saved_path = agent.save_checkpoint(ckpt_path)

        # Modify a parameter, then reload
        with torch.no_grad():
            for p in agent.network.parameters():
                p.fill_(0.0)
                break

        agent.load_checkpoint(saved_path)
        # After loading, parameter should no longer be all zeros
        loaded = list(agent.network.parameters())[0]
        assert not torch.all(loaded == 0.0)
