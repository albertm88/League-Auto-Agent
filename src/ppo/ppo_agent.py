"""PPO agent: ties together the network, rollout buffer, and optimiser.

This module implements the Proximal Policy Optimisation (PPO-clip) algorithm
as described in Schulman et al. (2017).  The agent:

1. Collects experience using :meth:`PPOAgent.select_action`.
2. Stores each transition via :meth:`PPOAgent.store_transition`.
3. Runs a PPO update via :meth:`PPOAgent.update` when the buffer is full.
4. Optionally saves / loads checkpoints.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.ppo.network import PPONetwork
from src.ppo.replay_buffer import RolloutBuffer


class PPOAgent:
    """PPO agent with a fused visual + LLM guidance policy.

    Parameters
    ----------
    cfg:
        The ``ppo`` section of the YAML configuration dictionary.
    device:
        Torch device to use (``"cpu"`` or ``"cuda"``).
    logger:
        Optional logger instance.
    """

    def __init__(
        self,
        cfg: Dict,
        device: str = "cpu",
        logger=None,
    ) -> None:
        self._cfg = cfg
        self._logger = logger
        self._device = torch.device(device)
        self._update_count = 0

        obs_shape: Tuple[int, int, int] = tuple(cfg["obs_shape"])  # type: ignore[assignment]
        llm_dim: int = cfg.get("llm_feature_dim", 8)
        num_actions: int = cfg.get("num_actions", 20)
        hidden_dim: int = cfg.get("hidden_dim", 256)

        self.network = PPONetwork(
            obs_shape=obs_shape,
            llm_feature_dim=llm_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
        ).to(self._device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=float(cfg.get("lr", 3e-4)),
            eps=1e-5,
        )

        self.buffer = RolloutBuffer(
            capacity=cfg.get("buffer_capacity", 2048),
            obs_shape=obs_shape,
            llm_feature_dim=llm_dim,
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            device=device,
        )

    # ------------------------------------------------------------------
    # Action selection (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        llm_features: np.ndarray,
    ) -> Tuple[int, float, float]:
        """Sample an action from the current policy.

        Parameters
        ----------
        obs:
            Visual observation ``(C, H, W)`` as a float32 array in [0, 1].
        llm_features:
            LLM guidance one-hot vector of shape ``(llm_feature_dim,)``.

        Returns
        -------
        action   : int
        log_prob : float
        value    : float
        """
        obs_t = torch.as_tensor(obs[np.newaxis], device=self._device)
        llm_t = torch.as_tensor(llm_features[np.newaxis], device=self._device)

        action_t, log_prob_t, _, value_t = self.network.get_action_and_value(
            obs_t, llm_t
        )
        return (
            int(action_t.item()),
            float(log_prob_t.item()),
            float(value_t.item()),
        )

    # ------------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------------

    def store_transition(
        self,
        obs: np.ndarray,
        llm_features: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Store a single transition in the rollout buffer."""
        self.buffer.add(
            obs=obs,
            llm_features=llm_features,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
        )

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Run one round of PPO updates using all stored transitions.

        Parameters
        ----------
        last_value:
            Bootstrap value for the state following the last stored
            transition (0 if the episode ended).

        Returns
        -------
        dict with keys ``policy_loss``, ``value_loss``, ``entropy``.
        """
        clip_epsilon: float = self._cfg.get("clip_epsilon", 0.2)
        value_coef: float = self._cfg.get("value_coef", 0.5)
        entropy_coef: float = self._cfg.get("entropy_coef", 0.01)
        max_grad_norm: float = self._cfg.get("max_grad_norm", 0.5)
        epochs: int = self._cfg.get("update_epochs", 4)
        batch_size: int = self._cfg.get("batch_size", 64)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for _ in range(epochs):
            for batch in self.buffer.get_batches(batch_size, last_value):
                (
                    obs_b,
                    llm_b,
                    actions_b,
                    old_log_probs_b,
                    advantages_b,
                    returns_b,
                ) = batch

                # Normalise observations to [0, 1]
                obs_b = obs_b.float() / 255.0 if obs_b.max() > 1.0 else obs_b.float()

                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    obs_b, llm_b, actions_b
                )

                # Policy loss (PPO-clip)
                ratio = torch.exp(new_log_probs - old_log_probs_b)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                policy_loss = -torch.min(
                    ratio * advantages_b,
                    clipped_ratio * advantages_b,
                ).mean()

                # Value loss
                values = values.squeeze(-1)
                value_loss = nn.functional.mse_loss(values, returns_b)

                # Total loss
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        self.buffer.clear()
        self._update_count += 1

        # Periodic checkpointing
        checkpoint_interval = self._cfg.get("checkpoint_interval", 100)
        if self._update_count % checkpoint_interval == 0:
            self.save_checkpoint()

        stats = {
            "policy_loss": total_policy_loss / max(num_batches, 1),
            "value_loss": total_value_loss / max(num_batches, 1),
            "entropy": total_entropy / max(num_batches, 1),
        }

        if self._logger:
            self._logger.info(
                "PPO update #%d | policy_loss=%.4f | value_loss=%.4f | entropy=%.4f",
                self._update_count,
                stats["policy_loss"],
                stats["value_loss"],
                stats["entropy"],
            )

        return stats

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save network weights and optimiser state to *path*.

        Returns the path where the checkpoint was saved.
        """
        if path is None:
            ckpt_dir = self._cfg.get("checkpoint_dir", "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, f"ppo_update_{self._update_count:06d}.pt")

        torch.save(
            {
                "update_count": self._update_count,
                "network_state": self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )
        if self._logger:
            self._logger.info("Checkpoint saved to '%s'.", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Restore network weights and optimiser state from a checkpoint."""
        checkpoint = torch.load(path, map_location=self._device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._update_count = checkpoint.get("update_count", 0)
        if self._logger:
            self._logger.info(
                "Checkpoint loaded from '%s' (update %d).",
                path,
                self._update_count,
            )
