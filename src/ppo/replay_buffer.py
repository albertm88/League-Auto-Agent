"""Rollout buffer for PPO.

Stores agent transitions (obs, action, log_prob, reward, done, value,
llm_features) collected during environment interaction and provides a
``get_batches`` method that computes GAE-lambda advantages and yields
mini-batches for the PPO update.
"""

from __future__ import annotations

from typing import Generator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-size circular buffer for PPO rollout data.

    Parameters
    ----------
    capacity:
        Maximum number of transitions to store before the buffer is
        considered full and ready for a PPO update.
    obs_shape:
        Shape of a single visual observation ``(C, H, W)``.
    llm_feature_dim:
        Dimension of the LLM guidance vector.
    gamma:
        Discount factor γ for computing returns.
    gae_lambda:
        GAE-lambda (λ) for advantage estimation.
    device:
        Torch device where tensors will be placed during training.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, int, int],
        llm_feature_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device | str = "cpu",
    ) -> None:
        self._capacity = capacity
        self._obs_shape = obs_shape
        self._llm_dim = llm_feature_dim
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._device = torch.device(device)

        self._ptr = 0
        self._size = 0

        # Pre-allocate NumPy arrays for speed
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.llm_features = np.zeros((capacity, llm_feature_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        llm_features: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Add a single transition to the buffer."""
        i = self._ptr
        self.obs[i] = obs
        self.llm_features[i] = llm_features
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value

        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def is_full(self) -> bool:
        return self._size >= self._capacity

    def clear(self) -> None:
        self._ptr = 0
        self._size = 0

    # ------------------------------------------------------------------
    # GAE computation & batch generation
    # ------------------------------------------------------------------

    def compute_advantages(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE-lambda advantages and discounted returns.

        Parameters
        ----------
        last_value:
            Value estimate of the state *after* the last stored transition
            (bootstrap value for non-terminal episodes).

        Returns
        -------
        advantages : np.ndarray  (size,)
        returns    : np.ndarray  (size,)
        """
        n = self._size
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1]
            delta = (
                self.rewards[t]
                + self._gamma * next_value * (1.0 - self.dones[t])
                - self.values[t]
            )
            # Zero out the GAE continuation when the current step is terminal,
            # i.e. done[t] == 1 means the episode ended at step t.
            gae = delta + self._gamma * self._gae_lambda * (1.0 - self.dones[t]) * gae
            advantages[t] = gae

        returns = advantages + self.values[:n]
        return advantages, returns

    def get_batches(
        self,
        batch_size: int,
        last_value: float,
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """Yield shuffled mini-batches of PPO training data.

        Each yielded tuple contains tensors:
        ``(obs, llm_features, actions, old_log_probs, advantages, returns)``

        Parameters
        ----------
        batch_size:
            Size of each mini-batch.
        last_value:
            Bootstrap value for advantage computation.
        """
        advantages, returns = self.compute_advantages(last_value)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = self._size
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]

            yield (
                torch.as_tensor(self.obs[batch_idx], device=self._device),
                torch.as_tensor(self.llm_features[batch_idx], device=self._device),
                torch.as_tensor(self.actions[batch_idx], device=self._device),
                torch.as_tensor(self.log_probs[batch_idx], device=self._device),
                torch.as_tensor(advantages[batch_idx], device=self._device),
                torch.as_tensor(returns[batch_idx], device=self._device),
            )
