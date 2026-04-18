"""PPO policy and value neural network.

Architecture
------------
* **Feature extractor**: Three convolutional layers that encode the visual
  observation (the resized game frame in CHW float format).
* **LLM fusion**: The LLM one-hot guidance vector is concatenated with the
  flattened CNN output and passed through a fully-connected layer.
* **Policy head**: Outputs logits over the discrete action space.
* **Value head**: Outputs a scalar state-value estimate.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PPONetwork(nn.Module):
    """Combined actor-critic network for the PPO agent.

    Parameters
    ----------
    obs_shape:
        Shape of the visual observation ``(C, H, W)``.
    llm_feature_dim:
        Dimension of the LLM guidance one-hot vector.
    num_actions:
        Number of discrete actions available to the agent.
    hidden_dim:
        Number of units in the shared fully-connected layer.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        llm_feature_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        c, h, w = obs_shape

        # ---- Visual encoder ------------------------------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output dim by doing a dry-forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out_dim = self.cnn(dummy).shape[1]

        # ---- Fusion layer --------------------------------------------------
        fused_dim = cnn_out_dim + llm_feature_dim
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
        )

        # ---- Actor / critic heads ------------------------------------------
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,
        llm_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action_logits, state_value)``.

        Parameters
        ----------
        obs:
            Visual observation tensor of shape ``(B, C, H, W)`` with values
            in ``[0, 1]``.
        llm_features:
            LLM guidance one-hot tensor of shape ``(B, llm_feature_dim)``.

        Returns
        -------
        action_logits : torch.Tensor  (B, num_actions)
        state_value   : torch.Tensor  (B, 1)
        """
        visual = self.cnn(obs)
        fused = torch.cat([visual, llm_features], dim=1)
        hidden = self.fc(fused)
        return self.policy_head(hidden), self.value_head(hidden)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        llm_features: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) an action and return full PPO quantities.

        Returns
        -------
        action      : sampled (or provided) action index – shape (B,)
        log_prob    : log-probability of *action* under the current policy
        entropy     : distribution entropy
        value       : state-value estimate
        """
        logits, value = self.forward(obs, llm_features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Policy head – smaller scale for initial entropy
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        # Value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
