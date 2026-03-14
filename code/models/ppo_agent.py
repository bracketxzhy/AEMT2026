"""Actor-critic network for bounded continuous PPO control."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim * 2)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        hidden = self.backbone(observations)
        alpha_raw, beta_raw = self.actor_head(hidden).chunk(2, dim=-1)
        alpha = F.softplus(alpha_raw) + 1.05
        beta = F.softplus(beta_raw) + 1.05
        value = self.critic_head(hidden).squeeze(-1)
        return alpha, beta, value

    def policy(self, observations: torch.Tensor) -> tuple[torch.distributions.Beta, torch.Tensor]:
        alpha, beta, value = self.forward(observations)
        distribution = torch.distributions.Beta(alpha, beta)
        return distribution, value

    def sample_action(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.policy(observations)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.policy(observations)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, value

    def act_deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        alpha, beta, _ = self.forward(observations)
        return alpha / (alpha + beta)
