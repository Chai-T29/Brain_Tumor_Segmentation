from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


def _build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int, last_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if last_activation is not None:
        layers.append(last_activation)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic policy network for TD3."""

    def __init__(self, embedding_dim: int, polygon_dim: int, action_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        input_dim = embedding_dim + polygon_dim
        self.net = _build_mlp(input_dim, tuple(hidden_sizes), action_dim)

    def forward(self, embedding: torch.Tensor, polygon_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([embedding, polygon_state], dim=-1)
        return torch.tanh(self.net(x))


class Critic(nn.Module):
    """Double Q-network used by TD3.

    Both Q-functions share the same input but maintain independent parameters.
    """

    def __init__(self, embedding_dim: int, polygon_dim: int, action_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        input_dim = embedding_dim + polygon_dim + action_dim
        self.q1 = _build_mlp(input_dim, tuple(hidden_sizes), 1)
        self.q2 = _build_mlp(input_dim, tuple(hidden_sizes), 1)

    def forward(self, embedding: torch.Tensor, polygon_state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([embedding, polygon_state, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

    def q1_forward(self, embedding: torch.Tensor, polygon_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([embedding, polygon_state, action], dim=-1)
        return self.q1(x)
