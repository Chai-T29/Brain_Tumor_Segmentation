from __future__ import annotations

from typing import Iterable, Sequence, Tuple

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


class EmbeddingEncoder(nn.Module):
    """Lightweight convolutional encoder that reduces [B, C, H, W] to a vector."""

    def __init__(self, embedding_shape: Tuple[int, int, int], projected_dim: int) -> None:
        super().__init__()
        channels, height, width = embedding_shape
        if height != width:
            raise ValueError("Embedding map must be square to reduce with strided convolutions.")
        if height < 1:
            raise ValueError("Embedding map must have positive spatial dimensions.")

        projected_dim = int(projected_dim)
        layers: list[nn.Module] = []
        in_channels = int(channels)
        size = int(height)

        i = 0
        while size > 1:
            kernel_size = 3 if size >= 3 else size
            factor = max(1, 3 - i)
            conv = nn.Conv2d(in_channels, factor * projected_dim, kernel_size=kernel_size, stride=1, padding=0, bias=True)
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            layers.append(conv)
            layers.append(nn.ReLU(inplace=True))

            size = size - (kernel_size - 1)
            if size <= 0:
                raise ValueError("Convolution stack collapsed spatial dimensions below 1. Check input shape.")
            layers.append(nn.LayerNorm((factor * projected_dim, size, size)))
            in_channels = factor * projected_dim
            i += 1

        self.net = nn.Sequential(*layers)
        self._flatten_dim = in_channels * max(1, size) * max(1, size)
        if self._flatten_dim != projected_dim:
            self.projector = nn.Linear(self._flatten_dim, projected_dim, bias=False)
            nn.init.kaiming_normal_(self.projector.weight, nonlinearity="relu")
        else:
            self.projector = nn.Identity()
        self.norm = nn.LayerNorm(projected_dim, elementwise_affine=False)

    def forward(self, embedding_map: torch.Tensor) -> torch.Tensor:
        if embedding_map.dim() != 4:
            raise ValueError("Expected embedding map with shape [B, C, H, W].")
        projected = self.net(embedding_map)
        flattened = projected.flatten(start_dim=1)
        flattened = self.projector(flattened)
        return self.norm(flattened)


class Actor(nn.Module):
    """Deterministic policy network with its own embedding encoder."""

    def __init__(
        self,
        embedding_shape: Tuple[int, int, int],
        polygon_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int],
        projected_dim: int,
        embedding_noise_std: float,
    ) -> None:
        super().__init__()
        self.encoder = EmbeddingEncoder(embedding_shape, projected_dim)
        input_dim = projected_dim + polygon_dim
        self.net = _build_mlp(input_dim, tuple(hidden_sizes), action_dim)
        self.embedding_noise_std = float(max(0.0, embedding_noise_std))

    def encode(self, embedding_map: torch.Tensor, apply_noise: bool = True) -> torch.Tensor:
        vec = self.encoder(embedding_map)
        if apply_noise and self.embedding_noise_std > 0.0:
            noise = torch.randn_like(vec) * self.embedding_noise_std
            vec = vec + noise
        return vec

    def forward_from_encoded(self, embedding_vec: torch.Tensor, polygon_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([embedding_vec, polygon_state], dim=-1)
        return torch.tanh(self.net(x))

    def forward(self, embedding_map: torch.Tensor, polygon_state: torch.Tensor, apply_noise: bool = True) -> torch.Tensor:
        encoded = self.encode(embedding_map, apply_noise=apply_noise)
        return self.forward_from_encoded(encoded, polygon_state)


class Critic(nn.Module):
    """Double Q-network with dedicated embedding encoders for each critic head."""

    def __init__(
        self,
        embedding_shape: Tuple[int, int, int],
        polygon_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int],
        projected_dim: int,
        embedding_noise_std: float,
    ) -> None:
        super().__init__()
        self.encoder = EmbeddingEncoder(embedding_shape, projected_dim)
        input_dim = projected_dim + polygon_dim + action_dim
        self.q1 = _build_mlp(input_dim, tuple(hidden_sizes), 1)
        self.q2 = _build_mlp(input_dim, tuple(hidden_sizes), 1)
        self.embedding_noise_std = float(max(0.0, embedding_noise_std))

    def _apply_noise(self, embedding_vec: torch.Tensor, apply_noise: bool) -> torch.Tensor:
        if apply_noise and self.embedding_noise_std > 0.0:
            noise = torch.randn_like(embedding_vec) * self.embedding_noise_std
            return embedding_vec + noise
        return embedding_vec

    def encode(self, embedding_map: torch.Tensor, apply_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.encoder(embedding_map)
        emb1 = self._apply_noise(base, apply_noise)
        emb2 = self._apply_noise(base, apply_noise)
        return emb1, emb2

    def encode_q1(self, embedding_map: torch.Tensor, apply_noise: bool = True) -> torch.Tensor:
        base = self.encoder(embedding_map)
        return self._apply_noise(base, apply_noise)

    def forward_from_encoded(
        self,
        encoded_q1: torch.Tensor,
        encoded_q2: torch.Tensor,
        polygon_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = torch.cat([encoded_q1, polygon_state, action], dim=-1)
        x2 = torch.cat([encoded_q2, polygon_state, action], dim=-1)
        q1 = self.q1(x1)
        q2 = self.q2(x2)
        return q1, q2

    def forward(self, embedding_map: torch.Tensor, polygon_state: torch.Tensor, action: torch.Tensor, apply_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        enc1, enc2 = self.encode(embedding_map, apply_noise=apply_noise)
        return self.forward_from_encoded(enc1, enc2, polygon_state, action)

    def q1_forward_from_encoded(self, encoded_q1: torch.Tensor, polygon_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([encoded_q1, polygon_state, action], dim=-1)
        return self.q1(x)
