from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class Transition:
    embedding: torch.Tensor
    polygon_state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    next_polygon_state: torch.Tensor | None
    done: torch.Tensor


class ReplayBuffer:
    """Simple replay buffer that stores tensors on CPU for fast sampling."""

    def __init__(self, capacity: int, embedding_dim: int, polygon_dim: int, action_dim: int, device: torch.device | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.polygon_dim = polygon_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cpu")

        self.embeddings = torch.zeros((capacity, embedding_dim), dtype=torch.float32)
        self.polygons = torch.zeros((capacity, polygon_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.discounts = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_polygons = torch.zeros((capacity, polygon_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)

        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        idx = self._position

        self.embeddings[idx].copy_(transition.embedding.detach().to(dtype=torch.float32, device="cpu"))
        self.polygons[idx].copy_(transition.polygon_state.detach().to(dtype=torch.float32, device="cpu"))
        self.actions[idx].copy_(transition.action.detach().to(dtype=torch.float32, device="cpu"))
        self.rewards[idx].copy_(transition.reward.detach().view(1).to(dtype=torch.float32, device="cpu"))
        self.discounts[idx].copy_(transition.discount.detach().view(1).to(dtype=torch.float32, device="cpu"))

        if transition.next_polygon_state is None:
            self.next_polygons[idx].zero_()
        else:
            self.next_polygons[idx].copy_(transition.next_polygon_state.detach().to(dtype=torch.float32, device="cpu"))

        self.dones[idx].copy_(transition.done.detach().view(1).to(dtype=torch.float32, device="cpu"))

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | None = None) -> Dict[str, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        batch_size = min(batch_size, self._size)

        indices = torch.randint(0, self._size, (batch_size,))
        target_device = device or self.device

        batch = {
            "embedding": self.embeddings[indices].to(target_device),
            "polygon": self.polygons[indices].to(target_device),
            "action": self.actions[indices].to(target_device),
            "reward": self.rewards[indices].to(target_device),
            "discount": self.discounts[indices].to(target_device),
            "next_polygon": self.next_polygons[indices].to(target_device),
            "done": self.dones[indices].to(target_device),
        }
        return batch
