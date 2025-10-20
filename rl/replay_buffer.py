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
    guidance_target: torch.Tensor | None = None


class ReplayBuffer:
    """Prioritised replay buffer that stores tensors on CPU for fast sampling."""

    def __init__(
        self,
        capacity: int,
        embedding_dim: int,
        polygon_dim: int,
        action_dim: int,
        alpha: float,
        beta_start: float,
        beta_steps: int,
        eps: float,
        device: torch.device | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.polygon_dim = polygon_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cpu")
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_steps = max(1, int(beta_steps))
        self.beta_increment = (1.0 - self.beta_start) / self.beta_steps
        self.beta = self.beta_start
        self.pr_eps = float(eps)

        self.embeddings = torch.zeros((capacity, embedding_dim), dtype=torch.float32)
        self.polygons = torch.zeros((capacity, polygon_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.discounts = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_polygons = torch.zeros((capacity, polygon_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        self.guidance_targets = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.guidance_mask = torch.zeros(capacity, dtype=torch.bool)
        self.priorities = torch.zeros(capacity, dtype=torch.float32)

        self._position = 0
        self._size = 0
        self._max_priority = 1.0

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
        if transition.guidance_target is None:
            self.guidance_targets[idx].zero_()
            self.guidance_mask[idx] = False
        else:
            target = transition.guidance_target.detach().to(dtype=torch.float32, device="cpu")
            if target.dim() > 1:
                target = target.view(-1)
            self.guidance_targets[idx].copy_(target)
            self.guidance_mask[idx] = True

        reward_abs = float(torch.abs(transition.reward.detach()).item())
        priority = max(self.pr_eps, reward_abs + self.pr_eps)
        self.priorities[idx] = priority
        self._max_priority = max(self._max_priority, priority)

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _sample_priorities(self, batch_size: int) -> torch.Tensor:
        priorities = self.priorities[: self._size].clamp_min(self.pr_eps)
        probs = priorities.pow(self.alpha)
        total = probs.sum()
        if not torch.isfinite(total) or total <= 0:
            probs.fill_(1.0 / float(self._size))
        else:
            probs /= total
        return probs

    def sample(self, batch_size: int, device: torch.device | None = None) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        batch_size = min(batch_size, self._size)

        probs = self._sample_priorities(batch_size)
        replacement = self._size < batch_size
        indices = torch.multinomial(probs, batch_size, replacement=replacement)
        target_device = device or self.device
        probs_indices = probs[indices]

        weights = (self._size * probs_indices).pow(-self.beta)
        weights /= weights.max().clamp(min=1e-6)
        weights = weights.to(dtype=torch.float32, device=target_device)
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = {
            "embedding": self.embeddings[indices].to(target_device),
            "polygon": self.polygons[indices].to(target_device),
            "action": self.actions[indices].to(target_device),
            "reward": self.rewards[indices].to(target_device),
            "discount": self.discounts[indices].to(target_device),
            "next_polygon": self.next_polygons[indices].to(target_device),
            "done": self.dones[indices].to(target_device),
            "guidance_target": self.guidance_targets[indices].to(target_device),
            "guidance_mask": self.guidance_mask[indices].to(target_device),
        }
        return batch, indices.to(device="cpu"), weights

    def update_priorities(self, indices: torch.Tensor, new_priorities: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        idx = indices.to(dtype=torch.long, device="cpu")
        priorities = torch.abs(new_priorities.detach().to(dtype=torch.float32, device="cpu")) + self.pr_eps
        self.priorities[idx] = priorities
        current_max = float(self.priorities[: self._size].max().item())
        if current_max > 0:
            self._max_priority = current_max
