from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class ReplayBatch:
    """Container returned when sampling from the replay memory."""

    state_images: torch.Tensor
    state_bboxes: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_state_images: torch.Tensor
    next_state_bboxes: torch.Tensor
    dones: torch.Tensor
    n_used: torch.Tensor


class PrioritizedReplayMemory:
    """GPU-friendly prioritized replay buffer with vectorised ingestion."""

    def __init__(self, capacity: int, alpha: float = 0.6, device: Optional[torch.device] = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self._initialized = False
        self._position = 0
        self._size = 0

        # Storage tensors will be created lazily on the first push.
        self._state_images: torch.Tensor
        self._state_bboxes: torch.Tensor
        self._actions: torch.Tensor
        self._rewards: torch.Tensor
        self._next_state_images: torch.Tensor
        self._next_state_bboxes: torch.Tensor
        self._dones: torch.Tensor
        self._n_used: torch.Tensor
        self._priorities: torch.Tensor = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def push_batch(
        self,
        state_images: torch.Tensor,
        state_bboxes: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_state_images: torch.Tensor,
        next_state_bboxes: torch.Tensor,
        dones: torch.Tensor,
        n_used: torch.Tensor,
        priorities: Optional[torch.Tensor] = None,
    ) -> None:
        """Insert a batch of transitions into the replay buffer."""

        if state_images.numel() == 0:
            return

        state_images = state_images.to(self.device)
        state_bboxes = state_bboxes.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_state_images = next_state_images.to(self.device)
        next_state_bboxes = next_state_bboxes.to(self.device)
        dones = dones.to(self.device)
        n_used = n_used.to(self.device)

        if not self._initialized:
            self._initialise_storage(
                state_images,
                state_bboxes,
                actions,
                rewards,
                next_state_images,
                next_state_bboxes,
                dones,
                n_used,
            )

        batch_size = state_images.size(0)
        indices = (torch.arange(batch_size, device=self.device) + self._position) % self.capacity

        self._state_images[indices] = state_images
        self._state_bboxes[indices] = state_bboxes
        self._actions[indices] = actions.view(batch_size, -1)
        self._rewards[indices] = rewards.view(batch_size, 1)
        self._next_state_images[indices] = next_state_images
        self._next_state_bboxes[indices] = next_state_bboxes
        self._dones[indices] = dones.view(batch_size, 1)
        self._n_used[indices] = n_used.view(batch_size, 1)

        if priorities is None:
            if self._size == 0:
                max_prio = torch.tensor(1.0, device=self.device)
            else:
                max_prio = torch.max(self._priorities[: self._size])
                if max_prio.item() <= 0:
                    max_prio = torch.tensor(1.0, device=self.device)
            priority_values = max_prio.expand(batch_size)
        else:
            priority_values = priorities.to(self.device).view(-1)

        self._priorities[indices] = torch.clamp(priority_values, min=1e-6)

        self._position = int((self._position + batch_size) % self.capacity)
        self._size = int(min(self._size + batch_size, self.capacity))

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[ReplayBatch, torch.Tensor, torch.Tensor]:
        """Sample a batch of experiences according to their priorities."""

        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        effective_size = self._size
        priorities = self._priorities[:effective_size].clamp(min=1e-6)
        scaled = priorities.pow(self.alpha)
        prob_sum = scaled.sum()
        if prob_sum.item() <= 0:
            probabilities = torch.ones_like(scaled) / float(effective_size)
        else:
            probabilities = scaled / prob_sum

        replacement = effective_size < batch_size
        indices = torch.multinomial(probabilities, batch_size, replacement=replacement)
        weights = (effective_size * probabilities[indices]).pow(-beta)
        weights = weights / weights.max()

        batch = ReplayBatch(
            state_images=self._state_images[indices],
            state_bboxes=self._state_bboxes[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_state_images=self._next_state_images[indices],
            next_state_bboxes=self._next_state_bboxes[indices],
            dones=self._dones[indices],
            n_used=self._n_used[indices],
        )
        return batch, indices, weights

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        if priorities.numel() == 0:
            return
        indices = indices.to(self.device).long().view(-1)
        updated = torch.clamp(priorities.to(self.device).view(-1), min=1e-6)
        self._priorities[indices] = updated

    def to(self, device: torch.device) -> None:
        device = torch.device(device)
        if self.device == device:
            return
        self.device = device
        self._priorities = self._priorities.to(device)
        if self._initialized:
            self._state_images = self._state_images.to(device)
            self._state_bboxes = self._state_bboxes.to(device)
            self._actions = self._actions.to(device)
            self._rewards = self._rewards.to(device)
            self._next_state_images = self._next_state_images.to(device)
            self._next_state_bboxes = self._next_state_bboxes.to(device)
            self._dones = self._dones.to(device)
            self._n_used = self._n_used.to(device)

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialise_storage(
        self,
        state_images: torch.Tensor,
        state_bboxes: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_state_images: torch.Tensor,
        next_state_bboxes: torch.Tensor,
        dones: torch.Tensor,
        n_used: torch.Tensor,
    ) -> None:
        batch_shape = (self.capacity,)
        self._state_images = torch.zeros(batch_shape + state_images.shape[1:], dtype=state_images.dtype, device=self.device)
        self._state_bboxes = torch.zeros(batch_shape + state_bboxes.shape[1:], dtype=state_bboxes.dtype, device=self.device)
        self._actions = torch.zeros(batch_shape + actions.view(actions.size(0), -1).shape[1:], dtype=actions.dtype, device=self.device)
        self._rewards = torch.zeros(batch_shape + rewards.view(rewards.size(0), -1).shape[1:], dtype=rewards.dtype, device=self.device)
        self._next_state_images = torch.zeros(batch_shape + next_state_images.shape[1:], dtype=next_state_images.dtype, device=self.device)
        self._next_state_bboxes = torch.zeros(batch_shape + next_state_bboxes.shape[1:], dtype=next_state_bboxes.dtype, device=self.device)
        self._dones = torch.zeros(batch_shape + dones.view(dones.size(0), -1).shape[1:], dtype=dones.dtype, device=self.device)
        self._n_used = torch.zeros(batch_shape + n_used.view(n_used.size(0), -1).shape[1:], dtype=n_used.dtype, device=self.device)
        self._initialized = True
