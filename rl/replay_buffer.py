from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import numpy as np


@dataclass
class Transition:
    embedding: torch.Tensor | Sequence[float] | Dict[str, float] | None
    polygon_state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    next_polygon_state: torch.Tensor | None
    done: torch.Tensor
    guided_target: torch.Tensor | None = None


class ReplayBuffer:
    """Prioritised replay buffer that stores embedding maps and polygon states on CPU."""

    def __init__(
        self,
        capacity: int,
        embedding_shape: tuple[int, int, int],
        polygon_dim: int,
        action_dim: int,
        alpha: float,
        beta_start: float,
        beta_steps: int,
        eps: float,
        device: torch.device | None = None,
        use_embedding_pointers: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if len(embedding_shape) != 3:
            raise ValueError("embedding_shape must be a tuple of (C, H, W).")

        self.capacity = capacity
        self.embedding_shape = tuple(int(v) for v in embedding_shape)
        self.polygon_dim = polygon_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cpu")
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_steps = max(1, int(beta_steps))
        self.beta_increment = (1.0 - self.beta_start) / self.beta_steps
        self.beta = self.beta_start
        self.pr_eps = float(eps)

        self.use_embedding_pointers = bool(use_embedding_pointers)
        if self.use_embedding_pointers:
            self.embeddings = [None] * capacity  # type: ignore[assignment]
        else:
            self.embeddings = torch.zeros((capacity, *self.embedding_shape), dtype=torch.float32)
        self.polygons = torch.zeros((capacity, polygon_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.discounts = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_polygons = torch.zeros((capacity, polygon_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        self.priorities = torch.zeros(capacity, dtype=torch.float32)
        self.guided_targets = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.guided_available = torch.zeros(capacity, dtype=torch.float32)

        self._position = 0
        self._size = 0
        self._max_priority = 1.0
        # Simple memmap cache to avoid repeated np.load calls in pointer mode
        self._mm_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        idx = self._position

        if self.use_embedding_pointers:
            self.embeddings[idx] = transition.embedding
        else:
            if transition.embedding is None:
                raise ValueError("Embedding tensor required when use_embedding_pointers is False.")
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

        if transition.guided_target is None:
            self.guided_targets[idx].zero_()
            self.guided_available[idx] = 0.0
        else:
            self.guided_targets[idx].copy_(transition.guided_target.detach().to(dtype=torch.float32, device="cpu"))
            self.guided_available[idx] = 1.0

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

        if self.use_embedding_pointers:
            emb_list = []
            for ptr in (self.embeddings[i] for i in indices):
                if isinstance(ptr, torch.Tensor):
                    emb_list.append(ptr.to(target_device, dtype=torch.float32))
                elif isinstance(ptr, np.ndarray):
                    # Make a writable copy to avoid PyTorch warning on non-writable views
                    np_copy = np.array(ptr, dtype=np.float32, copy=True)
                    emb_list.append(torch.from_numpy(np_copy).to(target_device))
                elif isinstance(ptr, dict):
                    path = ptr.get("path")
                    slice_idx = int(ptr.get("slice_index", 0))
                    if path is None:
                        raise ValueError("Embedding pointer missing 'path'.")
                    arr = self._get_memmap(str(path))
                    # Create a writable copy from the memmap slice
                    np_copy = np.array(arr[slice_idx], dtype=np.float32, copy=True)
                    emb = torch.from_numpy(np_copy).to(target_device)
                    emb_list.append(emb)
                elif isinstance(ptr, (tuple, list)):
                    if len(ptr) < 2:
                        raise ValueError("Embedding pointer sequence must contain (path, slice_index).")
                    path, slice_idx = ptr[0], int(ptr[1])
                    arr = self._get_memmap(str(path))
                    np_copy = np.array(arr[slice_idx], dtype=np.float32, copy=True)
                    emb = torch.from_numpy(np_copy).to(target_device)
                    emb_list.append(emb)
                else:
                    raise ValueError("Unsupported embedding pointer type.")
            embedding_batch = torch.stack(emb_list, dim=0)
        else:
            embedding_batch = self.embeddings[indices].to(target_device)

        batch = {
            "embedding": embedding_batch,
            "polygon": self.polygons[indices].to(target_device),
            "action": self.actions[indices].to(target_device),
            "reward": self.rewards[indices].to(target_device),
            "discount": self.discounts[indices].to(target_device),
            "next_polygon": self.next_polygons[indices].to(target_device),
            "done": self.dones[indices].to(target_device),
            "guided_target": self.guided_targets[indices].to(target_device),
            "guided_available": self.guided_available[indices].to(target_device).view(-1, 1),
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

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _get_memmap(self, path: str) -> np.ndarray:
        arr = self._mm_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            self._mm_cache[path] = arr
        return arr
