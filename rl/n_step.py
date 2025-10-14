from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import torch


@dataclass
class StepTuple:
    embedding: torch.Tensor
    polygon: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_polygon: torch.Tensor | None
    done: torch.Tensor


class NStepAccumulator:
    """Accumulates single-step transitions into n-step targets per environment."""

    def __init__(self, n_step: int, gamma: float, num_envs: int) -> None:
        if n_step <= 0:
            raise ValueError("n_step must be positive.")
        self.n_step = n_step
        self.gamma = gamma
        self.buffers: List[Deque[StepTuple]] = [deque() for _ in range(num_envs)]

    def push(self, env_idx: int, step: StepTuple) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]]:
        buffer = self.buffers[env_idx]
        buffer.append(step)
        transitions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]] = []

        def _pop_transition() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
            if not buffer:
                raise RuntimeError("Attempted to pop from an empty buffer.")
            reward_acc = torch.zeros_like(buffer[0].reward)
            device = buffer[0].reward.device
            discount = torch.tensor(1.0, dtype=buffer[0].reward.dtype, device=device)
            next_polygon = None
            done_flag = torch.zeros_like(buffer[0].done)

            steps = min(self.n_step, len(buffer))
            for i in range(steps):
                item = buffer[i]
                reward_acc = reward_acc + (self.gamma ** i) * item.reward
                if item.done.bool().item():
                    done_flag = item.done
                    next_polygon = item.next_polygon
                    discount = torch.tensor(0.0, dtype=buffer[0].reward.dtype, device=device)
                    steps = i + 1
                    break
            else:
                next_item = buffer[steps - 1]
                next_polygon = next_item.next_polygon
                discount = torch.tensor(self.gamma ** steps, dtype=buffer[0].reward.dtype, device=device)

            first = buffer.popleft()
            return first.embedding, first.polygon, first.action, reward_acc, next_polygon, done_flag, discount

        while buffer and (len(buffer) >= self.n_step or buffer[0].done.bool().item()):
            embedding, polygon, action, reward_acc, next_polygon, done_flag, discount = _pop_transition()
            transitions.append((embedding, polygon, action, reward_acc, next_polygon, done_flag, discount))

        if buffer and buffer[0].done.bool().item():
            while buffer:
                embedding, polygon, action, reward_acc, next_polygon, done_flag, discount = _pop_transition()
                transitions.append((embedding, polygon, action, reward_acc, next_polygon, done_flag, discount))

        return transitions

    def flush(self, env_idx: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]]:
        buffer = self.buffers[env_idx]
        transitions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]] = []
        while buffer:
            item = buffer.popleft()
            done_tensor = torch.ones_like(item.done)
            transitions.append(
                (
                    item.embedding,
                    item.polygon,
                    item.action,
                    item.reward,
                    item.next_polygon,
                    done_tensor,
                    torch.tensor(0.0, dtype=item.reward.dtype, device=item.reward.device),
                )
            )
        return transitions
