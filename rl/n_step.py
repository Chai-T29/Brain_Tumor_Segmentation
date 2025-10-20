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
    guided_target: torch.Tensor | None = None


class NStepAccumulator:
    """Accumulates single-step transitions into n-step targets per environment."""

    def __init__(self, n_step: int, gamma: float, num_envs: int) -> None:
        if n_step <= 0:
            raise ValueError("n_step must be positive.")
        self.n_step = n_step
        self.gamma = gamma
        self.buffers: List[Deque[StepTuple]] = [deque() for _ in range(num_envs)]

    def push(
        self,
        env_idx: int,
        step: StepTuple,
    ) -> List[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
        ]
    ]:
        buffer = self.buffers[env_idx]
        buffer.append(step)
        transitions: List[
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
            ]
        ] = []

        while buffer and (len(buffer) >= self.n_step or buffer[0].done.bool().item()):
            (
                embedding,
                polygon,
                action,
                reward_acc,
                next_polygon,
                done_flag,
                discount,
                guidance,
            ) = self._pop_transition(buffer, allow_partial=False)
            transitions.append((embedding, polygon, action, reward_acc, next_polygon, done_flag, discount, guidance))

        return transitions

    def flush(
        self,
        env_idx: int,
    ) -> List[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
        ]
    ]:
        buffer = self.buffers[env_idx]
        transitions: List[
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
            ]
        ] = []
        while buffer:
            (
                embedding,
                polygon,
                action,
                reward_acc,
                next_polygon,
                done_flag,
                discount,
                guidance,
            ) = self._pop_transition(buffer, allow_partial=True)
            transitions.append((embedding, polygon, action, reward_acc, next_polygon, done_flag, discount, guidance))
        return transitions

    def _pop_transition(
        self,
        buffer: Deque[StepTuple],
        allow_partial: bool,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        if not buffer:
            raise RuntimeError("Attempted to pop from an empty buffer.")

        reward_acc = torch.zeros_like(buffer[0].reward)
        device = buffer[0].reward.device
        dtype = buffer[0].reward.dtype
        discount = torch.tensor(1.0, dtype=dtype, device=device)
        next_polygon = None
        done_flag = torch.zeros_like(buffer[0].done)

        max_horizon = min(self.n_step, len(buffer))
        for i in range(max_horizon):
            item = buffer[i]
            reward_acc = reward_acc + (self.gamma ** i) * item.reward
            if item.done.bool().item():
                done_flag = item.done
                next_polygon = item.next_polygon
                discount = torch.tensor(0.0, dtype=dtype, device=device)
                max_horizon = i + 1
                break
        else:
            tail_item = buffer[max_horizon - 1]
            next_polygon = tail_item.next_polygon
            if next_polygon is not None:
                discount = torch.tensor(self.gamma ** max_horizon, dtype=dtype, device=device)
            else:
                # No next state available, treat as terminal.
                discount = torch.tensor(0.0, dtype=dtype, device=device)

        if not allow_partial and len(buffer) < self.n_step and not done_flag.bool().item():
            raise RuntimeError("Insufficient steps to pop transition without partial allowance.")

        first = buffer.popleft()
        return (
            first.embedding,
            first.polygon,
            first.action,
            reward_acc,
            next_polygon,
            done_flag,
            discount,
            first.guided_target,
        )
