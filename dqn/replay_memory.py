from collections import namedtuple
from typing import Iterable, List, Sequence

import numpy as np

import torch


class PrioritizedSample(list):
    """List-like container that also stores sampling metadata."""

    def __init__(self, experiences, indices, weights):
        super().__init__(experiences)
        self.indices = indices
        self.weights = weights

Experience = namedtuple('Experience', ('state','action','reward','next_state','done','n_used'))

class PrioritizedReplayMemory:
    """Prioritized replay buffer implementation with proportional prioritisation."""

    def __init__(self, capacity: int, alpha: float = 0.6, device: torch.device | None = None) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cpu")
        self.memory: List[Experience] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def _to_fp16_copy(self, t):
        if isinstance(t, torch.Tensor):
            # Keep on configured device; make independent storage; cast to fp16 for memory efficiency
            return t.detach().to(device=self.device, dtype=torch.float16).contiguous().clone()
        return t

    def _exp_to_float32_device(self, exp: Experience, device: torch.device) -> Experience:
        dev = device
        s_img, s_bbox = exp.state
        if isinstance(s_img, torch.Tensor):
            s_img = s_img.to(device=dev, dtype=torch.float32)
        if isinstance(s_bbox, torch.Tensor):
            s_bbox = s_bbox.to(device=dev, dtype=torch.float32)
        action = exp.action.to(device=dev) if isinstance(exp.action, torch.Tensor) else exp.action
        reward = (exp.reward.to(device=dev, dtype=torch.float32)
                  if (isinstance(exp.reward, torch.Tensor) and exp.reward.is_floating_point()) else exp.reward)
        if exp.next_state is not None:
            ns_img, ns_bbox = exp.next_state
            if isinstance(ns_img, torch.Tensor):
                ns_img = ns_img.to(device=dev, dtype=torch.float32)
            if isinstance(ns_bbox, torch.Tensor):
                ns_bbox = ns_bbox.to(device=dev, dtype=torch.float32)
            next_state = (ns_img, ns_bbox)
        else:
            next_state = None
        return Experience((s_img, s_bbox), action, reward, next_state, exp.done, exp.n_used)

    def push(self, *args, priority: float | None = None) -> None:
        """Save an experience with an optional priority value, storing large tensors as fp16 on the configured device."""
        state, action, reward, next_state, done, n_used = args
        # Convert tensors to fp16 on the configured device and create independent storage
        s_img, s_bbox = state
        s_img  = self._to_fp16_copy(s_img)
        s_bbox = self._to_fp16_copy(s_bbox)
        if next_state is not None:
            ns_img, ns_bbox = next_state
            ns_img  = self._to_fp16_copy(ns_img)
            ns_bbox = self._to_fp16_copy(ns_bbox)
            next_state = (ns_img, ns_bbox)
        # Rewards can be big tensors or scalars; keep scalar tensor and cast if float
        if isinstance(reward, torch.Tensor) and reward.is_floating_point():
            reward = reward.detach().to(device=self.device, dtype=torch.float16).contiguous().clone()
        # Actions are typically integer tensors; keep dtype as is and move to device
        if isinstance(action, torch.Tensor):
            action = action.detach().to(device=self.device).contiguous().clone()

        experience = Experience((s_img, s_bbox), action, reward, next_state, done, n_used)

        max_prio = self.priorities.max() if self.memory else 1.0
        priority_value = float(priority if priority is not None else max_prio)
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        self.priorities[self.position] = priority_value
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4, device: torch.device | None = None):
        """Sample a batch of experiences according to priorities. Returns float32 tensors on `device`."""
        if len(self.memory) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        priorities = self.priorities if len(self.memory) == self.capacity else self.priorities[: len(self.memory)]
        scaled_priorities = priorities ** self.alpha
        prob_sum = scaled_priorities.sum()
        probabilities = (np.ones_like(scaled_priorities) / len(scaled_priorities)) if prob_sum <= 0 else (scaled_priorities / prob_sum)

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        target_device = device if device is not None else self.device
        weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=target_device)
        # Convert stored fp16 experiences back to float32 on the requested device
        experiences = [self._exp_to_float32_device(self.memory[idx], target_device) for idx in indices]
        return PrioritizedSample(experiences, indices, weights_tensor)

    def update_priorities(self, indices: Sequence[int], priorities: Iterable[float]) -> None:
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < len(self.memory):
                self.priorities[idx] = float(max(prio, 1e-6))

    def __len__(self) -> int:
        return len(self.memory)


def torch_from_numpy(array: np.ndarray, device: torch.device | None = None):
    return torch.as_tensor(array, dtype=torch.float32, device=device or torch.device("cpu"))


# import random
# from collections import namedtuple, deque

# Experience = namedtuple('Experience',
#                         ('state', 'action', 'reward', 'next_state', 'done'))

# class ReplayMemory:
#     """A cyclic buffer of bounded size that holds the experiences observed recently."""

#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = deque([], maxlen=capacity)

#     def push(self, *args):
#         """Save an experience."""
#         self.memory.append(Experience(*args))

#     def sample(self, batch_size):
#         """Randomly sample a batch of experiences from memory."""
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)
