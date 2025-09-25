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

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.memory: List[Experience] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, *args, priority: float | None = None) -> None:
        """Save an experience with an optional priority value."""
        # Convert float tensors to float16 before storing in memory
        # The Experience tuple is: ('state','action','reward','next_state','done','n_used')
        # state: tuple (state_image, state_bbox)
        # action: tensor
        # reward: tensor
        # next_state: tuple or None
        # done: bool
        # n_used: int
        state, action, reward, next_state, done, n_used = args
        # Convert state images and bboxes to float16
        s_img, s_bbox = state
        s_img = torch.as_tensor(s_img, dtype=torch.float16) if isinstance(s_img, torch.Tensor) else s_img
        s_bbox = torch.as_tensor(s_bbox, dtype=torch.float16) if isinstance(s_bbox, torch.Tensor) else s_bbox
        # Convert action and reward to float16 if float tensor
        action = torch.as_tensor(action, dtype=torch.float16) if (isinstance(action, torch.Tensor) and action.is_floating_point()) else action
        reward = torch.as_tensor(reward, dtype=torch.float16) if (isinstance(reward, torch.Tensor) and reward.is_floating_point()) else reward
        # Convert next_state images and bboxes to float16 if not None
        if next_state is not None:
            ns_img, ns_bbox = next_state
            ns_img = torch.as_tensor(ns_img, dtype=torch.float16) if isinstance(ns_img, torch.Tensor) else ns_img
            ns_bbox = torch.as_tensor(ns_bbox, dtype=torch.float16) if isinstance(ns_bbox, torch.Tensor) else ns_bbox
            next_state = (ns_img, ns_bbox)
        experience = Experience((s_img, s_bbox), action, reward, next_state, done, n_used)
        max_prio = self.priorities.max() if self.memory else 1.0
        priority_value = priority if priority is not None else max_prio
        priority_value = float(priority_value)
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience

        self.priorities[self.position] = priority_value
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample a batch of experiences according to their priorities."""
        if len(self.memory) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: len(self.memory)]

        scaled_priorities = priorities ** self.alpha
        prob_sum = scaled_priorities.sum()
        if prob_sum <= 0:
            probabilities = np.ones_like(scaled_priorities) / len(scaled_priorities)
        else:
            probabilities = scaled_priorities / prob_sum

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        experiences = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        weights_tensor = torch_from_numpy(weights)
        # When sampling, convert float16 tensors back to float32 for training
        def convert_exp_float32(exp):
            # exp: Experience tuple
            s_img, s_bbox = exp.state
            s_img = s_img.to(dtype=torch.float32) if isinstance(s_img, torch.Tensor) and s_img.dtype == torch.float16 else s_img
            s_bbox = s_bbox.to(dtype=torch.float32) if isinstance(s_bbox, torch.Tensor) and s_bbox.dtype == torch.float16 else s_bbox
            action = exp.action.to(dtype=torch.float32) if (isinstance(exp.action, torch.Tensor) and exp.action.dtype == torch.float16) else exp.action
            reward = exp.reward.to(dtype=torch.float32) if (isinstance(exp.reward, torch.Tensor) and exp.reward.dtype == torch.float16) else exp.reward
            if exp.next_state is not None:
                ns_img, ns_bbox = exp.next_state
                ns_img = ns_img.to(dtype=torch.float32) if isinstance(ns_img, torch.Tensor) and ns_img.dtype == torch.float16 else ns_img
                ns_bbox = ns_bbox.to(dtype=torch.float32) if isinstance(ns_bbox, torch.Tensor) and ns_bbox.dtype == torch.float16 else ns_bbox
                next_state = (ns_img, ns_bbox)
            else:
                next_state = None
            return Experience((s_img, s_bbox), action, reward, next_state, exp.done, exp.n_used)

        experiences = [convert_exp_float32(exp) for exp in experiences]
        return PrioritizedSample(experiences, indices, weights_tensor)

    def update_priorities(self, indices: Sequence[int], priorities: Iterable[float]) -> None:
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < len(self.memory):
                self.priorities[idx] = float(max(prio, 1e-6))

    def __len__(self) -> int:
        return len(self.memory)


def torch_from_numpy(array: np.ndarray):
    # Always return float32 for compatibility with training
    return torch.as_tensor(array, dtype=torch.float32)


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
