import random
from collections import namedtuple, deque

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    """A cyclic buffer of bounded size that holds the experiences observed recently."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save an experience."""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
