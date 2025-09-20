import torch
import torch.optim as optim
import random
import math
from dqn.model import QNetwork
from dqn.replay_memory import ReplayMemory, Experience
import torch.nn.functional as F

class DQNAgent:
    """The DQN Agent that interacts with and learns from the environment."""
    def __init__(self, policy_net, target_net, num_actions, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=1000, memory_size=10000, batch_size=64, target_update=10):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = policy_net
        self.target_net = target_net

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)
        self.steps_done = 0

    def select_action(self, state, greedy=False):
        """Selects an action using an epsilon-greedy policy."""
        sample = random.random()
        # Epsilon decay
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if greedy or sample > eps_threshold:
            with torch.no_grad():
                # state is a tuple (image, bbox)
                image, bbox = state
                # Add a batch dimension if not present
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                if bbox.dim() == 1:
                    bbox = bbox.unsqueeze(0)
                image = image.to(self.device)
                bbox = bbox.to(self.device)
                return self.policy_net(image, bbox).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        """Performs a single step of the optimization."""
        if len(self.memory) < self.batch_size:
            return

        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Unpack the batch
        state_batch = batch.state
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = batch.next_state

        # Separate image and bbox from state
        image_batch = torch.stack([s[0] for s in state_batch]).to(self.device)
        bbox_batch = torch.stack([s[1] for s in state_batch]).to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(image_batch, bbox_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        non_final_mask_list = []
        non_final_next_states = []
        for next_state, done in zip(next_state_batch, batch.done):
            is_non_final = (next_state is not None) and (not done)
            non_final_mask_list.append(is_non_final)
            if is_non_final:
                non_final_next_states.append(next_state)

        non_final_mask = torch.tensor(non_final_mask_list, device=self.device, dtype=torch.bool)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states:
            non_final_next_images = torch.stack([s[0] for s in non_final_next_states]).to(self.device)
            non_final_next_bboxes = torch.stack([s[1] for s in non_final_next_states]).to(self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_images, non_final_next_bboxes).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        """Updates the target network with the policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
