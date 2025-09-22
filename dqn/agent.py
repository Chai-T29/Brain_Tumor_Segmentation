from collections import defaultdict, deque
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from dqn.replay_memory import Experience, PrioritizedReplayMemory


class DQNAgent:
    """The DQN (or DDQN) Agent that interacts with and learns from the environment."""

    def __init__(
        self,
        policy_net,
        target_net,
        num_actions,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 1000,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10,
    ) -> None:
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = max(1, epsilon_decay)
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = policy_net
        self.target_net = target_net

        self.memory = PrioritizedReplayMemory(memory_size)
        self.current_epsilon = max(0.0, epsilon_start)
        self.global_step = 0
        self.n_step = 3
        self.n_step_buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.n_step))
        self.beta = 0.4
        self.beta_increment_per_sampling = (1.0 - self.beta) / 100000
        # if self.epsilon_start <= 0:
        #     self._step_decay = 1.0
        # else:
        #     ratio = max(self.epsilon_end, 1e-12) / self.epsilon_start
        #     self._step_decay = ratio ** (1.0 / self.epsilon_decay)

    @property
    def device(self) -> torch.device:
        return next(self.policy_net.parameters()).device

    def set_episode_progress(self, episode_idx: int, total_episodes: Optional[int]) -> None:
        """Update epsilon-greedy exploration schedule for a new episode."""
        if total_episodes is None or total_episodes <= 1:
            progress = 0.0
        else:
            progress = min(max(episode_idx, 0), total_episodes - 1) / (total_episodes - 1)
        self.current_epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)

    def push_experience(self, state, action, reward, next_state, done, env_idx: int) -> None:
        buffer = self.n_step_buffers[env_idx]
        buffer.append((state, action, reward, next_state, done))

        if done:
            while buffer:
                transition = list(buffer)
                aggregated = self._aggregate_n_step(transition)
                self.memory.push(*aggregated)
                buffer.popleft()
            buffer.clear()
        elif len(buffer) == self.n_step:
            transition = list(buffer)
            aggregated = self._aggregate_n_step(transition)
            self.memory.push(*aggregated)
            buffer.popleft()

    def select_action(self, state, greedy: bool = False) -> torch.Tensor:
        """Selects actions for a batch of states using an epsilon-greedy policy."""
        image, bbox = state
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)

        batch_size = image.size(0)
        image = image.to(self.device)
        bbox = bbox.to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(image, bbox)
            greedy_actions = q_values.argmax(dim=1)

        if greedy:
            return greedy_actions.detach()

        epsilon = self.current_epsilon
        random_mask = torch.rand(batch_size, device=self.device) < epsilon
        random_actions = torch.randint(0, self.num_actions, (batch_size,), device=self.device)
        selected_actions = torch.where(random_mask, random_actions, greedy_actions)

        #self.current_epsilon = max(self.epsilon_end, self.current_epsilon * self._step_decay)
        self.global_step += 1
        fraction = min(1.0, self.global_step / self.epsilon_decay)
        # exponential decay between start and end
        self.current_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * ((1 - fraction) ** 3)
        return selected_actions.detach()

    def compute_loss(self) -> Optional[torch.Tensor]:
        """Computes the DQN loss without performing an optimization step."""
        if len(self.memory) < self.batch_size:
            return None

        experiences, indices, weights = self.memory.sample(self.batch_size, beta=self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        batch = Experience(*zip(*experiences))
        weights = weights.to(self.device)

        state_batch = batch.state
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = batch.next_state

        image_batch = torch.stack([s[0] for s in state_batch]).to(self.device)
        bbox_batch = torch.stack([s[1] for s in state_batch]).to(self.device)

        state_action_values = self.policy_net(image_batch, bbox_batch).gather(1, action_batch)

        non_final_mask_list = []
        non_final_next_states = []
        for next_state, done in zip(next_state_batch, batch.done):
            is_non_final = (next_state is not None) and (not done)
            non_final_mask_list.append(is_non_final)
            if is_non_final:
                non_final_next_states.append(next_state)

        non_final_mask = torch.tensor(non_final_mask_list, device=self.device, dtype=torch.bool)

        # Standard DQN Setup
        # next_state_values = torch.zeros(self.batch_size, device=self.device)
        # if non_final_next_states:
        #     non_final_next_images = torch.stack([s[0] for s in non_final_next_states]).to(self.device)
        #     non_final_next_bboxes = torch.stack([s[1] for s in non_final_next_states]).to(self.device)
        #     next_state_values[non_final_mask] = (
        #         self.target_net(non_final_next_images, non_final_next_bboxes).max(1)[0].detach()
        #     )

        # Double-DQN Setup (reduces overestimation bias)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        if non_final_next_states:
            non_final_next_images = torch.stack([s[0] for s in non_final_next_states]).to(self.device)
            non_final_next_bboxes = torch.stack([s[1] for s in non_final_next_states]).to(self.device)

            with torch.no_grad():
                # 1) Action selection uses the *policy* network
                q_next_policy = self.policy_net(non_final_next_images, non_final_next_bboxes)
                next_actions = q_next_policy.argmax(dim=1, keepdim=True)  # shape (M, 1)

                # 2) Action evaluation uses the *target* network
                q_next_target = self.target_net(non_final_next_images, non_final_next_bboxes)
                q_next_chosen = q_next_target.gather(1, next_actions).squeeze(1)  # shape (M,)

            next_state_values[non_final_mask] = q_next_chosen

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        td_target = expected_state_action_values.unsqueeze(1)
        loss_elements = F.smooth_l1_loss(state_action_values, td_target, reduction="none")
        loss = (loss_elements * weights.view(-1, 1)).mean()

        td_errors = (td_target - state_action_values).detach().squeeze(1).abs().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, td_errors)
        return loss

    def _aggregate_n_step(self, transitions):
        state, action, _, _, _ = transitions[0]
        cumulative_reward = torch.zeros_like(transitions[0][2])
        next_state = transitions[-1][3]
        done = transitions[-1][4]

        for idx, (_, _, reward, step_next_state, step_done) in enumerate(transitions):
            cumulative_reward += (self.gamma ** idx) * reward
            if step_done:
                next_state = None
                done = True
                break

        return state, action, cumulative_reward, next_state, done

    def update_target_net(self) -> None:
        """Updates the target network with the policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.policy_net.parameters()

    def state_dict(self) -> dict:
        return self.policy_net.state_dict()
