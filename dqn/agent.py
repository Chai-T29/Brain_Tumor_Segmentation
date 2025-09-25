from collections import defaultdict, deque
from typing import Dict, Iterable, Optional
import random

import torch
import torch.nn.functional as F

from dqn.replay_memory import Experience, PrioritizedReplayMemory


class DQNAgent:
    """Memory-optimized DQN Agent."""

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
        action_history_size: int = 5,
        movement_frequency: int = 3,
        diversity_penalty: float = 0.1,
        gpu_memory_fraction: float = 0.8,  # Keep most of replay buffer on GPU
    ) -> None:
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = max(1, epsilon_decay)
        self.batch_size = batch_size
        self.target_update = target_update
        self.gpu_memory_fraction = gpu_memory_fraction

        self.policy_net = policy_net
        self.target_net = target_net

        self.memory = PrioritizedReplayMemory(memory_size)
        self.current_epsilon = max(0.0, epsilon_start)
        self.global_step = 0
        self.n_step = 3
        self.n_step_buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.n_step))
        self.beta = 0.4
        self.beta_increment_per_sampling = (1.0 - self.beta) / 100000
        
        # Action diversity tracking
        self.action_history_size = action_history_size
        self.movement_frequency = movement_frequency
        self.diversity_penalty = diversity_penalty
        self.action_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=action_history_size))
        self.zoom_counters: Dict[int, int] = defaultdict(int)

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

    def _is_zoom_action(self, action: int) -> bool:
        """Check if action is a zoom action."""
        return action in [4, 5, 6, 7]

    def _is_movement_action(self, action: int) -> bool:
        """Check if action is a movement action."""
        return action in [0, 1, 2, 3]

    def _should_force_movement(self, env_idx: int) -> bool:
        """Check if we should force a movement action for diversity."""
        return self.zoom_counters[env_idx] >= self.movement_frequency

    def _update_action_tracking(self, env_idx: int, action: int):
        """Update action history and zoom counters."""
        self.action_history[env_idx].append(action)
        
        if self._is_zoom_action(action):
            self.zoom_counters[env_idx] += 1
        elif self._is_movement_action(action):
            self.zoom_counters[env_idx] = 0

    def _get_diversity_penalty(self, env_idx: int, action: int) -> float:
        """Calculate penalty for repeating actions too frequently."""
        if len(self.action_history[env_idx]) < 2:
            return 0.0
        
        recent_actions = list(self.action_history[env_idx])
        same_action_count = sum(1 for a in recent_actions[-3:] if a == action)
        
        if same_action_count >= 2:
            return self.diversity_penalty * same_action_count
        return 0.0

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
            # Reset tracking for this environment
            self.action_history[env_idx].clear()
            self.zoom_counters[env_idx] = 0
        elif len(buffer) == self.n_step:
            transition = list(buffer)
            aggregated = self._aggregate_n_step(transition)
            self.memory.push(*aggregated)
            buffer.popleft()

    def push_experience_batch(
        self,
        state_images: torch.Tensor,
        state_bboxes: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_state_images: torch.Tensor,
        next_state_bboxes: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Memory-optimized batch experience pushing."""
        
        if actions.dim() == 1:
            actions = actions.view(-1, 1)

        batch_size = state_images.size(0)
        
        # Decide whether to keep data on GPU or move to CPU based on memory settings
        # Keep a portion on GPU for faster access, rest on CPU
        gpu_batch_size = int(batch_size * self.gpu_memory_fraction)
        
        for env_idx in range(batch_size):
            done_flag = bool(done[env_idx].item())
            
            # Keep some data on GPU, some on CPU to balance memory usage
            if env_idx < gpu_batch_size:
                # Keep on GPU (faster but uses GPU memory)
                state = (state_images[env_idx].to(self.device), 
                        state_bboxes[env_idx].to(self.device))
                action = actions[env_idx].to(self.device).view(1, 1)
                reward = rewards[env_idx].to(self.device).view(1)
                next_state = None
                if not done_flag:
                    next_state = (
                        next_state_images[env_idx].to(self.device),
                        next_state_bboxes[env_idx].to(self.device),
                    )
            else:
                # Keep on CPU (saves GPU memory)
                state = (state_images[env_idx].cpu(), state_bboxes[env_idx].cpu())
                action = actions[env_idx].cpu().view(1, 1)
                reward = rewards[env_idx].cpu().view(1)
                next_state = None
                if not done_flag:
                    next_state = (
                        next_state_images[env_idx].cpu(),
                        next_state_bboxes[env_idx].cpu(),
                    )

            self.push_experience(
                state,
                action,
                reward,
                next_state,
                done_flag,
                env_idx=env_idx,
            )

    def select_action(self, state, env, greedy: bool = False) -> torch.Tensor:
        image, bbox = state
        if image.dim()==3: image=image.unsqueeze(0)
        if bbox.dim()==1:  bbox=bbox.unsqueeze(0)

        B = image.size(0)
        image = image.to(self.device); bbox = bbox.to(self.device)

        with torch.no_grad():
            q = self.policy_net(image, bbox)
            greedy_actions = q.argmax(dim=1)

        if greedy:
            for i in range(B):
                self._update_action_tracking(i, int(greedy_actions[i].item()))
            return greedy_actions.detach()

        explore_mask = torch.rand(B, device=self.device) < self.current_epsilon

        if not explore_mask.any():
            actions = greedy_actions
        else:
            pos_mask = env.positive_actions_mask()
            best_by_iou = env.best_action_by_iou(include_stop=True)

            actions = greedy_actions.clone()
            idx = torch.nonzero(explore_mask, as_tuple=False).squeeze(1)
            for i in idx.tolist():
                if env.has_tumor is not None and not bool(env.has_tumor[i]):
                    a = env._STOP_ACTION
                else:
                    if self._should_force_movement(i):
                        movement_actions = [0, 1, 2, 3]
                        pos_movements = [act for act in movement_actions if pos_mask[i][act]]
                        if pos_movements:
                            a = random.choice(pos_movements)
                        else:
                            a = random.choice(movement_actions)
                    else:
                        pos_idx = torch.nonzero(pos_mask[i], as_tuple=False).squeeze(1)
                        if pos_idx.numel() > 0:
                            available_actions = []
                            for act_idx in pos_idx.tolist():
                                penalty = self._get_diversity_penalty(i, act_idx)
                                if penalty == 0.0 or random.random() > penalty:
                                    available_actions.append(act_idx)
                            
                            if available_actions:
                                a = random.choice(available_actions)
                            else:
                                j = torch.randint(0, pos_idx.numel(), (1,), device=self.device)
                                a = pos_idx[j].item()
                        else:
                            a = int(best_by_iou[i].item())
                actions[i] = a

        for i in range(B):
            self._update_action_tracking(i, int(actions[i].item()))

        # Epsilon decay
        self.global_step += 1
        frac = min(1.0, self.global_step / self.epsilon_decay)
        self.current_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * ((1 - frac) ** 3)
        return actions.detach()

    def compute_loss(self) -> Optional[torch.Tensor]:
        sample_n = max(self.batch_size, 1000)  # Need at least 1000 samples
        if len(self.memory) < sample_n:
            return None

        sampled = self.memory.sample(sample_n, beta=self.beta)
        if isinstance(sampled, tuple):
            experiences, indices, weights = sampled
        else:
            experiences = list(sampled)
            indices = sampled.indices
            weights = sampled.weights
        
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        batch = Experience(*zip(*experiences))
        weights = weights.to(self.device)

        state_batch = batch.state
        action_batch = torch.cat([a.to(self.device) for a in batch.action])
        reward_batch = torch.cat([r.to(self.device) for r in batch.reward])
        next_state_batch = batch.next_state

        # Move state data to device as needed
        image_batch = torch.stack([s[0].to(self.device) for s in state_batch])
        bbox_batch = torch.stack([s[1].to(self.device) for s in state_batch])

        state_action_values = self.policy_net(image_batch, bbox_batch).gather(1, action_batch)

        non_final_mask_list = []
        non_final_next_states = []
        for next_state, done in zip(next_state_batch, batch.done):
            is_non_final = (next_state is not None) and (not done)
            non_final_mask_list.append(is_non_final)
            if is_non_final:
                non_final_next_states.append(next_state)

        non_final_mask = torch.tensor(non_final_mask_list, device=self.device, dtype=torch.bool)

        # Double-DQN Setup with memory optimization
        next_state_values = torch.zeros(sample_n, device=self.device)
        
        if non_final_next_states:
            non_final_next_images = torch.stack([s[0].to(self.device) for s in non_final_next_states])
            non_final_next_bboxes = torch.stack([s[1].to(self.device) for s in non_final_next_states])

            with torch.no_grad():
                q_next_policy = self.policy_net(non_final_next_images, non_final_next_bboxes)
                next_actions = q_next_policy.argmax(dim=1, keepdim=True)

                q_next_target = self.target_net(non_final_next_images, non_final_next_bboxes)
                q_next_chosen = q_next_target.gather(1, next_actions).squeeze(1)

            next_state_values[non_final_mask] = q_next_chosen
            
            # Clear intermediate tensors to save memory
            del non_final_next_images, non_final_next_bboxes, q_next_policy, q_next_target

        n_used_batch = torch.tensor([n for n in batch.n_used], device=self.device, dtype=torch.float32)
        expected_state_action_values = reward_batch + (self.gamma ** n_used_batch) * next_state_values
        td_target = expected_state_action_values.unsqueeze(1)
        loss_elements = F.smooth_l1_loss(state_action_values, td_target, reduction="none")
        loss = (loss_elements * weights.view(-1, 1)).mean()

        td_errors = (td_target - state_action_values).detach().squeeze(1).abs().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, td_errors)
        
        # Clear batch tensors to prevent memory buildup
        del image_batch, bbox_batch, state_action_values, td_target
        
        # Force GPU cache cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return loss

    def _aggregate_n_step(self, transitions):
        state, action, _, _, _ = transitions[0]
        cumulative_reward = torch.zeros_like(transitions[0][2])
        next_state = transitions[-1][3]
        done = transitions[-1][4]

        n_used = 0
        for idx, (_, _, reward, step_next_state, step_done) in enumerate(transitions):
            cumulative_reward += (self.gamma ** idx) * reward
            n_used += 1
            if step_done:
                next_state = None
                done = True
                break
        return state, action, cumulative_reward, next_state, done, n_used

    def update_target_net(self) -> None:
        """Updates the target network with the policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.policy_net.parameters()

    def state_dict(self) -> dict:
        return self.policy_net.state_dict()

    def cleanup_memory(self):
        """Force cleanup of tracked data structures."""
        # Clear action tracking for inactive environments
        active_envs = set()
        for env_idx in list(self.action_history.keys()):
            if len(self.action_history[env_idx]) > 0:
                active_envs.add(env_idx)
        
        # Remove tracking for environments that haven't been used recently
        if len(active_envs) > 100:  # Arbitrary threshold
            for env_idx in list(self.action_history.keys()):
                if env_idx not in active_envs:
                    del self.action_history[env_idx]
                    if env_idx in self.zoom_counters:
                        del self.zoom_counters[env_idx]