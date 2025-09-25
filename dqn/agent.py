from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from dqn.replay_memory import PrioritizedReplayMemory


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
        learn_every: int = 4,
        min_buffer_size: int = 256,
        updates_per_step: int = 2,
    ) -> None:
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = max(1, epsilon_decay)
        self.batch_size = batch_size
        self.sample_batch_size = batch_size
        self.target_update = target_update
        self.learn_every = max(1, learn_every)
        self.updates_per_step = max(1, updates_per_step)
        self.min_buffer_size = max(self.batch_size, min_buffer_size)

        self.policy_net = policy_net
        self.target_net = target_net

        self.memory = PrioritizedReplayMemory(memory_size, device=self.device)
        self.current_epsilon = max(0.0, epsilon_start)
        self.global_step = 0
        self.n_step = 3
        self.n_step_buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.n_step))
        self.beta = 0.4
        self.beta_increment_per_sampling = (1.0 - self.beta) / 100000
        self._steps_since_update = 0

    # ------------------------------------------------------------------
    # Properties & simple helpers
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.policy_net.parameters()).device

    def sync_memory_device(self) -> None:
        """Ensure the replay buffer tensors live on the same device as the networks."""
        self.memory.to(self.device)

    def ready_to_learn(self) -> bool:
        enough_samples = len(self.memory) >= max(self.min_buffer_size, self.sample_batch_size)
        return enough_samples and self._steps_since_update >= self.learn_every

    def reset_update_tracker(self) -> None:
        self._steps_since_update = 0

    def set_episode_progress(self, episode_idx: int, total_episodes: Optional[int]) -> None:
        """Update epsilon-greedy exploration schedule for a new episode."""
        if total_episodes is None or total_episodes <= 1:
            progress = 0.0
        else:
            progress = min(max(episode_idx, 0), total_episodes - 1) / (total_episodes - 1)
        self.current_epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)

    # ------------------------------------------------------------------
    # Replay buffer ingestion
    # ------------------------------------------------------------------
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
        """Push a batch of experiences to the replay buffer without CPU copies."""

        state_images = state_images.detach()
        state_bboxes = state_bboxes.detach()
        actions = actions.detach()
        rewards = rewards.detach()
        next_state_images = next_state_images.detach()
        next_state_bboxes = next_state_bboxes.detach()
        done = done.detach()

        if actions.dim() == 1:
            actions = actions.view(-1, 1)
        if rewards.dim() == 1:
            rewards = rewards.view(-1, 1)

        aggregated = []
        batch_size = state_images.size(0)
        for env_idx in range(batch_size):
            done_flag = bool(done[env_idx].item())
            state = (state_images[env_idx], state_bboxes[env_idx])
            action = actions[env_idx].view(1, -1)
            reward = rewards[env_idx].view(1)
            next_state = None
            if not done_flag:
                next_state = (
                    next_state_images[env_idx],
                    next_state_bboxes[env_idx],
                )

            buffer = self.n_step_buffers[env_idx]
            buffer.append((state, action, reward, next_state, done_flag))

            if done_flag:
                while buffer:
                    transition = list(buffer)
                    aggregated.append(self._aggregate_n_step(transition))
                    buffer.popleft()
                buffer.clear()
            elif len(buffer) == self.n_step:
                transition = list(buffer)
                aggregated.append(self._aggregate_n_step(transition))
                buffer.popleft()

        if not aggregated:
            return

        state_images_batch = torch.stack([exp[0][0] for exp in aggregated])
        state_bboxes_batch = torch.stack([exp[0][1] for exp in aggregated])
        actions_batch = torch.cat([exp[1] for exp in aggregated]).long()
        rewards_batch = torch.cat([exp[2] for exp in aggregated]).view(-1, 1)

        next_images_list = []
        next_bboxes_list = []
        dones_list = []
        n_used_list = []
        for state, action, reward, next_state, done_flag, n_used in aggregated:
            if next_state is None:
                next_images_list.append(torch.zeros_like(state[0]))
                next_bboxes_list.append(torch.zeros_like(state[1]))
            else:
                next_images_list.append(next_state[0])
                next_bboxes_list.append(next_state[1])
            dones_list.append(torch.tensor([[done_flag]], device=self.device, dtype=torch.bool))
            n_used_list.append(torch.tensor([[float(n_used)]], device=self.device))

        next_state_images_batch = torch.stack(next_images_list)
        next_state_bboxes_batch = torch.stack(next_bboxes_list)
        dones_batch = torch.cat(dones_list)
        n_used_batch = torch.cat(n_used_list)

        self.memory.push_batch(
            state_images_batch,
            state_bboxes_batch,
            actions_batch,
            rewards_batch,
            next_state_images_batch,
            next_state_bboxes_batch,
            dones_batch,
            n_used_batch,
        )
        self._steps_since_update += len(aggregated)

    # ------------------------------------------------------------------
    # Action selection & optimisation
    # ------------------------------------------------------------------
    def select_action(self, state, env, greedy: bool = False) -> torch.Tensor:
        image, bbox = state
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)

        B = image.size(0)
        image = image.to(self.device)
        bbox = bbox.to(self.device)

        with torch.no_grad():
            q = self.policy_net(image, bbox)
            greedy_actions = q.argmax(dim=1)  # (B,)

        # Force STOP when IoU >= threshold (and tumor present/active)
        actions = greedy_actions.clone()
        stop_at_threshold = None
        if getattr(env, "last_iou", None) is not None:
            tumor_present = (
                env.has_tumor
                if getattr(env, "has_tumor", None) is not None
                else torch.ones(B, device=self.device, dtype=torch.bool)
            )
            active = (
                env.active_mask
                if getattr(env, "active_mask", None) is not None
                else torch.ones(B, device=self.device, dtype=torch.bool)
            )
            stop_at_threshold = (
                env.last_iou.to(self.device) >= float(env.iou_threshold)
            ) & tumor_present & active
            actions = torch.where(stop_at_threshold, torch.full_like(actions, env._STOP_ACTION), actions)
        else:
            tumor_present = torch.ones(B, device=self.device, dtype=torch.bool)
            active = torch.ones(B, device=self.device, dtype=torch.bool)
            stop_at_threshold = torch.zeros(B, device=self.device, dtype=torch.bool)

        if greedy:
            return actions.detach()

        # Which rows explore this step?
        explore_mask = torch.rand(B, device=self.device) < self.current_epsilon

        if explore_mask.any():
            pos_mask = env.positive_actions_mask()  # (B, 8) on-device
            any_pos = pos_mask.any(dim=1)  # (B,)
            best_by_iou = env.best_action_by_iou(include_stop=True)  # (B,)

            below_threshold = (
                env.last_iou.to(self.device) < float(env.iou_threshold)
            ) if getattr(env, "last_iou", None) is not None else torch.zeros(
                B, dtype=torch.bool, device=self.device
            )
            tumor_present = tumor_present
            active = active
            explore_rows = explore_mask & (~stop_at_threshold)

            # Case A: rows needing full random action (no positives, below threshold, tumor present & active)
            random_needed = explore_rows & (~any_pos) & below_threshold & tumor_present & active
            idx_random = torch.nonzero(random_needed, as_tuple=False).squeeze(1)
            if idx_random.numel() > 0:
                actions[idx_random] = torch.randint(0, 8, (idx_random.numel(),), device=self.device)

            # Case B: rows with positives → sample uniformly among positive actions (fully vectorized)
            pos_needed = explore_rows & any_pos
            idx_pos = torch.nonzero(pos_needed, as_tuple=False).squeeze(1)
            if idx_pos.numel() > 0:
                probs = pos_mask[pos_needed].float()  # (M, 8)
                chosen = torch.multinomial(probs, num_samples=1).squeeze(1)  # (M,)
                actions[idx_pos] = chosen

            # Case C: remaining explore rows (e.g., no tumor) → fallback to oracle (may be STOP)
            remaining = explore_rows & ~(random_needed | pos_needed)
            idx_rem = torch.nonzero(remaining, as_tuple=False).squeeze(1)
            if idx_rem.numel() > 0:
                actions[idx_rem] = best_by_iou[idx_rem]

        # epsilon decay
        self.global_step += 1
        frac = min(1.0, self.global_step / self.epsilon_decay)
        self.current_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * ((1 - frac) ** 3)
        return actions.detach()

    def compute_loss(self) -> Optional[torch.Tensor]:
        """Computes the DQN loss without performing an optimization step."""

        if len(self.memory) < max(self.sample_batch_size, self.min_buffer_size):
            return None

        sample_n = min(self.sample_batch_size, len(self.memory))
        batch, indices, weights = self.memory.sample(sample_n, beta=self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        state_images = batch.state_images.to(self.device)
        state_bboxes = batch.state_bboxes.to(self.device)
        actions = batch.actions.to(self.device).long()
        rewards = batch.rewards.to(self.device)
        next_state_images = batch.next_state_images.to(self.device)
        next_state_bboxes = batch.next_state_bboxes.to(self.device)
        dones = batch.dones.to(self.device).view(-1).bool()
        n_used = batch.n_used.to(self.device).view(-1)
        weights = weights.to(self.device).view(-1, 1)

        state_action_values = self.policy_net(state_images, state_bboxes).gather(1, actions)

        next_state_values = torch.zeros(sample_n, device=self.device)
        non_final_mask = ~dones
        if non_final_mask.any():
            next_images = next_state_images[non_final_mask]
            next_bboxes = next_state_bboxes[non_final_mask]
            with torch.no_grad():
                q_next_policy = self.policy_net(next_images, next_bboxes)
                next_actions = q_next_policy.argmax(dim=1, keepdim=True)
                q_next_target = self.target_net(next_images, next_bboxes)
                q_next_selected = q_next_target.gather(1, next_actions).squeeze(1)
            next_state_values[non_final_mask] = q_next_selected

        gamma_power = self.gamma ** n_used
        expected_state_action_values = rewards.view(-1) + gamma_power * next_state_values
        td_target = expected_state_action_values.unsqueeze(1)

        loss_elements = F.smooth_l1_loss(state_action_values, td_target, reduction="none")
        loss = (loss_elements * weights).mean()

        td_errors = (td_target - state_action_values).detach().abs().view(-1)
        self.memory.update_priorities(indices, td_errors + 1e-6)
        return loss

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def update_target_net(self) -> None:
        """Updates the target network with the policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.policy_net.parameters()

    def state_dict(self) -> dict:
        return self.policy_net.state_dict()

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

