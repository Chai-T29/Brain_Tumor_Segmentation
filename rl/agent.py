from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import optim

from .networks import Actor, Critic


@dataclass
class NoiseScheduleConfig:
    sigma_init: float = 1.0
    sigma_final: float = 0.1
    steps: int = 500000


@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    n_step: int = 3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    policy_delay: int = 2
    exploration_noise: NoiseScheduleConfig = field(default_factory=NoiseScheduleConfig)
    target_policy_noise_std: float = 0.2
    target_policy_noise_clip: float = 0.5
    max_grad_norm: float = 10.0
    embedding_noise_std: float = 0.01


class TD3Agent:
    """TD3 agent with optional embedding noise to emulate DrQ-style augmentation."""

    def __init__(
        self,
        embedding_dim: int,
        polygon_dim: int,
        action_dim: int,
        config: TD3Config,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cpu")

        self.actor = Actor(embedding_dim, polygon_dim, action_dim).to(self.device)
        self.actor_target = Actor(embedding_dim, polygon_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(embedding_dim, polygon_dim, action_dim).to(self.device)
        self.critic_target = Critic(embedding_dim, polygon_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.total_updates = 0
        self._interaction_count = 0
        self.warmup_steps = 0

    def to(self, device: torch.device) -> "TD3Agent":
        self.device = device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        return self

    def _augment_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.config.embedding_noise_std <= 0:
            return embedding
        noise = torch.randn_like(embedding) * self.config.embedding_noise_std
        return embedding + noise

    @torch.no_grad()
    def act(
        self,
        embedding: torch.Tensor,
        polygon_state: torch.Tensor,
        deterministic: bool = False,
        apply_embedding_noise: bool = True,
    ) -> torch.Tensor:
        self.actor.eval()
        emb = self._augment_embedding(embedding) if apply_embedding_noise else embedding
        action = self.actor(emb, polygon_state)
        if not deterministic:
            sigma = self._current_exploration_sigma()
            if sigma > 0:
                noise = torch.randn_like(action) * sigma
                action = action + noise
            self._interaction_count += embedding.size(0)
        return action.clamp_(-1.0, 1.0)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()

        embedding = batch["embedding"]
        polygon = batch["polygon"]
        action = batch["action"]
        reward = batch["reward"]
        discount = batch["discount"]
        next_polygon = batch["next_polygon"]

        # Critics update -----------------------------------------------------
        emb_aug = self._augment_embedding(embedding)
        current_q1, current_q2 = self.critic(emb_aug, polygon, action)

        with torch.no_grad():
            next_emb_aug = self._augment_embedding(embedding)
            target_action = self.actor_target(next_emb_aug, next_polygon)
            if self.config.target_policy_noise_std > 0:
                noise = torch.randn_like(target_action) * self.config.target_policy_noise_std
                noise = noise.clamp_(-self.config.target_policy_noise_clip, self.config.target_policy_noise_clip)
                target_action = target_action + noise
            target_action = target_action.clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_emb_aug, next_polygon, target_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = reward + discount * target_q

        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        metrics: Dict[str, float] = {"critic_loss": float(critic_loss.item())}

        # Actor update -------------------------------------------------------
        update_actor = (self.total_updates + 1) % max(1, self.config.policy_delay) == 0
        actor_loss_value: Optional[float] = None
        if update_actor:
            actor_action = self.actor(emb_aug, polygon)
            actor_loss = -self.critic.q1_forward(emb_aug, polygon, actor_action).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_opt.step()
            actor_loss_value = float(actor_loss.item())
            metrics["actor_loss"] = actor_loss_value

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        self.total_updates += 1
        return metrics

    def _soft_update(self, net: torch.nn.Module, target_net: torch.nn.Module) -> None:
        tau = self.config.tau
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.lerp_(param.data, tau)

    def _current_exploration_sigma(self) -> float:
        schedule = self.config.exploration_noise
        if self._interaction_count < self.warmup_steps:
            return float(schedule.sigma_init)
        progressed = self._interaction_count - self.warmup_steps
        if schedule.steps <= 0:
            return float(schedule.sigma_final)
        progress = min(1.0, progressed / float(schedule.steps))
        return float(schedule.sigma_init + (schedule.sigma_final - schedule.sigma_init) * progress)

    def set_warmup_steps(self, warmup_steps: int) -> None:
        self.warmup_steps = max(0, int(warmup_steps))
