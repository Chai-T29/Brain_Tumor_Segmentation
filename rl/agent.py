from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
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
    embedding_projected_dim: int = 512
    pr_alpha: float = 0.6
    pr_beta_start: float = 0.4
    pr_beta_steps: int = 200000
    pr_eps: float = 1e-6
    actor_hidden_sizes: tuple[int, ...] = (512, 512)
    critic_hidden_sizes: tuple[int, ...] = (512, 512)
    guided_exploration: bool = False
    guidance_scale: float = 0.2


class EmbeddingProjector(nn.Module):
    """Fixed random projection that flattens spatial features into a compact vector."""

    def __init__(self, in_shape: Tuple[int, int, int], out_dim: int) -> None:
        super().__init__()
        channels, height, width = in_shape
        self.in_dim = int(channels * height * width)
        self.out_dim = int(out_dim)
        self.fc = nn.Linear(self.in_dim, self.out_dim, bias=False)
        nn.init.normal_(self.fc.weight, mean=0.0, std=1.0 / math.sqrt(self.out_dim))
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flattened = x.flatten(start_dim=1)
        return self.fc(flattened)


class TD3Agent(nn.Module):
    """TD3 agent with optional embedding noise to emulate DrQ-style augmentation."""

    def __init__(
        self,
        embedding_shape: Tuple[int, int, int],
        polygon_dim: int,
        action_dim: int,
        config: TD3Config,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cpu")
        channel_dim, _, _ = embedding_shape

        self.embedding_projector = EmbeddingProjector(
            in_shape=embedding_shape,
            out_dim=int(config.embedding_projected_dim),
        ).to(self.device)
        self.embedding_norm = nn.LayerNorm(int(config.embedding_projected_dim), elementwise_affine=False).to(self.device)
        self.embedding_dim = int(config.embedding_projected_dim)

        self.actor = Actor(self.embedding_dim, polygon_dim, action_dim, self.config.actor_hidden_sizes).to(self.device)
        self.actor_target = Actor(self.embedding_dim, polygon_dim, action_dim, self.config.actor_hidden_sizes).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.embedding_dim, polygon_dim, action_dim, self.config.critic_hidden_sizes).to(self.device)
        self.critic_target = Critic(self.embedding_dim, polygon_dim, action_dim, self.config.critic_hidden_sizes).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.total_updates = 0
        self._interaction_count = 0
        self.warmup_steps = 0

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device_arg = kwargs.get("device", None)
        if device_arg is None and len(args) == 1:
            device_arg = args[0]
        if isinstance(device_arg, torch.device):
            self.device = device_arg
        return module

    def preprocess_embeddings(self, embedding_map: torch.Tensor) -> torch.Tensor:
        """Project encoder feature maps into a compact vector representation."""
        if embedding_map.dim() != 4:
            raise ValueError("Expected embedding map with shape [B, C, H, W].")
        embedding_map = embedding_map.to(self.device)
        projected = self.embedding_projector(embedding_map)
        normalized = self.embedding_norm(projected)
        return normalized

    def _augment_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.config.embedding_noise_std <= 0:
            return embedding
        noise = torch.randn_like(embedding) * self.config.embedding_noise_std
        return embedding + noise

    def _guided_exploration_mean(
        self,
        embedding: torch.Tensor,
        polygon_state: torch.Tensor,
        base_action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a guidance vector using the critic gradient to bias exploration."""

        if not self.config.guided_exploration:
            return base_action

        guidance_scale = float(self.config.guidance_scale)
        if guidance_scale <= 0.0:
            return base_action

        base = base_action.detach()
        with torch.enable_grad():
            action_var = base.clone().requires_grad_(True)
            emb = embedding.detach()
            poly = polygon_state.detach()
            # Critic gradients point toward higher Q; normalise to get a direction.
            q1 = self.critic.q1_forward(emb, poly, action_var)
            grad = torch.autograd.grad(q1.sum(), action_var, retain_graph=False, allow_unused=False)[0]

        if grad is None:
            return base_action

        grad = grad.detach()
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        guidance = grad / grad_norm
        guided_mean = base + guidance_scale * guidance
        return guided_mean.clamp(-1.0, 1.0)

    @torch.no_grad()
    def act(
        self,
        embedding: torch.Tensor,
        polygon_state: torch.Tensor,
        deterministic: bool = False,
        apply_embedding_noise: bool = True,
    ) -> torch.Tensor:
        self.actor.eval()
        if self.config.guided_exploration:
            self.critic.eval()
        emb = self._augment_embedding(embedding) if apply_embedding_noise else embedding
        action = self.actor(emb, polygon_state)
        if not deterministic:
            sigma = self._current_exploration_sigma()
            if sigma > 0:
                mean_action = action
                if self.config.guided_exploration:
                    mean_action = self._guided_exploration_mean(emb, polygon_state, action)
                noise = torch.randn_like(action) * sigma
                action = mean_action + noise
            self._interaction_count += embedding.size(0)
        return action.clamp_(-1.0, 1.0)

    def update(self, batch: Dict[str, torch.Tensor], weights: Optional[torch.Tensor] = None) -> Dict[str, float | torch.Tensor]:
        self.actor.train()
        self.critic.train()

        embedding = batch["embedding"]
        polygon = batch["polygon"]
        action = batch["action"]
        reward = batch["reward"]
        discount = batch["discount"]
        next_polygon = batch["next_polygon"]
        if weights is None:
            weights = torch.ones_like(reward)
        if weights.dim() == 1:
            weights = weights.view(-1, 1)
        weights = weights.to(embedding.device)

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

        td_error1 = target_value - current_q1
        td_error2 = target_value - current_q2
        critic_loss = (weights * td_error1.pow(2)).mean() + (weights * td_error2.pow(2)).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        metrics: Dict[str, float | torch.Tensor] = {"critic_loss": float(critic_loss.item())}

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

        td_errors = 0.5 * (torch.abs(td_error1.detach()) + torch.abs(td_error2.detach()))
        td_errors = td_errors.flatten()
        metrics["td_errors"] = td_errors

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
