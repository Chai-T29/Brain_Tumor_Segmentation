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
    """Learnable sequence of 3x3 convolutions that reduce H×W to 1×1."""

    def __init__(self, in_shape: Tuple[int, int, int], out_dim: int) -> None:
        super().__init__()
        channels, height, width = in_shape
        if height != width:
            raise ValueError("Embedding map must be square to reduce with 3x3 convolutions.")
        if height < 1:
            raise ValueError("Embedding map must have positive spatial dimensions.")

        self.out_dim = int(out_dim)
        size = int(height)
        in_channels = int(channels)
        layers: list[nn.Module] = []

        # Repeatedly apply 3x3 conv (stride 1, no padding) until spatial size reaches 1×1.
        while size > 1:
            if size < 3:
                # Fallback: collapse remaining spatial extent with kernel matching current size.
                kernel_size = size
            else:
                kernel_size = 3

            conv = nn.Conv2d(in_channels, self.out_dim, kernel_size=kernel_size, stride=1, padding=0, bias=True)
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            layers.append(conv)
            layers.append(nn.ReLU(inplace=True))

            size = size - (kernel_size - 1)
            if size <= 0:
                raise ValueError("Convolution stack collapsed spatial dimensions below 1. Check input shape.")

            norm_shape = (self.out_dim, size, size)
            layers.append(nn.LayerNorm(norm_shape))
            in_channels = self.out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Expected embedding map with shape [B, C, H, W].")
        if len(self.net) == 0:
            return x
        return self.net(x)


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
        projected_map = self.embedding_projector(embedding_map)
        flattened = projected_map.flatten(start_dim=1)
        normalized = self.embedding_norm(flattened)
        return normalized

    def _augment_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.config.embedding_noise_std <= 0:
            return embedding
        if embedding.dim() != 2:
            raise ValueError("Expected embedding tensor with shape [B, D].")
        batch, dim = embedding.shape
        base = embedding.view(batch, dim, 1, 1)
        noise = torch.randn_like(base) * self.config.embedding_noise_std
        perturbed = base + noise
        return perturbed.view(batch, dim)

    def _guided_exploration_adjust(
        self,
        embedding: torch.Tensor,
        polygon_state: torch.Tensor,
        noisy_action: torch.Tensor,
    ) -> torch.Tensor:
        """Adjust a noisy action using the critic gradient if it improves value."""

        if not self.config.guided_exploration:
            return noisy_action

        guidance_scale = float(self.config.guidance_scale)
        if guidance_scale <= 0.0:
            return noisy_action

        emb = embedding.detach()
        poly = polygon_state.detach()

        with torch.enable_grad():
            action_var = noisy_action.detach().clone().requires_grad_(True)
            q1, q2 = self.critic(emb, poly, action_var)
            q_min = torch.minimum(q1, q2)
            grad = torch.autograd.grad(q_min.sum(), action_var, retain_graph=False, allow_unused=False)[0]

        if grad is None:
            return noisy_action

        grad = grad.detach()
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = guidance_scale * (grad / grad_norm)
        candidate_action = (noisy_action + direction).clamp(-1.0, 1.0)

        with torch.no_grad():
            q_min_original = q_min.detach()
            q1_new, q2_new = self.critic(emb, poly, candidate_action)
            q_min_new = torch.minimum(q1_new, q2_new)

        improved = (q_min_new > q_min_original).view(-1, 1)
        return torch.where(improved, candidate_action, noisy_action)

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
                noise = torch.randn_like(action) * sigma
                noisy_action = action + noise
                if self.config.guided_exploration:
                    noisy_action = self._guided_exploration_adjust(emb, polygon_state, noisy_action)
                action = noisy_action
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
