from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import optim

from .networks import Actor, Critic


@dataclass
class GuidanceScheduleConfig:
    scale_init: float = 0.7
    scale_final: float = 0.0
    steps: int = 500000


@dataclass
class LRScheduleConfig:
    """Linear learning-rate decay schedule.

    lr = lr_init + (lr_final - lr_init) * clamp(step/steps, 0, 1)
    """
    lr_init: float
    lr_final: float
    steps: int = 500000


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
    true_guided_exploration: bool = False
    guidance_schedule: GuidanceScheduleConfig = field(default_factory=GuidanceScheduleConfig)
    guided_actor_loss: bool = False
    guided_actor_loss_weight: float = 0.01
    # guidance_mode options: "critic_guidance", "true_guidance", "mixed", "true_guidance_post_noise", "random_true_guidance"
    guidance_mode: str = "true_guidance"
    mixed_guidance_steps: int = 500000
    actor_lr_schedule: Optional[LRScheduleConfig] = None
    critic_lr_schedule: Optional[LRScheduleConfig] = None
    use_embedding_pointers: bool = False


class TD3Agent(nn.Module):
    """TD3 agent with optional embedding noise to emulate DrQ-style augmentation."""

    def __init__(
        self,
        embedding_shape: Tuple[int, int, int],
        polygon_dim: int,
        action_dim: int,
        config: TD3Config,
        device: torch.device | None = None,
        embedding_pointer_mode: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cpu")
        self.embedding_shape = tuple(int(v) for v in embedding_shape)
        self.embedding_dim = int(config.embedding_projected_dim)
        self.use_embedding_pointers = bool(embedding_pointer_mode)

        self.actor = Actor(
            embedding_shape=self.embedding_shape,
            polygon_dim=polygon_dim,
            action_dim=action_dim,
            hidden_sizes=self.config.actor_hidden_sizes,
            projected_dim=self.embedding_dim,
            embedding_noise_std=self.config.embedding_noise_std,
        ).to(self.device)
        self.actor_target = Actor(
            embedding_shape=self.embedding_shape,
            polygon_dim=polygon_dim,
            action_dim=action_dim,
            hidden_sizes=self.config.actor_hidden_sizes,
            projected_dim=self.embedding_dim,
            embedding_noise_std=self.config.embedding_noise_std,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(
            embedding_shape=self.embedding_shape,
            polygon_dim=polygon_dim,
            action_dim=action_dim,
            hidden_sizes=self.config.critic_hidden_sizes,
            projected_dim=self.embedding_dim,
            embedding_noise_std=self.config.embedding_noise_std,
        ).to(self.device)
        self.critic_target = Critic(
            embedding_shape=self.embedding_shape,
            polygon_dim=polygon_dim,
            action_dim=action_dim,
            hidden_sizes=self.config.critic_hidden_sizes,
            projected_dim=self.embedding_dim,
            embedding_noise_std=self.config.embedding_noise_std,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.total_updates = 0
        self._interaction_count = 0
        self.warmup_steps = 0
        self.current_epoch = 0
        # Normalise guidance mode and keep legacy flags in sync for backward compatibility.
        normalized_mode = str(self.config.guidance_mode).lower()
        valid_modes = {
            "critic_guidance",
            "true_guidance",
            "mixed",
            "true_guidance_post_noise",
            "random_true_guidance",
        }
        if normalized_mode not in valid_modes:
            raise ValueError(
                "guidance_mode must be one of 'critic_guidance', 'true_guidance', "
                "'mixed', or 'true_guidance_post_noise'."
            )
        self.config.guidance_mode = normalized_mode
        self.config.guided_exploration = normalized_mode in ("critic_guidance", "mixed")
        self.config.true_guided_exploration = normalized_mode in (
            "true_guidance",
            "true_guidance_post_noise",
            "mixed",
            "random_true_guidance",
        )
        self.config.mixed_guidance_steps = int(max(0, self.config.mixed_guidance_steps))

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device_arg = kwargs.get("device", None)
        if device_arg is None and len(args) == 1:
            device_arg = args[0]
        if isinstance(device_arg, torch.device):
            self.device = device_arg
        return module

    def _guided_exploration_adjust(
        self,
        embedding_map: torch.Tensor,
        polygon_state: torch.Tensor,
        noisy_action: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """Adjust a noisy action using the critic gradient if it improves value."""

        if not self.config.guided_exploration:
            return noisy_action

        guidance_scale = self._current_guidance_scale()
        if guidance_scale <= 0.0:
            return noisy_action

        poly = polygon_state.to(self.device).detach()
        embedding_map = embedding_map.to(self.device).detach()

        with torch.enable_grad():
            action_var = noisy_action.detach().clone().requires_grad_(True)
            encoded_q1 = self.critic.encode_q1(embedding_map, apply_noise=False)
            encoded_q1 = encoded_q1.detach()
            q1 = self.critic.q1_forward_from_encoded(encoded_q1, poly, action_var)
            q_min = q1
            grad = torch.autograd.grad(q_min.sum(), action_var, retain_graph=False, allow_unused=False)[0]

        if grad is None:
            return noisy_action

        grad = grad.detach()
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = guidance_scale * (grad / grad_norm)
        candidate_action = (noisy_action + direction).clamp(-1.0, 1.0)

        with torch.no_grad():
            q_min_original = q_min.detach()
            q1_new = self.critic.q1_forward_from_encoded(encoded_q1, poly, candidate_action)
            q_min_new = q1_new

        improved = (q_min_new > q_min_original).view(-1, 1)
        return torch.where(improved, candidate_action, noisy_action)

    @torch.no_grad()
    def act(
        self,
        embedding: torch.Tensor,
        polygon_state: torch.Tensor,
        deterministic: bool = False,
        apply_embedding_noise: bool = True,
        guided_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.actor.eval()
        mode = self._resolve_guidance_mode()
        use_gradient_guidance = mode == "critic_guidance"
        use_true_pre = mode in {"true_guidance", "random_true_guidance"}
        use_true_post = mode == "true_guidance_post_noise"
        use_random_true = mode == "random_true_guidance"

        embedding = embedding.to(self.device)
        polygon_state = polygon_state.to(self.device)

        if use_gradient_guidance:
            self.critic.eval()

        encoded = self.actor.encode(embedding, apply_noise=apply_embedding_noise)
        action = self.actor.forward_from_encoded(encoded, polygon_state)

        guidance_scale_value = self._current_guidance_scale()
        guidance_scale_tensor: Optional[torch.Tensor] = None
        if (use_true_pre or use_true_post or use_random_true) and guided_targets is not None:
            if use_random_true:
                guidance_scale_tensor = torch.rand(action.size(0), device=action.device, dtype=action.dtype).view(-1, 1)
            else:
                guidance_scale_tensor = torch.full((action.size(0), 1), guidance_scale_value, device=action.device, dtype=action.dtype)

        if use_true_pre and guided_targets is not None and guidance_scale_tensor is not None:
            if torch.any(guidance_scale_tensor > 0.0):
                tgt = guided_targets.to(action.device).clamp(-1.0, 1.0)
                action = (action + guidance_scale_tensor * (tgt - action)).clamp(-1.0, 1.0)

        if not deterministic:
            sigma = self._current_exploration_sigma()
            if sigma > 0:
                noise = torch.randn_like(action) * sigma
                noisy_action = action + noise
                if use_gradient_guidance:
                    noisy_action = self._guided_exploration_adjust(embedding, polygon_state, noisy_action, sigma)
                elif use_true_post and guided_targets is not None and guidance_scale_tensor is not None:
                    if torch.any(guidance_scale_tensor > 0.0):
                        tgt = guided_targets.to(action.device).clamp(-1.0, 1.0)
                        noisy_action = (noisy_action + guidance_scale_tensor * (tgt - noisy_action)).clamp(-1.0, 1.0)
                action = noisy_action
            self._interaction_count += embedding.size(0)
        else:
            if use_true_post and guided_targets is not None and guidance_scale_tensor is not None:
                if torch.any(guidance_scale_tensor > 0.0):
                    tgt = guided_targets.to(action.device).clamp(-1.0, 1.0)
                    action = (action + guidance_scale_tensor * (tgt - action)).clamp(-1.0, 1.0)

        return action.clamp_(-1.0, 1.0)

    def update(self, batch: Dict[str, torch.Tensor], weights: Optional[torch.Tensor] = None) -> Dict[str, float | torch.Tensor]:
        self.actor.train()
        self.critic.train()

        # Apply LR schedules (linear decay) using epoch index as progress step
        if self.config.critic_lr_schedule is not None:
            lr_c = self._current_lr(self.config.critic_lr_schedule)
            for g in self.critic_opt.param_groups:
                g["lr"] = lr_c
        if self.config.actor_lr_schedule is not None:
            lr_a = self._current_lr(self.config.actor_lr_schedule)
            for g in self.actor_opt.param_groups:
                g["lr"] = lr_a

        embedding_map = batch["embedding"].to(self.device)
        polygon = batch["polygon"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        discount = batch["discount"].to(self.device)
        next_polygon = batch["next_polygon"].to(self.device)
        if weights is None:
            weights = torch.ones_like(reward)
        if weights.dim() == 1:
            weights = weights.view(-1, 1)
        weights = weights.to(self.device)

        # Critics update -----------------------------------------------------
        current_q1, current_q2 = self.critic.forward(embedding_map, polygon, action, apply_noise=True)

        with torch.no_grad():
            target_action = self.actor_target.forward(embedding_map, next_polygon, apply_noise=True)
            if self.config.target_policy_noise_std > 0:
                noise = torch.randn_like(target_action) * self.config.target_policy_noise_std
                noise = noise.clamp_(-self.config.target_policy_noise_clip, self.config.target_policy_noise_clip)
                target_action = target_action + noise
            target_action = target_action.clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target.forward(embedding_map, next_polygon, target_action, apply_noise=True)
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
        guided_target = batch.get("guided_target")
        if guided_target is not None:
            guided_target = guided_target.to(self.device)
        guided_available = batch.get("guided_available")
        if guided_available is not None:
            guided_available = guided_available.to(self.device)

        guided_actor_mse: Optional[torch.Tensor] = None

        skip_guided_loss = False
        if self.config.guidance_mode == "mixed" and self._interaction_count >= self.config.mixed_guidance_steps:
            skip_guided_loss = True

        if update_actor:
            actor_encoded = self.actor.encode(embedding_map, apply_noise=True)
            actor_action = self.actor.forward_from_encoded(actor_encoded, polygon)
            critic_encoded_q1 = self.critic.encode_q1(embedding_map, apply_noise=True)
            actor_loss = -self.critic.q1_forward_from_encoded(critic_encoded_q1, polygon, actor_action).mean()

            if (
                self.config.guided_actor_loss
                and guided_target is not None
                and guided_available is not None
                and not skip_guided_loss
            ):
                mask = guided_available.view(-1)
                if mask.sum() > 0:
                    diff = actor_action - guided_target
                    mse_per_sample = diff.pow(2).mean(dim=-1)
                    weighted_mse = (mse_per_sample * mask).sum() / mask.sum()
                    guided_actor_mse = weighted_mse
                    actor_loss = actor_loss + (self.config.guided_actor_loss_weight * weighted_mse)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_opt.step()
            actor_loss_value = float(actor_loss.item())
            metrics["actor_loss"] = actor_loss_value

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        if guided_actor_mse is not None:
            metrics["guided_actor_mse"] = float(guided_actor_mse.item())

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

    def _current_guidance_scale(self) -> float:
        schedule = self.config.guidance_schedule
        init = float(schedule.scale_init)
        final = float(schedule.scale_final)
        if self._interaction_count < self.warmup_steps:
            return init
        progressed = self._interaction_count - self.warmup_steps
        raw_steps = int(schedule.steps)
        if raw_steps <= 0:
            return final
        steps = max(1, raw_steps)
        progress = min(1.0, progressed / float(steps))
        return float(init + (final - init) * progress)

    def _current_lr(self, schedule: LRScheduleConfig) -> float:
        """Return current LR under a linear schedule based on epoch index."""
        step = float(self.current_epoch)
        steps = max(1.0, float(schedule.steps))
        progress = min(1.0, max(0.0, step / steps))
        return float(schedule.lr_init + (schedule.lr_final - schedule.lr_init) * progress)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(0, int(epoch))

    def _resolve_guidance_mode(self) -> str:
        mode = self.config.guidance_mode
        if mode == "mixed":
            if self._interaction_count < self.config.mixed_guidance_steps:
                return "true_guidance"
            return "critic_guidance"
        return mode

    @property
    def requires_guided_targets(self) -> bool:
        return self.config.guidance_mode in {
            "true_guidance",
            "true_guidance_post_noise",
            "mixed",
            "random_true_guidance",
        }

    def is_true_guidance_active(self) -> bool:
        return self._resolve_guidance_mode() in {
            "true_guidance",
            "true_guidance_post_noise",
            "random_true_guidance",
        }

    def set_warmup_steps(self, warmup_steps: int) -> None:
        self.warmup_steps = max(0, int(warmup_steps))
