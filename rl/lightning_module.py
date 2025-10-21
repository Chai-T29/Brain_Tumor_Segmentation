from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pytorch_lightning as pl
import torch
import imageio.v2 as imageio
import math

from .agent import (
    TD3Agent,
    TD3Config,
    NoiseScheduleConfig,
    GuidanceScheduleConfig,
    LRScheduleConfig,
)
from .environment import EnvironmentConfig, PolygonLocalizationEnv
from .n_step import NStepAccumulator, StepTuple
from .replay_buffer import ReplayBuffer, Transition


@dataclass
class TrainingConfig:
    update_batch_size: int = 128
    update_every_n_steps: int = 2
    collect_steps_per_batch: Optional[int] = None
    warmup_steps: int = 1000
    precision: str = "32-true"
    max_epochs: int = 40
    log_interval: int = 50


class TD3Lightning(pl.LightningModule):
    def __init__(
        self,
        embedding_shape: Tuple[int, int, int],
        env_cfg: Dict,
        algo_cfg: Dict,
        training_cfg: Dict,
        replay_capacity: int,
        logging_cfg: Optional[Dict] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if len(embedding_shape) != 3:
            raise ValueError("embedding_shape must be a tuple of (C, H, W).")
        self.embedding_shape = tuple(int(v) for v in embedding_shape)
        self.env_config = EnvironmentConfig(**env_cfg)
        self.environment = PolygonLocalizationEnv(self.env_config)
        self.line_action_dim = self.environment.line_action_dim
        env_config_copy = EnvironmentConfig(**asdict(self.env_config))
        self.val_env = PolygonLocalizationEnv(env_config_copy)
        self.test_env = PolygonLocalizationEnv(EnvironmentConfig(**asdict(self.env_config)))

        algo_cfg = dict(algo_cfg)
        if "exploration_noise" in algo_cfg and isinstance(algo_cfg["exploration_noise"], dict):
            algo_cfg["exploration_noise"] = NoiseScheduleConfig(**algo_cfg["exploration_noise"])
        if "actor_hidden_sizes" in algo_cfg:
            algo_cfg["actor_hidden_sizes"] = tuple(algo_cfg["actor_hidden_sizes"])
        if "critic_hidden_sizes" in algo_cfg:
            algo_cfg["critic_hidden_sizes"] = tuple(algo_cfg["critic_hidden_sizes"])
        if "guidance_schedule" in algo_cfg and isinstance(algo_cfg["guidance_schedule"], dict):
            algo_cfg["guidance_schedule"] = GuidanceScheduleConfig(**algo_cfg["guidance_schedule"])
        elif "guidance_scale" in algo_cfg:
            raw = algo_cfg.pop("guidance_scale")
            if isinstance(raw, dict):
                initial = raw.get("initial", raw.get("scale_init", 0.0))
                final = raw.get("final", raw.get("scale_final", 0.0))
                steps = raw.get("steps", raw.get("scale_steps", 0))
                algo_cfg["guidance_schedule"] = GuidanceScheduleConfig(
                    scale_init=float(initial),
                    scale_final=float(final),
                    steps=int(steps),
                )
            else:
                value = float(raw)
                algo_cfg["guidance_schedule"] = GuidanceScheduleConfig(
                    scale_init=value,
                    scale_final=value,
                    steps=1,
                )
        if "guidance_mode" not in algo_cfg:
            if algo_cfg.get("true_guided_exploration", False):
                algo_cfg["guidance_mode"] = "true_guidance"
            elif algo_cfg.get("guided_exploration", False):
                algo_cfg["guidance_mode"] = "critic_guidance"
            else:
                algo_cfg["guidance_mode"] = "critic_guidance"
        else:
            algo_cfg["guidance_mode"] = str(algo_cfg["guidance_mode"]).lower()
        if "mixed_guidance_steps" not in algo_cfg:
            algo_cfg["mixed_guidance_steps"] = 500000
        # LR schedules (optional)
        if "actor_lr_schedule" in algo_cfg and isinstance(algo_cfg["actor_lr_schedule"], dict):
            algo_cfg["actor_lr_schedule"] = LRScheduleConfig(**algo_cfg["actor_lr_schedule"])
        if "critic_lr_schedule" in algo_cfg and isinstance(algo_cfg["critic_lr_schedule"], dict):
            algo_cfg["critic_lr_schedule"] = LRScheduleConfig(**algo_cfg["critic_lr_schedule"])
        self.algo_config = TD3Config(**algo_cfg)
        self.training_config = TrainingConfig(**training_cfg)
        self.logging_cfg = logging_cfg or {}
        self.verbose = bool(verbose)
        self.test_gif_limit = int(self.logging_cfg.get("test_gif_limit", 10))
        self.test_gif_fps = int(self.logging_cfg.get("test_gif_fps", 4))
        gif_dir = self.logging_cfg.get("test_gif_dir", "lightning_logs/test_gifs")
        self._gif_output_dir = Path(gif_dir)
        self._test_episode_index = 0
        self._test_epoch_outputs: List[Dict[str, torch.Tensor]] = []

        polygon_dim = self.environment.state_dim
        self.agent = TD3Agent(
            embedding_shape=self.embedding_shape,
            polygon_dim=polygon_dim,
            action_dim=self.environment.action_dim,
            config=self.algo_config,
            embedding_pointer_mode=self.algo_config.use_embedding_pointers,
        )
        self.agent.set_warmup_steps(self.training_config.warmup_steps)

        self.embedding_shape = self.agent.embedding_shape
        self.replay = ReplayBuffer(
            capacity=replay_capacity,
            embedding_shape=self.embedding_shape,
            polygon_dim=polygon_dim,
            action_dim=self.environment.action_dim,
            alpha=self.algo_config.pr_alpha,
            beta_start=self.algo_config.pr_beta_start,
            beta_steps=self.algo_config.pr_beta_steps,
            eps=self.algo_config.pr_eps,
            use_embedding_pointers=self.algo_config.use_embedding_pointers,
        )
        self._global_step_interactions = 0

        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        self.agent.to(self.device)

    def on_validation_start(self) -> None:
        self.agent.to(self.device)

    def on_test_start(self) -> None:
        self.agent.to(self.device)

    def on_save_checkpoint(self, checkpoint: Dict[str, any]) -> None:
        checkpoint["actor_opt_state"] = self.agent.actor_opt.state_dict()
        checkpoint["critic_opt_state"] = self.agent.critic_opt.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, any]) -> None:
        actor_state = checkpoint.get("actor_opt_state")
        critic_state = checkpoint.get("critic_opt_state")
        if actor_state is not None:
            try:
                self.agent.actor_opt.load_state_dict(actor_state)
            except ValueError:
                print("[Warning] Actor optimizer state did not match current architecture; skipping load.")
        if critic_state is not None:
            try:
                self.agent.critic_opt.load_state_dict(critic_state)
            except ValueError:
                print("[Warning] Critic optimizer state did not match current architecture; skipping load.")

    def configure_optimizers(self):
        # Optimisers are managed internally by the agent.
        return []

    def training_step(self, batch, batch_idx: int):
        self.agent.set_epoch(self.current_epoch)
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        batch_size = images.size(0)

        embedding_maps = batch["embedding"].to(self.device, non_blocking=True)
        meta_batch = batch.get("meta")
        if isinstance(meta_batch, list):
            processed_meta: Dict[str, list] = {}
            for item in meta_batch:
                if isinstance(item, dict):
                    for key, value in item.items():
                        processed_meta.setdefault(key, []).append(value)
            meta_batch = processed_meta
        pointer_enabled = getattr(self.replay, "use_embedding_pointers", False)
        embedding_maps_cpu = None if pointer_enabled else embedding_maps.detach().cpu()

        def _resolve_meta_value(container, idx: int):
            if container is None:
                return None
            value = container
            if isinstance(value, (list, tuple)):
                if idx >= len(value):
                    return None
                value = value[idx]
            elif isinstance(value, np.ndarray):
                if value.size == 0:
                    return None
                value = value[idx]
            elif torch.is_tensor(value):
                if value.numel() == 0:
                    return None
                elem = value[idx]
                if elem.numel() == 1:
                    return elem.item()
                return elem.item()
            return value

        def _embedding_reference(idx: int):
            if not pointer_enabled:
                if embedding_maps_cpu is None:
                    raise RuntimeError("embedding_maps_cpu unavailable for pointer-disabled mode.")
                return embedding_maps_cpu[idx]
            if not isinstance(meta_batch, dict):
                return embedding_maps[idx].detach().cpu()
            path = meta_batch.get("embedding_mm_path")
            if path is None:
                path = meta_batch.get("embedding_path")
            slice_val = meta_batch.get("slice_index")
            path_resolved = _resolve_meta_value(path, idx)
            if path_resolved is None:
                return embedding_maps[idx].detach().cpu()
            slice_resolved = _resolve_meta_value(slice_val, idx)
            slice_resolved = int(slice_resolved) if slice_resolved is not None else 0
            return {"path": str(path_resolved), "slice_index": slice_resolved}

        target_polygon_cpu = batch.get("target_polygon_state")
        target_polygon = None
        if target_polygon_cpu is not None:
            target_polygon_cpu = target_polygon_cpu.to(torch.float32)
            target_polygon = target_polygon_cpu.to(self.device, non_blocking=True)
        if self.agent.requires_guided_targets and target_polygon is None:
            raise RuntimeError(
                "Selected guidance mode requires 'target_polygon_state', but the dataset did not provide it."
            )

        polygon_state_cpu = self.environment.reset(images.cpu(), masks.cpu())
        polygon_state = polygon_state_cpu.to(self.device)

        avg_gt_iou: float | None = None
        if self.verbose and target_polygon_cpu is not None:
            gt_iou_tensor = self._compute_target_iou(target_polygon_cpu)
            if gt_iou_tensor is not None:
                avg_gt_iou = float(gt_iou_tensor.mean().item())

        if self.verbose:
            start_msg = (
                f"[Verbose][Train] epoch={self.current_epoch} batch={batch_idx} start -- batch_size={batch_size}"
            )
            if avg_gt_iou is not None:
                start_msg += f" gt_iou={avg_gt_iou:.4f}"
            print(start_msg)

        accumulator = NStepAccumulator(
            n_step=self.algo_config.n_step,
            gamma=self.algo_config.gamma,
            num_envs=batch_size,
        )

        alive_mask = torch.ones(batch_size, dtype=torch.bool)
        cumulative_rewards = torch.zeros(batch_size, device=self.device)
        steps_taken = torch.zeros(batch_size, device=self.device)
        final_iou = torch.zeros(batch_size, device=self.device)
        success_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        transitions_added = 0
        batch_steps_completed = 0
        max_collect_steps = self.training_config.collect_steps_per_batch or self.env_config.max_steps

        updates_trigger = max(1, self.training_config.update_every_n_steps)
        performed_updates = 0
        critic_loss_sum = 0.0
        actor_loss_sum = 0.0
        critic_update_count = 0
        actor_update_count = 0
        guided_actor_mse_sum = 0.0
        guided_actor_mse_count = 0

        action_norm_total = 0.0
        distance_norm_total = 0.0
        angle_norm_total = 0.0
        action_measure_count = 0
        # Extra diagnostics: stop usage and guidance reliance
        stop_count_total = 0
        active_step_count = 0
        base_action_norm_total = 0.0
        base_action_count = 0
        guidance_diff_norm_total = 0.0
        guidance_diff_count = 0

        def _maybe_run_updates() -> None:
            nonlocal performed_updates, critic_loss_sum, actor_loss_sum
            nonlocal critic_update_count, actor_update_count
            nonlocal guided_actor_mse_sum, guided_actor_mse_count
            while performed_updates < batch_steps_completed // updates_trigger:
                if len(self.replay) < self.training_config.update_batch_size:
                    break
                batch_samples, batch_indices, batch_weights = self.replay.sample(
                    self.training_config.update_batch_size, device=self.device
                )
                metrics = self.agent.update(batch_samples, weights=batch_weights)
                critic_loss_sum += metrics["critic_loss"]
                critic_update_count += 1
                if "actor_loss" in metrics:
                    actor_loss_sum += metrics["actor_loss"]
                    actor_update_count += 1
                if "guided_actor_mse" in metrics:
                    guided_actor_mse_sum += float(metrics["guided_actor_mse"])
                    guided_actor_mse_count += 1
                if "td_errors" in metrics:
                    self.replay.update_priorities(batch_indices, metrics["td_errors"])
                performed_updates += 1

        actor_encoding_cache: torch.Tensor | None = None
        critic_q1_cache: torch.Tensor | None = None
        cache_version = -1
        cached_guidance_mode: str | None = None

        def _refresh_actor_cache(current_mode: str) -> None:
            nonlocal actor_encoding_cache, critic_q1_cache, cache_version, cached_guidance_mode
            with torch.no_grad():
                actor_encoding_cache = self.agent.actor.encode(embedding_maps, apply_noise=False).detach()
                cache_version = self.agent.total_updates
                cached_guidance_mode = current_mode
                if current_mode == "critic_guidance":
                    critic_q1_cache = self.agent.critic.encode_q1(embedding_maps, apply_noise=False).detach()
                else:
                    critic_q1_cache = None

        def _ensure_caches() -> None:
            nonlocal actor_encoding_cache, critic_q1_cache, cache_version, cached_guidance_mode
            current_mode = self.agent._resolve_guidance_mode()
            if actor_encoding_cache is None or cache_version != self.agent.total_updates:
                _refresh_actor_cache(current_mode)
            else:
                if current_mode != cached_guidance_mode:
                    cached_guidance_mode = current_mode
                    if current_mode == "critic_guidance":
                        with torch.no_grad():
                            critic_q1_cache = self.agent.critic.encode_q1(embedding_maps, apply_noise=False).detach()
                    else:
                        critic_q1_cache = None

        for _ in range(max_collect_steps):
            if not alive_mask.any():
                break

            batch_steps_completed += 1
            active_indices = alive_mask.nonzero(as_tuple=False).squeeze(1)
            prev_polygon_cpu = polygon_state_cpu.clone()

            actions = torch.zeros(batch_size, self.environment.action_dim, device=self.device)

            guidance_targets_full = None
            if target_polygon is not None:
                current_iou = None
                if self.environment.last_iou is not None:
                    current_iou = self.environment.last_iou.detach().to(self.device, dtype=polygon_state.dtype)
                guidance_targets_full = self._compute_true_guidance_targets(
                    polygon_state,
                    target_polygon,
                    current_iou=current_iou,
                )

            base_policy_actions = None
            guided_subset = None
            active_idx_device: torch.Tensor | None = None
            if active_indices.numel() > 0:
                active_idx_device = active_indices.to(self.device)
                if guidance_targets_full is not None:
                    guided_subset = torch.index_select(guidance_targets_full, 0, active_idx_device)

                _ensure_caches()

                actor_encoded_active = torch.index_select(actor_encoding_cache, 0, active_idx_device)
                polygon_active = torch.index_select(polygon_state, 0, active_idx_device)

                with torch.no_grad():
                    base_policy_actions = self.agent.actor.forward_from_encoded(
                        actor_encoded_active,
                        polygon_active,
                    )
                    base_action_norm_total += float(base_policy_actions.norm(dim=-1).sum().item())
                    base_action_count += int(base_policy_actions.size(0))

                embedding_active = torch.index_select(embedding_maps, 0, active_idx_device)
                critic_encoded_active = None
                if critic_q1_cache is not None:
                    critic_encoded_active = torch.index_select(critic_q1_cache, 0, active_idx_device)

                chosen_actions = self.agent.act_from_encoded(
                    actor_encoded_active,
                    embedding_active,
                    polygon_active,
                    deterministic=False,
                    apply_embedding_noise=not self.agent.is_true_guidance_active(),
                    guided_targets=guided_subset,
                    encoded_q1=critic_encoded_active,
                )
                actions.index_copy_(0, active_idx_device, chosen_actions)
            actions_cpu = actions.detach().cpu()

            # Measure norms only over active envs to avoid bias from zeroed actions
            if active_indices.numel() > 0:
                if active_idx_device is None:
                    active_idx_device = active_indices.to(self.device)
                active_actions = torch.index_select(actions, 0, active_idx_device)
                action_norm_total += float(active_actions.norm(dim=-1).sum().item())
                action_measure_count += int(active_actions.size(0))
                if self.line_action_dim > 0:
                    line_actions = active_actions[:, : self.line_action_dim].view(-1, self.environment.num_lines, 2)
                    distance_norm_total += float(torch.linalg.norm(line_actions[..., 0], dim=-1).sum().item())
                    angle_norm_total += float(torch.linalg.norm(line_actions[..., 1], dim=-1).sum().item())

            alive_before = alive_mask.clone()
            next_polygon_cpu, reward_cpu, done_cpu, info = self.environment.step(actions_cpu)

            # Step-level diagnostics
            active_step_count += int(alive_before.sum().item())
            manual_stop = info.get("manual_stop")
            if manual_stop is not None:
                # Count only for active envs this step
                stop_count_total += int((manual_stop & alive_before).sum().item())

            if guided_subset is not None and base_policy_actions is not None and guided_subset.numel() > 0:
                # Measure how far the base policy is from guided targets on active envs
                gd = (guided_subset - base_policy_actions).detach()
                guidance_diff_norm_total += float(gd.norm(dim=-1).sum().item())
                guidance_diff_count += int(gd.size(0))

            if self.verbose:
                avg_iou_val = float(info["iou"].mean().item())
                avg_delta_val = float(info["delta_iou"].mean().item())
                avg_reward_val = float(reward_cpu.mean().item())
                log_msg = (
                    f"[Verbose][Train] epoch={self.current_epoch} batch={batch_idx} step={batch_steps_completed} "
                    f"avg_iou={avg_iou_val:.4f} delta_iou={avg_delta_val:.4f} reward={avg_reward_val:.4f}"
                )
                if avg_gt_iou is not None:
                    log_msg += f" gt_iou={avg_gt_iou:.4f}"
                print(log_msg)

            reward = reward_cpu.to(self.device)
            done_bool = done_cpu.to(torch.bool)
            success = info["success"].to(torch.bool)
            success_device = success.to(self.device)
            iou = info["iou"].to(self.device)

            if active_indices.numel() > 0:
                if active_idx_device is None:
                    active_idx_device = active_indices.to(self.device)
                reward_active = torch.index_select(reward, 0, active_idx_device)
                cumulative_rewards.index_add_(0, active_idx_device, reward_active)
                step_increments = torch.ones_like(reward_active)
                steps_taken.index_add_(0, active_idx_device, step_increments)

            newly_done = done_bool & alive_before
            if newly_done.any():
                final_iou = torch.where(newly_done.to(self.device), iou, final_iou)
            success_flags = success_flags | success_device

            for env_idx in active_indices.tolist():
                reward_tensor = reward_cpu[env_idx].view(1)
                done_tensor = done_cpu[env_idx].view(1).to(torch.float32)
                next_polygon_single = None if done_bool[env_idx].item() else next_polygon_cpu[env_idx]
                guidance_tensor = None
                if guidance_targets_full is not None:
                    guidance_tensor = guidance_targets_full[env_idx].detach().cpu()
                step_tuple = StepTuple(
                    embedding=_embedding_reference(env_idx),
                    polygon=prev_polygon_cpu[env_idx],
                    action=actions_cpu[env_idx],
                    reward=reward_tensor,
                    next_polygon=next_polygon_single,
                    done=done_tensor,
                    guided_target=guidance_tensor,
                )
                aggregated = accumulator.push(env_idx, step_tuple)
                for (
                    embedding_t,
                    polygon_t,
                    action_t,
                    reward_t,
                    next_polygon_t,
                    done_flag_t,
                    discount_t,
                    guidance_t,
                ) in aggregated:
                    transition = Transition(
                        embedding=embedding_t,
                        polygon_state=polygon_t,
                        action=action_t,
                        reward=reward_t.view(1),
                        discount=discount_t.view(1),
                        next_polygon_state=next_polygon_t,
                        done=done_flag_t.view(1),
                        guided_target=guidance_t,
                    )
                    self.replay.add(transition)
                    transitions_added += 1

            polygon_state_cpu = next_polygon_cpu
            polygon_state = polygon_state_cpu.to(self.device)

            alive_mask = alive_before & (~done_bool)
            self._global_step_interactions += int(active_indices.numel())

            _maybe_run_updates()

        leftover_transitions_added = 0
        for env_idx in range(batch_size):
            leftovers = accumulator.flush(env_idx)
            for (
                embedding_t,
                polygon_t,
                action_t,
                reward_t,
                next_polygon_t,
                done_flag_t,
                discount_t,
                guidance_t,
            ) in leftovers:
                transition = Transition(
                    embedding=embedding_t,
                    polygon_state=polygon_t,
                    action=action_t,
                    reward=reward_t.view(1),
                    discount=discount_t.view(1),
                    next_polygon_state=next_polygon_t,
                    done=done_flag_t.view(1),
                    guided_target=guidance_t,
                )
                self.replay.add(transition)
                transitions_added += 1
                leftover_transitions_added += 1

        if leftover_transitions_added > 0:
            _maybe_run_updates()

        if self.environment.last_iou is not None:
            last_iou = self.environment.last_iou.to(self.device)
            final_iou = torch.where(alive_mask.to(self.device), last_iou, final_iou)

        _maybe_run_updates()

        norm_count = max(1, action_measure_count)
        stop_rate = (stop_count_total / max(1, active_step_count)) if active_step_count > 0 else 0.0
        sigma_val = 0.0
        try:
            sigma_val = float(self.agent._current_exploration_sigma())
        except Exception:
            sigma_val = 0.0
        guidance_scale_val = 0.0
        try:
            guidance_scale_val = float(self.agent._current_guidance_scale())
        except Exception:
            guidance_scale_val = 0.0
        guided_actor_mse_mean = (
            guided_actor_mse_sum / guided_actor_mse_count if guided_actor_mse_count > 0 else 0.0
        )
        norm_metrics = {
            "train/action_norm": action_norm_total / norm_count,
            "train/distance_norm": distance_norm_total / norm_count,
            "train/angle_norm": angle_norm_total / norm_count,
            "train/base_action_norm": (base_action_norm_total / max(1, base_action_count)),
            "train/guided_diff_norm": (guidance_diff_norm_total / max(1, guidance_diff_count)),
            "train/stop_rate": stop_rate,
            "train/exploration_sigma": sigma_val,
            "train/guidance_scale": guidance_scale_val,
        }

        mean_critic_loss = (
            float(critic_loss_sum / critic_update_count) if critic_update_count > 0 else 0.0
        )
        mean_actor_loss = (
            float(actor_loss_sum / actor_update_count) if actor_update_count > 0 else 0.0
        )

        self.log_dict(
            {
                "train/avg_reward": cumulative_rewards.mean(),
                "train/avg_steps": steps_taken.mean(),
                "train/final_iou": final_iou.mean(),
                "train/success_rate": success_flags.float().mean(),
                "train/critic_loss": mean_critic_loss,
                "train/actor_loss": mean_actor_loss,
                "train/buffer_size": float(len(self.replay)),
                "train/updates": float(performed_updates),
                "train/transitions": float(transitions_added),
                "train/guided_actor_mse": guided_actor_mse_mean,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        self.log_dict(norm_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        if self.verbose:
            final_iou_mean = float(final_iou.mean().detach().cpu().item())
            end_msg = (
                f"[Verbose][Train] epoch={self.current_epoch} batch={batch_idx} complete -- final_iou={final_iou_mean:.4f}"
            )
            if avg_gt_iou is not None:
                end_msg += f" gt_iou={avg_gt_iou:.4f}"
            print(end_msg)

        return torch.tensor(mean_critic_loss, device=self.device)

    @staticmethod
    def _wrap_degrees(delta: torch.Tensor) -> torch.Tensor:
        """Wrap degree differences to [-180, 180]."""
        return (delta + 180.0).remainder(360.0) - 180.0



    def _compute_true_guidance_targets(
        self,
        current_state: torch.Tensor,
        target_state: torch.Tensor,
        current_iou: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute normalised action targets, including stop guidance when IoU is high."""
        device = current_state.device
        if target_state.device != device:
            target_state = target_state.to(device)

        num_lines = self.environment.num_lines
        if current_state.size(-1) < num_lines * 2 + 2 or target_state.size(-1) < num_lines * 2 + 2:
            raise ValueError("Current/target polygon state does not match expected dimensions.")

        distance_scale = max(1e-6, float(self.env_config.line_distance_step_scale))
        angle_step_rad = max(1e-6, math.radians(self.env_config.line_angle_step_scale_deg))
        center_step_scale = max(1e-6, float(self.env_config.center_step_scale))

        current_dist = current_state[:, :num_lines]
        current_angle = current_state[:, num_lines : 2 * num_lines]
        current_center = current_state[:, 2 * num_lines : 2 * num_lines + 2]

        target_dist = target_state[:, :num_lines]
        target_angle = target_state[:, num_lines : 2 * num_lines]
        target_center = target_state[:, 2 * num_lines : 2 * num_lines + 2]

        dist_diff = target_dist - current_dist
        angle_diff_deg = self._wrap_degrees(target_angle - current_angle)
        center_diff = target_center - current_center

        distance_actions = torch.clamp(dist_diff / distance_scale, min=-1.0, max=1.0)
        angle_diff_rad = torch.deg2rad(angle_diff_deg)
        angle_actions = torch.clamp(angle_diff_rad / angle_step_rad, min=-1.0, max=1.0)

        center_actions = torch.clamp(center_diff / center_step_scale, min=-1.0, max=1.0)

        distance_threshold = distance_scale
        angle_threshold = float(self.env_config.line_angle_step_scale_deg)

        dist_close = (dist_diff.abs() <= distance_threshold).all(dim=1)
        angle_close = (angle_diff_deg.abs() <= angle_threshold).all(dim=1)
        center_close = (center_diff.abs() <= center_step_scale).all(dim=1)
        close_enough = dist_close & angle_close & center_close

        stop_tensor = torch.full(
            (current_state.size(0), 1),
            -1.0,
            device=device,
            dtype=distance_actions.dtype,
        )
        stop_condition = torch.zeros(current_state.size(0), dtype=torch.bool, device=device)
        if current_iou is not None:
            iou_values = current_iou.detach().to(device=device, dtype=distance_actions.dtype).view(-1)
            high_threshold = float(self.env_config.iou_high_threshold)
            stop_condition = close_enough & (iou_values >= high_threshold)

        stop_tensor = torch.where(
            stop_condition.unsqueeze(1),
            torch.ones_like(stop_tensor),
            stop_tensor,
        )

        line_actions = torch.stack([distance_actions, angle_actions], dim=-1).reshape(current_state.size(0), num_lines * 2)
        guided_full = torch.cat([line_actions, center_actions, stop_tensor], dim=1)
        return guided_full.clamp(-1.0, 1.0)

    def _compute_target_iou(self, target_state: torch.Tensor | None) -> torch.Tensor | None:
        """Return IoU achieved by the provided target polygon state."""
        if target_state is None:
            return None

        env = self.environment
        if env.line_distances is None or env.line_angle_offsets is None:
            return None

        num_lines = env.num_lines
        device = env.line_distances.device
        dtype = env.line_distances.dtype

        target_state = target_state.to(device=device, dtype=dtype)

        saved_distances = env.line_distances.clone()
        saved_angles = env.line_angle_offsets.clone()
        saved_offsets = env.center_offsets.clone() if env.center_offsets is not None else None
        saved_positions = env.center_positions.clone() if env.center_positions is not None else None
        saved_vertices = env.vertices.clone() if env.vertices is not None else None
        saved_last_iou = env.last_iou.clone() if env.last_iou is not None else None

        env.line_distances.copy_(target_state[:, :num_lines])
        env.line_angle_offsets.copy_(torch.deg2rad(target_state[:, num_lines : 2 * num_lines]))

        if env.center_offsets is not None:
            env.center_offsets.copy_(target_state[:, 2 * num_lines : 2 * num_lines + 2])
            if env.base_center is not None:
                env.center_positions = env.base_center.unsqueeze(0) + env.center_offsets

        normals = env._compute_normals()
        poly_masks = env._rasterize_polytope(normals, env.line_distances)
        iou = env._calculate_iou_from_masks(poly_masks)

        env.line_distances.copy_(saved_distances)
        env.line_angle_offsets.copy_(saved_angles)
        if saved_offsets is not None and env.center_offsets is not None:
            env.center_offsets.copy_(saved_offsets)
        if saved_positions is not None:
            env.center_positions = saved_positions
        else:
            env.center_positions = None
        env.vertices = saved_vertices
        env.last_iou = saved_last_iou

        return iou

    def validation_step(self, batch, batch_idx: int):
        metrics, _ = self._simulate_environment(self.val_env, batch, deterministic=True, record=False)
        self._log_metrics(metrics, prefix="val")
        if self.verbose:
            mean_iou = float(metrics["mean_iou"].detach().cpu().item())
            avg_reward = float(metrics["avg_reward"].detach().cpu().item())
            print(
                f"[Verbose][Val] epoch={self.current_epoch} batch={batch_idx} avg_reward={avg_reward:.4f} mean_iou={mean_iou:.4f}"
            )
        return metrics

    def on_test_epoch_start(self) -> None:
        self._gif_output_dir.mkdir(parents=True, exist_ok=True)
        self._test_episode_index = 0
        self._test_epoch_outputs = []

    def test_step(self, batch, batch_idx: int):
        record = self._test_episode_index < self.test_gif_limit
        metrics, frames = self._simulate_environment(self.test_env, batch, deterministic=True, record=record)
        self._log_metrics(metrics, prefix="test")
        self._test_epoch_outputs.append(metrics)
        if self.verbose:
            mean_iou = float(metrics["mean_iou"].detach().cpu().item())
            avg_reward = float(metrics["avg_reward"].detach().cpu().item())
            print(
                f"[Verbose][Test] batch={batch_idx} avg_reward={avg_reward:.4f} mean_iou={mean_iou:.4f}"
            )

        if record and frames:
            reward_value = metrics["avg_reward"].detach().cpu().item()
            meta_batch = batch.get("meta") if isinstance(batch, dict) else None
            identifier = self._resolve_meta_value(meta_batch, "group_key")
            slice_idx = self._resolve_meta_value(meta_batch, "slice_index")
            if identifier is None:
                identifier = f"episode_{self._test_episode_index:03d}"
            if slice_idx is None:
                slice_idx = self._test_episode_index
            gif_name = (
                f"{identifier}_slice_{int(slice_idx):03d}_reward_{reward_value:.2f}.gif"
            )
            imageio.mimsave(self._gif_output_dir / gif_name, frames, fps=self.test_gif_fps)
            self._test_episode_index += 1

        return metrics

    def on_test_epoch_end(self) -> None:
        if not self._test_epoch_outputs:
            return
        device = self.device
        avg_reward = torch.stack([o["avg_reward"].detach().to(device) for o in self._test_epoch_outputs]).mean()
        mean_iou = torch.stack([o["mean_iou"].detach().to(device) for o in self._test_epoch_outputs]).mean()
        self.log("test/avg_reward_epoch", avg_reward, prog_bar=True)
        self.log("test/mean_iou_epoch", mean_iou, prog_bar=False)

    def _simulate_environment(
        self,
        env: PolygonLocalizationEnv,
        batch: Dict[str, torch.Tensor],
        deterministic: bool,
        record: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[List[np.ndarray]]]:
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        embedding_maps = batch["embedding"].to(self.device, non_blocking=True)
        target_polygon = None
        target_polygon_batch = batch.get("target_polygon_state")
        if target_polygon_batch is not None:
            target_polygon = target_polygon_batch.to(self.device, dtype=torch.float32)
        elif self.agent.requires_guided_targets:
            raise RuntimeError("Guided targets required by selected mode but not provided in dataset batch.")

        state_cpu = env.reset(images.cpu(), masks.cpu())
        state = state_cpu.to(self.device)

        batch_size = images.size(0)
        cumulative_rewards = torch.zeros(batch_size, device=self.device)
        steps_taken = torch.zeros(batch_size, device=self.device)
        success_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        iou_values: List[torch.Tensor] = []
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        frames: List[np.ndarray] = []

        if record:
            frame = env.render(index=0, mode="rgb_array")
            if frame is not None:
                frames.append(frame)

        max_steps = self.env_config.max_steps
        for _ in range(max_steps):
            if not active_mask.any():
                break

            actions = torch.zeros(batch_size, env.action_dim, device=self.device)
            active_indices = active_mask.nonzero(as_tuple=False).squeeze(1)

            guidance_targets_full = None
            if target_polygon is not None and active_indices.numel() > 0:
                current_iou = None
                if env.last_iou is not None:
                    current_iou = env.last_iou.detach().to(self.device, dtype=state.dtype)
                guidance_targets_full = self._compute_true_guidance_targets(
                    state,
                    target_polygon,
                    current_iou=current_iou,
                )

            guided_subset = None
            if guidance_targets_full is not None and active_indices.numel() > 0:
                guided_subset = guidance_targets_full[active_indices]

            selected_actions = self.agent.act(
                embedding_maps[active_indices],
                state[active_indices],
                deterministic=deterministic,
                apply_embedding_noise=False,
                guided_targets=guided_subset,
            )
            actions[active_indices] = selected_actions

            next_state_cpu, rewards_cpu, done_cpu, info = env.step(actions.detach().cpu())

            rewards = rewards_cpu.to(self.device)
            done_bool = done_cpu.to(torch.bool).to(self.device)
            success = info["success"].to(torch.bool).to(self.device)
            iou = info["iou"].to(self.device)

            cumulative_rewards += rewards
            steps_taken += active_mask.float()
            success_flags = success_flags | success
            iou_values.append(iou)

            state = next_state_cpu.to(self.device)
            active_mask = active_mask & (~done_bool)

            if record:
                frame = env.render(index=0, mode="rgb_array")
                if frame is not None:
                    frames.append(frame)

        mean_iou = torch.cat(iou_values).mean() if iou_values else torch.tensor(0.0, device=self.device)
        metrics = {
            "avg_reward": cumulative_rewards.mean(),
            "episode_length": steps_taken.mean(),
            "mean_iou": mean_iou,
            "success_rate": success_flags.float().mean(),
        }
        return metrics, (frames if record else None)

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], prefix: str) -> None:
        for key, value in metrics.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device)
            # NaN guard
            if torch.isnan(value).any():
                value = torch.nan_to_num(value, nan=0.0)
            if (prefix == "val") and (key == "mean_iou"):
                self.log(
                    "val_mean_iou",
                    value.detach(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )
                continue
            self.log(
                f"{prefix}/{key}",
                value.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=(key == "avg_reward"),
                sync_dist=True,
                add_dataloader_idx=False,
            )


    @staticmethod
    def _resolve_meta_value(meta_batch: Optional[Dict[str, Any]], key: str) -> Optional[Any]:
        if not isinstance(meta_batch, dict):
            return None
        value = meta_batch.get(key)
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return value.flatten()[0].item()
        return value
