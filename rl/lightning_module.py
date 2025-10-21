from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pytorch_lightning as pl
import torch
import imageio.v2 as imageio
import math

from .agent import TD3Agent, TD3Config, NoiseScheduleConfig, GuidanceScheduleConfig
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
        if "guidance_schedule" in algo_cfg and isinstance(algo_cfg["guidance_schedule"], dict):
            algo_cfg["guidance_schedule"] = GuidanceScheduleConfig(**algo_cfg["guidance_schedule"])
        if "actor_hidden_sizes" in algo_cfg:
            algo_cfg["actor_hidden_sizes"] = tuple(algo_cfg["actor_hidden_sizes"])
        if "critic_hidden_sizes" in algo_cfg:
            algo_cfg["critic_hidden_sizes"] = tuple(algo_cfg["critic_hidden_sizes"])
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
        )
        self.agent.set_warmup_steps(self.training_config.warmup_steps)

        self.embedding_dim = self.agent.embedding_dim
        self.replay = ReplayBuffer(
            capacity=replay_capacity,
            embedding_dim=self.embedding_dim,
            polygon_dim=polygon_dim,
            action_dim=self.environment.action_dim,
            alpha=self.algo_config.pr_alpha,
            beta_start=self.algo_config.pr_beta_start,
            beta_steps=self.algo_config.pr_beta_steps,
            eps=self.algo_config.pr_eps,
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
            self.agent.actor_opt.load_state_dict(actor_state)
        if critic_state is not None:
            self.agent.critic_opt.load_state_dict(critic_state)

    def configure_optimizers(self):
        # Optimisers are managed internally by the agent.
        return []

    def training_step(self, batch, batch_idx: int):
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        batch_size = images.size(0)

        embedding_maps = batch["embedding"].to(self.device, non_blocking=True)
        with torch.no_grad():
            embeddings = self.agent.preprocess_embeddings(embedding_maps)
        embeddings_cpu = embeddings.detach().cpu()

        target_polygon_cpu = batch.get("target_polygon_state")
        target_polygon = None
        if target_polygon_cpu is not None:
            target_polygon_cpu = target_polygon_cpu.to(torch.float32)
            target_polygon = target_polygon_cpu.to(self.device, non_blocking=True)
        if self.agent.config.true_guided_exploration and target_polygon is None:
            raise RuntimeError(
                "true_guided_exploration is enabled but dataset did not supply 'target_polygon_state'."
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
        guidance_scale_sum = 0.0
        guidance_scale_count = 0

        action_norm_total = 0.0
        distance_norm_total = 0.0
        angle_norm_total = 0.0
        action_measure_count = 0

        def _maybe_run_updates() -> None:
            nonlocal performed_updates, critic_loss_sum, actor_loss_sum
            nonlocal critic_update_count, actor_update_count, guidance_scale_sum, guidance_scale_count
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
                guidance_value = metrics.get("guidance_scale")
                if guidance_value is not None:
                    if isinstance(guidance_value, torch.Tensor):
                        guidance_scalar = float(guidance_value.detach().mean().item())
                    else:
                        guidance_scalar = float(guidance_value)
                    self.log(
                        "train/guidance_scale_step",
                        guidance_scalar,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                        sync_dist=False,
                    )
                    guidance_scale_sum += guidance_scalar
                    guidance_scale_count += 1
                if "td_errors" in metrics:
                    self.replay.update_priorities(batch_indices, metrics["td_errors"])
                performed_updates += 1

        for _ in range(max_collect_steps):
            if not alive_mask.any():
                break

            batch_steps_completed += 1
            active_indices = alive_mask.nonzero(as_tuple=False).squeeze(1)
            prev_polygon_cpu = polygon_state_cpu.clone()

            actions = torch.zeros(batch_size, self.environment.action_dim, device=self.device)

            guidance_targets_full = None
            if self.agent.config.true_guided_exploration and target_polygon is not None:
                current_iou = None
                if self.environment.last_iou is not None:
                    current_iou = self.environment.last_iou.detach().to(self.device, dtype=polygon_state.dtype)
                guidance_targets_full = self._compute_true_guidance_targets(
                    polygon_state,
                    target_polygon,
                    current_iou=current_iou,
                )

            guided_subset = guidance_targets_full[active_indices] if guidance_targets_full is not None else None
            chosen_actions, _ = self.agent.act(
                embeddings[active_indices],
                polygon_state[active_indices],
                deterministic=False,
                apply_embedding_noise=False if self.agent.config.true_guided_exploration else True,
                guided_targets=guided_subset,
                return_base_action=True,
            )
            actions[active_indices] = chosen_actions
            actions_cpu = actions.detach().cpu()
            guidance_targets_cpu = guidance_targets_full.detach().cpu() if guidance_targets_full is not None else None

            action_norm_total += float(actions.norm(dim=-1).sum().item())
            action_measure_count += actions.size(0)
            if self.line_action_dim > 0:
                line_actions = actions[:, : self.line_action_dim].view(batch_size, self.environment.num_lines, 2)
                distance_norm_total += float(torch.linalg.norm(line_actions[..., 0], dim=-1).sum().item())
                angle_norm_total += float(torch.linalg.norm(line_actions[..., 1], dim=-1).sum().item())

            alive_before = alive_mask.clone()
            next_polygon_cpu, reward_cpu, done_cpu, info = self.environment.step(actions_cpu)

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

            cumulative_rewards[active_indices] += reward[active_indices]
            steps_taken[active_indices] += 1.0

            newly_done = done_bool & alive_before
            if newly_done.any():
                final_iou = torch.where(newly_done.to(self.device), iou, final_iou)
            success_flags = success_flags | success_device

            for env_idx in active_indices.tolist():
                guidance_target_single = None
                if guidance_targets_cpu is not None:
                    guidance_target_single = guidance_targets_cpu[env_idx]
                reward_tensor = reward_cpu[env_idx].view(1)
                done_tensor = done_cpu[env_idx].view(1).to(torch.float32)
                next_polygon_single = None if done_bool[env_idx].item() else next_polygon_cpu[env_idx]
                step_tuple = StepTuple(
                    embedding=embeddings_cpu[env_idx],
                    polygon=prev_polygon_cpu[env_idx],
                    action=actions_cpu[env_idx],
                    reward=reward_tensor,
                    next_polygon=next_polygon_single,
                    done=done_tensor,
                    guidance_target=guidance_target_single,
                )
                aggregated = accumulator.push(env_idx, step_tuple)
                for embedding_t, polygon_t, action_t, reward_t, next_polygon_t, done_flag_t, discount_t, guidance_target_t in aggregated:
                    transition = Transition(
                        embedding=embedding_t,
                        polygon_state=polygon_t,
                        action=action_t,
                        reward=reward_t.view(1),
                        discount=discount_t.view(1),
                        next_polygon_state=next_polygon_t,
                        done=done_flag_t.view(1),
                        guidance_target=guidance_target_t,
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
            for embedding_t, polygon_t, action_t, reward_t, next_polygon_t, done_flag_t, discount_t, guidance_target_t in leftovers:
                transition = Transition(
                    embedding=embedding_t,
                    polygon_state=polygon_t,
                    action=action_t,
                    reward=reward_t.view(1),
                    discount=discount_t.view(1),
                    next_polygon_state=next_polygon_t,
                    done=done_flag_t.view(1),
                    guidance_target=guidance_target_t,
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
        norm_metrics = {
            "train/action_norm": action_norm_total / norm_count,
            "train/distance_norm": distance_norm_total / norm_count,
            "train/angle_norm": angle_norm_total / norm_count,
        }

        mean_critic_loss = (
            float(critic_loss_sum / critic_update_count) if critic_update_count > 0 else 0.0
        )
        mean_actor_loss = (
            float(actor_loss_sum / actor_update_count) if actor_update_count > 0 else 0.0
        )
        guidance_avg = guidance_scale_sum / guidance_scale_count if guidance_scale_count > 0 else 0.0

        self.log_dict(
            {
                "train/avg_reward": cumulative_rewards.mean(),
                "train/avg_steps": steps_taken.mean(),
                "train/final_iou": final_iou.mean(),
                "train/success_rate": success_flags.float().mean(),
                "train/critic_loss": mean_critic_loss,
                "train/actor_loss": mean_actor_loss,
                "train/guidance_scale": guidance_avg,
                "train/buffer_size": float(len(self.replay)),
                "train/updates": float(performed_updates),
                "train/transitions": float(transitions_added),
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

        diff_x = center_diff[:, 0]
        diff_y = center_diff[:, 1]
        up_action = torch.clamp(-diff_y / center_step_scale, min=0.0, max=1.0)
        down_action = torch.clamp(diff_y / center_step_scale, min=0.0, max=1.0)
        left_action = torch.clamp(-diff_x / center_step_scale, min=0.0, max=1.0)
        right_action = torch.clamp(diff_x / center_step_scale, min=0.0, max=1.0)
        center_actions = torch.stack([up_action, down_action, left_action, right_action], dim=1)

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
        with torch.no_grad():
            embeddings = self.agent.preprocess_embeddings(embedding_maps)

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
            selected_actions = self.agent.act(
                embeddings[active_indices],
                state[active_indices],
                deterministic=deterministic,
                apply_embedding_noise=False,
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
