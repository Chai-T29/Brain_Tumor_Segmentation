from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pytorch_lightning as pl
import torch

from .agent import TD3Agent, TD3Config, NoiseScheduleConfig
from .encoder import build_encoder, EncoderConfig, EfficientNetEncoder
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
        encoder_cfg: Dict,
        env_cfg: Dict,
        algo_cfg: Dict,
        training_cfg: Dict,
        replay_capacity: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder, self.encoder_config = build_encoder(encoder_cfg)
        self.env_config = EnvironmentConfig(**env_cfg)
        self.environment = PolygonLocalizationEnv(self.env_config)

        algo_cfg = dict(algo_cfg)
        if "exploration_noise" in algo_cfg and isinstance(algo_cfg["exploration_noise"], dict):
            algo_cfg["exploration_noise"] = NoiseScheduleConfig(**algo_cfg["exploration_noise"])
        self.algo_config = TD3Config(**algo_cfg)
        self.training_config = TrainingConfig(**training_cfg)

        polygon_dim = self.env_config.num_sides * 4
        self.agent = TD3Agent(
            embedding_dim=self.encoder.embedding_dim,
            polygon_dim=polygon_dim,
            action_dim=self.environment.action_dim,
            config=self.algo_config,
        )

        self.replay = ReplayBuffer(
            capacity=replay_capacity,
            embedding_dim=self.encoder.embedding_dim,
            polygon_dim=polygon_dim,
            action_dim=self.environment.action_dim,
        )
        self._global_step_interactions = 0

        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        self.agent.to(self.device)
        self.encoder.to(self.device)

    def configure_optimizers(self):
        # Optimisers are managed internally by the agent.
        return []

    def training_step(self, batch, batch_idx: int):
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        batch_size = images.size(0)

        # Compute base embeddings without augmentation.
        with torch.no_grad():
            embeddings = self.encoder.embed_without_noise(images)
        embeddings_cpu = embeddings.detach().cpu()

        polygon_state_cpu = self.environment.reset(images.cpu(), masks.cpu())
        polygon_state = polygon_state_cpu.to(self.device)

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
        max_collect_steps = self.training_config.collect_steps_per_batch or self.env_config.max_steps

        for _ in range(max_collect_steps):
            if not alive_mask.any():
                break

            active_indices = alive_mask.nonzero(as_tuple=False).squeeze(1)
            prev_polygon_cpu = polygon_state_cpu.clone()

            actions = torch.zeros(batch_size, self.environment.action_dim, device=self.device)
            chosen_actions = self.agent.act(embeddings[active_indices], polygon_state[active_indices], deterministic=False)
            actions[active_indices] = chosen_actions
            actions_cpu = actions.detach().cpu()

            alive_before = alive_mask.clone()
            next_polygon_cpu, reward_cpu, done_cpu, info = self.environment.step(actions_cpu)

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
                )
                aggregated = accumulator.push(env_idx, step_tuple)
                for embedding_t, polygon_t, action_t, reward_t, next_polygon_t, done_flag_t, discount_t in aggregated:
                    transition = Transition(
                        embedding=embedding_t,
                        polygon_state=polygon_t,
                        action=action_t,
                        reward=reward_t.view(1),
                        discount=discount_t.view(1),
                        next_polygon_state=next_polygon_t,
                        done=done_flag_t.view(1),
                    )
                    self.replay.add(transition)
                    transitions_added += 1

            polygon_state_cpu = next_polygon_cpu
            polygon_state = polygon_state_cpu.to(self.device)

            alive_mask = alive_before & (~done_bool)
            self._global_step_interactions += int(active_indices.numel())

        for env_idx in range(batch_size):
            leftovers = accumulator.flush(env_idx)
            for embedding_t, polygon_t, action_t, reward_t, next_polygon_t, done_flag_t, discount_t in leftovers:
                transition = Transition(
                    embedding=embedding_t,
                    polygon_state=polygon_t,
                    action=action_t,
                    reward=reward_t.view(1),
                    discount=discount_t.view(1),
                    next_polygon_state=next_polygon_t,
                    done=done_flag_t.view(1),
                )
                self.replay.add(transition)
                transitions_added += 1

        if self.environment.last_iou is not None:
            last_iou = self.environment.last_iou.to(self.device)
            final_iou = torch.where(alive_mask.to(self.device), last_iou, final_iou)

        updates_trigger = max(1, self.training_config.update_every_n_steps)
        updates_to_run = transitions_added // updates_trigger

        critic_losses = []
        actor_losses = []
        performed_updates = 0
        if self._global_step_interactions >= self.training_config.warmup_steps:
            for _ in range(updates_to_run):
                if len(self.replay) < self.training_config.update_batch_size:
                    break
                batch_samples = self.replay.sample(self.training_config.update_batch_size, device=self.device)
                metrics = self.agent.update(batch_samples)
                critic_losses.append(metrics["critic_loss"])
                if "actor_loss" in metrics:
                    actor_losses.append(metrics["actor_loss"])
                performed_updates += 1

        mean_critic_loss = float(sum(critic_losses) / len(critic_losses)) if critic_losses else 0.0
        mean_actor_loss = float(sum(actor_losses) / len(actor_losses)) if actor_losses else 0.0

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
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        return torch.tensor(mean_critic_loss, device=self.device)
