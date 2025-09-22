import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import imageio.v2 as imageio
import pytorch_lightning as pl
import torch

from dqn.agent import DQNAgent
from dqn.environment import TumorLocalizationEnv
from dqn.model import QNetwork


class DQNLightning(pl.LightningModule):
    """PyTorch Lightning module for the DQN agent with vectorised environment interactions."""

    def __init__(
        self,
        batch_size: int = 16,
        lr: float = 1e-4,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 1000,
        memory_size: int = 10000,
        target_update: int = 10,
        max_steps: int = 100,
        grad_clip: float = 1.0,
        lr_gamma: float = 0.995,
        val_interval: int = 1,
        test_gif_limit: int = 10,
        test_gif_dir: str = "lightning_logs/test_gifs",
        test_gif_fps: int = 4,
        iou_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.policy_net = QNetwork(num_actions=9)
        self.target_net = QNetwork(num_actions=9)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.agent = DQNAgent(
            policy_net=self.policy_net,
            target_net=self.target_net,
            num_actions=9,
            gamma=self.hparams.gamma,
            epsilon_start=self.hparams.eps_start,
            epsilon_end=self.hparams.eps_end,
            epsilon_decay=self.hparams.eps_decay,
            memory_size=self.hparams.memory_size,
            batch_size=self.hparams.batch_size,
            target_update=self.hparams.target_update,
        )

        self.train_env = TumorLocalizationEnv(max_steps=self.hparams.max_steps, iou_threshold=self.hparams.iou_threshold)
        self.val_env = TumorLocalizationEnv(max_steps=self.hparams.max_steps, iou_threshold=self.hparams.iou_threshold)
        self.test_env = TumorLocalizationEnv(max_steps=self.hparams.max_steps, iou_threshold=self.hparams.iou_threshold)

        self._opt_steps = 0
        self._gif_output_dir = Path(self.hparams.test_gif_dir)
        self._test_episode_index = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    # def on_train_epoch_start(self) -> None:
    #     self.agent.set_episode_progress(self.current_epoch, getattr(self.trainer, "max_epochs", None))

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)

        start_time = time.perf_counter()
        state = self.train_env.reset(images, masks)
        batch_size = images.size(0)

        cumulative_rewards = torch.zeros(batch_size, device=self.device)
        steps_taken = torch.zeros(batch_size, device=self.device)
        iou_values = []
        losses = []
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(self.hparams.max_steps):
            actions = self.agent.select_action(state)
            next_state, rewards, done, info = self.train_env.step(actions)

            rewards_device = rewards.to(self.device)
            cumulative_rewards += rewards_device
            steps_taken += active_mask.float()
            iou_values.append(info["iou"].to(self.device))

            for idx in range(batch_size):
                state_sample = (
                    state[0][idx].detach().cpu().clone(),
                    state[1][idx].detach().cpu().clone(),
                )
                action_sample = actions[idx].view(1, 1).detach().cpu()
                reward_sample = rewards_device[idx].view(1).detach().cpu()
                next_state_sample: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                if not bool(done[idx].item()):
                    next_state_sample = (
                        next_state[0][idx].detach().cpu().clone(),
                        next_state[1][idx].detach().cpu().clone(),
                    )
                self.agent.push_experience(
                    state_sample,
                    action_sample,
                    reward_sample,
                    next_state_sample,
                    bool(done[idx].item()),
                )

            loss = self.agent.compute_loss()
            if loss is not None:
                optimizer.zero_grad()
                self.manual_backward(loss)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.hparams.grad_clip)
                optimizer.step()
                losses.append(loss.detach())
                self._opt_steps += 1
                if self._opt_steps % self.hparams.target_update == 0:
                    self.agent.update_target_net()

            state = next_state
            active_mask = active_mask & ~done.to(self.device)
            if not active_mask.any():
                break

        episode_time = time.perf_counter() - start_time

        mean_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)
        mean_iou = torch.cat(iou_values).mean() if iou_values else torch.tensor(0.0, device=self.device)
        avg_reward = cumulative_rewards.mean()
        avg_length = steps_taken.mean()
        step_time = episode_time / max(1.0, float(steps_taken.max().item()))

        epsilon_tensor = torch.tensor(self.agent.current_epsilon, device=self.device)
        step_time_tensor = torch.tensor(step_time, device=self.device)

        self.log("train/episode_reward", avg_reward, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/episode_length", avg_length, on_epoch=True, sync_dist=True)
        self.log("train/loss", mean_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mean_iou", mean_iou, on_epoch=True, sync_dist=True)
        self.log("train/epsilon", epsilon_tensor, on_epoch=True, sync_dist=True)
        self.log("train/step_time_sec", step_time_tensor, on_epoch=True, sync_dist=True)

        return mean_loss

    def on_train_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if isinstance(schedulers, list):
                for scheduler in schedulers:
                    scheduler.step()
            else:
                schedulers.step()

        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", torch.tensor(current_lr, device=self.device), on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Validation & testing helpers
    # ------------------------------------------------------------------
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        metrics, _ = self._simulate_environment(self.val_env, batch, greedy=True, record=False)
        self._log_metrics(metrics, prefix="val")
        return metrics

    def on_test_epoch_start(self) -> None:
        self._gif_output_dir.mkdir(parents=True, exist_ok=True)
        self._test_episode_index = 0

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        record = self._test_episode_index < self.hparams.test_gif_limit
        metrics, frames = self._simulate_environment(
            self.test_env,
            batch,
            greedy=True,
            record=record,
        )
        self._log_metrics(metrics, prefix="test")

        if record and frames:
            reward_value = metrics["avg_reward"].item()
            gif_name = f"episode_{self._test_episode_index:03d}_reward_{reward_value:.2f}.gif"
            imageio.mimsave(self._gif_output_dir / gif_name, frames, fps=self.hparams.test_gif_fps)
            self._test_episode_index += 1

        return metrics

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_gamma)
        return [optimizer], [scheduler]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _simulate_environment(
        self,
        env: TumorLocalizationEnv,
        batch: Dict[str, torch.Tensor],
        greedy: bool,
        record: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[list]]:
        images = batch["image"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)

        state = env.reset(images, masks)
        batch_size = images.size(0)
        cumulative_rewards = torch.zeros(batch_size, device=self.device)
        steps_taken = torch.zeros(batch_size, device=self.device)
        iou_values = []
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        frames = []

        if record:
            frames.append(env.render(index=0, mode="rgb_array"))

        for _ in range(self.hparams.max_steps):
            actions = self.agent.select_action(state, greedy=greedy)
            next_state, rewards, done, info = env.step(actions)

            rewards_device = rewards.to(self.device)
            cumulative_rewards += rewards_device
            steps_taken += active_mask.float()
            iou_values.append(info["iou"].to(self.device))

            state = next_state

            if record:
                frames.append(env.render(index=0, mode="rgb_array"))

            active_mask = active_mask & ~done.to(self.device)
            if not active_mask.any():
                break

        metrics = {
            "avg_reward": cumulative_rewards.mean(),
            "episode_length": steps_taken.mean(),
            "mean_iou": torch.cat(iou_values).mean() if iou_values else torch.tensor(0.0, device=self.device),
        }
        return metrics, frames if record else None

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], prefix: str) -> None:
        for name, value in metrics.items():
            log_name = f"{prefix}/{name}"
            self.log(log_name, value, on_step=False, on_epoch=True, prog_bar=(name == "avg_reward"), sync_dist=True)

