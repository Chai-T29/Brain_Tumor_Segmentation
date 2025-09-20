import time
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from data.dataset import BrainTumorDataset
from dqn.agent import DQNAgent
from dqn.environment import TumorLocalizationEnv
from dqn.model import QNetwork


class RLDataset(Dataset):
    """Dataset wrapper that controls how many RL iterations Lightning performs."""

    def __init__(self, steps: int) -> None:
        self.steps = max(1, steps)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, idx: int) -> int:
        return idx


class DQNLightning(pl.LightningModule):
    """PyTorch Lightning module for the DQN agent."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        lr: float = 1e-4,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 1000,
        memory_size: int = 10000,
        target_update: int = 10,
        max_steps: int = 100,
        train_sample_size: int = 256,
        val_sample_size: int = 128,
        val_interval: int = 10,
        val_episodes: int = 5,
        val_split: float = 0.1,
        test_split: float = 0.1,
        grad_clip: float = 1.0,
        lr_gamma: float = 0.995,
        seed: int = 42,
        test_gif_limit: int = 100,
        test_gif_dir: str = "lightning_logs/test_gifs",
        test_gif_fps: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.train_env: Optional[TumorLocalizationEnv] = None
        self.val_env: Optional[TumorLocalizationEnv] = None
        self.test_env: Optional[TumorLocalizationEnv] = None

        self.train_state: Optional[Any] = None
        self.val_state: Optional[Any] = None
        self.test_state: Optional[Any] = None

        self.policy_net = QNetwork(num_actions=9)
        self.target_net = QNetwork(num_actions=9)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.agent: Optional[DQNAgent] = None
        self._train_episode_reward = 0.0
        self._train_episode_steps = 0
        self._test_episode_index = 0
        self._gif_output_dir = Path(self.hparams.test_gif_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = BrainTumorDataset(data_dir=self.hparams.data_dir)
        total_samples = len(dataset)
        if total_samples == 0:
            raise RuntimeError("BrainTumorDataset is empty. Please verify the dataset path and contents.")

        if total_samples < 3:
            raise RuntimeError("BrainTumorDataset must contain at least three samples to create train/val/test splits.")

        base_train = 1
        base_val = 1
        base_test = 1
        remaining = total_samples - (base_train + base_val + base_test)

        val_ratio = max(self.hparams.val_split, 0.0)
        test_ratio = max(self.hparams.test_split, 0.0)
        train_ratio = max(0.0, 1.0 - val_ratio - test_ratio)
        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0:
            ratio_sum = 1.0
        train_ratio /= ratio_sum
        val_ratio /= ratio_sum
        test_ratio /= ratio_sum

        extra_train = int(round(remaining * train_ratio))
        extra_val = int(round(remaining * val_ratio))
        extra_test = remaining - extra_train - extra_val

        train_size = base_train + max(0, extra_train)
        val_size = base_val + max(0, extra_val)
        test_size = base_test + max(0, extra_test)

        while train_size + val_size + test_size < total_samples:
            train_size += 1

        while train_size + val_size + test_size > total_samples:
            if train_size > base_train:
                train_size -= 1
            elif val_size > base_val:
                val_size -= 1
            elif test_size > base_test:
                test_size -= 1
            else:
                break

        if test_size < 1:
            test_size = 1
            if train_size > base_train:
                train_size -= 1
            else:
                val_size = max(1, val_size - 1)

        if val_size < 1:
            val_size = 1
            if train_size > base_train:
                train_size -= 1
            else:
                test_size = max(1, test_size - 1)

        total_assigned = train_size + val_size + test_size
        if total_assigned != total_samples:
            diff = total_samples - total_assigned
            train_size = max(1, train_size + diff)

        if train_size + val_size >= total_samples:
            test_size = 1
            train_size = max(1, total_samples - val_size - test_size)
        else:
            test_size = total_samples - train_size - val_size

        if test_size < 1:
            test_size = 1
            train_size = max(1, total_samples - val_size - test_size)
        generator = torch.Generator().manual_seed(self.hparams.seed)
        splits = random_split(dataset, [train_size, val_size, total_samples - train_size - val_size], generator=generator)
        self.train_dataset, self.val_dataset, self.test_dataset = splits

        self.train_env = TumorLocalizationEnv(self.train_dataset, max_steps=self.hparams.max_steps)
        self.val_env = TumorLocalizationEnv(self.val_dataset, max_steps=self.hparams.max_steps)
        self.test_env = TumorLocalizationEnv(self.test_dataset, max_steps=self.hparams.max_steps)

        num_actions = self.train_env.action_space.n
        if self.policy_net.fc2.out_features != num_actions:
            self.policy_net.fc2 = torch.nn.Linear(self.policy_net.fc2.in_features, num_actions)
            self.target_net.fc2 = torch.nn.Linear(self.target_net.fc2.in_features, num_actions)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        self.agent = DQNAgent(
            policy_net=self.policy_net,
            target_net=self.target_net,
            num_actions=num_actions,
            learning_rate=self.hparams.lr,
            gamma=self.hparams.gamma,
            epsilon_start=self.hparams.eps_start,
            epsilon_end=self.hparams.eps_end,
            epsilon_decay=self.hparams.eps_decay,
            memory_size=self.hparams.memory_size,
            batch_size=self.hparams.batch_size,
            target_update=self.hparams.target_update,
        )

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        self.agent.set_episode_progress(self.current_epoch, getattr(self.trainer, "max_epochs", None))
        self._train_episode_reward = 0.0
        self._train_episode_steps = 0
        indices = self._sample_indices(self.train_dataset, self.hparams.train_sample_size)
        self.train_env.set_active_indices(indices)
        self.train_state = self.train_env.reset()

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        del batch  # The RL loop drives training; dataloader batch is unused.
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        losses = []
        ious = []
        episode_start = time.perf_counter()

        for _ in range(self.hparams.max_steps):
            action = self.agent.select_action(self.train_state)
            next_state, reward, done, info = self.train_env.step(action.item())
            self._train_episode_reward += reward
            self._train_episode_steps += 1
            ious.append(info.get("iou", 0.0))

            reward_tensor = torch.tensor([reward], device=self.device)
            memory_next_state = None if done else next_state
            self.agent.push_experience(self.train_state, action, reward_tensor, memory_next_state, done)
            self.train_state = next_state

            loss = self.agent.compute_loss()
            if loss is not None:
                optimizer.zero_grad()
                self.manual_backward(loss)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.hparams.grad_clip)
                optimizer.step()
                losses.append(loss.detach())
                if (self.global_step + 1) % self.hparams.target_update == 0:
                    self.agent.update_target_net()

            if done:
                break

        episode_time = time.perf_counter() - episode_start
        mean_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)
        mean_iou = sum(ious) / max(1, len(ious))
        step_time = episode_time / max(1, self._train_episode_steps)

        reward_tensor = torch.tensor(self._train_episode_reward, device=self.device)
        length_tensor = torch.tensor(self._train_episode_steps, device=self.device, dtype=torch.float32)
        mean_iou_tensor = torch.tensor(mean_iou, device=self.device)
        epsilon_tensor = torch.tensor(self.agent.current_epsilon, device=self.device)
        step_time_tensor = torch.tensor(step_time, device=self.device)

        self.log("train/episode_reward", reward_tensor, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/episode_length", length_tensor, on_epoch=True, sync_dist=True)
        self.log("train/loss", mean_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mean_iou", mean_iou_tensor, on_epoch=True, sync_dist=True)
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

        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            optimizer = optimizers[0]
        else:
            optimizer = optimizers
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", torch.tensor(current_lr, device=self.device), on_epoch=True, prog_bar=False, sync_dist=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self) -> None:
        indices = self._sample_indices(self.val_dataset, self.hparams.val_sample_size)
        self.val_env.set_active_indices(indices)
        self.val_state = self.val_env.reset()

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        del batch
        start_time = time.perf_counter()
        episode_reward = 0.0
        steps = 0
        ious = []

        state = self.val_state
        for _ in range(self.hparams.max_steps):
            action = self.agent.select_action(state, greedy=True)
            next_state, reward, done, info = self.val_env.step(action.item())
            episode_reward += reward
            ious.append(info.get("iou", 0.0))
            state = next_state
            steps += 1
            if done:
                break

        self.val_state = self.val_env.reset()
        episode_time = time.perf_counter() - start_time
        mean_iou = sum(ious) / max(1, len(ious))
        metrics = {
            "val/avg_reward": torch.tensor(episode_reward, device=self.device),
            "val/episode_length": torch.tensor(steps, device=self.device),
            "val/mean_iou": torch.tensor(mean_iou, device=self.device),
            "val/step_time_sec": torch.tensor(episode_time / max(1, steps), device=self.device),
        }
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True, prog_bar=name == "val/avg_reward", sync_dist=True)
        return metrics

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------
    def on_test_epoch_start(self) -> None:
        total = len(self.test_dataset)
        limit = min(total, self.hparams.test_gif_limit)
        generator = torch.Generator().manual_seed(self.hparams.seed + 2024)
        permutation = torch.randperm(total, generator=generator).tolist()
        self._test_indices = permutation[:limit]
        self.test_env.set_active_indices(self._test_indices, sequential=True)
        self.test_state = self.test_env.reset()
        self._gif_output_dir.mkdir(parents=True, exist_ok=True)
        self._test_episode_index = 0

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        del batch
        frames = []
        episode_reward = 0.0
        steps = 0
        ious = []

        state = self.test_state
        frames.append(self.test_env.render(mode="rgb_array"))
        for _ in range(self.hparams.max_steps):
            action = self.agent.select_action(state, greedy=True)
            next_state, reward, done, info = self.test_env.step(action.item())
            frames.append(self.test_env.render(mode="rgb_array"))
            episode_reward += reward
            ious.append(info.get("iou", 0.0))
            state = next_state
            steps += 1
            if done:
                break

        gif_name = f"episode_{self._test_episode_index:03d}_reward_{episode_reward:.2f}.gif"
        imageio.mimsave(self._gif_output_dir / gif_name, frames, fps=self.hparams.test_gif_fps)
        self._test_episode_index += 1
        self.test_state = self.test_env.reset()

        mean_iou = sum(ious) / max(1, len(ious))
        metrics = {
            "test/avg_reward": torch.tensor(episode_reward, device=self.device),
            "test/episode_length": torch.tensor(steps, device=self.device),
            "test/mean_iou": torch.tensor(mean_iou, device=self.device),
        }
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True, prog_bar=name == "test/avg_reward", sync_dist=True)
        return metrics

    # ------------------------------------------------------------------
    # Dataloaders & optimizers
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_gamma)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(RLDataset(steps=1), batch_size=1)

    def val_dataloader(self):
        episodes = max(1, self.hparams.val_episodes)
        return DataLoader(RLDataset(steps=episodes), batch_size=1)

    def test_dataloader(self):
        total = len(getattr(self, "_test_indices", [])) or len(self.test_dataset)
        total = max(1, min(total, self.hparams.test_gif_limit))
        return DataLoader(RLDataset(steps=total), batch_size=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_indices(dataset, sample_size: int):
        if dataset is None or sample_size is None or sample_size <= 0:
            return None
        total = len(dataset)
        if total == 0:
            return None
        sample_size = min(sample_size, total)
        return torch.randperm(total)[:sample_size].tolist()
