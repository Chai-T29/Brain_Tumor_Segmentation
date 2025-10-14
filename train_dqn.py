from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.data_module import BrainTumorDataModule
from rl.lightning_module import TD3Lightning


class LightweightGPUUtilization(pl.Callback):
    """Logs lightweight throughput metrics for quick performance sanity checks."""

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._t_batch_start = 0.0
        self._t_prev_end = 0.0
        self._data_time = 0.0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        import time

        now_time = time.perf_counter()
        self._data_time = 0.0 if self._t_prev_end == 0.0 else max(0.0, now_time - self._t_prev_end)
        self._t_batch_start = now_time

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        import time

        if self._t_batch_start == 0.0:
            return
        now = time.perf_counter()
        compute_time = max(0.0, now - self._t_batch_start)
        total_time = compute_time + self._data_time
        util_ratio = compute_time / total_time if total_time > 0 else 0.0

        gs = trainer.global_step or 0
        if gs % self.log_every_n_steps == 0:
            pl_module.log("util/compute_time", compute_time, prog_bar=False, on_step=True, logger=True)
            pl_module.log("util/data_time", self._data_time, prog_bar=False, on_step=True, logger=True)
            pl_module.log("util/compute_ratio", util_ratio, prog_bar=True, on_step=True, logger=True)

        self._t_prev_end = now


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    config = load_config("config.yaml")
    seed = config.get("seed", 42)
    _seed_everything(seed)

    data_cfg = config.get("data", {})
    encoder_cfg = config.get("encoder", {})
    env_cfg = config.get("environment", {})
    algo_cfg = dict(config.get("algorithm", {}))
    training_cfg = config.get("training", {})
    logging_cfg = config.get("logging", {})

    data_module = BrainTumorDataModule(
        data_dir=data_cfg.get("data_dir", "MU-Glioma-Post/"),
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 0),
        persistent_workers=data_cfg.get("persistent_workers", False),
        pin_memory=data_cfg.get("pin_memory", False),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
        seed=seed,
        include_empty_masks=data_cfg.get("include_empty_masks", False),
        encoder_config=encoder_cfg,
        embedding_batch_size=data_cfg.get("embedding_batch_size", 128),
        embedding_device=data_cfg.get("embedding_device"),
    )

    data_module.setup(stage="fit")
    if data_module.embedding_dim is None:
        raise RuntimeError("Failed to prepare embedding memmaps; embedding_dim is undefined.")

    replay_capacity = int(algo_cfg.pop("replay_capacity", 200000))

    model = TD3Lightning(
        embedding_dim=data_module.embedding_dim,
        env_cfg=env_cfg,
        algo_cfg=algo_cfg,
        training_cfg=training_cfg,
        replay_capacity=replay_capacity,
        logging_cfg=logging_cfg,
    )

    logger = TensorBoardLogger(
        save_dir=logging_cfg.get("log_dir", "lightning_logs"),
        name=logging_cfg.get("logger_name", "td3_agent"),
    )

    checkpoint_dir = Path(logger.log_dir) / logging_cfg.get("checkpoint_dir", "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        monitor="train/final_iou",
        dirpath=str(checkpoint_dir),
        filename="td3-epoch{epoch:02d}-fiou{train/final_iou:.3f}",
        mode="max",
        save_top_k=3,
    )

    util_interval = logging_cfg.get("util_monitor_interval", 100)
    callbacks = [LightweightGPUUtilization(log_every_n_steps=util_interval), checkpoint_callback]

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=training_cfg.get("max_epochs", 40),
        logger=logger,
        callbacks=callbacks,
        precision=training_cfg.get("precision", "32-true"),
        enable_model_summary=True,
        log_every_n_steps=training_cfg.get("log_interval", 50),
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
