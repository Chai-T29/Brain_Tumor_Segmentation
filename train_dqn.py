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


# --- SimpleMetricCheckpoint callback inserted here ---
class SimpleMetricCheckpoint(pl.Callback):
    """Minimal best/last saver that doesn't rely on ModelCheckpoint internals."""
    def __init__(self, dirpath: str, monitor: str, mode: str = "max", save_last: bool = True) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.best = None
        self.save_last = save_last
        self.dirpath.mkdir(parents=True, exist_ok=True)
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = getattr(trainer, "callback_metrics", {})
        value = metrics.get(self.monitor, None)
        if value is None:
            return
        try:
            if isinstance(value, torch.Tensor):
                score = float(value.detach().cpu().item())
            else:
                score = float(value)
        except Exception as e:
            return
        improved = False
        if self.best is None:
            improved = True
        else:
            improved = (score > self.best) if self.mode == "max" else (score < self.best)
        if improved:
            self.best = score
            best_path = self.dirpath / f"best-epoch{int(pl_module.current_epoch):02d}-{self.monitor.replace('/', '_')}{score:.4f}.ckpt"
            trainer.save_checkpoint(str(best_path))
        if self.save_last:
            last_path = self.dirpath / "last.ckpt"
            trainer.save_checkpoint(str(last_path))

from data.data_module import BrainTumorDataModule
from rl.lightning_module import TD3Lightning




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
    verbose = bool(config.get("verbose", False))
    _seed_everything(seed)

    data_cfg = config.get("data", {})
    encoder_cfg = config.get("encoder", {})
    env_cfg = config.get("environment", {})
    algo_cfg = dict(config.get("algorithm", {}))
    training_cfg = config.get("training", {})
    logging_cfg = config.get("logging", {})

    if verbose:
        print("[Verbose] Initialising data module...")
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
        environment_config=env_cfg,
    )

    data_module.setup(stage="fit")
    if data_module.embedding_shape is None:
        raise RuntimeError("Failed to prepare embedding memmaps; embedding_shape is undefined.")

    if verbose:
        print("[Verbose] Data module ready. Building model...")

    replay_capacity = int(algo_cfg.pop("replay_capacity", 200000))

    model = TD3Lightning(
        embedding_shape=data_module.embedding_shape,
        env_cfg=env_cfg,
        algo_cfg=algo_cfg,
        training_cfg=training_cfg,
        replay_capacity=replay_capacity,
        logging_cfg=logging_cfg,
        verbose=verbose,
    )

    if verbose:
        print("[Verbose] Configuring trainer...")

    logger = TensorBoardLogger(
        save_dir=logging_cfg.get("log_dir", "lightning_logs"),
        name=logging_cfg.get("logger_name", "td3_agent"),
    )

    checkpoint_dir = Path(logger.log_dir) / Path(logging_cfg.get("checkpoint_dir", "checkpoints")).name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Monitor the exact key logged by the LightningModule: "val_mean_iou"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mean_iou",
        dirpath=str(checkpoint_dir),
        filename="td3-epoch{epoch:02d}-val_miou{val_mean_iou:.2f}",
        mode="max",
        save_top_k=3,
        save_last=True,
        save_on_train_epoch_end=False,
    )

    # Create simple metric-based checkpoint saver
    simple_ckpt = SimpleMetricCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor="val_mean_iou",
        mode="max",
        save_last=True,
    )

    callbacks = [
        checkpoint_callback,
        simple_ckpt,
    ]

    if verbose:
        print("[Verbose] Starting training loop...")

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=training_cfg.get("max_epochs", 40),
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        precision=training_cfg.get("precision", "32-true"),
        enable_model_summary=True,
        log_every_n_steps=training_cfg.get("log_interval", 50),
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=data_module)

    if verbose:
        print("[Verbose] Training completed.")


if __name__ == "__main__":
    main()
