from pathlib import Path
import time

import yaml
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.data_module import BrainTumorDataModule
from dqn.lightning_model import DQNLightning
from dqn.environment import TumorLocalizationEnv


class LightweightGPUUtilization(pl.Callback):
    """Logs low-overhead throughput/utilization proxies every N steps.
    - compute_time: on_train_batch_start → on_train_batch_end
    - data_time: previous batch end → current batch start
    - util_ratio: compute_time / (compute_time + data_time)
    - mps_mem_mb: torch.mps.current_allocated_memory() / 1e6 (best-effort)
    """
    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._t_batch_start = None
        self._t_prev_end = None
        self._data_time = 0.0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        self._data_time = 0.0 if self._t_prev_end is None else max(0.0, now - self._t_prev_end)
        self._t_batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        if self._t_batch_start is None:
            return
        compute_time = max(0.0, now - self._t_batch_start)
        total_time = compute_time + (self._data_time or 0.0)
        util_ratio = (compute_time / total_time) if total_time > 0 else 0.0

        gs = trainer.global_step or 0
        if gs % self.log_every_n_steps == 0:
            pl_module.log("util/compute_time", compute_time, prog_bar=False, on_step=True, logger=True)
            pl_module.log("util/data_time", self._data_time or 0.0, prog_bar=False, on_step=True, logger=True)
            pl_module.log("util/compute_ratio", util_ratio, prog_bar=True, on_step=True, logger=True)
            # Best-effort MPS memory (may not exist on all PyTorch/MPS versions)
            try:
                if torch.backends.mps.is_available() and hasattr(torch.mps, "current_allocated_memory"):
                    mps_mem_mb = float(torch.mps.current_allocated_memory()) / 1_000_000.0
                    pl_module.log("util/mps_mem_mb", mps_mem_mb, prog_bar=False, on_step=True, logger=True)
            except Exception:
                pass

        self._t_prev_end = now
        
    def on_train_epoch_end(self, trainer, pl_module):
        # Print a simple utilization summary at epoch end
        try:
            compute_ratio = trainer.callback_metrics.get("util/compute_ratio")
            mem_mb = trainer.callback_metrics.get("util/mps_mem_mb")
            if compute_ratio is not None:
                ratio_val = float(compute_ratio.detach().cpu().item())
                msg = f"[Utilization] Epoch {trainer.current_epoch}: compute_ratio={ratio_val:.3f}"
                if mem_mb is not None:
                    mem_val = float(mem_mb.detach().cpu().item())
                    msg += f", mps_mem={mem_val:.1f} MB"
                print(msg, flush=True)
        except Exception:
            pass

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def main():
    config = load_config("config.yaml")
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    env_cfg = config.get("environment", {})
    logging_cfg = config.get("logging", {})

    import numpy as np, random, pytorch_lightning as pl, torch
    seed = training_cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)  # optional, if you really need it

    data_module = BrainTumorDataModule(
        data_dir=data_cfg.get("data_dir", "MU-Glioma-Post/"),
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 0),
        persistent_workers=data_cfg.get("persistent_workers", False),
        pin_memory=data_cfg.get("pin_memory", False),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
        seed=training_cfg.get("seed", 42),
        include_empty_masks=data_cfg.get("include_empty_masks", False),
    )

    env_common = dict(
        max_steps=training_cfg.get("max_steps", 100),
        iou_threshold=env_cfg.get("iou_threshold", 0.8),
        step_size=env_cfg.get("step_size", 10.0),
        scale_factor=env_cfg.get("scale_factor", 1.1),
        min_bbox_size=env_cfg.get("min_bbox_size", 10.0),
        initial_mode=env_cfg.get("initial_mode", "random_corners"),
        initial_margin=env_cfg.get("initial_margin", 8.0),
        reward_clip_range=tuple(env_cfg.get("reward_clip_range", [-6.0, 6.0])),
        delta_iou=env_cfg.get("delta_iou", 2.5),
        current_iou=env_cfg.get("current_iou", 0.5),
        stop_reward_success=env_cfg.get("stop_reward_success", 4.0),
        stop_reward_no_tumor=env_cfg.get("stop_reward_no_tumor", 2.0),
        stop_reward_false=env_cfg.get("stop_reward_false", -3.0),
        time_penalty=env_cfg.get("time_penalty", 0.01),
        hold_penalty=env_cfg.get("hold_penalty", 0.5),
    )
    train_env = TumorLocalizationEnv(**env_common)
    val_env = TumorLocalizationEnv(**env_common)
    test_env = TumorLocalizationEnv(**env_common)

    model = DQNLightning(
        batch_size=data_cfg.get("batch_size", 16),
        lr=training_cfg.get("lr", 1e-4),
        gamma=training_cfg.get("gamma", 0.99),
        eps_start=training_cfg.get("eps_start", 1.0),
        eps_end=training_cfg.get("eps_end", 0.05),
        eps_decay=training_cfg.get("eps_decay", 2000),
        memory_size=training_cfg.get("memory_size", 50000),
        target_update=training_cfg.get("target_update", 10),
        max_steps=training_cfg.get("max_steps", 100),
        grad_clip=training_cfg.get("grad_clip", 1.0),
        lr_gamma=training_cfg.get("lr_gamma", 0.995),
        val_interval=training_cfg.get("val_interval", 1),
        test_gif_limit=logging_cfg.get("test_gif_limit", 10),
        test_gif_dir=logging_cfg.get("test_gif_dir", "lightning_logs/test_gifs"),
        test_gif_fps=logging_cfg.get("test_gif_fps", 4),
        iou_threshold=env_cfg.get("iou_threshold", 0.8),
        train_env=train_env,
        val_env=val_env,
        test_env=test_env,
    )

    logger_version = logging_cfg.get("version")
    logger = TensorBoardLogger(
        save_dir=logging_cfg.get("log_dir", "lightning_logs"),
        name=logging_cfg.get("logger_name", "dqn_agent"),
        version=logger_version,
    )

    checkpoint_subdir = Path(logging_cfg.get("checkpoint_dir", "checkpoints")).name
    checkpoint_dir = Path(logger.log_dir) / checkpoint_subdir

    checkpoint_callback = ModelCheckpoint(
        monitor="val_avg_reward",
        dirpath=str(checkpoint_dir),
        filename="dqn-epoch{epoch:02d}-reward{val_avg_reward:.2f}",
        save_top_k=3,
        mode="max",
    )

    early_stopping = EarlyStopping(
        monitor="val_avg_reward",
        min_delta=1e-3,
        patience=logging_cfg.get("early_stopping_patience", 20),
        mode="max",
        verbose=True,
    )

    util_interval = logging_cfg.get("util_monitor_interval", 50)  # configurable, defaults to 50
    util_cb = LightweightGPUUtilization(log_every_n_steps=util_interval)

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=training_cfg.get("max_epochs", 200),
        check_val_every_n_epoch=model.hparams.val_interval,
        callbacks=[checkpoint_callback, early_stopping, util_cb],
        logger=logger,
        precision=training_cfg.get("precision", "32-true"),
        log_every_n_steps=1,
        enable_model_summary=True,
        deterministic=True,
    )
    #torch.set_float32_matmul_precision('medium')
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
