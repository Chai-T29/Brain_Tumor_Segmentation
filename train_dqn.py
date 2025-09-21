from pathlib import Path

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.data_module import BrainTumorDataModule
from dqn.lightning_model import DQNLightning


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def main():
    config = load_config("config.yaml")

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    env_cfg = config.get("environment", {})
    logging_cfg = config.get("logging", {})

    data_module = BrainTumorDataModule(
        data_dir=data_cfg.get("data_dir", "MU-Glioma-Post/"),
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 0),
        persistent_workers=data_cfg.get("persistent_workers", False),
        pin_memory=data_cfg.get("pin_memory", False),
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
        seed=training_cfg.get("seed", 42),
    )

    model = DQNLightning(
        data_dir=data_cfg.get("data_dir", "MU-Glioma-Post/"),
        batch_size=training_cfg.get("batch_size", data_cfg.get("batch_size", 16)),
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
        seed=training_cfg.get("seed", 42),
        val_interval=training_cfg.get("val_interval", 1),
        test_gif_limit=logging_cfg.get("test_gif_limit", 10),
        test_gif_dir=logging_cfg.get("test_gif_dir", "lightning_logs/test_gifs"),
        test_gif_fps=logging_cfg.get("test_gif_fps", 4),
        iou_threshold=env_cfg.get("iou_threshold", 0.8),
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
        monitor="val/avg_reward",
        dirpath=str(checkpoint_dir),
        filename="dqn-epoch{epoch:02d}-reward{val_avg_reward:.2f}",
        save_top_k=3,
        mode="max",
    )

    early_stopping = EarlyStopping(
        monitor="val/avg_reward",
        min_delta=1e-3,
        patience=logging_cfg.get("early_stopping_patience", 20),
        mode="max",
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=training_cfg.get("max_epochs", 200),
        check_val_every_n_epoch=model.hparams.val_interval,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=1,
        enable_model_summary=True,
        deterministic=True,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module, ckpt_path="best")
    trainer.test(model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
