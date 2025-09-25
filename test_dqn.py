from pathlib import Path
from typing import Dict, Iterable

import imageio
import torch
import yaml

import pytorch_lightning as pl
from data.data_module import BrainTumorDataModule
from dqn.lightning_model import DQNLightning
from dqn.environment import TumorLocalizationEnv


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _sorted_checkpoint_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    checkpoints = sorted(directory.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return checkpoints


def find_best_checkpoint(
    log_dir: Path,
    logger_name: str,
    checkpoint_subdir: str,
    versions: Iterable[Path] | None = None,
) -> Path | None:
    logger_root = log_dir / logger_name

    if versions is None:
        candidate_versions = sorted(
            (path for path in logger_root.glob("version_*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    else:
        candidate_versions = [path for path in versions if path.is_dir()]

    for version_dir in candidate_versions:
        checkpoint_dir = version_dir / checkpoint_subdir
        checkpoints = _sorted_checkpoint_paths(checkpoint_dir)
        if checkpoints:
            return checkpoints[0]

    return None


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

    log_dir = Path(logging_cfg.get("log_dir", "lightning_logs"))
    logger_name = logging_cfg.get("logger_name", "dqn_agent")
    checkpoint_subdir = Path(logging_cfg.get("checkpoint_dir", "checkpoints")).name

    requested_version = logging_cfg.get("version")
    versions = None
    if requested_version is not None:
        version_dir = log_dir / logger_name / f"version_{requested_version}"
        versions = [version_dir]

    best_checkpoint_path = find_best_checkpoint(log_dir, logger_name, checkpoint_subdir, versions=versions)

    if best_checkpoint_path is None:
        fallback_dir = Path(logging_cfg.get("checkpoint_dir", "checkpoints"))
        checkpoints = _sorted_checkpoint_paths(fallback_dir)
        if checkpoints:
            best_checkpoint_path = checkpoints[0]

    if not best_checkpoint_path:
        print("No checkpoints found. Please train a model first.")
        return

    print(f"Loading model from: {best_checkpoint_path}")
    model = DQNLightning.load_from_checkpoint(str(best_checkpoint_path))

    data_module = BrainTumorDataModule(
        data_dir=data_cfg.get("data_dir", "MU-Glioma-Post/"),
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 0),
        persistent_workers=data_cfg.get("persistent_workers", False),
        pin_memory=False,
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
        seed=training_cfg.get("seed", 42),
    )

    # Construct test environment from config and attach to model
    resize_shape = env_cfg.get("resize_shape")
    if resize_shape is not None:
        resize_shape = tuple(resize_shape)

    env_common = dict(
        max_steps=training_cfg.get("max_steps", 100),
        iou_threshold=env_cfg.get("iou_threshold", 0.8),
        step_size=env_cfg.get("step_size", 10.0),
        scale_factor=env_cfg.get("scale_factor", 1.1),
        min_bbox_size=env_cfg.get("min_bbox_size", 10.0),
        initial_mode=env_cfg.get("initial_mode", "random_corners"),
        initial_margin=env_cfg.get("initial_margin", 8.0),
        reward_clip_range=tuple(env_cfg.get("reward_clip_range", [-6.0, 6.0])),
        stop_reward_success=env_cfg.get("stop_reward_success", 4.0),
        stop_reward_no_tumor=env_cfg.get("stop_reward_no_tumor", 2.0),
        stop_reward_false=env_cfg.get("stop_reward_false", -3.0),
        time_penalty=env_cfg.get("time_penalty", 0.01),
        hold_penalty=env_cfg.get("hold_penalty", 0.5),
        resize_shape=resize_shape,
    )
    test_env = TumorLocalizationEnv(**env_common)
    model.test_env = test_env

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        deterministic=True,
    )
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()
