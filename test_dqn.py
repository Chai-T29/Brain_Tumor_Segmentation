from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import yaml
import re

from data.data_module import BrainTumorDataModule
from rl.lightning_module import TD3Lightning


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def _extract_val_miou(path: Path) -> float:
    name = path.name
    m = re.search(r"val_miou([0-9.]+)", name)  # PL ModelCheckpoint filenames
    if not m:
        m = re.search(r"val_mean_iou([0-9.]+)", name)  # SimpleMetricCheckpoint fallback
    if not m:
        m = re.search(r"miou([0-9.]+)", name)  # last resort
    return float(m.group(1)[:-1]) if m else float("-inf")

def _sorted_checkpoint_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    checkpoints = list(directory.glob("*.ckpt"))
    checkpoints.sort(
        key=lambda p: (_extract_val_miou(p), p.stat().st_mtime),
        reverse=True,
    )
    return checkpoints


def find_checkpoint(log_dir: Path, logger_name: str, checkpoint_subdir: str) -> Optional[Path]:
    logger_root = log_dir / logger_name
    if not logger_root.exists():
        return None

    candidate_versions = sorted(
        (path for path in logger_root.glob("version_*") if path.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for version_dir in candidate_versions:
        ckpt_dir = version_dir / checkpoint_subdir
        checkpoints = _sorted_checkpoint_paths(ckpt_dir)
        if checkpoints:
            return checkpoints[0]
    return None


def main() -> None:
    config = load_config("config.yaml")
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    logging_cfg = config.get("logging", {})

    datamodule = BrainTumorDataModule(
        data_dir=data_cfg.get("data_dir", "MU-Glioma-Post/"),
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 0),
        persistent_workers=data_cfg.get("persistent_workers", False),
        pin_memory=data_cfg.get("pin_memory", False),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
        seed=config.get("seed", 42),
        include_empty_masks=data_cfg.get("include_empty_masks", False),
        encoder_config=config.get("encoder", {}),
        embedding_batch_size=data_cfg.get("embedding_batch_size", 128),
        embedding_device=data_cfg.get("embedding_device"),
    )

    log_dir = Path(logging_cfg.get("log_dir", "lightning_logs"))
    logger_name = logging_cfg.get("logger_name", "td3_agent")
    checkpoint_subdir = logging_cfg.get("checkpoint_dir", "checkpoints")

    checkpoint_path = find_checkpoint(log_dir, logger_name, checkpoint_subdir)
    if checkpoint_path is None:
        fallback_dir = Path(checkpoint_subdir)
        checkpoints = _sorted_checkpoint_paths(fallback_dir)
        checkpoint_path = checkpoints[0] if checkpoints else None

    if checkpoint_path is None:
        print("No checkpoints found. Please train the model first.")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    model = TD3Lightning.load_from_checkpoint(str(checkpoint_path))
    datamodule.setup(stage="test")

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        precision=training_cfg.get("precision", "32-true"),
        accelerator="auto",
        devices="auto",
        enable_model_summary=False,
    )

    test_results = trainer.test(model=model, datamodule=datamodule, verbose=False)

    if not test_results:
        print("No test results returned.")
        return

    print("Evaluation metrics:")
    for metric, value in sorted(test_results[0].items()):
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
