from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

from data.data_module import BrainTumorDataModule
from rl.environment import PolygonLocalizationEnv, EnvironmentConfig
from rl.lightning_module import TD3Lightning


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _sorted_checkpoint_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    checkpoints = sorted(directory.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
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


def evaluate(model: TD3Lightning, datamodule: BrainTumorDataModule, device: torch.device) -> Dict[str, float]:
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    env = PolygonLocalizationEnv(EnvironmentConfig(**model.hparams["env_cfg"]))
    agent = model.agent.to(device)
    encoder = model.encoder.to(device)

    total_success = 0.0
    total_iou = 0.0
    total_steps = 0.0
    total_samples = 0

    for batch in test_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.no_grad():
            embeddings = encoder.embed_without_noise(images)

        polygon_state_cpu = env.reset(images.cpu(), masks.cpu())
        polygon_state = polygon_state_cpu.to(device)
        alive = torch.ones(images.size(0), dtype=torch.bool)
        final_iou = torch.zeros(images.size(0), device=device)
        success_flags = torch.zeros(images.size(0), dtype=torch.bool, device=device)
        steps_taken = torch.zeros(images.size(0), device=device)

        for _ in range(env.config.max_steps):
            if not alive.any():
                break
            active_idx = alive.nonzero(as_tuple=False).squeeze(1)
            actions = torch.zeros(images.size(0), env.action_dim, device=device)
            deterministic_actions = agent.act(
                embeddings[active_idx],
                polygon_state[active_idx],
                deterministic=True,
                apply_embedding_noise=False,
            )
            actions[active_idx] = deterministic_actions

            next_polygon_cpu, _, done_cpu, info = env.step(actions.cpu())
            polygon_state = next_polygon_cpu.to(device)

            steps_taken[active_idx] += 1.0
            success_flags = success_flags | info["success"].to(device=device)
            newly_done = done_cpu & alive
            if newly_done.any():
                final_iou = torch.where(newly_done.to(device), info["iou"].to(device), final_iou)
            alive = alive & (~done_cpu)

        if env.last_iou is not None:
            final_iou = torch.where(alive.to(device), env.last_iou.to(device), final_iou)

        total_success += success_flags.float().sum().item()
        total_iou += final_iou.sum().item()
        total_steps += steps_taken.sum().item()
        total_samples += images.size(0)

    if total_samples == 0:
        return {"success_rate": 0.0, "mean_iou": 0.0, "avg_steps": 0.0}

    return {
        "success_rate": total_success / total_samples,
        "mean_iou": total_iou / total_samples,
        "avg_steps": total_steps / total_samples,
    }


def main() -> None:
    config = load_config("config.yaml")
    data_cfg = config.get("data", {})
    logging_cfg = config.get("logging", {})

    datamodule = BrainTumorDataModule(
        data_dir=data_cfg.get("data_dir", "MU-Glioma-Post/"),
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 0),
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
        seed=config.get("seed", 42),
        include_empty_masks=data_cfg.get("include_empty_masks", False),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate(model, datamodule, device)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
