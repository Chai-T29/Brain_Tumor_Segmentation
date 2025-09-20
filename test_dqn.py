import glob
import os
from pathlib import Path
from typing import Dict

import imageio
import torch
import yaml

from data.data_module import BrainTumorDataModule
from dqn.environment import TumorLocalizationEnv
from dqn.lightning_model import DQNLightning


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def find_best_checkpoint(checkpoint_dir: str) -> str | None:
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoints:
        return None

    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def main():
    config = load_config("config.yaml")
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    env_cfg = config.get("environment", {})
    logging_cfg = config.get("logging", {})

    checkpoint_dir = logging_cfg.get("checkpoint_dir", "lightning_logs/checkpoints")
    best_checkpoint_path = find_best_checkpoint(checkpoint_dir)

    if not best_checkpoint_path:
        print("No checkpoints found. Please train a model first.")
        return

    print(f"Loading model from: {best_checkpoint_path}")
    model = DQNLightning.load_from_checkpoint(best_checkpoint_path)
    model.eval()
    model.to(torch.device("cpu"))

    data_module = BrainTumorDataModule(
        data_dir=data_cfg.get("data_dir", "MU-Glioma-Post/"),
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 0),
        persistent_workers=False,
        pin_memory=False,
        val_split=data_cfg.get("val_split", 0.1),
        test_split=data_cfg.get("test_split", 0.1),
        seed=training_cfg.get("seed", 42),
    )
    data_module.setup("test")

    val_loader = data_module.val_dataloader()
    if val_loader is None:
        print("Validation dataloader is not available.")
        return

    batch = next(iter(val_loader))
    images = batch["image"][:1]
    masks = batch["mask"][:1]

    env = TumorLocalizationEnv(max_steps=model.hparams.max_steps, iou_threshold=env_cfg.get("iou_threshold", 0.8))
    state = env.reset(images, masks)

    total_reward = 0.0
    frames = [env.render(index=0, mode="rgb_array")]
    done = torch.zeros(images.size(0), dtype=torch.bool)

    while not done.all():
        action = model.agent.select_action(state, greedy=True)
        next_state, rewards, done, _ = env.step(action)
        frames.append(env.render(index=0, mode="rgb_array"))
        total_reward += float(rewards[0].item())
        state = next_state

    env.close()

    output_dir = Path(logging_cfg.get("test_gif_dir", "lightning_logs/test_gifs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "dqn_test_episode.gif"

    print(f"Test episode finished with a total reward of: {total_reward:.2f}")
    print(f"Saving video to {video_path}")
    imageio.mimsave(video_path, frames, fps=logging_cfg.get("test_gif_fps", 4))


if __name__ == "__main__":
    main()
