import torch
import os
import glob
import imageio
from dqn.lightning_model import DQNLightning
from dqn.environment import TumorLocalizationEnv
from data.dataset import BrainTumorDataset
from torch.utils.data import random_split

def find_best_checkpoint(checkpoint_dir):
    """Finds the checkpoint with the lowest validation loss."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if not checkpoints:
        return None
    
    best_checkpoint = None
    best_loss = float('inf')
    for ckpt in checkpoints:
        try:
            val_loss_str = os.path.basename(ckpt).split('val_loss=')[1]
            val_loss = float(val_loss_str.split('.ckpt')[0])
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = ckpt
        except (ValueError, IndexError):
            continue
            
    return best_checkpoint

def main():
    CHECKPOINT_DIR = 'lightning_logs/checkpoints'
    DATA_DIR = "MU-Glioma-Post/"
    VIDEO_PATH = "dqn_test_episode.gif"

    best_checkpoint_path = find_best_checkpoint(CHECKPOINT_DIR)

    if not best_checkpoint_path:
        print("No checkpoints found. Please train a model first.")
        return

    print(f"Loading model from: {best_checkpoint_path}")
    model = DQNLightning.load_from_checkpoint(best_checkpoint_path)
    model.setup()
    model.eval()

    full_dataset = BrainTumorDataset(data_dir=DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_env = TumorLocalizationEnv(val_dataset)

    state = test_env.reset()
    done = False
    total_reward = 0
    frames = []

    while not done:
        frame = test_env.render(mode='rgb_array')
        frames.append(frame)
        action = model.agent.select_action(state, greedy=True)
        state, reward, done, _ = test_env.step(action.item())
        total_reward += reward

    print(f"Test episode finished with a total reward of: {total_reward}")
    test_env.close()

    print(f"Saving video to {VIDEO_PATH}")
    imageio.mimsave(VIDEO_PATH, frames, fps=5)

if __name__ == "__main__":
    main()