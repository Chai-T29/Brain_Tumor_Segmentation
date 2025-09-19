import pytorch_lightning as pl
import torch
from dqn.model import QNetwork
from dqn.agent import DQNAgent
from dqn.environment import TumorLocalizationEnv
from data.dataset import BrainTumorDataset
from torch.utils.data import DataLoader, Dataset, random_split

class RLDataset(Dataset):
    """A dummy dataset for RL training with PyTorch Lightning."""
    def __init__(self, steps):
        self.steps = steps

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        return idx

class DQNLightning(pl.LightningModule):
    """PyTorch Lightning module for the DQN agent."""
    def __init__(self, data_dir, batch_size=64, lr=1e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.1, eps_decay=1000,
                 memory_size=10000, target_update=10, max_steps=100):
        super().__init__()
        self.save_hyperparameters()

        self.total_reward = 0
        self.episode_reward = 0
        self.val_episode_reward = 0
        
        self.train_env = None
        self.val_env = None
        self.train_state = None
        self.val_state = None

        # Create networks here so they are part of the LightningModule's state_dict
        self.policy_net = QNetwork(num_actions=9) # num_actions will be updated in setup
        self.target_net = QNetwork(num_actions=9) # num_actions will be updated in setup
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.agent = None # Agent will be created in setup


    def setup(self, stage=None):
        full_dataset = BrainTumorDataset(data_dir=self.hparams.data_dir)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        
        self.train_env = TumorLocalizationEnv(self.train_dataset, max_steps=self.hparams.max_steps)
        self.val_env = TumorLocalizationEnv(self.val_dataset, max_steps=self.hparams.max_steps)
        
        self.train_state = self.train_env.reset()
        self.val_state = self.val_env.reset()
        
        # Update num_actions for the networks based on the environment
        self.policy_net.fc2 = torch.nn.Linear(self.policy_net.fc2.in_features, self.train_env.action_space.n)
        self.target_net.fc2 = torch.nn.Linear(self.target_net.fc2.in_features, self.train_env.action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.agent = DQNAgent(policy_net=self.policy_net,
                              target_net=self.target_net,
                              num_actions=self.train_env.action_space.n,
                              learning_rate=self.hparams.lr,
                              gamma=self.hparams.gamma,
                              epsilon_start=self.hparams.eps_start,
                              epsilon_end=self.hparams.eps_end,
                              epsilon_decay=self.hparams.eps_decay,
                              memory_size=self.hparams.memory_size,
                              batch_size=self.hparams.batch_size,
                              target_update=self.hparams.target_update)

    def forward(self, x):
        image, bbox = x
        return self.policy_net(image, bbox)

    def training_step(self, batch, batch_idx):
        action = self.agent.select_action(self.train_state)
        next_state, reward, done, _ = self.train_env.step(action.item())
        
        self.episode_reward += reward

        self.agent.memory.push(self.train_state, action, torch.tensor([reward], device=self.device), next_state, done)

        self.train_state = next_state
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self.train_state = self.train_env.reset()

        loss = self.agent.optimize_model()

        if self.global_step % self.hparams.target_update == 0:
            self.agent.update_target_net()
            
        self.log_dict({
            'total_reward': self.total_reward,
            'episode_reward': self.episode_reward,
            'steps': self.agent.steps_done
        })
        
        if loss is not None:
            self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        action = self.agent.select_action(self.val_state, greedy=True)
        next_state, reward, done, _ = self.val_env.step(action.item())
        
        self.val_episode_reward += reward

        self.val_state = next_state
        if done:
            self.log('val_loss', -self.val_episode_reward)
            self.val_episode_reward = 0
            self.val_state = self.val_env.reset()

    def configure_optimizers(self):
        return self.agent.optimizer

    def train_dataloader(self):
        """Get train loader."""
        return DataLoader(RLDataset(steps=self.train_env.max_steps))

    def val_dataloader(self):
        """Get validation loader."""
        return DataLoader(RLDataset(steps=self.val_env.max_steps))