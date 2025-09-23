import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


class NoisyLinear(nn.Module):
    """Factorised Gaussian noisy linear layer as described in NoisyNet-DQN."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma_weight = self.sigma_init / math.sqrt(self.in_features)
        sigma_bias = self.sigma_init / math.sqrt(self.out_features)
        self.weight_sigma.data.fill_(sigma_weight)
        self.bias_sigma.data.fill_(sigma_bias)

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

class QNetwork(nn.Module):
    """Q-Network for the DQN Agent."""
    def __init__(self, num_actions=9):
        super(QNetwork, self).__init__()
        # CNN for the image. Assumes input image is resized to 224x224.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute flattened conv output size dynamically for 224x224 input
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 224, 224)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            conv_out_size = x.view(1, -1).size(1)

        # MLP for the bounding box coordinates (x, y, w, h)
        self.fc_bbox = nn.Linear(4, 128)

        # The input size for the final MLP is the sum of the flattened CNN output and the bbox MLP output.
        self.fc1 = nn.Linear(conv_out_size + 128, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, image, bbox):
        """Forward pass through the network."""
        # Process image with CNN
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten

        # Process bounding box with MLP
        y = F.relu(self.fc_bbox(bbox))

        # Concatenate image and bbox features
        z = torch.cat((x, y), dim=1)
        
        # Final MLP for Q-values
        z = F.relu(self.fc1(z))
        q_values = self.fc2(z)
        return q_values

class DuelingQNetwork(nn.Module):
    """Dueling Q-Network for the DQN Agent."""
    def __init__(self, num_actions=9):
        super(DuelingQNetwork, self).__init__()
        # CNN for the image. Assumes input image is resized to 224x224.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute flattened conv output size dynamically for 224x224 input
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 224, 224)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            conv_out_size = x.view(1, -1).size(1)

        # MLP for the bounding box coordinates (x, y, w, h)
        self.fc_bbox = nn.Linear(4, 128)

        # Shared fully connected layer (same as original fc1)
        self.fc1 = nn.Linear(conv_out_size + 128, 512)

        # Separate heads for value and advantage
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)       # outputs scalar V(s)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)  # outputs A(s,a) for all actions
        )

    def forward(self, image, bbox):
        """Forward pass through the network."""
        # Process image with CNN
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten

        # Process bounding box with MLP
        y = F.relu(self.fc_bbox(bbox))

        # Concatenate image and bbox features
        z = torch.cat((x, y), dim=1)
        z = F.relu(self.fc1(z))

        # Value and Advantage streams
        value = self.value_stream(z)                # shape (B, 1)
        advantage = self.advantage_stream(z)        # shape (B, num_actions)

        # Combine into Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class DuelingQNetworkHF(nn.Module):
    """Dueling Q-Network with Hugging Face pretrained backbone (frozen)."""

    def __init__(self, num_actions=9, model_name="microsoft/resnet18"):
        super().__init__()

        # Load pretrained backbone
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get feature dimension from backbone's pooled output
        # Works for most HF vision models that expose `pooler_output` or `last_hidden_state`
        dummy = torch.zeros(1, 3, 224, 224)  # HF models typically expect 224x224
        with torch.no_grad():
            out = self.backbone(dummy)
            if hasattr(out, "pooler_output"):
                feat_dim = out.pooler_output.size(-1)
            else:
                feat_dim = out.last_hidden_state[:, 0].size(-1)
        self.feat_dim = feat_dim

        # Bounding box MLP
        self.fc_bbox = nn.Linear(4, 128)

        # Shared fusion layer
        self.fc1 = nn.Linear(self.feat_dim + 128, 512)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, image, bbox):
        # Assumes image is [B, C, H, W] in torch.Tensor
        if image.size(1) == 1:
            image = image.repeat(1, 3, 1, 1)

        # HF backbones expect pixel_values preprocessed
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(image.device)

        with torch.no_grad():  # frozen backbone
            out = self.backbone(pixel_values)
            if hasattr(out, "pooler_output"):
                x = out.pooler_output  # [B, feat_dim]
            else:
                x = out.last_hidden_state[:, 0]  # CLS token for transformers

        # Process bounding box
        y = F.relu(self.fc_bbox(bbox))

        # Fuse
        z = torch.cat((x, y), dim=1)
        z = F.relu(self.fc1(z))

        # Dueling heads
        value = self.value_stream(z)           # (B, 1)
        advantage = self.advantage_stream(z)   # (B, num_actions)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
 
class NoisyDuelingQNetwork(nn.Module):
    """Dueling Q-Network variant that uses NoisyNet linear layers."""

    def __init__(self, num_actions: int = 9) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute flattened conv output size dynamically for 224x224 input
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 224, 224)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            conv_out_size = x.view(1, -1).size(1)

        self.fc_bbox = nn.Linear(4, 128)
        self.fc1 = nn.Linear(conv_out_size + 128, 512)

        self.value_stream = nn.Sequential(
            NoisyLinear(512, 256),
            nn.ReLU(),
            NoisyLinear(256, 1),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(512, 256),
            nn.ReLU(),
            NoisyLinear(256, num_actions),
        )

    def forward(self, image: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        y = F.relu(self.fc_bbox(bbox))
        z = torch.cat((x, y), dim=1)
        z = F.relu(self.fc1(z))

        value = self.value_stream(z)
        advantage = self.advantage_stream(z)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
