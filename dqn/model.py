import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Q-Network for the DQN Agent."""
    def __init__(self, num_actions=9):
        super(QNetwork, self).__init__()
        # CNN for the image. Assumes input image is resized to 84x84.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # MLP for the bounding box coordinates (x, y, w, h)
        self.fc_bbox = nn.Linear(4, 128)

        # The input size for the final MLP is the sum of the flattened CNN output and the bbox MLP output.
        # CNN output for 84x84 input: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(3136 + 128, 512)
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
