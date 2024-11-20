import torch.nn as nn
import torch.nn.functional as F


class Connect4DQN(nn.Module):
    """
    Deep Q-Network for Connect Four
    Input: Board state (3x6x7)
    Output: Q-values for each possible action (7 columns)
    """

    def __init__(self, input_channels=3):
        super(Connect4DQN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Calculate the size of flattened features
        self.flat_features = 128 * 6 * 7

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)  # 7 possible actions (columns)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Print input shape for debugging
        # print(f"Input shape: {x.shape}")

        # Ensure input is the correct shape (batch_size, channels, height, width)
        if len(x.shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch_size, channels, height, width), got shape {x.shape}"
            )

        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))

        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))

        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the output
        x = x.view(-1, self.flat_features)

        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output layer (no activation as these are Q-values)
        x = self.fc3(x)

        return x
