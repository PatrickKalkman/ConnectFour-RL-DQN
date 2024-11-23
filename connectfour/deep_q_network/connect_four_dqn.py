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

        # Second convolutional layer (now 64->64 instead of 64->128)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer (now 64->64 instead of 128->128)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Adjust flat_features size for 64 channels instead of 128
        self.flat_features = 64 * 6 * 7

        # Rest remains the same
        self.fc1 = nn.Linear(self.flat_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Input validation remains same
        if len(x.shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch_size, channels, height, width), got shape {x.shape}"
            )

        x = F.relu(self.bn1(self.conv1(x)))

        # Second conv block with residual
        identity2 = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + identity2  # First residual connection

        # Third conv block with residual
        x = F.relu(self.bn3(self.conv3(x)))
        x = x + identity2  # Second residual connection

        # Rest remains the same
        x = x.view(-1, self.flat_features)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
