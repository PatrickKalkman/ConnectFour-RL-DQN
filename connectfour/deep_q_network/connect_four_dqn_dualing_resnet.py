import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4DQN(nn.Module):
    def __init__(self, input_channels=3):
        super(Connect4DQN, self).__init__()

        self.flat_features = input_channels * 6 * 7

        # Wider layers with consistent width initially
        self.fc1 = nn.Linear(self.flat_features, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)

        # Split into advantage and value streams
        self.advantage_hidden = nn.Linear(512, 256)
        self.advantage_output = nn.Linear(256, 7)

        self.value_hidden = nn.Linear(512, 256)
        self.value_output = nn.Linear(256, 1)

        # Layer normalization for better training stability
        self.ln1 = nn.LayerNorm(1024)
        self.ln2 = nn.LayerNorm(1024)
        self.ln3 = nn.LayerNorm(512)
        self.ln_adv = nn.LayerNorm(256)
        self.ln_val = nn.LayerNorm(256)

        # Lower dropout rate
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch_size, channels, height, width), got shape {x.shape}"
            )

        # Flatten input
        x = x.view(-1, self.flat_features)

        # First dense block with residual
        identity1 = self.fc1(x)
        x = self.ln1(identity1)
        x = F.gelu(x)
        x = self.dropout(x)

        # Second dense block with residual
        identity2 = self.fc2(x)
        x = self.ln2(identity2)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x + identity1  # Residual connection

        # Third dense block
        x = self.fc3(x)
        x = self.ln3(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x + F.linear(
            identity2, self.fc3.weight, self.fc3.bias
        )  # Residual connection

        # Advantage stream
        advantage = self.advantage_hidden(x)
        advantage = self.ln_adv(advantage)
        advantage = F.gelu(advantage)
        advantage = self.dropout(advantage)
        advantage = self.advantage_output(advantage)

        # Value stream
        value = self.value_hidden(x)
        value = self.ln_val(value)
        value = F.gelu(value)
        value = self.dropout(value)
        value = self.value_output(value)

        # Combine value and advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
