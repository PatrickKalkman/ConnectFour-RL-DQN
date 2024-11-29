import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4DQN(nn.Module):
    def __init__(self, input_channels=3):
        super(Connect4DQN, self).__init__()
        
        self.flat_features = input_channels * 6 * 7
        
        # Narrower layers with progressive reduction
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Advantage and value streams with smaller dimensions
        self.advantage_output = nn.Linear(256, 7)
        self.value_output = nn.Linear(256, 1)
        
        # Single layer normalization for stability
        self.ln1 = nn.LayerNorm(512)
        
        # Minimal dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch_size, channels, height, width), got shape {x.shape}"
            )
        
        # Flatten input
        x = x.view(-1, self.flat_features)
        
        # Simplified forward path with single residual
        identity = self.fc1(x)
        x = self.ln1(identity)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = x + F.linear(identity, self.fc2.weight, self.fc2.bias)
        
        # Simplified advantage and value streams
        advantage = self.advantage_output(x)
        value = self.value_output(x)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values