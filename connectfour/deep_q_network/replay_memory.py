import random
from collections import deque

import numpy as np
import torch


class ReplayMemory:
    """
    Replay Memory Buffer for storing and sampling experiences
    Stores (state, action, reward, next_state, done) tuples
    """

    def __init__(self, capacity, device):
        """
        Initialize Replay Memory

        Args:
            capacity (int): Maximum size of the replay buffer
            device (torch.device): Device to store tensors on (CPU/GPU)
        """
        self.memory = deque(
            maxlen=capacity
        )  # Using deque with max length for automatic FIFO
        self.device = device

    def push(self, state, action, reward, next_state, done):
        """
        Add an experience to memory

        Args:
            state: Current state (board position)
            action: Action taken (column chosen)
            reward: Reward received
            next_state: Resulting state
            done: Whether the game ended
        """
        # Convert numpy arrays to tensors if necessary
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(self.device)

        # Store experience tuple
        self.memory.append(
            (
                state,
                torch.LongTensor([action]).to(self.device),
                torch.FloatTensor([reward]).to(self.device),
                next_state,
                torch.BoolTensor([done]).to(self.device),
            )
        )

    def sample(self, batch_size):
        """
        Sample a batch of experiences from memory

        Args:
            batch_size (int): Size of batch to sample

        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        # Ensure we have enough samples
        if len(self) < batch_size:
            batch_size = len(self)

        # Sample random experiences
        experiences = random.sample(self.memory, batch_size)

        # Transpose the batch to get aligned batches
        batch = list(zip(*experiences))

        # Stack tensors for batch processing
        states = torch.stack(batch[0])
        actions = torch.cat(batch[1])
        rewards = torch.cat(batch[2])
        next_states = torch.stack(batch[3])
        dones = torch.cat(batch[4])

        return (states, actions, rewards, next_states, dones)

    def can_sample(self, batch_size):
        """Check if enough samples are available"""
        return len(self) >= batch_size

    def __len__(self):
        """Return current size of memory"""
        return len(self.memory)

    def clear(self):
        """Clear the memory buffer"""
        self.memory.clear()
