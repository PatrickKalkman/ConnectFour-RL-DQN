import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss

# from connectfour.deep_q_network.connect_four_dqn_convolutional import Connect4DQN
from connectfour.deep_q_network.connect_four_dqn_dense import Connect4DQN
from connectfour.deep_q_network.replay_memory import ReplayMemory


class DQNAgent:
    """
    Deep Q-Learning Agent for Connect Four
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        learning_rate=1e-4,
        memory_capacity=10000,
        batch_size=64,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        tau=1e-3,  # Soft update parameter
    ):
        """
        Initialize DQN Agent

        Args:
            state_dim (tuple): Dimensions of state (channels, height, width)
            action_dim (int): Number of possible actions (columns)
            device (torch.device): Device to use for tensor operations
            learning_rate (float): Learning rate for optimizer
            memory_capacity (int): Size of replay memory
            batch_size (int): Size of training batch
            gamma (float): Discount factor for future rewards
            epsilon_start (float): Starting value of epsilon for ε-greedy policy
            epsilon_end (float): Minimum value of epsilon
            epsilon_decay (float): Decay rate of epsilon
            tau (float): Soft update parameter for target network
        """
        self.device = device
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # Initialize epsilon for ε-greedy policy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Create online and target networks
        self.policy_net = Connect4DQN(state_dim[0]).to(device)
        self.target_net = Connect4DQN(state_dim[0]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Initialize replay memory
        self.memory = ReplayMemory(memory_capacity, device)

        # Training step counter
        self.training_steps = 0

    def select_action(self, state, valid_moves):
        """Select action using epsilon-greedy policy"""
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                if isinstance(state, torch.Tensor):
                    state_tensor = state.unsqueeze(0)  # Already on device
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                q_values = self.policy_net(state_tensor)

                # Mask invalid moves
                mask = torch.ones(self.action_dim, device=self.device) * float("-inf")
                mask[valid_moves] = 0
                q_values = q_values + mask

                return q_values.max(1)[1].item()
        else:
            return np.random.choice(valid_moves)

    def train_step(self):
        """
        Perform one step of training

        Returns:
            float: Loss value if training occurred, None otherwise
        """
        if not self.memory.can_sample(self.batch_size):
            return None

        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Debug prints
        # print("Batch shapes:")
        # print(f"States shape: {states.shape}")
        # print(f"Actions shape: {actions.shape}")
        # print(f"Rewards shape: {rewards.shape}")
        # print(f"Next states shape: {next_states.shape}")
        # print(f"Dones shape: {dones.shape}")

        # Ensure states have correct shape
        if len(states.shape) != 4:
            raise ValueError(
                f"Expected states to have shape (batch_size, channels, height, width), got {states.shape}"
            )

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            # Set Q-value to 0 for terminal states
            next_q_values[dones] = 0.0

        # Compute target Q values
        target_q_values = rewards + (self.gamma * next_q_values)

        # Compute loss
        loss = mse_loss(current_q_values, target_q_values.unsqueeze(1))

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Increment step counter
        self.training_steps += 1

        # Update target network
        if self.training_steps % 100 == 0:  # Update target network every 100 steps
            self.soft_update_target_network()

        # Decay epsilon
        self.epsilon = min(
            1.0, max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        )

        return loss.item()

    def soft_update_target_network(self):
        """Soft update of target network parameters"""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path):
        """Save model parameters"""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
            },
            path,
        )

    def load(self, path):
        """Load model parameters"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
