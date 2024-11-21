import json
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from pettingzoo.classic import connect_four_v3

from connectfour.deep_q_network.dqn_agent import DQNAgent


@dataclass
class TrainingConfig:
    episodes: int = 300_000
    log_interval: int = 250
    save_interval: int = 10_000
    render_interval: int = 600_000
    render_delay: float = 0.1
    model_path: str = "models/dqn_agent_random_first_player.pth"
    metrics_path: str = "metrics/dqn_training_metrics_random_first_player"
    # DQN specific parameters
    batch_size: int = 256
    memory_capacity: int = 750_000
    learning_rate: float = 1e-4
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.999995


class DQNTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Create directories
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)

        # Initialize metrics tracking
        self.recent_results = deque(maxlen=1000)
        self.metrics_history = {
            "episodes": [],
            "overall_win_rate": [],
            "recent_win_rate": [],
            "epsilon": [],
            "draws_ratio": [],
            "loss": [],
        }

        # Initialize environment to get dimensions
        temp_env = connect_four_v3.env()
        temp_env.reset()

        # Get state dimensions from observation space
        obs, _, _, _, _ = temp_env.last()
        # board_shape = obs["observation"].shape
        self.state_dim = (3, 6, 7)  # Channels, Height, Width
        self.action_dim = temp_env.action_space("player_1").n
        temp_env.close()

        # Initialize DQN agent
        self.agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
            learning_rate=config.learning_rate,
            memory_capacity=config.memory_capacity,
            batch_size=config.batch_size,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
        )

    def _preprocess_observation(self, obs):
        """Convert PettingZoo observation to DQN input format"""
        board = obs["observation"]  # Shape is (6, 7, 2)

        # We'll use the first channel of the observation
        board_2d = board[:, :, 0]  # Now shape is (6, 7)

        # Create 3-channel representation
        player_pieces = (board_2d == 1).astype(np.float32)
        opponent_pieces = (board_2d == -1).astype(np.float32)

        # Create valid moves channel with same shape as board
        valid_moves = np.zeros((6, 7), dtype=np.float32)
        for i, valid in enumerate(obs["action_mask"]):
            if valid:
                valid_moves[0, i] = 1  # Only mark the top row position

        # Stack channels
        state = np.stack([player_pieces, opponent_pieces, valid_moves])
        return state

    def _get_valid_moves(self, obs):
        """Get list of valid moves from observation"""
        return [i for i, valid in enumerate(obs["action_mask"]) if valid]

    def _update_stats(self, reward: float):
        if reward > 0:
            self.wins += 1
            self.recent_results.append("W")
        elif reward < 0:
            self.losses += 1
            self.recent_results.append("L")
        else:
            self.draws += 1
            self.recent_results.append("D")

    def train(self):
        for episode in range(self.config.episodes):
            # Set render mode
            render_mode = (
                "human"
                if episode % self.config.render_interval == 0 and episode > 0
                else None
            )
            self.env = connect_four_v3.env(render_mode=render_mode)
            self.env.reset()
            episode_reward = 0
            episode_loss = 0
            training_steps = 0

            # Keep track of previous state and action for learning
            previous_state = None
            previous_action = None

            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()

                # Convert rewards for second player perspective
                if agent == "player_0":  # Our agent (second player)
                    reward = reward
                else:
                    reward = -reward

                # Preprocess state
                current_state = self._preprocess_observation(observation)

                if termination or truncation:
                    action = None
                    # Learn from the final state
                    if previous_state is not None and agent == "player_0":
                        self.agent.memory.push(
                            previous_state, previous_action, reward, current_state, True
                        )
                        loss = self.agent.train_step()
                        if loss is not None:
                            episode_loss += loss
                            training_steps += 1
                    self._update_stats(reward)
                else:
                    # Learn from previous state-action pair if it exists
                    if previous_state is not None and agent == "player_0":
                        self.agent.memory.push(
                            previous_state,
                            previous_action,
                            reward,
                            current_state,
                            False,
                        )
                        loss = self.agent.train_step()
                        if loss is not None:
                            episode_loss += loss
                            training_steps += 1

                    if agent == "player_0":  # Our DQN agent (second player)
                        valid_moves = self._get_valid_moves(observation)
                        action = self.agent.select_action(current_state, valid_moves)
                        previous_state = current_state
                        previous_action = action
                        episode_reward += reward
                    else:  # Random opponent (second player)
                        valid_moves = self._get_valid_moves(observation)
                        action = np.random.choice(valid_moves)

                self.env.step(action)

            if render_mode == "human":
                time.sleep(self.config.render_delay)

            # Log progress
            if episode % self.config.log_interval == 0:
                avg_loss = episode_loss / training_steps if training_steps > 0 else 0
                self._log_progress(episode, avg_loss)

            # Save model
            if episode % self.config.save_interval == 0:
                self.agent.save(self.config.model_path)
                self.save_metrics(episode)

            self.env.close()

        self.save_metrics("last")

    def _log_progress(self, episode: int, loss: float):
        # Calculate metrics
        total_games = self.wins + self.losses + self.draws
        win_rate = self.wins / total_games if total_games > 0 else 0
        win_percentage = win_rate * 100

        recent_wins = self.recent_results.count("W")
        recent_total = len(self.recent_results)
        recent_percentage = (
            (recent_wins / recent_total * 100) if recent_total > 0 else 0
        )

        draws_ratio = (self.draws / total_games * 100) if total_games > 0 else 0

        # Store metrics
        self.metrics_history["episodes"].append(episode)
        self.metrics_history["overall_win_rate"].append(win_percentage)
        self.metrics_history["recent_win_rate"].append(recent_percentage)
        self.metrics_history["epsilon"].append(self.agent.epsilon)
        self.metrics_history["draws_ratio"].append(draws_ratio)
        self.metrics_history["loss"].append(loss)

        # Print current metrics
        print(f"Episode: {episode}")
        print(f"Overall Win Rate: {win_rate:.2f} ({win_percentage:.1f}%)")
        print(f"Last {recent_total} games: {recent_percentage:.1f}%")
        print(f"Wins: {self.wins}, Losses: {self.losses}, Draws: {self.draws}")
        print(f"Epsilon: {self.agent.epsilon:.3f}")

        # Enhanced loss reporting
        if loss > 0:  # Only log when we have valid loss
            if len(self.metrics_history["loss"]) > 1:
                last_100_losses = self.metrics_history["loss"][-100:]
                avg_loss = sum(last_100_losses) / len(last_100_losses)
                min_loss = min(last_100_losses)
                max_loss = max(last_100_losses)
                print(f"Current Loss: {loss:.6f}")
                print(f"Avg Loss (last 100): {avg_loss:.6f}")
                print(f"Min/Max Loss (last 100): {min_loss:.6f}/{max_loss:.6f}")
            else:
                print(f"Initial Loss: {loss:.6f}")

        print("Playing as First Player")
        print("-" * 50)

    def save_metrics(self, episode):
        with open(f"{self.config.metrics_path}_{episode}.json", "w") as f:
            json.dump(self.metrics_history, f)


def main():
    config = TrainingConfig()
    trainer = DQNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
