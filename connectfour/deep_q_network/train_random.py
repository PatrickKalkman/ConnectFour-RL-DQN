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
    batch_size: int = 128
    memory_capacity: int = 750_000
    learning_rate: float = 5e-5
    gamma: float = 0.99
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
            "fps": [],  # Added FPS tracking
        }

        # Initialize environment dimensions
        temp_env = connect_four_v3.env()
        temp_env.reset()
        obs, _, _, _, _ = temp_env.last()
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
        """Faster preprocessing with less memory allocation"""
        board = torch.from_numpy(obs["observation"][:, :, 0]).to(self.device)
        # Reuse tensors instead of creating new ones
        if not hasattr(self, "valid_moves_tensor"):
            self.valid_moves_tensor = torch.zeros((6, 7), device=self.device)

        self.valid_moves_tensor.zero_()
        self.valid_moves_tensor[
            0, [i for i, valid in enumerate(obs["action_mask"]) if valid]
        ] = 1

        return torch.stack(
            [
                (board == 1).float(),  # Player pieces
                (board == -1).float(),  # Opponent pieces
                self.valid_moves_tensor,
            ]
        )

    def _get_valid_moves(self, obs):
        """Get list of valid moves from observation"""
        return [i for i, valid in enumerate(obs["action_mask"]) if valid]

    def _update_stats(self, reward: float):
        """Debug version of stats update"""
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
        training_interval = 4
        steps_since_train = 0
        episode_times = []

        for episode in range(self.config.episodes):
            episode_start = time.time()
            self.env = connect_four_v3.env(render_mode=None)
            self.env.reset()

            episode_loss = 0
            training_steps = 0
            previous_state = None
            previous_action = None
            game_done = False  # Flag to track if game outcome was already recorded

            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, _ = self.env.last()

                # Only process reward for our agent and only once per game
                if (
                    (termination or truncation)
                    and not game_done
                    and agent == "player_0"
                ):
                    self._update_stats(reward)
                    game_done = True

                # Convert rewards for agent perspective
                if agent == "player_0":  # Our agent
                    current_state = self._preprocess_observation(observation)

                    if termination or truncation:
                        action = None
                        if previous_state is not None:
                            self.agent.memory.push(
                                previous_state,
                                previous_action,
                                reward,  # Use original reward
                                current_state,
                                True,
                            )
                            steps_since_train += 1
                            if steps_since_train >= training_interval:
                                loss = self.agent.train_step()
                                if loss is not None:
                                    episode_loss += loss
                                    training_steps += 1
                                steps_since_train = 0
                    else:
                        if previous_state is not None:
                            self.agent.memory.push(
                                previous_state,
                                previous_action,
                                reward,  # Use original reward
                                current_state,
                                False,
                            )
                            steps_since_train += 1
                            if steps_since_train >= training_interval:
                                loss = self.agent.train_step()
                                if loss is not None:
                                    episode_loss += loss
                                    training_steps += 1
                                steps_since_train = 0

                        valid_moves = self._get_valid_moves(observation)
                        action = self.agent.select_action(current_state, valid_moves)
                        previous_state = current_state
                        previous_action = action
                else:  # Random opponent
                    if not (termination or truncation):
                        valid_moves = self._get_valid_moves(observation)
                        action = np.random.choice(valid_moves)
                    else:
                        action = None

                self.env.step(action)

            # End of episode processing
            episode_end = time.time()
            episode_times.append(episode_end - episode_start)

            if episode % self.config.log_interval == 0 and episode > 0:
                recent_episodes = min(1000, episode)
                recent_times = episode_times[-recent_episodes:]
                avg_time = sum(recent_times) / len(recent_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0

                avg_loss = episode_loss / training_steps if training_steps > 0 else 0
                self._log_progress(episode, avg_loss, fps)

            if episode % self.config.save_interval == 0:
                self.agent.save(self.config.model_path)
                self.save_metrics(episode)

            self.env.close()

        self.save_metrics("last")

    def _log_progress(self, episode: int, loss: float, fps: float):
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
        self.metrics_history["fps"].append(fps)

        # Print metrics
        print(f"FPS: {fps:.2f}")
        print(f"Episode: {episode}")
        print(f"Overall Win Rate: {win_rate:.2f} ({win_percentage:.1f}%)")
        print(f"Last {recent_total} games: {recent_percentage:.1f}%")
        print(f"Wins: {self.wins}, Losses: {self.losses}, Draws: {self.draws}")
        print(f"Epsilon: {self.agent.epsilon:.3f}")

        if loss > 0:
            if len(self.metrics_history["loss"]) > 1:
                last_100_losses = self.metrics_history["loss"][-100:]
                print(f"Current Loss: {loss:.6f}")
                print(
                    f"Avg Loss (last 100): {sum(last_100_losses) / len(last_100_losses):.6f}"
                )
                print(
                    f"Min/Max Loss (last 100): {min(last_100_losses):.6f}/{max(last_100_losses):.6f}"
                )
            else:
                print(f"Initial Loss: {loss:.6f}")

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
