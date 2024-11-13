import json
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from pettingzoo.classic import connect_four_v3

from connectfour.sparse_matrix.minimax_player import MinimaxPlayer
from connectfour.sparse_matrix.q_learning_agent import QLearningAgent


@dataclass
class TrainingConfig:
    episodes: int = 500_000  # Increased episodes for deeper learning
    log_interval: int = 2000
    save_interval: int = 30_000
    render_interval: int = 30_000
    render_delay: float = 0.2
    model_path: str = "models/agent_mm_deep.npy"  # New path for enhanced model
    metrics_path: str = "metrics/training_metrics_deep_curriculum.json"


class CurriculumTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.agent = QLearningAgent()

        # Modified minimax players with higher depths
        self.minimax_players = {
            "easy": MinimaxPlayer(depth=2),  # Increased from 1
            "medium": MinimaxPlayer(depth=3),  # Increased from 2
            "hard": MinimaxPlayer(depth=4),  # Increased from 3
            "expert": MinimaxPlayer(depth=5),  # New expert level
            "master": MinimaxPlayer(depth=6),  # New master level
        }

        self.recent_results = deque(maxlen=1000)
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)

        self.metrics_history = {
            "episodes": [],
            "overall_win_rate": [],
            "recent_win_rate": [],
            "q_table_size": [],
            "epsilon": [],
            "draws_ratio": [],
            "learning_progress": [],
            "difficulty": [],
        }

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

    def _get_opponent_move(self, observation, episode: int):
        """Enhanced curriculum with higher depths and smoother progression"""
        # Early random phase
        random_prob = min(0.9, max(0.3, 1.0 - (episode / (self.config.episodes * 0.3))))
        if np.random.random() < random_prob:
            valid_actions = [
                i for i, mask in enumerate(observation["action_mask"]) if mask == 1
            ]
            return np.random.choice(valid_actions)

        # Get recent performance
        recent_wins = self.recent_results.count("W")
        recent_total = len(self.recent_results)
        recent_winrate = (recent_wins / recent_total) if recent_total > 0 else 0

        # Enhanced difficulty progression with higher depths
        if episode < self.config.episodes * 0.1:  # First 30%
            valid_actions = [
                i for i, mask in enumerate(observation["action_mask"]) if mask == 1
            ]
            return np.random.choice(valid_actions)
        elif episode < self.config.episodes * 0.2 or recent_winrate < 0.15:
            # Next 15% or if struggling: Easy (depth 2)
            return self.minimax_players["easy"].choose_action(observation)
        elif episode < self.config.episodes * 0.6 and recent_winrate > 0.2:
            # Next 15% if doing okay: Medium (depth 3)
            return self.minimax_players["medium"].choose_action(observation)
        elif episode < self.config.episodes * 0.75 and recent_winrate > 0.15:
            # Next 15% if still doing okay: Hard (depth 4)
            return self.minimax_players["hard"].choose_action(observation)
        elif episode < self.config.episodes * 0.9 and recent_winrate > 0.12:
            # Next 15% if maintaining performance: Expert (depth 5)
            return self.minimax_players["expert"].choose_action(observation)
        elif recent_winrate > 0.1:
            # Final episodes if not struggling too much: Master (depth 6)
            return self.minimax_players["master"].choose_action(observation)
        else:
            # Fallback to hard if struggling
            return self.minimax_players["hard"].choose_action(observation)

    def _get_current_difficulty(self, episode: int) -> str:
        """Enhanced difficulty level reporting"""
        recent_wins = self.recent_results.count("W")
        recent_total = len(self.recent_results)
        recent_winrate = (recent_wins / recent_total) if recent_total > 0 else 0

        if episode < self.config.episodes * 0.3:
            return "random"
        elif episode < self.config.episodes * 0.45 or recent_winrate < 0.15:
            return "easy"
        elif episode < self.config.episodes * 0.6 and recent_winrate > 0.2:
            return "medium"
        elif episode < self.config.episodes * 0.75 and recent_winrate > 0.15:
            return "hard"
        elif episode < self.config.episodes * 0.9 and recent_winrate > 0.12:
            return "expert"
        elif recent_winrate > 0.1:
            return "master"
        else:
            return "hard"

    def _evaluate_position(self, observation) -> float:
        board = observation["observation"]
        score = 0

        # Reward center control (increased importance)
        center_col = board[:, 3]
        score += 0.15 * np.sum(center_col[:, 0] == 1)  # Our pieces in center

        # Adjacent center columns
        adjacent_center_cols = board[:, [2, 4]]
        score += 0.1 * np.sum(adjacent_center_cols[:, :, 0] == 1)

        # Reward connected pieces with enhanced scoring
        for row in range(6):
            for col in range(7):
                if board[row, col, 0] == 1:  # Our piece
                    # Horizontal connections
                    if col < 6 and board[row, col + 1, 0] == 1:
                        score += 0.4
                        # Three in a row horizontally
                        if col < 5 and board[row, col + 2, 0] == 1:
                            score += 0.6

                    # Vertical connections
                    if row < 5 and board[row + 1, col, 0] == 1:
                        score += 0.4
                        # Three in a row vertically
                        if row < 4 and board[row + 2, col, 0] == 1:
                            score += 0.6

                    # Diagonal connections
                    if row < 5 and col < 6 and board[row + 1, col + 1, 0] == 1:
                        score += 0.4
                        # Three in a row diagonally
                        if row < 4 and col < 5 and board[row + 2, col + 2, 0] == 1:
                            score += 0.6

                    if row < 5 and col > 0 and board[row + 1, col - 1, 0] == 1:
                        score += 0.4
                        # Three in a row diagonally
                        if row < 4 and col > 1 and board[row + 2, col - 2, 0] == 1:
                            score += 0.6

        return score

    def train(self):
        start_time = time.time()
        episode_times = deque(maxlen=100)  # Track last 100 episodes

        try:
            for episode in range(self.config.episodes):
                episode_start = time.time()

                render_mode = (
                    "human" if episode % self.config.render_interval == 0 else None
                )
                env = connect_four_v3.env(render_mode=render_mode)
                env.reset()

                previous_observation = None
                previous_action = None
                game_finished = False

                while not game_finished:
                    for agent in env.agent_iter():
                        observation, reward, termination, truncation, info = env.last()

                        if termination or truncation:
                            game_finished = True
                            action = None
                            final_reward = reward if agent == "player_0" else -reward

                            # Enhanced final reward based on position evaluation
                            position_value = self._evaluate_position(observation)
                            final_reward += position_value

                            if previous_observation is not None:
                                self.agent.learn(
                                    previous_observation,
                                    previous_action,
                                    final_reward,
                                    observation,
                                    True,
                                )
                            self._update_stats(final_reward)
                            break

                        if previous_observation is not None and agent == "player_0":
                            # Add position evaluation to intermediate rewards
                            position_reward = self._evaluate_position(observation)
                            self.agent.learn(
                                previous_observation,
                                previous_action,
                                reward + position_reward,
                                observation,
                                False,
                            )

                        if agent == "player_0":  # Our learning agent
                            action = self.agent.choose_action(observation)
                            previous_observation = observation
                            previous_action = action
                        else:  # Curriculum opponent
                            action = self._get_opponent_move(observation, episode)

                        env.step(action)

                env.close()
                self.agent.decay_epsilon()

                # Track episode timing
                episode_end = time.time()
                episode_duration = episode_end - episode_start
                episode_times.append(episode_duration)

                if episode % self.config.log_interval == 0:
                    self._log_progress(episode)

                    # Calculate speeds
                    total_time = time.time() - start_time
                    avg_speed = (episode + 1) / total_time

                    recent_avg_duration = sum(episode_times) / len(episode_times)
                    recent_speed = (
                        1 / recent_avg_duration if recent_avg_duration > 0 else 0
                    )

                    print(f"Overall Speed: {avg_speed:.2f} episodes/second")
                    print(f"Recent Speed: {recent_speed:.2f} episodes/second")
                    print(f"Current episode duration: {episode_duration:.3f} seconds")
                    print("=" * 50)

                if episode % self.config.save_interval == 0 and episode > 0:
                    print(f"\nSaving model at episode {episode}")
                    self.agent.save(self.config.model_path)
                    self.save_metrics()
        finally:
            print("\nTiming statistics for different difficulty levels:")
            for difficulty, minimax_player in self.minimax_players.items():
                print(f"\n=== {difficulty.upper()} Difficulty ===")
                minimax_player.print_timing_stats()

        self.agent.save(self.config.model_path)
        self.save_metrics()

    def _log_progress(self, episode: int):
        total_games = self.wins + self.losses + self.draws
        win_rate = self.wins / total_games if total_games > 0 else 0
        win_percentage = win_rate * 100

        recent_wins = self.recent_results.count("W")
        recent_total = len(self.recent_results)
        recent_percentage = (
            (recent_wins / recent_total * 100) if recent_total > 0 else 0
        )

        draws_ratio = (self.draws / total_games * 100) if total_games > 0 else 0
        q_table_size = self.agent.q_table.get_size()
        current_difficulty = self._get_current_difficulty(episode)

        # Store metrics
        self.metrics_history["episodes"].append(episode)
        self.metrics_history["overall_win_rate"].append(win_percentage)
        self.metrics_history["recent_win_rate"].append(recent_percentage)
        self.metrics_history["q_table_size"].append(q_table_size)
        self.metrics_history["epsilon"].append(self.agent.epsilon)
        self.metrics_history["draws_ratio"].append(draws_ratio)
        self.metrics_history["difficulty"].append(current_difficulty)

        # Calculate learning progress
        if len(self.metrics_history["q_table_size"]) > 1:
            new_states = q_table_size - self.metrics_history["q_table_size"][-2]
            learning_progress = (
                (new_states / q_table_size * 100) if q_table_size > 0 else 0
            )
        else:
            learning_progress = 0
        self.metrics_history["learning_progress"].append(learning_progress)

        print("\n" + "=" * 50)
        print(f"Episode: {episode}")
        print(f"Current Difficulty: {current_difficulty}")
        print(f"Overall Win Rate: {win_rate:.2f} ({win_percentage:.1f}%)")
        print(f"Last {recent_total} games: {recent_percentage:.1f}%")
        print(f"Wins: {self.wins}, Losses: {self.losses}, Draws: {self.draws}")
        print(f"Epsilon: {self.agent.epsilon:.3f}")
        print(f"Q-table size: {q_table_size}")
        print("=" * 50)

    def save_metrics(self):
        with open(self.config.metrics_path, "w") as f:
            json.dump(self.metrics_history, f)


def main():
    config = TrainingConfig()
    trainer = CurriculumTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
