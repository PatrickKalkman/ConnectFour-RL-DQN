import json
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from pettingzoo.classic import connect_four_v3
from q_learning_agent import QLearningAgent


@dataclass
class TrainingConfig:
    episodes: int = 500_000
    log_interval: int = 5000
    save_interval: int = 50_000
    render_interval: int = 600_000
    render_delay: float = 0.1
    model_path: str = "models/agent_random_second_player.npy"  # Updated path
    metrics_path: str = (
        "metrics/training_metrics_random_second_player.json"  # Updated path
    )


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.agent = QLearningAgent()
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)
        self.recent_results = deque(maxlen=1000)

        self.metrics_history = {
            "episodes": [],
            "overall_win_rate": [],
            "recent_win_rate": [],
            "q_table_size": [],
            "epsilon": [],
            "draws_ratio": [],
            "learning_progress": [],
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

    def train(self):
        for episode in range(self.config.episodes):
            render_mode = (
                "human"
                if episode % self.config.render_interval == 0 and episode > 0
                else None
            )
            self.env = connect_four_v3.env(render_mode=render_mode)
            self.env.reset()
            episode_reward = 0

            # Keep track of previous state and action for learning
            previous_observation = None
            previous_action = None

            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()

                # Convert rewards to second player's perspective
                if agent == "player_1":  # This is our agent now
                    reward = reward  # Keep reward as is
                else:
                    reward = -reward  # Invert first player's reward

                if termination or truncation:
                    action = None
                    # Learn from the final state
                    if previous_observation is not None and agent == "player_1":
                        self.agent.learn(
                            previous_observation,
                            previous_action,
                            reward,
                            observation,
                            True,  # Game is done
                        )
                    self._update_stats(reward)
                else:
                    # Learn from previous state-action pair if it exists
                    if previous_observation is not None and agent == "player_1":
                        self.agent.learn(
                            previous_observation,
                            previous_action,
                            reward,
                            observation,
                            False,  # Game is not done
                        )

                    if agent == "player_1":  # Our learning agent (now second player)
                        action = self.agent.choose_action(observation)
                        previous_observation = observation
                        previous_action = action
                        episode_reward += reward
                    else:  # Random opponent (first player)
                        mask = observation["action_mask"]
                        valid_actions = [i for i in range(len(mask)) if mask[i] == 1]
                        action = np.random.choice(valid_actions)

                self.env.step(action)

            if render_mode == "human":
                time.sleep(self.config.render_delay)

            self.agent.decay_epsilon()

            if episode % self.config.log_interval == 0:
                self._log_progress(episode)

            if episode % self.config.save_interval == 0:
                self.agent.save(self.config.model_path)

            self.env.close()

        self.save_metrics()

    def _log_progress(self, episode: int):
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
        q_table_size = self.agent.q_table.get_size()

        # Store metrics
        self.metrics_history["episodes"].append(episode)
        self.metrics_history["overall_win_rate"].append(win_percentage)
        self.metrics_history["recent_win_rate"].append(recent_percentage)
        self.metrics_history["q_table_size"].append(q_table_size)
        self.metrics_history["epsilon"].append(self.agent.epsilon)
        self.metrics_history["draws_ratio"].append(draws_ratio)

        # Calculate learning progress
        if len(self.metrics_history["q_table_size"]) > 1:
            new_states = q_table_size - self.metrics_history["q_table_size"][-2]
            learning_progress = (
                (new_states / q_table_size * 100) if q_table_size > 0 else 0
            )
        else:
            learning_progress = 0
        self.metrics_history["learning_progress"].append(learning_progress)

        # Print current metrics
        print(f"Episode: {episode}")
        print(f"Overall Win Rate: {win_rate:.2f} ({win_percentage:.1f}%)")
        print(f"Last {recent_total} games: {recent_percentage:.1f}%")
        print(f"Wins: {self.wins}, Losses: {self.losses}, Draws: {self.draws}")
        print(f"Epsilon: {self.agent.epsilon:.2f}")
        print(f"Q-table size: {q_table_size}")
        print("Playing as Second Player")  # Added indicator
        print("-" * 50)

    def save_metrics(self):
        with open(self.config.metrics_path, "w") as f:
            json.dump(self.metrics_history, f)


def main():
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
