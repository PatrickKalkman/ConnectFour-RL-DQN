import json
import os
from collections import deque
from dataclasses import dataclass

from pettingzoo.classic import connect_four_v3

from connectfour.sparse_matrix.minimax_player import MinimaxPlayer
from connectfour.sparse_matrix.q_learning_agent import QLearningAgent


@dataclass
class TrainingConfig:
    episodes: int = 150_000
    log_interval: int = 100
    save_interval: int = 20_000
    render_interval: int = 20_000
    render_delay: float = 0.5
    model_path: str = "models/agent_mm.npy"
    metrics_path: str = "metrics/training_metrics.json"


class MinimaxTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.agent = QLearningAgent()
        self.minimax = MinimaxPlayer(depth=4)  # Adjust depth for difficulty
        self.recent_results = deque(maxlen=1000)
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

        self.metrics_history = {
            "episodes": [],
            "overall_win_rate": [],
            "recent_win_rate": [],
            "q_table_size": [],
            "epsilon": [],
            "draws_ratio": [],
            "learning_progress": [],
        }

    def train(self):
        try:
            for episode in range(self.config.episodes):
                # Create and reset environment for each episode
                render_mode = "human" if episode % self.config.render_interval == 0 else None
                env = connect_four_v3.env(render_mode=render_mode)
                env.reset()

                # Keep track of previous state and action for learning
                previous_observation = None
                previous_action = None
                game_finished = False

                while not game_finished:
                    for agent in env.agent_iter():
                        observation, reward, termination, truncation, info = env.last()

                        if termination or truncation:
                            game_finished = True
                            action = None
                            if previous_observation is not None and agent == "player_0":
                                self.agent.learn(
                                    previous_observation,
                                    previous_action,
                                    reward,
                                    observation,
                                    True,
                                )
                            self._update_stats(reward)
                            break

                        if previous_observation is not None and agent == "player_0":
                            self.agent.learn(
                                previous_observation,
                                previous_action,
                                reward,
                                observation,
                                False,
                            )

                        if agent == "player_0":  # Our learning agent
                            action = self.agent.choose_action(observation)
                            previous_observation = observation
                            previous_action = action
                        else:  # Minimax opponent
                            action = self.minimax.choose_action(observation)

                        env.step(action)

                # Close environment after each episode
                env.close()

                # Update epsilon and save progress
                self.agent.decay_epsilon()

                if episode % self.config.log_interval == 0:
                    self._log_progress(episode)

                if episode % self.config.save_interval == 0:
                    self.agent.save(self.config.model_path)

        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
        finally:
            self.save_metrics()

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

        # Calculate learning progress (new states since last interval)
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
        print("-" * 50)

    def save_metrics(self):
        with open(self.config.metrics_path, "w") as f:
            json.dump(self.metrics_history, f)


def main():
    config = TrainingConfig()

    trainer = MinimaxTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
