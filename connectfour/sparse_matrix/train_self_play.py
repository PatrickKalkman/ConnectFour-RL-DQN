import json
import os
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
from pettingzoo.classic import connect_four_v3

from connectfour.sparse_matrix.metrics_tracker import MetricsTracker
from connectfour.sparse_matrix.q_learning_agent import AgentConfig, QLearningAgent


@dataclass
class TrainingConfig:
    episodes: int = 200_000
    log_interval: int = 5000
    save_interval: int = 50_000
    render_interval: int = 200_000
    render_delay: float = 0.1
    model_path: str = "models/agent_self_play.npy"
    model_random_path_player1: str = "models/agent_random_first_player.npy"
    model_random_path_player2: str = "models/agent_random_second_player.npy"
    metrics_path: str = "metrics/training_metrics_self_play.json"
    plot_path: str = "metrics/training_plots"
    moving_average_window: int = 1000
    streak_threshold: int = 5


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

        # Initialize metrics tracker
        self.metrics = MetricsTracker(window_size=config.moving_average_window)

        # Create two instances of the agent for self-play
        agent_config = AgentConfig()
        agent_config.initial_epsilon = 0.35
        agent_config.epsilon_decay = 0.999995

        # Initialize agents
        self.agent1 = QLearningAgent.load(
            config.model_random_path_player1, agent_config
        )  # player_0
        self.agent2 = QLearningAgent.load(
            config.model_random_path_player2, agent_config
        )  # player_1

        print("Agent1 q table size:", self.agent1.q_table.get_size())
        print("Agent2 q table size:", self.agent2.q_table.get_size())

        # Create necessary directories
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)
        os.makedirs(self.config.plot_path, exist_ok=True)

    def _get_game_outcome(self, reward: float) -> str:
        """Convert reward to game outcome string."""
        if reward > 0:
            return "W"
        elif reward < 0:
            return "L"
        return "D"

    def _log_progress(self, episode: int):
        """Log training progress and metrics."""
        metrics = self.metrics.get_current_metrics()
        if not metrics:
            return

        # Print current metrics
        print(f"\nEpisode: {episode}")
        print(
            f"Win Rate: {metrics['win_rate']:.2f}% (Recent: {metrics['recent_win_rate']:.1f}%)"
        )
        print(f"Draw Rate: {metrics['draw_rate']:.1f}%")
        print(f"Average Game Length: {metrics['avg_game_length']:.1f} moves")
        print(f"Action Entropy: {metrics['action_entropy']:.3f}")
        print(f"Longest Win Streak: {metrics['longest_win_streak']}")
        print(
            f"Q-table sizes - Agent1: {self.agent1.q_table.get_size()}, Agent2: {self.agent2.q_table.get_size()}"
        )
        print(
            f"Epsilons - Agent1: {self.agent1.epsilon:.3f}, Agent2: {self.agent2.epsilon:.3f}"
        )
        print("-" * 50)

    def _save_model_and_metrics(self):
        """Save model weights and training metrics."""
        # Save both agents
        self.agent1.save(self.config.model_path)
        self.agent2.save(self.config.model_path.replace(".npy", "_agent2.npy"))

        # Save metrics
        with open(self.config.metrics_path, "w") as f:
            json.dump(self.metrics.metrics_history, f)

    def _calculate_td_error(self, agent, prev_state, action, reward, next_state, done):
        """Calculate TD error for a given transition."""
        if prev_state is None or action is None:
            return 0.0

        current_q = agent.q_table.get_value(prev_state, action)
        if done:
            target_q = reward
        else:
            next_action = agent.choose_action(next_state)
            next_q = agent.q_table.get_value(next_state, next_action)
            target_q = reward + agent.config.discount_factor * next_q

        return abs(target_q - current_q)

    def _process_game_end(
        self, agent, observation, reward, prev_observation, prev_action
    ):
        """Process end of game state and calculate final TD error."""
        td_error = 0.0
        if prev_observation is not None:
            td_error = self._calculate_td_error(
                agent, prev_observation, prev_action, reward, observation, True
            )
            agent.learn(prev_observation, prev_action, reward, observation, True)
        return td_error

    def train(self):
        """Main training loop."""
        try:
            for episode in range(self.config.episodes):
                # Run single game episode
                self._run_single_game(episode)

                # Decay epsilon for both agents
                self.agent1.decay_epsilon()
                self.agent2.decay_epsilon()

                # Log progress at intervals
                if episode % self.config.log_interval == 0:
                    self._log_progress(episode)

                # Save models and metrics at intervals
                if episode % self.config.save_interval == 0 and episode > 0:
                    self._save_model_and_metrics()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            # Save final models and metrics
            self._save_model_and_metrics()
            print("\nTraining completed")

    def _run_single_game(self, episode: int):
        """Run a single game episode."""
        render_mode = (
            "human"
            if episode % self.config.render_interval == 0 and episode > 0
            else None
        )
        env = connect_four_v3.env(render_mode=render_mode)
        env.reset()

        # Game state tracking
        game_length = 0
        agent1_moves = []
        agent2_moves = []
        td_errors = []

        # Previous state tracking
        prev_observation1 = None
        prev_action1 = None
        prev_observation2 = None
        prev_action2 = None

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if done:
                action = None
                if agent == "player_0":
                    # Process end game for both agents
                    td_error1 = self._process_game_end(
                        self.agent1,
                        observation,
                        reward,
                        prev_observation1,
                        prev_action1,
                    )
                    td_error2 = self._process_game_end(
                        self.agent2,
                        observation,
                        -reward,  # Opposite reward for agent2
                        prev_observation2,
                        prev_action2,
                    )
                    td_errors.extend([td_error1, td_error2])

                    # Update metrics
                    outcome = self._get_game_outcome(reward)
                    self.metrics.update_game_metrics(
                        winner=outcome,
                        game_length=game_length,
                        agent1_moves=agent1_moves,
                        agent2_moves=agent2_moves,
                        final_td_error=np.mean(td_errors),
                        exploration_rate=self.agent1.epsilon,
                        agent1_qtable=self.agent1.q_table,
                        agent2_qtable=self.agent2.q_table,
                    )
            else:
                # Regular game step
                if agent == "player_0":
                    # Agent 1's turn
                    action = self.agent1.choose_action(observation)
                    if prev_observation1 is not None:
                        td_error = self._calculate_td_error(
                            self.agent1,
                            prev_observation1,
                            prev_action1,
                            reward,
                            observation,
                            False,
                        )
                        td_errors.append(td_error)
                        self.agent1.learn(
                            prev_observation1, prev_action1, reward, observation, False
                        )

                    prev_observation1 = observation
                    prev_action1 = action
                    agent1_moves.append(action)
                    game_length += 1
                else:
                    # Agent 2's turn
                    action = self.agent2.choose_action(observation)
                    if prev_observation2 is not None:
                        td_error = self._calculate_td_error(
                            self.agent2,
                            prev_observation2,
                            prev_action2,
                            reward,
                            observation,
                            False,
                        )
                        td_errors.append(td_error)
                        self.agent2.learn(
                            prev_observation2, prev_action2, reward, observation, False
                        )

                    prev_observation2 = observation
                    prev_action2 = action
                    agent2_moves.append(action)

            env.step(action)
            if render_mode == "human":
                time.sleep(self.config.render_delay)

        env.close()
        return game_length, agent1_moves, agent2_moves

    def get_training_summary(self) -> Dict:
        """Get a summary of the training metrics."""
        metrics = self.metrics.get_current_metrics()
        return {
            "total_games": self.metrics.total_games,
            "final_win_rate": metrics["win_rate"],
            "final_draw_rate": metrics["draw_rate"],
            "longest_win_streak": metrics["longest_win_streak"],
            "final_q_table_size": self.agent1.q_table.get_size(),
            "final_epsilon": self.agent1.epsilon,
            "avg_game_length": metrics["avg_game_length"],
            "final_action_entropy": metrics["action_entropy"],
        }

    def save_metrics(self):
        """Save all metrics to file."""
        self._save_model_and_metrics()


def main():
    """Main entry point for training."""
    config = TrainingConfig()
    trainer = Trainer(config)

    print("Starting training...")
    trainer.train()

    summary = trainer.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
