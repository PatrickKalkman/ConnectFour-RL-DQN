import json
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from pettingzoo.classic import connect_four_v3

from connectfour.sparse_matrix.q_learning_agent import QLearningAgent


@dataclass
class TrainingConfig:
    episodes: int = 70_000
    log_interval: int = 100
    save_interval: int = 1000
    render_interval: int = 20_000
    render_delay: float = 0.5
    model_path: str = "models/agent_self_play.npy"
    model_random_path: str = "models/agent_random.npy"
    metrics_path: str = "metrics/training_metrics_self_play.json"


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.wins = 0
        self.losses = 0
        self.draws = 0
        # Create two instances of the agent for self-play
        self.agent1 = QLearningAgent()  # player_0
        self.agent2 = QLearningAgent()  # player_1
        self.agent1.load(self.config.model_path)
        self.recent_results = deque(maxlen=1000)
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)

        # New metrics for self-play
        self.avg_game_length = deque(maxlen=1000)  # Track game complexity
        self.agent1_moves = deque(maxlen=1000)  # Track move distribution
        self.agent2_moves = deque(maxlen=1000)
        self.decisive_victories = deque(maxlen=1000)  # Track quick wins

        # Store metrics history
        self.metrics_history = {
            "episodes": [],
            "agent1_win_rate": [],
            "recent_agent1_win_rate": [],
            "draws_rate": [],
            "agent1_q_table_size": [],
            "agent2_q_table_size": [],
            "agent1_epsilon": [],
            "agent2_epsilon": [],
            "avg_game_length": [],
            "decisive_victory_rate": [],  # Wins in less than 10 moves
            "move_diversity": [],  # How varied are the agents' moves
        }

    def _update_stats(
        self, reward: float, game_length: int, agent1_moves: list, agent2_moves: list
    ):
        # Update basic stats
        if reward > 0:
            self.wins += 1
            self.recent_results.append("W")
        elif reward < 0:
            self.losses += 1
            self.recent_results.append("L")
        else:
            self.draws += 1
            self.recent_results.append("D")

        # Update additional metrics
        self.avg_game_length.append(game_length)
        self.agent1_moves.extend(agent1_moves)
        self.agent2_moves.extend(agent2_moves)
        self.decisive_victories.append(1 if game_length < 10 and reward != 0 else 0)

    def _calculate_state_space_overlap(self):
        # Get all states for both agents
        agent1_states = set(self.agent1.q_table.get_all_states())
        agent2_states = set(self.agent2.q_table.get_all_states())

        # Print debugging information
        print("\nDebugging State Space Overlap:")
        print(f"Agent1 states count: {len(agent1_states)}")
        print(f"Agent2 states count: {len(agent2_states)}")

        # Print a few example states from each agent
        print("\nExample states from Agent1:")
        for state in list(agent1_states)[:2]:
            print(np.array(state))

        print("\nExample states from Agent2:")
        for state in list(agent2_states)[:2]:
            print(np.array(state))

        # Calculate overlap
        overlap = len(agent1_states.intersection(agent2_states))
        total_states = len(agent1_states.union(agent2_states))

        print(f"\nOverlap count: {overlap}")
        print(f"Total unique states: {total_states}")

        return (overlap / total_states * 100) if total_states > 0 else 0

    def train(self):
        for episode in range(self.config.episodes):
            render_mode = (
                "human" if episode % self.config.render_interval == 0 else None
            )
            self.env = connect_four_v3.env(render_mode=render_mode)
            self.env.reset()

            # Keep track of previous states and actions for both agents
            previous_observation1 = None
            previous_action1 = None
            previous_observation2 = None
            previous_action2 = None

            game_length = 0
            agent1_moves = []
            agent2_moves = []

            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()

                if termination or truncation:
                    action = None
                    # Learn from the final state for both agents
                    if agent == "player_0" and previous_observation1 is not None:
                        self.agent1.learn(
                            previous_observation1,
                            previous_action1,
                            reward,
                            observation,
                            True,
                        )
                        self.agent2.learn(
                            previous_observation2,
                            previous_action2,
                            -reward,  # Opposite reward for agent2
                            observation,
                            True,
                        )
                    reward_to_add = reward if agent == "player_0" else -reward
                    self._update_stats(
                        reward_to_add, game_length, agent1_moves, agent2_moves
                    )
                else:
                    # Learn from previous state-action pair if it exists
                    if agent == "player_0" and previous_observation1 is not None:
                        self.agent1.learn(
                            previous_observation1,
                            previous_action1,
                            reward,
                            observation,
                            False,
                        )
                    elif agent == "player_1" and previous_observation2 is not None:
                        self.agent2.learn(
                            previous_observation2,
                            previous_action2,
                            reward,
                            observation,
                            False,
                        )

                    # Choose actions for both agents using their respective policies
                    if agent == "player_0":
                        action = self.agent1.choose_action(observation)
                        previous_observation1 = observation
                        previous_action1 = action
                        agent1_moves.append(action)
                        game_length += 1
                    else:  # player_1
                        action = self.agent2.choose_action(observation)
                        previous_observation2 = observation
                        previous_action2 = action
                        agent2_moves.append(action)

                self.env.step(action)

                if render_mode == "human":
                    time.sleep(self.config.render_delay)

            # Decay epsilon for both agents
            self.agent1.decay_epsilon()
            self.agent2.decay_epsilon()

            if episode % self.config.log_interval == 0:
                self._log_progress(episode)

            if episode % self.config.save_interval == 0:
                # Save both agents, but primary agent (agent1) to the main path
                self.agent1.save(self.config.model_path)
                self.agent2.save(self.config.model_path.replace(".npy", "_agent2.npy"))

            self.env.close()

        self.save_metrics()

    def _calculate_move_diversity(self, recent_moves):
        # Calculate entropy of move distribution to measure strategy diversity
        if not recent_moves:
            return 0
        move_counts = np.bincount(recent_moves)
        probabilities = move_counts / len(recent_moves)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy / np.log2(7)  # Normalize by max entropy (7 possible moves)

    def _log_progress(self, episode: int):
        total_games = self.wins + self.losses + self.draws
        agent1_win_rate = self.wins / total_games if total_games > 0 else 0

        recent_wins = self.recent_results.count("W")
        recent_total = len(self.recent_results)
        recent_win_rate = (recent_wins / recent_total * 100) if recent_total > 0 else 0

        # Calculate advanced metrics
        avg_game_length = (
            sum(self.avg_game_length) / len(self.avg_game_length)
            if self.avg_game_length
            else 0
        )
        decisive_rate = (
            sum(self.decisive_victories) / len(self.decisive_victories)
            if self.decisive_victories
            else 0
        )
        move_diversity = (
            self._calculate_move_diversity(list(self.agent1_moves))
            + self._calculate_move_diversity(list(self.agent2_moves))
        ) / 2

        # Store metrics
        self.metrics_history["episodes"].append(episode)
        self.metrics_history["agent1_win_rate"].append(agent1_win_rate * 100)
        self.metrics_history["recent_agent1_win_rate"].append(recent_win_rate)
        self.metrics_history["draws_rate"].append(
            self.draws / total_games * 100 if total_games > 0 else 0
        )
        self.metrics_history["agent1_q_table_size"].append(
            self.agent1.q_table.get_size()
        )
        self.metrics_history["agent2_q_table_size"].append(
            self.agent2.q_table.get_size()
        )
        self.metrics_history["agent1_epsilon"].append(self.agent1.epsilon)
        self.metrics_history["agent2_epsilon"].append(self.agent2.epsilon)
        self.metrics_history["avg_game_length"].append(avg_game_length)
        self.metrics_history["decisive_victory_rate"].append(decisive_rate * 100)
        self.metrics_history["move_diversity"].append(move_diversity * 100)

        # Print current metrics
        print(f"Episode: {episode}")
        print(f"Agent1 Win Rate: {agent1_win_rate:.2f} ({recent_win_rate:.1f}% recent)")
        print(f"Draws: {self.draws / total_games * 100:.1f}%")
        print(f"Average Game Length: {avg_game_length:.1f} moves")
        print(f"Decisive Victory Rate: {decisive_rate * 100:.1f}%")
        print(f"Move Diversity: {move_diversity * 100:.1f}%")
        print(
            f"Q-table sizes - Agent1: {self.agent1.q_table.get_size()}, Agent2: {self.agent2.q_table.get_size()}"
        )
        print(
            f"Epsilons - Agent1: {self.agent1.epsilon:.2f}, Agent2: {self.agent2.epsilon:.2f}"
        )
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
