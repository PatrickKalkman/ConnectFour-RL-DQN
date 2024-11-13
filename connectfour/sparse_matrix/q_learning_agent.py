from dataclasses import dataclass
from typing import Optional

import numpy as np

from connectfour.sparse_matrix.sparse_q_table import SparseQTable


@dataclass
class AgentConfig:
    learning_rate: float = 0.2  # Increased from 0.1 for faster learning
    discount_factor: float = 0.87  # Reduced from 0.99 to focus more on immediate rewards
    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.999991  # Slower decay
    min_epsilon: float = 0.15  # Increased from 0.01 for more exploration


class QLearningAgent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.q_table = SparseQTable()
        self.epsilon = self.config.initial_epsilon

    def choose_action(self, observation: dict) -> int:
        action_mask = observation["action_mask"]
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]

        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        state_key = self.q_table._get_state_key(observation)
        q_values = self.q_table.q_values[state_key]

        valid_q_values = [(action, q_values[action]) for action in valid_actions]
        max_q = max(q for _, q in valid_q_values)

        best_actions = [action for action, q in valid_q_values if q == max_q]
        return np.random.choice(best_actions)

    def learn(
        self, state: dict, action: int, reward: float, next_state: dict, done: bool
    ) -> None:
        current_q = self.q_table.get_value(state, action)

        if not done:
            next_action_mask = next_state["action_mask"]
            next_valid_actions = [
                i for i, valid in enumerate(next_action_mask) if valid
            ]

            next_q_values = [
                self.q_table.get_value(next_state, a) for a in next_valid_actions
            ]
            max_next_q = max(next_q_values) if next_q_values else 0

            target_q = reward + self.config.discount_factor * max_next_q
        else:
            target_q = reward

        new_q = current_q + self.config.learning_rate * (target_q - current_q)
        self.q_table.set_value(state, action, new_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.config.min_epsilon, self.epsilon * self.config.epsilon_decay
        )

    def save(self, filename: str) -> None:
        self.q_table.save(filename)

    @classmethod
    def load(
        cls, filename: str, config: Optional[AgentConfig] = None
    ) -> "QLearningAgent":
        agent = cls(config)
        agent.q_table = SparseQTable.load(filename)
        return agent
