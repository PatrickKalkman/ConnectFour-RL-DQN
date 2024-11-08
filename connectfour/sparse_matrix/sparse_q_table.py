from collections import defaultdict
from typing import Dict, Tuple

import numpy as np


class SparseQTable:
    def __init__(self, default_value: float = 0.0):
        self.q_values: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.full(7, default_value)  # 7 possible actions in Connect Four
        )
        self.default_value = default_value

    def _get_state_key(self, observation: Dict) -> Tuple:
        # Get board state from observation (6x7x2 array)
        board_state = observation["observation"]

        # Convert to tuple of tuples (immutable) for dictionary key
        # Combine both channels into single board representation
        state_key = tuple(
            tuple(
                1
                if board_state[i, j, 0] == 1
                else 2  # Current player's pieces
                if board_state[i, j, 1] == 1
                else 0  # Opponent's pieces  # Empty spaces
                for j in range(7)
            )
            for i in range(6)
        )
        return state_key

    def get_value(self, observation: Dict, action: int) -> float:
        state_key = self._get_state_key(observation)
        return self.q_values[state_key][action]

    def set_value(self, observation: Dict, action: int, value: float) -> None:
        state_key = self._get_state_key(observation)
        if not np.isclose(value, self.default_value):
            self.q_values[state_key][action] = value

    def get_size(self) -> int:
        return len(self.q_values)

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.config.min_epsilon, self.epsilon * self.config.epsilon_decay
        )

    def save(self, filename: str) -> None:
        # Convert defaultdict to regular dict for saving
        save_dict = {k: v for k, v in self.q_values.items()}
        np.save(filename, save_dict)

    @classmethod
    def load(cls, filename: str) -> "SparseQTable":
        table = cls()
        loaded_dict = np.load(filename, allow_pickle=True).item()
        # Update the defaultdict with loaded values
        table.q_values.update(loaded_dict)
        return table
