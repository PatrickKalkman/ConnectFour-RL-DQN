from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from pettingzoo.classic import connect_four_v3


class SparseQTable:
    def __init__(self, default_value: float = 0.0):
        """
        Initialize sparse Q-table for Connect Four.

        Args:
            default_value: Value to return for previously unseen states
        """
        # Use defaultdict to automatically handle new states
        self.q_values: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.full(7, default_value)  # 7 possible actions in Connect Four
        )
        self.default_value = default_value

    def _get_state_key(self, observation: Dict) -> Tuple:
        """
        Convert PettingZoo observation to immutable state key.

        Args:
            observation: PettingZoo observation dictionary

        Returns:
            Tuple representation of the state
        """
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
        """
        Get Q-value for a state-action pair.

        Args:
            observation: PettingZoo observation
            action: Column to play (0-6)

        Returns:
            Q-value for the state-action pair
        """
        state_key = self._get_state_key(observation)
        return self.q_values[state_key][action]

    def set_value(self, observation: Dict, action: int, value: float) -> None:
        """
        Set Q-value for a state-action pair.

        Args:
            observation: PettingZoo observation
            action: Column to play (0-6)
            value: New Q-value to set
        """
        state_key = self._get_state_key(observation)
        if not np.isclose(value, self.default_value):
            self.q_values[state_key][action] = value

    def get_size(self) -> int:
        """
        Get number of states in the Q-table.

        Returns:
            Number of states stored
        """
        return len(self.q_values)
