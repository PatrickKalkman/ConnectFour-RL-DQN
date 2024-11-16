from collections import deque
from typing import Dict, List

import numpy as np


class MetricsTracker:
    def __init__(
        self,
        window_size: int = 1000,
        save_interval: int = 100,
    ):  # Save metrics every N episodes
        # Basic metrics
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_games = 0

        # Windows for rolling statistics
        self.window_size = window_size
        self.recent_results = deque(maxlen=window_size)
        self.game_lengths = deque(maxlen=window_size)

        # Streak tracking
        self.current_streak = 0
        self.longest_win_streak = 0

        # Learning metrics - Using smaller window sizes for expensive computations
        self.action_entropy_window = deque(maxlen=100)
        self.exploration_rates = deque(maxlen=100)

        # Q-table size tracking
        self.prev_state_count = 0

        # Simple moving averages for performance
        self.running_game_length = 0
        self.running_entropy = 0
        self.alpha = 0.01  # Update rate for running averages

        # History tracking
        self.save_interval = save_interval
        self.metrics_history = {
            "episodes": [],
            "win_rate": [],
            "recent_win_rate": [],
            "draw_rate": [],
            "avg_game_length": [],
            "action_entropy": [],
            "exploration_rate": [],
            "longest_win_streak": [],
            "q_table_sizes": [],
        }

    def _fast_entropy(self, moves: List[int]) -> float:
        """Optimized entropy calculation using numpy."""
        if not moves:
            return 0.0
        counts = np.bincount(moves, minlength=7)
        probabilities = counts[counts > 0] / len(moves)
        return -np.sum(probabilities * np.log2(probabilities))

    def _get_qtable_size(self, q_table) -> int:
        """Simply return the number of states in the Q-table."""
        return len(q_table.q_values)

    def update_game_metrics(
        self,
        winner: str,
        game_length: int,
        agent1_moves: List[int],
        agent2_moves: List[int],
        final_td_error: float,
        exploration_rate: float,
        agent1_qtable,
        agent2_qtable,
    ) -> None:
        """Highly optimized update method with periodic history saving."""
        # Update basic counters
        self.total_games += 1

        # Update win/loss/draw counts and streaks
        if winner == "W":
            self.wins += 1
            self.current_streak += 1
            self.longest_win_streak = max(self.longest_win_streak, self.current_streak)
        elif winner == "L":
            self.losses += 1
            self.current_streak = 0
        else:  # Draw
            self.draws += 1
            self.current_streak = 0

        # Update recent results and game length
        self.recent_results.append(winner)
        self.game_lengths.append(game_length)

        # Update running averages
        if self.running_game_length == 0:
            self.running_game_length = game_length
        else:
            self.running_game_length = (
                1 - self.alpha
            ) * self.running_game_length + self.alpha * game_length

        # Calculate entropy and update running average
        current_entropy = self._fast_entropy(agent1_moves)
        if self.running_entropy == 0:
            self.running_entropy = current_entropy
        else:
            self.running_entropy = (
                1 - self.alpha
            ) * self.running_entropy + self.alpha * current_entropy

        # Store exploration rate
        self.exploration_rates.append(exploration_rate)

        # Store Q-table sizes
        qtable_size = self._get_qtable_size(agent1_qtable)

        # Update history at intervals
        if self.total_games % self.save_interval == 0:
            current_metrics = self.get_current_metrics()
            self.metrics_history["episodes"].append(self.total_games)
            self.metrics_history["win_rate"].append(current_metrics["win_rate"])
            self.metrics_history["recent_win_rate"].append(
                current_metrics["recent_win_rate"]
            )
            self.metrics_history["draw_rate"].append(current_metrics["draw_rate"])
            self.metrics_history["avg_game_length"].append(
                current_metrics["avg_game_length"]
            )
            self.metrics_history["action_entropy"].append(
                current_metrics["action_entropy"]
            )
            self.metrics_history["exploration_rate"].append(
                current_metrics["exploration_rate"]
            )
            self.metrics_history["longest_win_streak"].append(
                current_metrics["longest_win_streak"]
            )
            self.metrics_history["q_table_sizes"].append(qtable_size)

    def get_current_metrics(self) -> Dict:
        """Fast metrics calculation focusing on essential metrics."""
        if not self.recent_results:
            return {}

        recent_games = len(self.recent_results)
        recent_wins = sum(1 for result in self.recent_results if result == "W")

        return {
            "win_rate": (self.wins / self.total_games * 100)
            if self.total_games > 0
            else 0,
            "recent_win_rate": (recent_wins / recent_games * 100)
            if recent_games > 0
            else 0,
            "draw_rate": (self.draws / self.total_games * 100)
            if self.total_games > 0
            else 0,
            "avg_game_length": self.running_game_length,
            "action_entropy": self.running_entropy,
            "longest_win_streak": self.longest_win_streak,
            "exploration_rate": np.mean(self.exploration_rates)
            if self.exploration_rates
            else 0,
        }
