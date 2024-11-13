import time
from collections import defaultdict
from functools import wraps
from typing import List, Optional, Tuple

import numpy as np

class MinimaxPlayer:
    def __init__(self, depth: int = 4):
        self.depth = depth  # How many moves to look ahead

    def _get_valid_moves(self, board: np.ndarray) -> List[int]:
        """Return columns where a piece can be dropped"""
        return [col for col in range(7) if board[0][col] == 0]

    def _drop_piece(
        self, board: np.ndarray, col: int, player: int
    ) -> Tuple[bool, np.ndarray]:
        temp_board = board.copy()
        for row in range(5, -1, -1):
            if temp_board[row][col] == 0:
                temp_board[row][col] = player
                return True, temp_board
        return False, board

    def _check_winner(self, board: np.ndarray, player: int) -> bool:
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if all(board[row][col + i] == player for i in range(4)):
                    return True

        # Check vertical
        for row in range(3):
            for col in range(7):
                if all(board[row + i][col] == player for i in range(4)):
                    return True

        # Check diagonal (positive slope)
        for row in range(3):
            for col in range(4):
                if all(board[row + i][col + i] == player for i in range(4)):
                    return True

        # Check diagonal (negative slope)
        for row in range(3, 6):
            for col in range(4):
                if all(board[row - i][col + i] == player for i in range(4)):
                    return True

        return False

    def _evaluate_window(self, window: List[int], player: int) -> int:
        opponent = 3 - player  # Convert between 1 and 2

        score = 0
        piece_count = window.count(player)
        empty_count = window.count(0)
        opponent_count = window.count(opponent)

        if piece_count == 4:
            score += 100
        elif piece_count == 3 and empty_count == 1:
            score += 5
        elif piece_count == 2 and empty_count == 2:
            score += 2
        if opponent_count == 3 and empty_count == 1:
            score -= 4

        return score

    def _evaluate_position(self, board: np.ndarray, player: int) -> int:
        score = 0

        # Center column preference
        center_array = [int(board[i][3]) for i in range(6)]
        center_count = center_array.count(player)
        score += center_count * 3

        # Horizontal
        for row in range(6):
            for col in range(4):
                window = [int(board[row][col + i]) for i in range(4)]
                score += self._evaluate_window(window, player)

        # Vertical
        for row in range(3):
            for col in range(7):
                window = [int(board[row + i][col]) for i in range(4)]
                score += self._evaluate_window(window, player)

        # Diagonal (positive slope)
        for row in range(3):
            for col in range(4):
                window = [int(board[row + i][col + i]) for i in range(4)]
                score += self._evaluate_window(window, player)

        # Diagonal (negative slope)
        for row in range(3, 6):
            for col in range(4):
                window = [int(board[row - i][col + i]) for i in range(4)]
                score += self._evaluate_window(window, player)

        return score

    def _minimax(
        self,
        board: np.ndarray,
        depth: int,
        alpha: int,
        beta: int,
        maximizing_player: bool,
        player: int,
    ) -> Tuple[int, Optional[int]]:
        """
        Minimax algorithm with alpha-beta pruning
        Returns: (score, column)
        """
        valid_moves = self._get_valid_moves(board)

        # Check terminal conditions
        if self._check_winner(board, player):
            return (100000, None) if maximizing_player else (-100000, None)
        if self._check_winner(board, 3 - player):
            return (-100000, None) if maximizing_player else (100000, None)
        if len(valid_moves) == 0:
            return (0, None)
        if depth == 0:
            return (self._evaluate_position(board, player), None)

        if maximizing_player:
            value = -np.inf
            column = np.random.choice(valid_moves)
            for col in valid_moves:
                success, board_copy = self._drop_piece(board, col, player)
                if success:
                    new_score, _ = self._minimax(
                        board_copy, depth - 1, alpha, beta, False, player
                    )
                    if new_score > value:
                        value = new_score
                        column = col
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
            return value, column
        else:
            value = np.inf
            column = np.random.choice(valid_moves)
            for col in valid_moves:
                success, board_copy = self._drop_piece(board, col, 3 - player)
                if success:
                    new_score, _ = self._minimax(
                        board_copy, depth - 1, alpha, beta, True, player
                    )
                    if new_score < value:
                        value = new_score
                        column = col
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
            return value, column

    def choose_action(self, observation: dict) -> int:
        """Choose the best action using minimax"""
        board = np.zeros((6, 7))

        # Convert observation to our board format
        for i in range(6):
            for j in range(7):
                if observation["observation"][i, j, 0] == 1:
                    board[i][j] = 1
                elif observation["observation"][i, j, 1] == 1:
                    board[i][j] = 2

        # Use minimax to find the best move
        _, best_col = self._minimax(board, self.depth, -np.inf, np.inf, True, 2)
        return best_col

    def print_timing_stats(self):
        print("\nTiming Statistics:")
        for method in [
            self.choose_action,
            self._minimax,
            self._evaluate_position,
            self._check_winner,
            self._drop_piece,
        ]:
            times = method.time_stats[method.__name__]
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                calls = len(times)
                print(f"{method.__name__}:")
                print(f"  Calls: {calls}")
                print(f"  Average time: {avg_time * 1000:.3f}ms")
                print(f"  Max time: {max_time * 1000:.3f}ms")
