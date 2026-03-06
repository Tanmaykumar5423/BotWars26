import numpy as np


class RuleBot:
    """
    A simple rule-based Connect 4 Plus bot.

    Priority order:
      1. Win immediately if possible.
      2. Block the opponent's winning move.
      3. Avoid columns that give the opponent a win on their next turn.
      4. Prefer the centre column, then adjacent columns.
    """

    def __init__(self):
        pass

    # ── public interface ─────────────────────────────────────────────────

    def act(self, observation):
        board = self._build_board(observation["observation"])
        action_mask = observation["action_mask"]
        valid = [c for c in range(7) if action_mask[c] == 1]

        # 1. Win if we can
        for col in valid:
            if self._is_winning_move(board, col, piece=1):
                return col

        # 2. Block opponent win
        for col in valid:
            if self._is_winning_move(board, col, piece=2):
                return col

        # 3. Filter out columns that set up an opponent win
        safe = []
        for col in valid:
            row = self._drop_row(board, col)
            if row < 0:
                continue
            board[row][col] = 1  # simulate our move
            gives_win = False
            if row - 1 >= 0 and board[row - 1][col] == 0:
                board[row - 1][col] = 2
                if self._check_four(board, 2):
                    gives_win = True
                board[row - 1][col] = 0
            board[row][col] = 0
            if not gives_win:
                safe.append(col)

        candidates = safe if safe else valid

        # 4. Prefer centre columns (3 > 2,4 > 1,5 > 0,6)
        preference = [3, 2, 4, 1, 5, 0, 6]
        for col in preference:
            if col in candidates:
                return col

        return candidates[0]

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_board(obs):
        """1 = my pieces, 2 = opponent, 3 = neutral, 0 = empty."""
        board = np.zeros((6, 7), dtype=np.int8)
        board[obs[:, :, 0] == 1] = 1
        board[obs[:, :, 1] == 1] = 2
        board[obs[:, :, 2] == 1] = 3
        return board

    @staticmethod
    def _drop_row(board, col):
        """Row where a piece would land, or -1 if full."""
        for r in range(5, -1, -1):
            if board[r][col] == 0:
                return r
        return -1

    def _is_winning_move(self, board, col, piece):
        row = self._drop_row(board, col)
        if row < 0:
            return False
        board[row][col] = piece
        won = self._check_four(board, piece)
        board[row][col] = 0
        return won

    @staticmethod
    def _check_four(board, piece):
        """Return True if piece has four in a row."""
        for r in range(6):
            for c in range(4):
                if (
                    board[r][c] == piece
                    and board[r][c + 1] == piece
                    and board[r][c + 2] == piece
                    and board[r][c + 3] == piece
                ):
                    return True
        for c in range(7):
            for r in range(3):
                if (
                    board[r][c] == piece
                    and board[r + 1][c] == piece
                    and board[r + 2][c] == piece
                    and board[r + 3][c] == piece
                ):
                    return True
        for c in range(4):
            for r in range(3):
                if (
                    board[r][c] == piece
                    and board[r + 1][c + 1] == piece
                    and board[r + 2][c + 2] == piece
                    and board[r + 3][c + 3] == piece
                ):
                    return True
        for c in range(4):
            for r in range(3, 6):
                if (
                    board[r][c] == piece
                    and board[r - 1][c + 1] == piece
                    and board[r - 2][c + 2] == piece
                    and board[r - 3][c + 3] == piece
                ):
                    return True
        return False
