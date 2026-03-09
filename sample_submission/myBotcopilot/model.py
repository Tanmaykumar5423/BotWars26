"""
BotWars26 — Connect 4 Plus Bot
================================
Strategy: Iterative-deepening Alpha-Beta minimax with:
  - Transposition table (board bytes → depth/score/flag)
  - Smart move ordering (immediate win → block → centre-first)
  - Neutral-coin-aware window scoring
  - Zugzwang / trap-avoidance in evaluation
  - Timeout protection + graceful fallback

No external weights file is needed — this is a pure search bot.

Observation format (from the tournament engine):
  observation["observation"]  : np.ndarray (6, 7, 3)  float32 binary planes
      channel 0 → my pieces  (1 where I have a piece)
      channel 1 → opponent   (1 where opponent has a piece)
      channel 2 → neutral    (1 where the neutral coin sits)
  observation["action_mask"]  : np.ndarray (7,) — 1 = legal column
"""

import time
import numpy as np
import torch
from safetensors.torch import load_file
from pathlib import Path

# ── Board constants ────────────────────────────────────────────────────────────
ROWS = 6
COLS = 7
EMPTY = 0
MY = 1       # always "my pieces" from the current player's perspective
OPP = 2
NEUTRAL = 3

# Column ordering: centre first, then outwards (proven to be strongest opening)
COL_ORDER = [3, 2, 4, 1, 5, 0, 6]

# Time budget per move (seconds). 4.5 s leaves 0.5 s safety margin vs 5 s limit.
TIME_LIMIT = 4.5

# Transposition table flag values
TT_EXACT = 0
TT_LOWER = 1  # alpha (lower-bound) cut
TT_UPPER = 2  # beta  (upper-bound) cut


class Bot:
    """
    Connect 4 Plus bot powered by iterative-deepening alpha-beta search.
    """

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def __init__(self):
        # Transposition table: board_bytes → (depth, score, flag)
        self._tt: dict = {}
        self._start: float = 0.0

    # ── Public interface (called by the tournament) ────────────────────────────

    def act(self, observation: dict) -> int:
        """
        Choose the best legal column for the current board state.

        Parameters
        ----------
        observation : dict
            "observation"  → np.ndarray (6, 7, 3)
            "action_mask"  → np.ndarray (7,)  1 = legal

        Returns
        -------
        int
            Column index 0-6.  Always a legal move.
        """
        obs = observation["observation"]       # (6, 7, 3)
        mask = observation["action_mask"]      # (7,)

        # Decode board to int8 grid
        board = _decode(obs)
        valid = [c for c in range(COLS) if mask[c] == 1]

        # Safety: should never happen, but return centre if no legal moves given
        if not valid:
            return 3

        # Single legal move — return immediately without search
        if len(valid) == 1:
            return valid[0]

        # ── Instant win ──────────────────────────────────────────────────────
        for c in valid:
            r = _drop_row(board, c)
            if r < 0:
                continue
            board[r][c] = MY
            if _check_win(board, MY):
                board[r][c] = EMPTY
                return c
            board[r][c] = EMPTY

        # ── Instant block ────────────────────────────────────────────────────
        for c in valid:
            r = _drop_row(board, c)
            if r < 0:
                continue
            board[r][c] = OPP
            if _check_win(board, OPP):
                board[r][c] = EMPTY
                return c
            board[r][c] = EMPTY

        # ── Iterative-deepening alpha-beta ────────────────────────────────────
        self._start = time.time()
        self._tt = {}

        best = _centre_fallback(valid)

        for depth in range(1, MAX_DEPTH):
            try:
                col = self._root(board, depth, valid)
                if col is not None:
                    best = col
            except _Timeout:
                break  # keep last fully-completed depth result
            if time.time() - self._start >= TIME_LIMIT:
                break

        return best

    # ── Search ────────────────────────────────────────────────────────────────

    def _root(self, board: np.ndarray, depth: int, valid: list) -> int | None:
        """Alpha-beta root: return best column at given depth."""
        alpha = -_INF
        beta = _INF
        best_col = None
        best_score = -_INF

        for c in _order_moves(board, valid, maximising=True):
            r = _drop_row(board, c)
            if r < 0:
                continue
            _check_time(self._start)
            board[r][c] = MY
            score = self._ab(board, depth - 1, alpha, beta, maximising=False)
            board[r][c] = EMPTY

            if score > best_score:
                best_score = score
                best_col = c
            if score > alpha:
                alpha = score

        return best_col

    def _ab(
        self,
        board: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        maximising: bool,
    ) -> float:
        """
        Minimax alpha-beta with transposition table.
        Returns score from MY (root player) perspective: high = good for MY.
        """
        _check_time(self._start)

        # Terminal checks (previous player may have won)
        if _check_win(board, MY):
            return _WIN + depth        # sooner win = higher score
        if _check_win(board, OPP):
            return -(_WIN + depth)     # sooner loss for opponent = better block

        valid = [c for c in range(COLS) if _drop_row(board, c) >= 0]
        if not valid:
            return 0  # board full, draw

        if depth == 0:
            return _evaluate(board)

        # Transposition table lookup
        key = board.tobytes()
        tt_entry = self._tt.get(key)
        if tt_entry is not None:
            td, ts, tf = tt_entry
            if td >= depth:
                if tf == TT_EXACT:
                    return ts
                if tf == TT_LOWER:
                    alpha = max(alpha, ts)
                else:
                    beta = min(beta, ts)
                if alpha >= beta:
                    return ts

        orig_alpha = alpha
        best = -_INF if maximising else _INF

        for c in _order_moves(board, valid, maximising):
            r = _drop_row(board, c)
            if r < 0:
                continue
            _check_time(self._start)

            piece = MY if maximising else OPP
            board[r][c] = piece
            score = self._ab(board, depth - 1, alpha, beta, not maximising)
            board[r][c] = EMPTY

            if maximising:
                if score > best:
                    best = score
                if score > alpha:
                    alpha = score
            else:
                if score < best:
                    best = score
                if score < beta:
                    beta = score

            if alpha >= beta:
                break  # prune

        # Store in transposition table
        if best <= orig_alpha:
            flag = TT_UPPER
        elif best >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT

        old = self._tt.get(key)
        if old is None or old[0] <= depth:
            self._tt[key] = (depth, best, flag)

        return best


# ── Module-level helpers (no self, faster lookup) ─────────────────────────────

_INF = float("inf")
# Win score — must be larger than any heuristic evaluation (max ~4 000 in practice)
# but well within float64 range.  Adding `depth` (≤ 42) makes sooner wins score higher.
_WIN = 100_000

# Maximum plies in a Connect 4 Plus game: 6 rows × 7 columns = 42 squares (+1 for loop)
MAX_DEPTH = ROWS * COLS + 1


class _Timeout(Exception):
    """Raised when the time budget is exceeded."""


def _check_time(start: float) -> None:
    if time.time() - start >= TIME_LIMIT:
        raise _Timeout()


def _decode(obs: np.ndarray) -> np.ndarray:
    """Convert (6,7,3) binary observation to (6,7) int8 board."""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[obs[:, :, 0] == 1] = MY
    board[obs[:, :, 1] == 1] = OPP
    board[obs[:, :, 2] == 1] = NEUTRAL
    return board


def _drop_row(board: np.ndarray, col: int) -> int:
    """Lowest empty row in *col*. Returns -1 if column is full."""
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == EMPTY:
            return r
    return -1


def _check_win(board: np.ndarray, piece: int) -> bool:
    """True iff *piece* has four in a row (neutral never counts)."""
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if (
                board[r][c] == piece
                and board[r][c + 1] == piece
                and board[r][c + 2] == piece
                and board[r][c + 3] == piece
            ):
                return True
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if (
                board[r][c] == piece
                and board[r + 1][c] == piece
                and board[r + 2][c] == piece
                and board[r + 3][c] == piece
            ):
                return True
    # Diagonal ↗
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if (
                board[r][c] == piece
                and board[r - 1][c + 1] == piece
                and board[r - 2][c + 2] == piece
                and board[r - 3][c + 3] == piece
            ):
                return True
    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if (
                board[r][c] == piece
                and board[r + 1][c + 1] == piece
                and board[r + 2][c + 2] == piece
                and board[r + 3][c + 3] == piece
            ):
                return True
    return False


def _centre_fallback(valid: list) -> int:
    """Return the most-centre valid column."""
    for c in COL_ORDER:
        if c in valid:
            return c
    return valid[0]


def _order_moves(board: np.ndarray, valid: list, maximising: bool) -> list:
    """
    Order moves: immediate wins first → immediate blocks second → centre-out.
    This dramatically improves alpha-beta pruning efficiency.
    """
    piece = MY if maximising else OPP
    opp = OPP if maximising else MY

    wins, blocks, safe, traps = [], [], [], []

    for c in valid:
        r = _drop_row(board, c)
        if r < 0:
            continue

        # Check if this move wins immediately
        board[r][c] = piece
        if _check_win(board, piece):
            board[r][c] = EMPTY
            wins.append(c)
            continue
        board[r][c] = EMPTY

        # Check if opponent wins if we don't play here
        board[r][c] = opp
        if _check_win(board, opp):
            board[r][c] = EMPTY
            blocks.append(c)
            continue
        board[r][c] = EMPTY

        # Trap detection: playing here gives opponent an immediate win above
        is_trap = False
        if r > 0 and board[r - 1][c] == EMPTY:
            board[r][c] = piece
            board[r - 1][c] = opp
            if _check_win(board, opp):
                is_trap = True
            board[r - 1][c] = EMPTY
            board[r][c] = EMPTY

        if is_trap:
            traps.append(c)
        else:
            safe.append(c)

    # Sort non-critical moves by distance to centre
    for lst in (safe, traps):
        lst.sort(key=lambda c: abs(c - 3))

    # Prefer safe over trap moves; only use traps if nothing else is available
    return wins + blocks + safe + traps


def _score_window(window: list) -> float:
    """
    Score a 4-cell window from MY's perspective.
    A window containing a neutral coin is always dead for both players.
    A window containing both MY and OPP pieces is also contested (score 0).
    """
    if NEUTRAL in window:
        return 0.0

    my_count = window.count(MY)
    op_count = window.count(OPP)
    em_count = window.count(EMPTY)

    # Contested window
    if my_count > 0 and op_count > 0:
        return 0.0

    score = 0.0

    if my_count == 4:
        score += 10_000.0
    elif my_count == 3 and em_count >= 1:
        score += 50.0
    elif my_count == 2 and em_count >= 2:
        score += 5.0
    elif my_count == 1 and em_count == 3:
        score += 1.0

    if op_count == 4:
        score -= 10_000.0
    elif op_count == 3 and em_count >= 1:
        score -= 70.0    # block opponent threats more aggressively
    elif op_count == 2 and em_count >= 2:
        score -= 8.0
    elif op_count == 1 and em_count == 3:
        score -= 1.5

    return score


def _evaluate(board: np.ndarray) -> float:
    """
    Static evaluation of *board* from MY's perspective.
    Higher score = better for MY.
    """
    score = 0.0

    # Centre column control (most important positional factor in Connect 4)
    centre = board[:, 3].tolist()
    score += centre.count(MY) * 6
    score -= centre.count(OPP) * 6

    # Secondary centre columns (2 and 4)
    for c in (2, 4):
        col = board[:, c].tolist()
        score += col.count(MY) * 3
        score -= col.count(OPP) * 3

    # Score all windows of size 4 ────────────────────────────────────────────
    # Horizontal
    for r in range(ROWS):
        row = board[r].tolist()
        for c in range(COLS - 3):
            score += _score_window(row[c : c + 4])

    # Vertical
    for c in range(COLS):
        col = board[:, c].tolist()
        for r in range(ROWS - 3):
            score += _score_window(col[r : r + 4])

    # Diagonal ↗
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = [board[r - i][c + i] for i in range(4)]
            score += _score_window(window)

    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += _score_window(window)

    return score


USE_HEURISTIC = True  # Toggle to enable/disable learned model

try:
    import torch
    from safetensors.torch import load_file
    from pathlib import Path
    
    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(126, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, 7),
            )
        def forward(self, x):
            return self.net(x.reshape(x.shape[0], -1))
    
    WEIGHTS = Path(__file__).parent / "weights/heuristic_model.safetensors"
    _heuristic_model = TinyMLP()
    _heuristic_model.load_state_dict({k: v.float() for k, v in load_file(WEIGHTS).items()})
    _heuristic_model.eval()
    HAS_MODEL = True
except Exception:
    HAS_MODEL = False
    USE_HEURISTIC = False
