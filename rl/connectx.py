"""Small ConnectX helpers shared by the RL training scripts."""

import numpy as np


ROWS = 6
COLUMNS = 7
INAROW = 4
EMPTY = 0


def opponent(mark):
    return 1 if mark == 2 else 2


def valid_moves(board, columns=COLUMNS):
    return [c for c in range(columns) if board[c] == EMPTY]


def next_open_row(board, column, rows=ROWS, columns=COLUMNS):
    for row in range(rows - 1, -1, -1):
        if board[row * columns + column] == EMPTY:
            return row
    return -1


def drop_piece(board, column, mark, rows=ROWS, columns=COLUMNS):
    row = next_open_row(board, column, rows, columns)
    if row < 0:
        return None, -1
    next_board = board[:]
    next_board[row * columns + column] = mark
    return next_board, row


def is_winning_move(board, row, column, mark, rows=ROWS, columns=COLUMNS, inarow=INAROW):
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r = row + sign * dr
            c = column + sign * dc
            while 0 <= r < rows and 0 <= c < columns and board[r * columns + c] == mark:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= inarow:
            return True
    return False


def board_to_tensor(board, mark, rows=ROWS, columns=COLUMNS):
    """Return [own, opponent, legal-column] planes from current player's view."""
    arr = np.array(board, dtype=np.int8).reshape(rows, columns)
    own = (arr == mark).astype(np.float32)
    opp = (arr == opponent(mark)).astype(np.float32)
    legal = np.zeros((rows, columns), dtype=np.float32)
    for col in valid_moves(board, columns):
        legal[:, col] = 1.0
    return np.stack([own, opp, legal], axis=0)


def feature_rows_to_tensors(X, rows=ROWS, columns=COLUMNS):
    """Convert existing 56-dim relative features to CNN planes."""
    raw = X[:, : rows * columns].reshape(-1, rows, columns)
    own = (raw > 0.5).astype(np.float32)
    opp = (raw < -0.5).astype(np.float32)
    legal = np.zeros_like(own, dtype=np.float32)
    legal[:, :, :] = (raw[:, 0:1, :] == 0).astype(np.float32)
    return np.stack([own, opp, legal], axis=1).astype(np.float32)


def mirror_tensor_planes(planes):
    return planes[..., ::-1].copy()


def mirror_policy(policy):
    return policy[::-1].copy()
