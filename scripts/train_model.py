"""Train a value model on self-play data and export as submission-ready agent.

Trains a small 2-layer neural network using numpy only. The model predicts
negamax evaluation scores from board features. After training, the weights
are serialized into a standalone submission file.

Usage:
    python scripts/train_model.py --data data/selfplay.npz --epochs 2000 --hidden 256
"""

import argparse
import os
import sys

import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(np.float32)


class ValueModel:
    """Simple 2-layer neural network: input -> hidden(ReLU) -> output(tanh)."""

    def __init__(self, input_dim, hidden_dim=256):
        scale1 = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1).astype(np.float32) * scale2
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        return tanh(self.z2)

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1 @ self.W2 + self.b2
        return tanh(z2)

    def train_step(self, X, y, lr, weight_decay=1e-4):
        batch_size = X.shape[0]
        pred = self.forward(X)

        y_scaled = y / 1e6
        y_scaled = np.clip(y_scaled, -1, 1)

        d_out = (pred - y_scaled.reshape(-1, 1)) * tanh_deriv(self.z2)
        self.dW2 = (self.h1.T @ d_out) / batch_size + weight_decay * self.W2
        self.db2 = d_out.sum(axis=0) / batch_size
        d_h1 = d_out @ self.W2.T
        d_z1 = d_h1 * relu_deriv(self.z1)
        self.dW1 = (self.X.T @ d_z1) / batch_size + weight_decay * self.W1
        self.db1 = d_z1.sum(axis=0) / batch_size

        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

        loss = float(((pred.flatten() - y_scaled) ** 2).mean())
        return loss

    def to_dict(self):
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "hidden_dim": self.W1.shape[1],
        }


def generate_submission(model_path, output_path):
    """Generate a self-contained submission.py with embedded model weights."""
    import json

    with open(model_path, "r") as f:
        model_data = json.load(f)

    hidden_dim = model_data["hidden_dim"]
    W1 = model_data["W1"]
    b1 = model_data["b1"]
    W2 = model_data["W2"]
    b2 = model_data["b2"]

    submission_code = f'''"""ML-enhanced ConnectX agent with embedded neural network weights."""
import math
import random
import time

EMPTY = 0
WIN_SCORE = 1000000
DEFAULT_TIME_LIMIT = 1.65
MAX_SEARCH_DEPTH = 8
TIME_MARGIN = 0.25
HIDDEN_DIM = {hidden_dim}

W1 = {repr(W1)}
b1 = {repr(b1)}
W2 = {repr(W2)}
b2 = {repr(b2)}


def opponent(mark):
    return 1 if mark == 2 else 2


def valid_moves(board, columns):
    return [c for c in range(columns) if board[c] == EMPTY]


def ordered_columns(columns):
    center = columns // 2
    return sorted(range(columns), key=lambda c: abs(c - center))


def next_open_row(board, column, rows, columns):
    for row in range(rows - 1, -1, -1):
        if board[row * columns + column] == EMPTY:
            return row
    return -1


def drop_piece(board, column, mark, rows, columns):
    row = next_open_row(board, column, rows, columns)
    if row < 0:
        return None, -1
    next_board = board[:]
    next_board[row * columns + column] = mark
    return next_board, row


def is_winning_move(board, row, column, mark, rows, columns, inarow):
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


def find_winning_move(board, mark, rows, columns, inarow):
    for column in ordered_columns(columns):
        if board[column] != EMPTY:
            continue
        next_board, row = drop_piece(board, column, mark, rows, columns)
        if row >= 0 and is_winning_move(next_board, row, column, mark, rows, columns, inarow):
            return column
    return None


def _encode_board(board, mark, rows, columns):
    flat = [0.0] * (rows * columns)
    for i, cell in enumerate(board):
        if cell == mark:
            flat[i] = 1.0
        elif cell == 0:
            flat[i] = 0.0
        else:
            flat[i] = -1.0
    return flat


def _extract_features(board, mark, rows, columns, inarow):
    features = _encode_board(board, mark, rows, columns)
    engineered = [0.0] * 14
    center = columns // 2
    center_col = [board[r * columns + center] for r in range(rows)]
    engineered[0] = float(sum(1 for v in center_col if v == mark))
    engineered[1] = float(sum(1 for v in center_col if v == opponent(mark)))

    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        if dr == 0 and dc == 1:
            count_range = columns - inarow + 1
        elif dr == 1 and dc == 0:
            count_range = rows - inarow + 1
        else:
            count_range = rows - inarow + 1
        for start_r in range(count_range if dr else rows):
            for start_c in range(count_range if dc else columns):
                window = []
                valid = True
                for k in range(inarow):
                    r = start_r + k * dr
                    c = start_c + k * dc
                    if not (0 <= r < rows and 0 <= c < columns):
                        valid = False
                        break
                    window.append(board[r * columns + c])
                if not valid:
                    continue
                my_count = window.count(mark)
                opp_count = window.count(opponent(mark))
                empty_count = window.count(0)
                if my_count and opp_count:
                    continue
                if my_count == inarow - 1 and empty_count == 1:
                    engineered[2] += 1.0
                if my_count == inarow - 2 and empty_count == 2:
                    engineered[4] += 1.0
                if opp_count == inarow - 1 and empty_count == 1:
                    engineered[3] += 1.0
                if opp_count == inarow - 2 and empty_count == 2:
                    engineered[5] += 1.0

    engineered[6] = float(sum(1 for c in board[:columns] if c == 0))
    engineered[7] = 1.0 if find_winning_move(board, mark, rows, columns, inarow) is not None else 0.0
    engineered[8] = 1.0 if find_winning_move(board, opponent(mark), rows, columns, inarow) is not None else 0.0
    engineered[9] = float(sum(1 for c in board[:columns] if c == mark))
    engineered[10] = float(sum(1 for c in board[:columns] if c == opponent(mark)))
    engineered[11] = float(board.index(mark)) if mark in board else float(columns)
    engineered[12] = float(board.index(opponent(mark))) if opponent(mark) in board else float(columns)
    engineered[13] = len(board) / float(rows * columns)
    return features + engineered


def relu_val(x):
    return x if x > 0 else 0.0


def tanh_val(x):
    if x > 5.0:
        return 1.0
    if x < -5.0:
        return -1.0
    import math
    exp2x = math.exp(2.0 * x)
    return (exp2x - 1.0) / (exp2x + 1.0)


def nn_predict(features):
    hd = HIDDEN_DIM
    h1 = [0.0] * hd
    for j in range(hd):
        s = b1[j]
        for i in range(len(features)):
            s += features[i] * W1[i][j]
        h1[j] = relu_val(s)
    s = b2[0]
    for j in range(hd):
        s += h1[j] * W2[j][0]
    return tanh_val(s)


def evaluate_board(board, mark, rows, columns, inarow):
    features = _extract_features(board, mark, rows, columns, inarow)
    raw = nn_predict(features)
    return int(raw * WIN_SCORE)


def score_move_for_ordering(board, column, mark, rows, columns, inarow):
    next_board, row = drop_piece(board, column, mark, rows, columns)
    if row < 0:
        return -math.inf
    if is_winning_move(next_board, row, column, mark, rows, columns, inarow):
        return WIN_SCORE
    danger = -500000 if find_winning_move(next_board, opponent(mark), rows, columns, inarow) is not None else 0
    center_bonus = 20 - abs(column - columns // 2) * 3
    return danger + center_bonus + evaluate_board(next_board, mark, rows, columns, inarow)


def order_moves(board, moves, mark, rows, columns, inarow):
    return sorted(
        moves,
        key=lambda c: score_move_for_ordering(board, c, mark, rows, columns, inarow),
        reverse=True,
    )


class SearchTimeout(Exception):
    pass


def negamax(board, mark, depth, alpha, beta, rows, columns, inarow, deadline, cache):
    if time.time() >= deadline:
        raise SearchTimeout
    moves = valid_moves(board, columns)
    if not moves:
        return 0, None
    immediate = find_winning_move(board, mark, rows, columns, inarow)
    if immediate is not None:
        return WIN_SCORE + depth, immediate
    if depth == 0:
        return evaluate_board(board, mark, rows, columns, inarow), None
    original_alpha = alpha
    key = (tuple(board), mark, depth)
    cached = cache.get(key)
    if cached is not None:
        return cached
    best_score = -math.inf
    best_col = moves[0]
    for column in order_moves(board, moves, mark, rows, columns, inarow):
        next_board, row = drop_piece(board, column, mark, rows, columns)
        if row < 0:
            continue
        if is_winning_move(next_board, row, column, mark, rows, columns, inarow):
            score = WIN_SCORE + depth
        else:
            child_score, _ = negamax(
                next_board, opponent(mark), depth - 1, -beta, -alpha,
                rows, columns, inarow, deadline, cache,
            )
            score = -child_score
        if score > best_score:
            best_score = score
            best_col = column
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    if alpha > original_alpha and best_score < beta:
        cache[key] = (best_score, best_col)
    return best_score, best_col


def choose_action(board, mark, rows, columns, inarow, time_limit=DEFAULT_TIME_LIMIT):
    moves = valid_moves(board, columns)
    if not moves:
        return 0
    winning = find_winning_move(board, mark, rows, columns, inarow)
    if winning is not None:
        return winning
    blocking = find_winning_move(board, opponent(mark), rows, columns, inarow)
    if blocking is not None:
        return blocking
    deadline = time.time() + time_limit
    best_col = moves[0]
    for col in ordered_columns(columns):
        if col in moves:
            best_col = col
            break
    cache = {{}}
    for depth in range(1, MAX_SEARCH_DEPTH + 1):
        try:
            _, column = negamax(
                board, mark, depth, -math.inf, math.inf,
                rows, columns, inarow, deadline, cache,
            )
            if column is not None and column in moves:
                best_col = column
        except SearchTimeout:
            break
    return best_col if best_col in moves else random.choice(moves)


def agent(observation, configuration):
    board = list(observation.board)
    timeout = getattr(configuration, "timeout", None)
    time_limit = DEFAULT_TIME_LIMIT
    if isinstance(timeout, (int, float)):
        time_limit = min(time_limit, max(0.10, timeout - TIME_MARGIN))
    return choose_action(
        board, observation.mark, configuration.rows, configuration.columns,
        configuration.inarow, time_limit=time_limit,
    )
'''

    with open(output_path, "w") as f:
        f.write(submission_code)
    print(f"Generated submission file: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to selfplay.npz")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-output", default="models/model.json")
    parser.add_argument("--submission-output", default="submission_ml.py")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    data = np.load(args.data)
    X = data["X"]
    scores = data["scores"]
    moves = data["moves"]

    n_samples = X.shape[0]
    n_val = int(n_samples * args.val_split)
    indices = np.random.permutation(n_samples)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, y_train = X[train_idx], scores[train_idx]
    X_val, y_val = X[val_idx], scores[val_idx]

    print(f"Data: {n_samples} samples, {n_val} val, {len(train_idx)} train")
    print(f"  X shape: {X.shape}, input_dim={X.shape[1]}")
    print(f"  scores: mean={scores.mean():.1f}, std={scores.std():.1f}")

    model = ValueModel(X.shape[1], args.hidden)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(args.epochs):
        perm = np.random.permutation(len(train_idx))
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_idx), args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            current_lr = args.lr / (1.0 + 0.01 * epoch)
            loss = model.train_step(X_batch, y_batch, current_lr, weight_decay=1e-4)
            total_loss += loss
            n_batches += 1

        if (epoch + 1) % 100 == 0:
            val_pred = model.predict(X_val)
            val_loss = float(((val_pred.flatten() - y_val / 1e6) ** 2).mean())
            avg_loss = total_loss / n_batches
            print(f"  epoch {epoch + 1}/{args.epochs}: train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {
                    "W1": model.W1.copy(),
                    "b1": model.b1.copy(),
                    "W2": model.W2.copy(),
                    "b2": model.b2.copy(),
                }

    print(f"Best val loss: {best_val_loss:.6f}")

    os.makedirs(os.path.dirname(args.model_output) or ".", exist_ok=True)
    import json

    model_data = {
        "W1": best_weights["W1"].tolist(),
        "b1": best_weights["b1"].tolist(),
        "W2": best_weights["W2"].tolist(),
        "b2": best_weights["b2"].tolist(),
        "hidden_dim": args.hidden,
    }

    with open(args.model_output, "w") as f:
        json.dump(model_data, f)
    print(f"Saved model weights to {args.model_output}")

    generate_submission(args.model_output, args.submission_output)
    print(f"Done. Run: python scripts/evaluate_agents.py --validate-submission {args.submission_output}")


if __name__ == "__main__":
    main()
