"""Train value + policy models on self-play data and export as submission-ready agent.

Policy model: predicts which column the search would choose (7-class classification).
Value model: predicts negamax evaluation score (regression).

At inference time, both models run at the root to order moves:
  - Policy gives P(best move | column) for each candidate
  - Value gives estimated score after playing each column
  - Combined score = policy_prob * value_weight + normalized_value * weight

This corresponds to Phase 3 of MODEL_PLAN: "ML move ordering + ML leaf evaluation".

Usage:
    python scripts/train_model.py --data data/selfplay.npz --epochs 3000 --hidden 128
"""

import argparse
import json
import math
import os

import numpy as np

NUM_COLUMNS = 7

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(np.float32)


def softmax(x):
    """Numerically stable softmax for 2D array."""
    x_max = x.max(axis=1, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, labels):
    """Mean cross-entropy loss."""
    n = len(labels)
    log_probs = np.log(probs[np.arange(n), labels] + 1e-10)
    return -log_probs.mean()


class PolicyModel:
    """56 -> hidden(ReLU) -> 7 (softmax logits)."""

    def __init__(self, input_dim, hidden_dim=128):
        scale1 = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, NUM_COLUMNS).astype(np.float32) * scale2
        self.b2 = np.zeros(NUM_COLUMNS, dtype=np.float32)

    def forward(self, X):
        """Returns softmax probabilities."""
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        self.logits = self.h1 @ self.W2 + self.b2
        return softmax(self.logits)

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        logits = h1 @ self.W2 + self.b2
        return softmax(logits)

    def train_step(self, X, labels, lr, weight_decay=1e-4):
        """One step of cross-entropy gradient descent."""
        batch_size = X.shape[0]
        probs = self.forward(X)

        # d_logits = probs - one_hot(labels)
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), labels] -= 1.0
        d_logits /= batch_size

        self.dW2 = self.h1.T @ d_logits + weight_decay * self.W2
        self.db2 = d_logits.sum(axis=0)
        d_h1 = d_logits @ self.W2.T
        d_z1 = d_h1 * relu_deriv(self.z1)
        self.dW1 = self.X.T @ d_z1 + weight_decay * self.W1
        self.db1 = d_z1.sum(axis=0)

        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

        return cross_entropy(probs, labels)

    def to_dict(self):
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "hidden_dim": self.W1.shape[1],
        }


class ValueModel:
    """56 -> hidden(ReLU) -> 1 (tanh)."""

    def __init__(self, input_dim, hidden_dim=128):
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
        return np.tanh(self.z2)

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        h1 = relu(z1)
        z2 = h1 @ self.W2 + self.b2
        return np.tanh(z2)

    def train_step(self, X, y, lr, weight_decay=1e-4):
        batch_size = X.shape[0]
        pred = self.forward(X)
        y_scaled = np.clip(y / 1e6, -1, 1)

        d_out = (pred - y_scaled.reshape(-1, 1)) * (1.0 - pred ** 2)
        self.dW2 = self.h1.T @ d_out / batch_size + weight_decay * self.W2
        self.db2 = d_out.sum(axis=0) / batch_size
        d_h1 = d_out @ self.W2.T
        d_z1 = d_h1 * relu_deriv(self.z1)
        self.dW1 = self.X.T @ d_z1 / batch_size + weight_decay * self.W1
        self.db1 = d_z1.sum(axis=0) / batch_size

        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

        return float(((pred.flatten() - y_scaled) ** 2).mean())

    def to_dict(self):
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "hidden_dim": self.W1.shape[1],
        }


# ---------------------------------------------------------------------------
# Submission generation
# ---------------------------------------------------------------------------

def generate_submission(policy_data, value_data, output_path):
    """Generate a self-contained submission.py with embedded model weights."""
    policy_hidden = policy_data["hidden_dim"]
    value_hidden = value_data["hidden_dim"]

    pW1 = policy_data["W1"]
    pb1 = policy_data["b1"]
    pW2 = policy_data["W2"]
    pb2 = policy_data["b2"]

    vW1 = value_data["W1"]
    vb1 = value_data["b1"]
    vW2 = value_data["W2"]
    vb2 = value_data["b2"]

    # Build the embedded weight constants
    submission_code = f'''"""ML-enhanced ConnectX agent: value + policy for move ordering."""
import math
import random
import time

EMPTY = 0
WIN_SCORE = 1000000
DEFAULT_TIME_LIMIT = 1.65
MAX_SEARCH_DEPTH = 8
TIME_MARGIN = 0.25
POLICY_HIDDEN = {policy_hidden}
VALUE_HIDDEN = {value_hidden}
POLICY_W1 = {repr(pW1)}
POLICY_B1 = {repr(pb1)}
POLICY_W2 = {repr(pW2)}
POLICY_B2 = {repr(pb2)}
VALUE_W1 = {repr(vW1)}
VALUE_B1 = {repr(vb1)}
VALUE_W2 = {repr(vW2)}
VALUE_B2 = {repr(vb2)}


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
    nb = board[:]
    nb[row * columns + column] = mark
    return nb, row


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
    for col in ordered_columns(columns):
        if board[col] != EMPTY:
            continue
        nb, row = drop_piece(board, col, mark, rows, columns)
        if row >= 0 and is_winning_move(nb, row, col, mark, rows, columns, inarow):
            return col
    return None


def _extract_features(board, mark, rows, columns, inarow):
    """Extract 56-dim feature vector: 42 raw + 14 engineered."""
    features = [0.0] * (rows * columns)
    for i, cell in enumerate(board):
        if cell == mark:
            features[i] = 1.0
        elif cell == 0:
            features[i] = 0.0
        else:
            features[i] = -1.0

    eng = [0.0] * 14
    center = columns // 2
    center_col = [board[r * columns + center] for r in range(rows)]
    eng[0] = float(sum(1 for v in center_col if v == mark))
    eng[1] = float(sum(1 for v in center_col if v == opponent(mark)))

    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        if dr == 0 and dc == 1:
            cr = columns - inarow + 1
        elif dr == 1 and dc == 0:
            cr = rows - inarow + 1
        else:
            cr = rows - inarow + 1
        for sr in range(cr if dr else rows):
            for sc in range(cr if dc else columns):
                win = []
                ok = True
                for k in range(inarow):
                    r2 = sr + k * dr
                    c2 = sc + k * dc
                    if not (0 <= r2 < rows and 0 <= c2 < columns):
                        ok = False
                        break
                    win.append(board[r2 * columns + c2])
                if not ok:
                    continue
                mc = win.count(mark)
                oc = win.count(opponent(mark))
                ec = win.count(0)
                if mc and oc:
                    continue
                if mc == inarow - 1 and ec == 1:
                    eng[2] += 1.0
                if mc == inarow - 2 and ec == 2:
                    eng[4] += 1.0
                if oc == inarow - 1 and ec == 1:
                    eng[3] += 1.0
                if oc == inarow - 2 and ec == 2:
                    eng[5] += 1.0

    eng[6] = float(sum(1 for c in board[:columns] if c == 0))
    eng[7] = 1.0 if find_winning_move(board, mark, rows, columns, inarow) is not None else 0.0
    eng[8] = 1.0 if find_winning_move(board, opponent(mark), rows, columns, inarow) is not None else 0.0
    eng[9] = float(sum(1 for c in board[:columns] if c == mark))
    eng[10] = float(sum(1 for c in board[:columns] if c == opponent(mark)))
    eng[11] = float(board.index(mark)) if mark in board else float(columns)
    eng[12] = float(board.index(opponent(mark))) if opponent(mark) in board else float(columns)
    eng[13] = len(board) / float(rows * columns)
    return features + eng


def _relu(x):
    return max(0.0, x)


def _softmax_7(logits):
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    s = sum(e)
    return [v / s for v in e]


def policy_predict(features):
    hd = POLICY_HIDDEN
    h1 = [0.0] * hd
    for j in range(hd):
        s = POLICY_B1[j]
        for i in range(len(features)):
            s += features[i] * POLICY_W1[i][j]
        h1[j] = _relu(s)
    logits = [0.0] * 7
    for j in range(7):
        s = POLICY_B2[j]
        for i in range(hd):
            s += h1[i] * POLICY_W2[i][j]
        logits[j] = s
    return _softmax_7(logits)


def value_predict(features):
    hd = VALUE_HIDDEN
    h1 = [0.0] * hd
    for j in range(hd):
        s = VALUE_B1[j]
        for i in range(len(features)):
            s += features[i] * VALUE_W1[i][j]
        h1[j] = _relu(s)
    s = VALUE_B2[0]
    for j in range(hd):
        s += h1[j] * VALUE_W2[j][0]
    if s > 5.0:
        return 1.0
    if s < -5.0:
        return -1.0
    e = math.exp(2.0 * s)
    return (e - 1.0) / (e + 1.0)


def evaluate_board(board, mark, rows, columns, inarow):
    """Fast hand-crafted evaluation for search leaves."""
    score = 0
    center = columns // 2
    cc = [board[r * columns + center] for r in range(rows)]
    score += cc.count(mark) * 8 - cc.count(opponent(mark)) * 8
    for row in range(rows):
        base = row * columns
        for col in range(columns - inarow + 1):
            w = board[base + col:base + col + inarow]
            mine = w.count(mark)
            theirs = w.count(opponent(mark))
            empt = w.count(0)
            if mine and theirs:
                pass
            elif mine == inarow:
                score += WIN_SCORE
            elif theirs == inarow:
                score -= WIN_SCORE
            elif mine:
                if mine == inarow - 1 and empt == 1:
                    score += 900
                elif mine == inarow - 2 and empt == 2:
                    score += 80
                else:
                    score += mine * mine
            elif theirs:
                if theirs == inarow - 1 and empt == 1:
                    score -= 1200
                elif theirs == inarow - 2 and empt == 2:
                    score -= 120
                else:
                    score -= theirs * theirs
    for col in range(columns):
        for row in range(rows - inarow + 1):
            w = [board[(row + i) * columns + col] for i in range(inarow)]
            mine = w.count(mark)
            theirs = w.count(opponent(mark))
            empt = w.count(0)
            if mine and theirs:
                pass
            elif mine == inarow:
                score += WIN_SCORE
            elif theirs == inarow:
                score -= WIN_SCORE
            elif mine:
                if mine == inarow - 1 and empt == 1:
                    score += 900
                elif mine == inarow - 2 and empt == 2:
                    score += 80
                else:
                    score += mine * mine
            elif theirs:
                if theirs == inarow - 1 and empt == 1:
                    score -= 1200
                elif theirs == inarow - 2 and empt == 2:
                    score -= 120
                else:
                    score -= theirs * theirs
    for row in range(rows - inarow + 1):
        for col in range(columns - inarow + 1):
            w = [board[(row + i) * columns + col + i] for i in range(inarow)]
            mine = w.count(mark)
            theirs = w.count(opponent(mark))
            empt = w.count(0)
            if mine and theirs:
                pass
            elif mine == inarow:
                score += WIN_SCORE
            elif theirs == inarow:
                score -= WIN_SCORE
            elif mine:
                if mine == inarow - 1 and empt == 1:
                    score += 900
                elif mine == inarow - 2 and empt == 2:
                    score += 80
                else:
                    score += mine * mine
            elif theirs:
                if theirs == inarow - 1 and empt == 1:
                    score -= 1200
                elif theirs == inarow - 2 and empt == 2:
                    score -= 120
                else:
                    score -= theirs * theirs
    for row in range(inarow - 1, rows):
        for col in range(columns - inarow + 1):
            w = [board[(row - i) * columns + col + i] for i in range(inarow)]
            mine = w.count(mark)
            theirs = w.count(opponent(mark))
            empt = w.count(0)
            if mine and theirs:
                pass
            elif mine == inarow:
                score += WIN_SCORE
            elif theirs == inarow:
                score -= WIN_SCORE
            elif mine:
                if mine == inarow - 1 and empt == 1:
                    score += 900
                elif mine == inarow - 2 and empt == 2:
                    score += 80
                else:
                    score += mine * mine
            elif theirs:
                if theirs == inarow - 1 and empt == 1:
                    score -= 1200
                elif theirs == inarow - 2 and empt == 2:
                    score -= 120
                else:
                    score -= theirs * theirs
    return score


def score_move_ml(board, column, mark, rows, columns, inarow, policy_probs=None):
    """ML-enhanced move ordering: policy on the current node, value on the child."""
    nb, row = drop_piece(board, column, mark, rows, columns)
    if row < 0:
        return -1e18
    if is_winning_move(nb, row, column, mark, rows, columns, inarow):
        return float(WIN_SCORE)

    if policy_probs is None:
        policy_probs = policy_predict(_extract_features(board, mark, rows, columns, inarow))
    policy_score = policy_probs[column]

    child_mark = opponent(mark)
    child_feats = _extract_features(nb, child_mark, rows, columns, inarow)
    value_score = -value_predict(child_feats)

    heur = evaluate_board(nb, mark, rows, columns, inarow)
    heur_norm = heur / WIN_SCORE

    # Policy ranks the move; value/heuristic estimate the resulting position.
    return policy_score * 0.45 + value_score * 0.35 + heur_norm * 0.20


def order_moves(board, moves, mark, rows, columns, inarow):
    policy_probs = policy_predict(_extract_features(board, mark, rows, columns, inarow))
    return sorted(
        moves,
        key=lambda c: score_move_ml(board, c, mark, rows, columns, inarow, policy_probs=policy_probs),
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
    imm = find_winning_move(board, mark, rows, columns, inarow)
    if imm is not None:
        return WIN_SCORE + depth, imm
    if depth == 0:
        heur = evaluate_board(board, mark, rows, columns, inarow)
        ml_value = value_predict(_extract_features(board, mark, rows, columns, inarow))
        return 0.7 * heur + 0.3 * (ml_value * WIN_SCORE), None
    orig_alpha = alpha
    key = (tuple(board), mark, depth)
    cached = cache.get(key)
    if cached is not None:
        return cached
    best_score = -math.inf
    best_col = moves[0]
    for column in order_moves(board, moves, mark, rows, columns, inarow):
        nb, row = drop_piece(board, column, mark, rows, columns)
        if row < 0:
            continue
        if is_winning_move(nb, row, column, mark, rows, columns, inarow):
            score = WIN_SCORE + depth
        else:
            cs, _ = negamax(nb, opponent(mark), depth - 1, -beta, -alpha,
                            rows, columns, inarow, deadline, cache)
            score = -cs
        if score > best_score:
            best_score = score
            best_col = column
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    if alpha > orig_alpha and best_score < beta:
        cache[key] = (best_score, best_col)
    return best_score, best_col


def choose_action(board, mark, rows, columns, inarow, time_limit=DEFAULT_TIME_LIMIT):
    moves = valid_moves(board, columns)
    if not moves:
        return 0
    win = find_winning_move(board, mark, rows, columns, inarow)
    if win is not None:
        return win
    block = find_winning_move(board, opponent(mark), rows, columns, inarow)
    if block is not None:
        return block

    deadline = time.time() + time_limit
    best_col = moves[0]
    for col in ordered_columns(columns):
        if col in moves:
            best_col = col
            break

    cache = {{}}
    for depth in range(1, MAX_SEARCH_DEPTH + 1):
        try:
            _, column = negamax(board, mark, depth, -math.inf, math.inf,
                                rows, columns, inarow, deadline, cache)
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to selfplay.npz")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden dim for both policy and value models")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-output", default="models/models.json")
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

    X_train, y_train, m_train = X[train_idx], scores[train_idx], moves[train_idx]
    X_val, y_val, m_val = X[val_idx], scores[val_idx], moves[val_idx]

    input_dim = X.shape[1]
    print(f"Data: {n_samples} samples, {n_val} val, {len(train_idx)} train")
    print(f"  X shape: {X.shape}, input_dim={input_dim}")
    print(f"  scores: mean={scores.mean():.1f}, std={scores.std():.1f}")
    print(f"  moves: {np.bincount(moves, minlength=7)}")

    policy_model = PolicyModel(input_dim, args.hidden)
    value_model = ValueModel(input_dim, args.hidden)

    best_policy_loss = float("inf")
    best_value_loss = float("inf")
    best_policy_w = None
    best_value_w = None

    for epoch in range(args.epochs):
        perm = np.random.permutation(len(train_idx))
        total_p_loss = 0.0
        total_v_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_idx), args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            m_batch = m_train[batch_idx]

            current_lr = args.lr / (1.0 + 0.005 * epoch)

            p_loss = policy_model.train_step(X_batch, m_batch, current_lr, weight_decay=1e-4)
            v_loss = value_model.train_step(X_batch, y_batch, current_lr, weight_decay=1e-4)

            total_p_loss += p_loss
            total_v_loss += v_loss
            n_batches += 1

        if (epoch + 1) % 200 == 0:
            p_probs = policy_model.predict(X_val)
            p_loss_val = cross_entropy(p_probs, m_val)

            v_pred = value_model.predict(X_val)
            v_loss_val = float(((v_pred.flatten() - np.clip(y_val / 1e6, -1, 1)) ** 2).mean())

            avg_p = total_p_loss / n_batches
            avg_v = total_v_loss / n_batches
            p_acc = float((p_probs.argmax(axis=1) == m_val).mean())

            print(f"  epoch {epoch+1}/{args.epochs}: "
                  f"policy_loss={avg_p:.4f}(val={p_loss_val:.4f}, acc={p_acc:.3f}) "
                  f"value_loss={avg_v:.6f}(val={v_loss_val:.6f})")

            if p_loss_val < best_policy_loss:
                best_policy_loss = p_loss_val
                best_policy_w = {
                    "W1": policy_model.W1.tolist(),
                    "b1": policy_model.b1.tolist(),
                    "W2": policy_model.W2.tolist(),
                    "b2": policy_model.b2.tolist(),
                }
            if v_loss_val < best_value_loss:
                best_value_loss = v_loss_val
                best_value_w = {
                    "W1": value_model.W1.tolist(),
                    "b1": value_model.b1.tolist(),
                    "W2": value_model.W2.tolist(),
                    "b2": value_model.b2.tolist(),
                }

    print(f"Best policy val loss: {best_policy_loss:.4f}")
    print(f"Best value val loss:  {best_value_loss:.6f}")

    os.makedirs(os.path.dirname(args.model_output) or ".", exist_ok=True)
    # If no improvement was recorded (e.g., few epochs), fall back to final weights
    if best_policy_w is None:
        best_policy_w = {
            "W1": policy_model.W1.tolist(),
            "b1": policy_model.b1.tolist(),
            "W2": policy_model.W2.tolist(),
            "b2": policy_model.b2.tolist(),
        }
    if best_value_w is None:
        best_value_w = {
            "W1": value_model.W1.tolist(),
            "b1": value_model.b1.tolist(),
            "W2": value_model.W2.tolist(),
            "b2": value_model.b2.tolist(),
        }
    best_policy_w["hidden_dim"] = args.hidden
    best_value_w["hidden_dim"] = args.hidden
    model_data = {
        "policy": best_policy_w,
        "value": best_value_w,
        "policy_hidden": args.hidden,
        "value_hidden": args.hidden,
    }
    with open(args.model_output, "w") as f:
        json.dump(model_data, f)
    print(f"Saved model weights to {args.model_output}")

    generate_submission(best_policy_w, best_value_w, args.submission_output)
    print(f"Done. Run: python scripts/evaluate_agents.py --validate-submission {args.submission_output}")


if __name__ == "__main__":
    main()
