"""Generate self-play training data using the strong search agent.

Plays the search agent against itself and extracts (board, score, move)
tuples from the root of each search tree. The negamax evaluation scores
serve as regression targets for learning a value function, while the
selected columns train a policy model.

Usage:
    python scripts/generate_selfplay_data.py --games 500 --depth 5 --output data/selfplay.npz
"""

import argparse
import os
import sys
import time

from kaggle_environments import make

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

import agents.search_agent as search_module


def _encode_board(board, mark, rows, columns):
    """Player-relative encoding: +1 current, -1 opponent, 0 empty."""
    flat = np.zeros(rows * columns, dtype=np.float32)
    for i, cell in enumerate(board):
        if cell == mark:
            flat[i] = 1.0
        elif cell == 0:
            flat[i] = 0.0
        else:
            flat[i] = -1.0
    return flat


def _extract_features(board, mark, rows, columns, inarow):
    """Extended feature vector: raw board + engineered features."""
    features = _encode_board(board, mark, rows, columns)

    engineered = np.zeros(14, dtype=np.float32)
    center = columns // 2
    center_col = [board[r * columns + center] for r in range(rows)]
    engineered[0] = sum(1 for v in center_col if v == mark)
    engineered[1] = sum(1 for v in center_col if v == search_module.opponent(mark))

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

                my = window.count(mark)
                opp = window.count(search_module.opponent(mark))
                empty = window.count(0)

                if my and opp:
                    continue
                if my == inarow - 1 and empty == 1:
                    engineered[2] += 1
                if my == inarow - 2 and empty == 2:
                    engineered[4] += 1
                if opp == inarow - 1 and empty == 1:
                    engineered[3] += 1
                if opp == inarow - 2 and empty == 2:
                    engineered[5] += 1

    engineered[6] = sum(1 for c in board[:columns] if c == 0)
    engineered[7] = 1.0 if search_module.find_winning_move(
        board, mark, rows, columns, inarow
    ) is not None else 0.0
    engineered[8] = 1.0 if search_module.find_winning_move(
        board, search_module.opponent(mark), rows, columns, inarow
    ) is not None else 0.0
    engineered[9] = sum(1 for c in board[:columns] if c == mark)
    engineered[10] = sum(1 for c in board[:columns] if c == search_module.opponent(mark))
    engineered[11] = board.index(mark) if mark in board else columns
    engineered[12] = board.index(search_module.opponent(mark)) if search_module.opponent(mark) in board else columns
    engineered[13] = len(board) / (rows * columns)

    return np.concatenate([features, engineered])


def play_game_and_extract(rows, columns, inarow, depth, time_limit):
    """Play one self-play game and return list of (features, score, move)."""
    env = make("connectx", debug=False)
    board = list(env.state[0].observation.board)
    mark = 1

    search_module.MAX_SEARCH_DEPTH = depth
    search_module.DEFAULT_TIME_LIMIT = time_limit

    trajectory = []
    while True:
        valid = search_module.valid_moves(board, columns)
        if not valid:
            break

        score, best_col = search_module.negamax(
            board,
            mark,
            depth,
            -float("inf"),
            float("inf"),
            rows,
            columns,
            inarow,
            time.time() + time_limit,
            {},
        )

        feats = _extract_features(board, mark, rows, columns, inarow)
        trajectory.append((feats, score, best_col if best_col in valid else valid[0]))

        board, row = search_module.drop_piece(board, best_col if best_col in valid else valid[0], mark, rows, columns)
        if row < 0:
            break

        if search_module.is_winning_move(board, row, best_col if best_col in valid else valid[0], mark, rows, columns, inarow):
            for i in range(len(trajectory)):
                f, s, m = trajectory[i]
                if i == len(trajectory) - 1:
                    trajectory[i] = (f, search_module.WIN_SCORE, m)
                else:
                    trajectory[i] = (f, -search_module.WIN_SCORE, m)
            break

        mark = search_module.opponent(mark)

    search_module.MAX_SEARCH_DEPTH = 8
    search_module.DEFAULT_TIME_LIMIT = 1.65
    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--depth", type=int, default=5, help="Search depth for data generation")
    parser.add_argument("--time-limit", type=float, default=0.5)
    parser.add_argument("--output", default="data/selfplay.npz")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    rows, columns, inarow = 6, 7, 4

    all_X, all_scores, all_moves = [], [], []
    start = time.time()

    for game_idx in range(args.games):
        if (game_idx + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (game_idx + 1) / elapsed * 60
            print(f"  game {game_idx + 1}/{args.games} ({rate:.0f} games/min, {len(all_X)} samples)")

        traj = play_game_and_extract(rows, columns, inarow, args.depth, args.time_limit)
        for feats, score, move in traj:
            all_X.append(feats)
            all_scores.append(score)
            all_moves.append(move)

    X = np.array(all_X, dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    moves = np.array(all_moves, dtype=np.int32)

    np.savez(args.output, X=X, scores=scores, moves=moves)
    print(f"\nSaved {len(X)} samples to {args.output}")
    print(f"  X shape: {X.shape}")
    print(f"  scores: mean={scores.mean():.1f}, std={scores.std():.1f}, min={scores.min():.1f}, max={scores.max():.1f}")
    print(f"  moves: {len(moves)} actions, columns distribution: {np.bincount(moves, minlength=7)}")


if __name__ == "__main__":
    main()
