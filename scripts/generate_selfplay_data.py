"""Generate stronger self-play training data for ConnectX.

This version exports richer supervision than the original script:
  - root search score for the current position
  - best move label
  - per-column root scores for every legal action
  - legal-move mask
  - optional mirrored augmentation
  - optional random opening moves for broader state coverage

Usage:
    python scripts/generate_selfplay_data.py --games 500 --depth 7 --output data/selfplay.npz
"""

import argparse
import os
import random
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from kaggle_environments import make

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
    engineered[10] = sum(
        1 for c in board[:columns] if c == search_module.opponent(mark)
    )
    engineered[11] = board.index(mark) if mark in board else columns
    opp_mark = search_module.opponent(mark)
    engineered[12] = board.index(opp_mark) if opp_mark in board else columns
    engineered[13] = len(board) / (rows * columns)

    return np.concatenate([features, engineered])


def _mirror_board(board, rows, columns):
    mirrored = [0] * len(board)
    for row in range(rows):
        base = row * columns
        for col in range(columns):
            mirrored[base + (columns - 1 - col)] = board[base + col]
    return mirrored


def _mirror_vector(vec):
    return vec[::-1].copy()


def _evaluate_all_moves(board, mark, rows, columns, inarow, depth):
    """Compute root scores for every legal move from the current position."""
    valid_moves = search_module.valid_moves(board, columns)
    move_scores = np.full(columns, -float(search_module.WIN_SCORE), dtype=np.float32)
    valid_mask = np.zeros(columns, dtype=np.float32)
    cache = {}

    for col in valid_moves:
        valid_mask[col] = 1.0
        next_board, row = search_module.drop_piece(board, col, mark, rows, columns)
        if row < 0:
            continue
        if search_module.is_winning_move(
            next_board, row, col, mark, rows, columns, inarow
        ):
            score = search_module.WIN_SCORE + depth
        elif depth <= 1:
            score = search_module.evaluate_board(
                next_board, mark, rows, columns, inarow
            )
        else:
            child_score, _ = search_module.negamax(
                next_board,
                search_module.opponent(mark),
                depth - 1,
                -float("inf"),
                float("inf"),
                rows,
                columns,
                inarow,
                time.time() + 3600.0,
                cache,
            )
            score = -child_score
        move_scores[col] = float(score)

    best_col = max(valid_moves, key=lambda c: move_scores[c])
    best_score = float(move_scores[best_col])
    return best_score, best_col, move_scores, valid_mask


def _record_sample(samples, board, mark, rows, columns, inarow, score, best_col, move_scores, valid_mask):
    features = _extract_features(board, mark, rows, columns, inarow)
    samples.append(
        {
            "features": features.astype(np.float32),
            "score": np.float32(score),
            "move": np.int32(best_col),
            "move_scores": move_scores.astype(np.float32),
            "valid_mask": valid_mask.astype(np.float32),
        }
    )


def _append_with_mirror(samples, board, mark, rows, columns, inarow, score, best_col, move_scores, valid_mask, mirror_augment):
    _record_sample(
        samples, board, mark, rows, columns, inarow, score, best_col, move_scores, valid_mask
    )
    if not mirror_augment:
        return

    mirrored_board = _mirror_board(board, rows, columns)
    mirrored_move = columns - 1 - best_col
    mirrored_scores = _mirror_vector(move_scores)
    mirrored_mask = _mirror_vector(valid_mask)
    _record_sample(
        samples,
        mirrored_board,
        mark,
        rows,
        columns,
        inarow,
        score,
        mirrored_move,
        mirrored_scores,
        mirrored_mask,
    )


def play_game_and_extract(rows, columns, inarow, depth, random_opening_moves, mirror_augment, rng):
    """Play one self-play game and return a list of rich training samples."""
    env = make("connectx", debug=False)
    board = list(env.state[0].observation.board)
    mark = 1

    search_module.MAX_SEARCH_DEPTH = depth
    samples = []
    opening_budget = rng.randint(0, random_opening_moves) if random_opening_moves > 0 else 0
    ply = 0

    while True:
        valid = search_module.valid_moves(board, columns)
        if not valid:
            break

        score, best_col, move_scores, valid_mask = _evaluate_all_moves(
            board, mark, rows, columns, inarow, depth
        )
        _append_with_mirror(
            samples,
            board,
            mark,
            rows,
            columns,
            inarow,
            score,
            best_col,
            move_scores,
            valid_mask,
            mirror_augment,
        )

        if ply < opening_budget:
            chosen_col = rng.choice(valid)
        else:
            chosen_col = best_col if best_col in valid else valid[0]

        board, row = search_module.drop_piece(board, chosen_col, mark, rows, columns)
        if row < 0:
            break

        if search_module.is_winning_move(
            board, row, chosen_col, mark, rows, columns, inarow
        ):
            break

        mark = search_module.opponent(mark)
        ply += 1

    search_module.MAX_SEARCH_DEPTH = 8
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument(
        "--depth", type=int, default=6, help="Search depth for data generation"
    )
    parser.add_argument(
        "--random-opening-moves",
        type=int,
        default=2,
        help="Sample 0..N random opening plies each game for broader coverage",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for opening perturbations"
    )
    parser.add_argument(
        "--no-mirror-augment",
        action="store_true",
        help="Disable left-right mirroring augmentation",
    )
    parser.add_argument("--output", default="data/selfplay.npz")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    rows, columns, inarow = 6, 7, 4
    rng = random.Random(args.seed)
    mirror_augment = not args.no_mirror_augment

    all_X = []
    all_scores = []
    all_moves = []
    all_move_scores = []
    all_valid_masks = []
    start = time.time()

    for game_idx in range(args.games):
        if (game_idx + 1) % 25 == 0:
            elapsed = time.time() - start
            rate = (game_idx + 1) / max(elapsed, 1e-6) * 60
            print(
                f"  game {game_idx + 1}/{args.games} "
                f"({rate:.0f} games/min, {len(all_X)} samples)"
            )

        game_samples = play_game_and_extract(
            rows,
            columns,
            inarow,
            args.depth,
            args.random_opening_moves,
            mirror_augment,
            rng,
        )
        for sample in game_samples:
            all_X.append(sample["features"])
            all_scores.append(sample["score"])
            all_moves.append(sample["move"])
            all_move_scores.append(sample["move_scores"])
            all_valid_masks.append(sample["valid_mask"])

    X = np.array(all_X, dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    moves = np.array(all_moves, dtype=np.int32)
    move_scores = np.array(all_move_scores, dtype=np.float32)
    valid_masks = np.array(all_valid_masks, dtype=np.float32)

    np.savez(
        args.output,
        X=X,
        scores=scores,
        moves=moves,
        move_scores=move_scores,
        valid_masks=valid_masks,
    )
    print(f"\nSaved {len(X)} samples to {args.output}")
    print(f"  X shape: {X.shape}")
    print(
        "  scores: "
        f"mean={scores.mean():.1f}, std={scores.std():.1f}, "
        f"min={scores.min():.1f}, max={scores.max():.1f}"
    )
    print(f"  moves distribution: {np.bincount(moves, minlength=7)}")
    legal_counts = valid_masks.sum(axis=1)
    print(
        "  legal moves/sample: "
        f"mean={legal_counts.mean():.2f}, min={legal_counts.min():.0f}, max={legal_counts.max():.0f}"
    )


if __name__ == "__main__":
    main()
