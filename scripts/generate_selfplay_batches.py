"""Generate self-play data in stable batches and save incrementally.

Usage:
  python scripts/generate_selfplay_batches.py --games 3000 --depth 6 --time-limit 0.5 --batch-save 50 --output data/selfplay_large.npz
"""
import argparse
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ensure local kaggle env
local_kag = ROOT / 'kaggle-environments-0.1.4'
if str(local_kag) not in sys.path:
    sys.path.insert(0, str(local_kag))

from scripts.generate_selfplay_data import play_game_and_extract
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=3000)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--time-limit', type=float, default=0.5)
    p.add_argument('--batch-save', type=int, default=50)
    p.add_argument('--output', type=str, default='data/selfplay_large.npz')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    rows, columns, inarow = 6, 7, 4
    all_X = []
    all_scores = []
    all_moves = []
    start_i = 0

    try:
        for i in range(args.games):
            traj = play_game_and_extract(rows, columns, inarow, args.depth, args.time_limit)
            for feats, score, move in traj:
                all_X.append(feats)
                all_scores.append(score)
                all_moves.append(move)
            if (i + 1) % args.batch_save == 0 or (i + 1) == args.games:
                np.savez(args.output, X=np.array(all_X, dtype=np.float32), scores=np.array(all_scores, dtype=np.float32), moves=np.array(all_moves, dtype=np.int32))
                print(f"  saved after {i+1}/{args.games} games: samples={len(all_X)} -> {args.output}")
    except Exception as e:
        # save partial
        np.savez(args.output, X=np.array(all_X, dtype=np.float32), scores=np.array(all_scores, dtype=np.float32), moves=np.array(all_moves, dtype=np.int32))
        print(f"Exception occurred: {e}. Saved partial data to {args.output}")
        raise

    print(f"Completed {args.games} games, total samples={len(all_X)}")


if __name__ == '__main__':
    main()
