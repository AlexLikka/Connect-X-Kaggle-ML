"""Quick arena for a checkpoint-backed MCTS player against a submission agent."""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from kaggle_environments import evaluate
import torch

from rl.mcts import run_mcts
from rl.model import load_checkpoint


def load_submission(path):
    namespace = {}
    with open(path, "r", encoding="utf-8") as handle:
        exec(compile(handle.read(), path, "exec"), namespace)
    return namespace["agent"]


def make_mcts_agent(checkpoint, simulations, device):
    model, _ = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()

    def agent(observation, configuration):
        board = list(observation.board)
        policy = run_mcts(
            model,
            board,
            observation.mark,
            simulations=simulations,
            device=device,
            add_noise=False,
        )
        for col in policy.argsort()[::-1]:
            if board[int(col)] == 0:
                return int(col)
        return 0

    return agent


def summarize(rows):
    wins = sum(1 for a, b in rows if a > b)
    losses = sum(1 for a, b in rows if a < b)
    draws = len(rows) - wins - losses
    mean = sum(a for a, _ in rows) / len(rows)
    return wins, losses, draws, mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", default="submission_ml_best.py")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=80)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    subject = make_mcts_agent(args.checkpoint, args.simulations, device)
    opponent = load_submission(args.opponent)

    first = evaluate("connectx", [subject, opponent], num_episodes=args.episodes)
    second_raw = evaluate("connectx", [opponent, subject], num_episodes=args.episodes)
    second = [[row[1], row[0]] for row in second_raw]
    wins, losses, draws, mean = summarize(first + second)
    print(
        f"episodes={len(first) + len(second)} wins={wins} losses={losses} "
        f"draws={draws} mean_reward={mean:.3f}"
    )


if __name__ == "__main__":
    main()
