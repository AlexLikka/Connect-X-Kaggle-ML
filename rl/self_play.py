"""Generate AlphaZero-style self-play data with MCTS."""

import argparse
import os

import numpy as np
import torch

from rl.connectx import COLUMNS, ROWS, board_to_tensor, drop_piece, is_winning_move, opponent, valid_moves
from rl.mcts import run_mcts, select_action_from_policy
from rl.model import load_checkpoint


def play_game(model, simulations, device, temperature_moves):
    board = [0] * (ROWS * COLUMNS)
    mark = 1
    states = []
    policies = []
    marks = []
    ply = 0

    while valid_moves(board):
        policy = run_mcts(
            model,
            board,
            mark,
            simulations=simulations,
            device=device,
            add_noise=True,
        )
        states.append(board_to_tensor(board, mark))
        policies.append(policy)
        marks.append(mark)

        temperature = 1.0 if ply < temperature_moves else 0.05
        move = select_action_from_policy(policy, temperature=temperature)
        if move not in valid_moves(board):
            move = int(np.argmax(policy))
        board, row = drop_piece(board, move, mark)
        if row >= 0 and is_winning_move(board, row, move, mark):
            winner = mark
            break
        mark = opponent(mark)
        ply += 1
    else:
        winner = 0

    values = []
    for m in marks:
        if winner == 0:
            values.append(0.0)
        elif winner == m:
            values.append(1.0)
        else:
            values.append(-1.0)
    return states, policies, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="rl/data/selfplay_az.npz")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=160)
    parser.add_argument("--temperature-moves", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    model, _ = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()

    all_states, all_policies, all_values = [], [], []
    for game in range(1, args.games + 1):
        states, policies, values = play_game(
            model, args.simulations, device, args.temperature_moves
        )
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)
        if game == 1 or game % 10 == 0:
            print(f"game {game}/{args.games}: samples={len(all_states)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        states=np.array(all_states, dtype=np.float32),
        policies=np.array(all_policies, dtype=np.float32),
        values=np.array(all_values, dtype=np.float32),
    )
    print(f"saved {len(all_states)} samples to {args.output}")


if __name__ == "__main__":
    main()
