"""Fast offline diagnostics for RL checkpoints before running arena."""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rl.connectx import feature_rows_to_tensors
from rl.mcts import predict, run_mcts
from rl.model import load_checkpoint


def load_rich_dataset(path):
    data = np.load(path)
    states = feature_rows_to_tensors(data["X"]).astype(np.float32)
    moves = data["moves"].astype(np.int64)
    values = np.clip(data["scores"] / 1_000_000.0, -1.0, 1.0).astype(np.float32)
    legal = states[:, 2, 0, :].astype(np.float32)
    return TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(moves),
        torch.from_numpy(values),
        torch.from_numpy(legal),
    )


def load_az_dataset(path):
    data = np.load(path)
    states = data["states"].astype(np.float32)
    policies = data["policies"].astype(np.float32)
    values = data["values"].astype(np.float32)
    legal = states[:, 2, 0, :].astype(np.float32)
    return TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(policies),
        torch.from_numpy(values),
        torch.from_numpy(legal),
    )


def eval_rich(model, loader, device):
    n = 0
    ce = 0.0
    mse = 0.0
    acc = 0.0
    with torch.no_grad():
        for xb, mb, vb, lb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            vb = vb.to(device)
            lb = lb.to(device)
            logits, value = model(xb, legal_mask=lb)
            batch_n = len(xb)
            n += batch_n
            ce += F.cross_entropy(logits, mb, reduction="sum").item()
            mse += F.mse_loss(value, vb, reduction="sum").item()
            acc += (logits.argmax(dim=1) == mb).float().sum().item()
    return ce / n, mse / n, acc / n


def eval_az(model, loader, device):
    n = 0
    ce = 0.0
    mse = 0.0
    acc = 0.0
    with torch.no_grad():
        for xb, pb, vb, lb in loader:
            xb = xb.to(device)
            pb = pb.to(device)
            vb = vb.to(device)
            lb = lb.to(device)
            logits, value = model(xb, legal_mask=lb)
            batch_n = len(xb)
            n += batch_n
            ce += (-(pb * F.log_softmax(logits, dim=1)).sum(dim=1).sum().item())
            mse += F.mse_loss(value, vb, reduction="sum").item()
            acc += (logits.argmax(dim=1) == pb.argmax(dim=1)).float().sum().item()
    return ce / n, mse / n, acc / n


def opening_probe(model, device, simulations):
    board = [0] * 42
    raw_policy, raw_value = predict(model, board, 1, device)
    mcts_policy = run_mcts(model, board, 1, simulations=simulations, device=device, add_noise=False)
    return raw_policy, raw_value, mcts_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rich-data", default="data/selfplay_rich.npz")
    parser.add_argument("--az-data", default="rl/data/selfplay_az_c128_b8_s320.npz")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--simulations", type=int, default=80)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    model, payload = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()

    rich_loader = DataLoader(load_rich_dataset(args.rich_data), batch_size=args.batch_size, shuffle=False)
    az_loader = DataLoader(load_az_dataset(args.az_data), batch_size=args.batch_size, shuffle=False)

    rich_ce, rich_mse, rich_acc = eval_rich(model, rich_loader, device)
    az_ce, az_mse, az_acc = eval_az(model, az_loader, device)
    raw_policy, raw_value, mcts_policy = opening_probe(model, device, args.simulations)

    print(f"checkpoint={args.checkpoint}")
    print(f"metadata={payload.get('metadata', {})}")
    print(
        f"rich_ce={rich_ce:.4f} rich_mse={rich_mse:.5f} rich_acc={rich_acc:.3f} "
        f"| az_ce={az_ce:.4f} az_mse={az_mse:.4f} az_acc={az_acc:.3f}"
    )
    print(
        f"opening_raw_argmax={int(np.argmax(raw_policy))} opening_raw_value={float(raw_value):.4f} "
        f"opening_raw_policy={[round(float(x), 4) for x in raw_policy]}"
    )
    print(
        f"opening_mcts_argmax={int(np.argmax(mcts_policy))} "
        f"opening_mcts_policy={[round(float(x), 4) for x in mcts_policy]}"
    )


if __name__ == "__main__":
    main()
