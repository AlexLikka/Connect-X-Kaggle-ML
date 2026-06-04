"""Export policy/value predictions from a checkpoint for later distillation."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rl.connectx import feature_rows_to_tensors
from rl.model import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/selfplay_rich.npz")
    parser.add_argument("--output", default="rl/data/distill_targets.npz")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model, _ = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device).eval()

    data = np.load(args.data)
    states = feature_rows_to_tensors(data["X"])
    legal = states[:, 2, 0, :]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(states), torch.from_numpy(legal.astype(np.float32))),
        batch_size=args.batch_size,
    )

    policies = []
    values = []
    with torch.no_grad():
        for xb, lb in loader:
            xb, lb = xb.to(device), lb.to(device)
            logits, value = model(xb, legal_mask=lb)
            policies.append(torch.softmax(logits, dim=1).cpu().numpy())
            values.append(value.cpu().numpy())

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        X=data["X"],
        policy_targets=np.concatenate(policies).astype(np.float32),
        value_targets=np.concatenate(values).astype(np.float32),
    )
    print(f"saved distillation targets to {args.output}")


if __name__ == "__main__":
    main()
