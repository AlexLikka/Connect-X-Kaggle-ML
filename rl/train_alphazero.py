"""Train the policy-value network on AlphaZero-style self-play records."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from rl.model import PolicyValueNet, load_checkpoint, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="npz from rl/self_play.py")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--output", default="rl/checkpoints/az.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    data = np.load(args.data)
    states = torch.from_numpy(data["states"].astype(np.float32))
    policies = torch.from_numpy(data["policies"].astype(np.float32))
    values = torch.from_numpy(data["values"].astype(np.float32))
    legal = states[:, 2, 0, :]
    loader = DataLoader(TensorDataset(states, policies, values, legal), batch_size=args.batch_size, shuffle=True)

    if args.init_checkpoint:
        model, _ = load_checkpoint(args.init_checkpoint, map_location=device)
    else:
        model = PolicyValueNet(channels=args.channels, blocks=args.blocks)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, pb, vb, lb in loader:
            xb, pb, vb, lb = xb.to(device), pb.to(device), vb.to(device), lb.to(device)
            logits, value = model(xb, legal_mask=lb)
            policy_loss = -(pb * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(value, vb)
            loss = policy_loss + args.value_weight * value_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(xb)
        print(f"epoch {epoch:03d}: loss={total / len(states):.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_checkpoint(args.output, model, opt, data=args.data, init_checkpoint=args.init_checkpoint)
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
