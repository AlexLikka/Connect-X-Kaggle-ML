"""Mixed training that keeps the rich search teacher as the anchor.

This is intended for the case where pure AlphaZero self-play fine-tuning drifts
away from the current strong f902rich-style teacher. Each step trains on one
batch from the rich data and one batch from MCTS self-play data, while model
selection stays anchored on held-out rich validation quality.
"""

import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from rl.connectx import feature_rows_to_tensors
from rl.model import PolicyValueNet, load_checkpoint, save_checkpoint


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rich-data", default="data/selfplay_rich.npz")
    parser.add_argument("--az-data", default="rl/data/selfplay_az_c128_b8_s320.npz")
    parser.add_argument("--init-checkpoint", default="rl/checkpoints/supervised_c128_b8.pt")
    parser.add_argument("--output", default="rl/checkpoints/mixed_c128_b8_iter1.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rich-value-weight", type=float, default=0.2)
    parser.add_argument("--az-policy-weight", type=float, default=0.25)
    parser.add_argument("--az-value-weight", type=float, default=0.05)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    rich_full = load_rich_dataset(args.rich_data)
    az_full = load_az_dataset(args.az_data)
    n_rich_val = int(len(rich_full) * args.val_split)
    n_az_val = int(len(az_full) * args.val_split)
    rich_train, rich_val = random_split(
        rich_full,
        [len(rich_full) - n_rich_val, n_rich_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    az_train, az_val = random_split(
        az_full,
        [len(az_full) - n_az_val, n_az_val],
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    rich_loader = DataLoader(rich_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    az_loader = DataLoader(az_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    rich_val_loader = DataLoader(rich_val, batch_size=args.batch_size, shuffle=False)
    az_val_loader = DataLoader(az_val, batch_size=args.batch_size, shuffle=False)

    if args.init_checkpoint:
        model, _ = load_checkpoint(args.init_checkpoint, map_location=device)
    else:
        model = PolicyValueNet(channels=args.channels, blocks=args.blocks)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = None

    print(
        f"rich_train={len(rich_train)} rich_val={len(rich_val)} "
        f"az_train={len(az_train)} az_val={len(az_val)} device={device} "
        f"az_policy_weight={args.az_policy_weight} az_value_weight={args.az_value_weight}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        az_cycle = itertools.cycle(az_loader)
        totals = {
            "loss": 0.0,
            "rich_policy": 0.0,
            "rich_value": 0.0,
            "az_policy": 0.0,
            "az_value": 0.0,
            "rich_acc": 0.0,
            "n": 0,
        }

        for xb, mb, vb, lb in rich_loader:
            axb, apb, avb, alb = next(az_cycle)

            xb = xb.to(device)
            mb = mb.to(device)
            vb = vb.to(device)
            lb = lb.to(device)
            axb = axb.to(device)
            apb = apb.to(device)
            avb = avb.to(device)
            alb = alb.to(device)

            rich_logits, rich_value = model(xb, legal_mask=lb)
            az_logits, az_value = model(axb, legal_mask=alb)

            rich_policy_loss = F.cross_entropy(rich_logits, mb)
            rich_value_loss = F.mse_loss(rich_value, vb)
            az_policy_loss = -(apb * F.log_softmax(az_logits, dim=1)).sum(dim=1).mean()
            az_value_loss = F.mse_loss(az_value, avb)
            loss = (
                rich_policy_loss
                + args.rich_value_weight * rich_value_loss
                + args.az_policy_weight * az_policy_loss
                + args.az_value_weight * az_value_loss
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            n = len(xb)
            totals["n"] += n
            totals["loss"] += loss.item() * n
            totals["rich_policy"] += rich_policy_loss.item() * n
            totals["rich_value"] += rich_value_loss.item() * n
            totals["az_policy"] += az_policy_loss.item() * n
            totals["az_value"] += az_value_loss.item() * n
            totals["rich_acc"] += (rich_logits.argmax(dim=1) == mb).float().sum().item()

        n = totals["n"]
        model.eval()
        rich_val_policy = 0.0
        rich_val_value = 0.0
        rich_val_acc = 0.0
        rich_val_n = 0
        az_val_policy = 0.0
        az_val_value = 0.0
        az_val_n = 0
        with torch.no_grad():
            for xb, mb, vb, lb in rich_val_loader:
                xb = xb.to(device)
                mb = mb.to(device)
                vb = vb.to(device)
                lb = lb.to(device)
                logits, value = model(xb, legal_mask=lb)
                batch_n = len(xb)
                rich_val_n += batch_n
                rich_val_policy += F.cross_entropy(logits, mb, reduction="sum").item()
                rich_val_value += F.mse_loss(value, vb, reduction="sum").item()
                rich_val_acc += (logits.argmax(dim=1) == mb).float().sum().item()
            for xb, pb, vb, lb in az_val_loader:
                xb = xb.to(device)
                pb = pb.to(device)
                vb = vb.to(device)
                lb = lb.to(device)
                logits, value = model(xb, legal_mask=lb)
                batch_n = len(xb)
                az_val_n += batch_n
                az_val_policy += (-(pb * F.log_softmax(logits, dim=1)).sum(dim=1).sum().item())
                az_val_value += F.mse_loss(value, vb, reduction="sum").item()

        rich_val_policy /= rich_val_n
        rich_val_value /= rich_val_n
        rich_val_acc /= rich_val_n
        az_val_policy /= az_val_n
        az_val_value /= az_val_n
        score = rich_val_policy + args.rich_value_weight * rich_val_value

        print(
            f"epoch {epoch:03d}: loss={totals['loss']/n:.4f} "
            f"rich_policy={totals['rich_policy']/n:.4f} "
            f"rich_value={totals['rich_value']/n:.5f} "
            f"az_policy={totals['az_policy']/n:.4f} "
            f"az_value={totals['az_value']/n:.4f} "
            f"rich_acc={totals['rich_acc']/n:.3f} "
            f"| val_rich_policy={rich_val_policy:.4f} "
            f"val_rich_value={rich_val_value:.5f} "
            f"val_rich_acc={rich_val_acc:.3f} "
            f"val_az_policy={az_val_policy:.4f} "
            f"val_az_value={az_val_value:.4f}"
        )

        if best_score is None or score < best_score:
            best_score = score
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            save_checkpoint(
                args.output,
                model,
                opt,
                rich_data=args.rich_data,
                az_data=args.az_data,
                init_checkpoint=args.init_checkpoint,
                rich_value_weight=args.rich_value_weight,
                az_policy_weight=args.az_policy_weight,
                az_value_weight=args.az_value_weight,
                val_rich_policy=rich_val_policy,
                val_rich_value=rich_val_value,
                val_rich_acc=rich_val_acc,
                val_az_policy=az_val_policy,
                val_az_value=az_val_value,
                epoch=epoch,
            )
            print(f"  saved best checkpoint to {args.output}")

    print(f"best mixed score={best_score:.5f} saved at {args.output}")


if __name__ == "__main__":
    main()
