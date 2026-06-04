"""Supervised pretraining for the ConnectX policy-value network."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from rl.connectx import feature_rows_to_tensors
from rl.model import PolicyValueNet, save_checkpoint


def soft_targets(move_scores, valid_masks, temperature):
    scaled = np.clip(move_scores / temperature, -80.0, 80.0)
    scaled = np.where(valid_masks > 0, scaled, -1e9)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scores = np.exp(scaled) * valid_masks
    denom = exp_scores.sum(axis=1, keepdims=True)
    fallback = valid_masks / np.clip(valid_masks.sum(axis=1, keepdims=True), 1.0, None)
    return np.where(denom > 0, exp_scores / np.clip(denom, 1e-10, None), fallback)


def load_dataset(path, policy_targets, temperature):
    data = np.load(path)
    planes = feature_rows_to_tensors(data["X"])
    values = np.clip(data["scores"] / 1_000_000.0, -1.0, 1.0).astype(np.float32)
    moves = data["moves"].astype(np.int64)

    if policy_targets == "soft" and "move_scores" in data and "valid_masks" in data:
        policy = soft_targets(
            data["move_scores"].astype(np.float32),
            data["valid_masks"].astype(np.float32),
            temperature,
        ).astype(np.float32)
        target_kind = "soft"
    else:
        policy = np.zeros((len(moves), 7), dtype=np.float32)
        policy[np.arange(len(moves)), moves] = 1.0
        target_kind = "hard"

    legal = planes[:, 2, 0, :].astype(np.float32)
    return planes, policy, values, legal, target_kind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/selfplay_rich.npz")
    parser.add_argument("--output", default="rl/checkpoints/supervised.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--policy-targets", choices=["hard", "soft"], default="hard")
    parser.add_argument("--policy-temperature", type=float, default=200.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    planes, policy, values, legal, target_kind = load_dataset(
        args.data, args.policy_targets, args.policy_temperature
    )
    idx = np.random.permutation(len(planes))
    n_val = int(len(idx) * args.val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = TensorDataset(
        torch.from_numpy(planes[train_idx]),
        torch.from_numpy(policy[train_idx]),
        torch.from_numpy(values[train_idx]),
        torch.from_numpy(legal[train_idx]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(planes[val_idx]),
        torch.from_numpy(policy[val_idx]),
        torch.from_numpy(values[val_idx]),
        torch.from_numpy(legal[val_idx]),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = PolicyValueNet(channels=args.channels, blocks=args.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")

    print(
        f"data={len(planes)} train={len(train_idx)} val={len(val_idx)} "
        f"targets={target_kind} device={device} model=channels{args.channels}_blocks{args.blocks}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, pb, vb, lb in train_loader:
            xb, pb, vb, lb = xb.to(device), pb.to(device), vb.to(device), lb.to(device)
            logits, value = model(xb, legal_mask=lb)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(pb * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value, vb)
            loss = policy_loss + args.value_weight * value_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item() * len(xb)

        model.eval()
        val_loss = val_policy = val_value = val_acc = 0.0
        with torch.no_grad():
            for xb, pb, vb, lb in val_loader:
                xb, pb, vb, lb = xb.to(device), pb.to(device), vb.to(device), lb.to(device)
                logits, value = model(xb, legal_mask=lb)
                log_probs = F.log_softmax(logits, dim=1)
                p_loss = -(pb * log_probs).sum(dim=1).mean()
                v_loss = F.mse_loss(value, vb)
                loss = p_loss + args.value_weight * v_loss
                val_loss += loss.item() * len(xb)
                val_policy += p_loss.item() * len(xb)
                val_value += v_loss.item() * len(xb)
                val_acc += (logits.argmax(dim=1) == pb.argmax(dim=1)).float().sum().item()

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        val_policy /= len(val_ds)
        val_value /= len(val_ds)
        val_acc /= len(val_ds)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch {epoch:03d}: train={train_loss:.4f} val={val_loss:.4f} "
                f"policy={val_policy:.4f} value={val_value:.5f} acc={val_acc:.3f}"
            )
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            save_checkpoint(
                args.output,
                model,
                opt,
                data=args.data,
                policy_targets=target_kind,
                val_loss=best_loss,
                epoch=epoch,
            )

    print(f"saved best checkpoint to {args.output} (val_loss={best_loss:.4f})")


if __name__ == "__main__":
    main()
