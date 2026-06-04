"""PyTorch policy-value network for ConnectX."""

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PolicyValueNet(nn.Module):
    def __init__(self, channels=96, blocks=6, rows=6, columns=7):
        super().__init__()
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.blocks = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * rows * columns, columns),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_fc1 = nn.Linear(2 * rows * columns, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x, legal_mask=None):
        x = self.res_blocks(self.stem(x))
        logits = self.policy_head(x)
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask <= 0, -1e9)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(self.value_conv(x)))))
        return logits, value.squeeze(-1)


def save_checkpoint(path, model, optimizer=None, **metadata):
    payload = {
        "state_dict": model.state_dict(),
        "model_config": {
            "channels": model.channels,
            "blocks": model.blocks,
            "rows": model.rows,
            "columns": model.columns,
        },
        "metadata": metadata,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    config = payload.get("model_config", {})
    model = PolicyValueNet(**config)
    model.load_state_dict(payload["state_dict"])
    return model, payload
