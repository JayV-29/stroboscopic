"""Two-layer MLP decoder with a calibrated Gaussian head."""
from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_dim: int, n_targets: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * n_targets))
        self.n_targets = n_targets

    def forward(self, x):
        out = self.net(x)
        mean, logvar = out.chunk(2, -1)
        return mean, logvar.clamp(-6, 4)

    def macs(self) -> int:
        return sum(m.weight.numel() for m in self.net if isinstance(m, nn.Linear))


def gaussian_nll(mean, logvar, target, mask):
    """Masked Gaussian negative log-likelihood, mean over valid entries."""
    nll = 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))
    m = mask.float()
    return (nll * m).sum() / m.sum().clamp(min=1.0)
