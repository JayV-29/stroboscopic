"""Phase-conditioned sampling head + straight-through Gumbel-Bernoulli."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplingHead(nn.Module):
    """Maps a context vector to the logit of 'fire a burst now'."""

    def __init__(self, in_dim: int, hidden: int = 32, init_rate: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # start near a sensible firing probability so the warm-up isn't degenerate
        with torch.no_grad():
            self.net[-1].bias.fill_(float(torch.logit(torch.tensor(init_rate))))
            self.net[-1].weight.mul_(0.1)

    def forward(self, x):
        return self.net(x)[..., 0]

    def macs(self) -> int:
        return sum(m.weight.numel() for m in self.net if isinstance(m, nn.Linear))


def gumbel_bernoulli(logit: torch.Tensor, tau: float = 1.0, hard: bool = True,
                     training: bool = True) -> torch.Tensor:
    """Straight-through relaxed Bernoulli sample in {0,1} with sigmoid gradient."""
    if not training:
        return (logit > 0).float()
    u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
    noise = torch.log(u) - torch.log(1 - u)            # logistic noise
    y = torch.sigmoid((logit + noise) / tau)
    if not hard:
        return y
    return (y > 0.5).float() + y - y.detach()


def anneal_tau(epoch: int, n_epochs: int, tau0: float = 1.0, tau1: float = 0.1) -> float:
    if n_epochs <= 1:
        return tau1
    frac = min(1.0, epoch / max(1, n_epochs - 1))
    return float(tau0 * (tau1 / tau0) ** frac)
