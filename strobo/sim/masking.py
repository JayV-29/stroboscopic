"""Burst masking: turn a per-tick fire decision into the sparse observation a
real duty-cycled front-end would return.

A burst fired at tick t reveals expensive[t : t+B] (B ticks x k samples).  Ticks
inside a burst are *refractory*: the policy cannot fire again until the burst
has completed.  All functions are causal and differentiable in the fire signal
so that a Gumbel-softmax policy can be trained end-to-end.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class BurstMasker:
    def __init__(self, burst_ticks: int = 4):
        self.B = burst_ticks

    def burst_values(self, expensive: torch.Tensor, t: int) -> torch.Tensor:
        """Values revealed by a burst starting at tick t: (Bt, B*k), zero-padded at the end."""
        Bt, T, k = expensive.shape
        end = min(t + self.B, T)
        v = expensive[:, t:end].reshape(Bt, -1)
        if end - t < self.B:
            v = F.pad(v, (0, (self.B - (end - t)) * k))
        return v


def fire_to_observed(fire: torch.Tensor, burst_ticks: int) -> torch.Tensor:
    """Expand a (Bt, T) fire signal to a (Bt, T) observation mask covering each burst."""
    Bt, T = fire.shape
    ker = torch.ones(1, 1, burst_ticks, device=fire.device, dtype=fire.dtype)
    x = F.pad(fire[:, None, :], (burst_ticks - 1, 0))
    obs = F.conv1d(x, ker)[:, 0]
    return obs.clamp(max=1.0)


def apply_fire_mask(expensive: torch.Tensor, fire: torch.Tensor, burst_ticks: int):
    """Return (masked expensive, observed-mask) for visualisation / classical baselines."""
    obs = fire_to_observed(fire, burst_ticks)
    return expensive * obs[..., None], obs
