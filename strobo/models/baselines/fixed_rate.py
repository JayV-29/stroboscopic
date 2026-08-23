"""Baseline 1: fixed-rate down-sampling.  Fire a burst every ``period_s`` seconds."""
from __future__ import annotations

import torch


class FixedRate:
    name = "fixed_rate"
    sweep_param = "period_s"

    def __init__(self, period_s: float = 1.0, phase_jitter: bool = True):
        self.period_s = period_s
        self.phase_jitter = phase_jitter

    def __call__(self, cheap, expensive, fs, burst_ticks):
        Bt, T, _ = cheap.shape
        period = max(burst_ticks, int(round(self.period_s * fs)))
        t = torch.arange(T, device=cheap.device)[None].expand(Bt, -1)
        if self.phase_jitter:
            off = torch.randint(0, period, (Bt, 1), device=cheap.device)
        else:
            off = torch.zeros(Bt, 1, dtype=torch.long, device=cheap.device)
        return ((t + off) % period == 0).float()

    @staticmethod
    def sweep():
        return [{"period_s": p} for p in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)]
