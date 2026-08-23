"""Baseline 2: send-on-delta (oracle transmitter-side threshold).

The classical send-on-delta scheme assumes the sensor is *read* continuously
and only a burst is transmitted/processed when the linear extrapolation from
the last burst deviates from the true signal by more than ``delta``.  In our
setting that is an optimistic baseline (it sees the expensive signal for free
when deciding); it upper-bounds what any threshold on the signal itself can do.
Energy is counted per transmitted burst, like every other method.
"""
from __future__ import annotations

import torch


class SendOnDelta:
    name = "send_on_delta"
    sweep_param = "delta"

    def __init__(self, delta: float = 0.5, max_gap_s: float = 6.0):
        self.delta = delta
        self.max_gap_s = max_gap_s

    def __call__(self, cheap, expensive, fs, burst_ticks):
        Bt, T, k = expensive.shape
        dev = expensive.device
        x = expensive[..., 0]                                     # first sample of each tick
        fire = torch.zeros(Bt, T, device=dev)
        last_val = torch.zeros(Bt, device=dev)
        slope = torch.zeros(Bt, device=dev)
        since = torch.full((Bt,), 10 ** 6, device=dev)
        refr = torch.zeros(Bt, device=dev)
        max_gap = int(self.max_gap_s * fs)
        for t in range(T):
            pred = last_val + slope * since
            err = (x[:, t] - pred).abs()
            f = ((err > self.delta) | (since >= max_gap)) & (refr <= 0)
            fire[:, t] = f.float()
            # update predictor with the burst (use burst end-points for slope)
            end = min(t + burst_ticks, T)
            v0, v1 = x[:, t], x[:, end - 1]
            new_slope = (v1 - v0) / max(end - 1 - t, 1)
            last_val = torch.where(f, v1, last_val)
            slope = torch.where(f, new_slope, slope)
            since = torch.where(f, torch.full_like(since, float(1 - (end - 1 - t))), since + 1)
            refr = torch.where(f, torch.full_like(refr, burst_ticks - 1), (refr - 1).clamp(min=0))
        return fire

    @staticmethod
    def sweep():
        return [{"delta": d} for d in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)]
