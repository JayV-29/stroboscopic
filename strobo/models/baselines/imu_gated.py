"""Baseline 4: IMU-gated sampling (what commercial wearables do).

Two fixed burst rates: ``period_rest_s`` while the 1-s ACC-magnitude standard
deviation is below ``act_thr`` and ``period_motion_s`` above it.  The plan's
"fire only when active" variant is the special case period_rest_s = inf.
Sweeping both periods traces its Pareto front.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class IMUGated:
    name = "imu_gated"
    sweep_param = "period_rest_s"

    def __init__(self, period_rest_s: float = 2.0, period_motion_s: float = 0.5, act_thr: float = 0.1):
        self.period_rest_s, self.period_motion_s, self.act_thr = period_rest_s, period_motion_s, act_thr

    def __call__(self, cheap, expensive, fs, burst_ticks):
        Bt, T, C = cheap.shape
        dev = cheap.device
        mag = cheap.norm(dim=-1)                                            # (B,T)
        w = int(fs)
        x = F.pad(mag[:, None], (w - 1, 0), mode="replicate")
        m1 = F.avg_pool1d(x, w, stride=1)[:, 0]
        m2 = F.avg_pool1d(x ** 2, w, stride=1)[:, 0]
        act = (m2 - m1 ** 2).clamp(min=0).sqrt()                            # causal 1-s std
        active = act > self.act_thr
        p_rest = float("inf") if self.period_rest_s == float("inf") else max(burst_ticks, int(round(self.period_rest_s * fs)))
        p_mot = max(burst_ticks, int(round(self.period_motion_s * fs)))
        fire = torch.zeros(Bt, T, device=dev)
        since = torch.full((Bt,), 10 ** 6, device=dev)
        for t in range(T):
            period = torch.where(active[:, t], torch.full_like(since, float(p_mot)), torch.full_like(since, float(p_rest)))
            f = since >= period
            fire[:, t] = f.float()
            since = torch.where(f, torch.ones_like(since), since + 1)
        return fire

    @staticmethod
    def sweep():
        out = []
        for pr in (0.5, 1.0, 2.0, 4.0, float("inf")):
            for pm in (0.25, 0.5, 1.0, 2.0):
                out.append({"period_rest_s": pr, "period_motion_s": pm})
        return out
