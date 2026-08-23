"""Baseline 3: event-triggered Kalman filter on (phase, frequency).

State x = [phi, omega]; phi advances by omega each tick.  Between bursts the
filter only predicts and its phase variance grows; a burst is fired when the
predicted phase standard deviation exceeds ``thr`` radians.  When a burst is
fired the observed 8-sample segment is matched against a cosine template at
the predicted phase to obtain a phase measurement (least-squares over a small
grid), which is then assimilated.  Fully causal.
"""
from __future__ import annotations

import math

import torch


class EventTriggeredKalman:
    name = "kalman"
    sweep_param = "thr"

    def __init__(self, thr: float = 0.5, q_phi: float = 1e-3, q_om: float = 1e-5,
                 r_meas: float = 0.2, f0: float = 1.2, max_gap_s: float = 6.0, n_grid: int = 16):
        self.thr, self.q_phi, self.q_om, self.r = thr, q_phi, q_om, r_meas
        self.f0, self.max_gap_s, self.n_grid = f0, max_gap_s, n_grid

    def _measure_phase(self, seg, phi_pred, omega, k):
        """seg: (B, L) burst samples; return phase estimate (B,) nearest to phi_pred."""
        Bt, L = seg.shape
        dev = seg.device
        # candidate phases around the prediction
        cand = phi_pred[:, None] + torch.linspace(-math.pi, math.pi, self.n_grid, device=dev)[None]
        j = torch.arange(L, device=dev, dtype=torch.float32) / k          # tick offsets
        arg = cand[:, :, None] + omega[:, None, None] * j[None, None]        # (B,G,L)
        tpl = torch.cos(arg)
        seg_c = seg - seg.mean(1, keepdim=True)
        tpl_c = tpl - tpl.mean(2, keepdim=True)
        corr = (seg_c[:, None] * tpl_c).sum(-1) / (seg_c.norm(dim=1)[:, None] * tpl_c.norm(dim=2) + 1e-6)
        best = corr.argmax(1)
        return cand.gather(1, best[:, None])[:, 0]

    def __call__(self, cheap, expensive, fs, burst_ticks):
        Bt, T, k = expensive.shape
        dev = expensive.device
        fire = torch.zeros(Bt, T, device=dev)
        phi = torch.zeros(Bt, device=dev)
        om = torch.full((Bt,), 2 * math.pi * self.f0 / fs, device=dev)
        P = torch.zeros(Bt, 2, 2, device=dev)
        P[:, 0, 0] = 10.0; P[:, 1, 1] = 1e-2
        Fm = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=dev)
        Q = torch.diag(torch.tensor([self.q_phi, self.q_om], device=dev))
        since = torch.zeros(Bt, device=dev)
        refr = torch.zeros(Bt, device=dev)
        max_gap = int(self.max_gap_s * fs)
        H = torch.tensor([[1.0, 0.0]], device=dev)
        for t in range(T):
            # predict
            phi = phi + om
            P = Fm @ P @ Fm.T + Q
            since = since + 1
            f = ((P[:, 0, 0].sqrt() > self.thr) | (since >= max_gap)) & (refr <= 0)
            fire[:, t] = f.float()
            if f.any():
                end = min(t + burst_ticks, T)
                seg = expensive[:, t:end].reshape(Bt, -1)
                z = self._measure_phase(seg, phi, om, k)
                innov = torch.remainder(z - phi + math.pi, 2 * math.pi) - math.pi
                S = P[:, 0, 0] + self.r
                K = P[:, :, 0] / S[:, None]                                # (B,2)
                upd = K * innov[:, None]
                phi = torch.where(f, phi + upd[:, 0], phi)
                om = torch.where(f, (om + upd[:, 1]).clamp(2 * math.pi * 0.5 / fs, 2 * math.pi * 3.5 / fs), om)
                I_KH = torch.eye(2, device=dev)[None] - K[:, :, None] @ H[None]
                P_new = I_KH @ P
                P = torch.where(f[:, None, None], P_new, P)
                since = torch.where(f, torch.zeros_like(since), since)
            refr = torch.where(f, torch.full_like(refr, burst_ticks - 1), (refr - 1).clamp(min=0))
            phi = torch.remainder(phi, 2 * math.pi)
        return fire

    @staticmethod
    def sweep():
        return [{"thr": v} for v in (0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)]
