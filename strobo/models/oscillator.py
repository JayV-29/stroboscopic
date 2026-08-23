"""Kuramoto-style latent oscillator bank.

State per oscillator i:   phase theta_i, angular frequency omega_i (rad/tick), amplitude a_i.
Update per tick:
    theta_i <- theta_i + omega_i + sum_j K_ij sin(theta_j - theta_i) + g_i(u)
    omega_i <- omega_i + h_i(u)                  (bounded to [f_min, f_max])
    a_i     <- sigmoid(m_i(u))
where u = [cheap-stream feature, last observed burst (masked)].

The bank also owns a per-oscillator 32-bin waveform template; the predicted
expensive-sensor waveform is sum_i a_i * w_i(theta_i).  The Fisher information
of a sample at the current phase with respect to (omega, theta) is
proportional to sum_i a_i^2 * w_i'(theta_i)^2, which is computed analytically
from the template and used as a sampling reward.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OscillatorBank(nn.Module):
    def __init__(self, n_osc: int = 12, in_dim: int = 32, obs_dim: int = 8, fs: float = 32.0,
                 f_min: float = 0.1, f_max: float = 4.0, n_bins: int = 32, k_out: int = 2):
        super().__init__()
        self.N, self.fs, self.n_bins, self.k_out = n_osc, fs, n_bins, k_out
        self.w_min, self.w_max = 2 * math.pi * f_min / fs, 2 * math.pi * f_max / fs
        # log-spaced initial frequencies spanning respiration -> fast gait / heartbeat
        f0 = torch.logspace(math.log10(f_min), math.log10(f_max), n_osc)
        self.register_buffer("omega0", 2 * math.pi * f0 / fs)
        self.K = nn.Parameter(0.01 * torch.randn(n_osc, n_osc))
        self.drive = nn.Linear(in_dim + obs_dim + 1, 3 * n_osc)   # (+1 for obs mask flag)
        nn.init.zeros_(self.drive.bias)
        with torch.no_grad():
            self.drive.weight.mul_(0.1)
        # waveform templates: (N, n_bins) for each of k_out outputs -> (N, k_out, n_bins)
        ang = torch.linspace(0, 2 * math.pi, n_bins + 1)[:-1]
        init = torch.stack([torch.cos(ang + 0.3 * j) for j in range(k_out)], 0)
        self.template = nn.Parameter(init[None].repeat(n_osc, 1, 1) * 0.5)
        # soft task-group weights for coherence
        self.group_logit = nn.Parameter(torch.zeros(n_osc))
        self.obs_dim, self.in_dim = obs_dim, in_dim

    # ---------------------------------------------------------------- state
    def init_state(self, batch: int, device=None):
        theta = torch.rand(batch, self.N, device=device) * 2 * math.pi
        omega = self.omega0[None].expand(batch, -1).clone()
        amp = torch.full((batch, self.N), 0.5, device=device)
        om_mean = omega.clone()
        om_sq = omega.clone() ** 2
        return {"theta": theta, "omega": omega, "amp": amp, "om_mean": om_mean, "om_sq": om_sq}

    def step(self, st: dict, feat: torch.Tensor, obs: torch.Tensor, obs_flag: torch.Tensor) -> dict:
        """One tick.  feat (B, in_dim); obs (B, obs_dim) last burst values (zeros if none);
        obs_flag (B, 1) = 1 if a fresh burst arrived this tick."""
        theta, omega, amp = st["theta"], st["omega"], st["amp"]
        u = torch.cat([feat, obs * obs_flag, obs_flag], -1)
        d = self.drive(u)
        d_theta, d_omega, a_logit = d.chunk(3, -1)
        diff = theta[:, None, :] - theta[:, :, None]                  # theta_j - theta_i
        coupling = (self.K[None] * torch.sin(diff)).sum(-1)
        theta = theta + omega + coupling + 0.5 * torch.tanh(d_theta)
        omega = (omega + 0.02 * omega * torch.tanh(d_omega)).clamp(self.w_min, self.w_max)
        amp = torch.sigmoid(a_logit)
        theta = torch.remainder(theta, 2 * math.pi)
        # 1-second EMA of omega statistics for the frequency-variance coherence
        alpha = 1.0 / self.fs
        om_mean = (1 - alpha) * st["om_mean"] + alpha * omega
        om_sq = (1 - alpha) * st["om_sq"] + alpha * omega ** 2
        return {"theta": theta, "omega": omega, "amp": amp, "om_mean": om_mean, "om_sq": om_sq}

    # ------------------------------------------------------------- read-outs
    def group_weights(self):
        return torch.softmax(self.group_logit, 0)

    def coherence(self, st: dict) -> torch.Tensor:
        """(B, 2): Kuramoto order parameter of the task group and the (normalised)
        frequency variance over the last second."""
        w = self.group_weights()[None]
        z = (w * torch.exp(1j * st["theta"].to(torch.complex64))).sum(-1)
        r = z.abs()
        var = (st["om_sq"] - st["om_mean"] ** 2).clamp(min=0)
        fvar = (w * var / (st["om_mean"] ** 2 + 1e-8)).sum(-1)
        return torch.stack([r, fvar], -1)

    def features(self, st: dict) -> torch.Tensor:
        """Phase-domain feature vector for the heads: (B, 4N)."""
        return torch.cat([torch.cos(st["theta"]), torch.sin(st["theta"]),
                          st["omega"] / self.w_max, st["amp"]], -1)

    def _lookup(self, theta: torch.Tensor):
        """Linear-interpolated template value and its phase derivative at theta.
        Returns w (B, N, k_out), dw (B, N, k_out)."""
        pos = theta / (2 * math.pi) * self.n_bins                    # (B, N)
        i0 = torch.floor(pos).long() % self.n_bins
        i1 = (i0 + 1) % self.n_bins
        frac = (pos - torch.floor(pos))[..., None]
        tpl = self.template.permute(0, 2, 1)                         # (N, n_bins, k_out)
        g0 = torch.gather(tpl[None].expand(theta.shape[0], -1, -1, -1), 2,
                          i0[..., None, None].expand(-1, -1, 1, self.k_out))[:, :, 0]
        g1 = torch.gather(tpl[None].expand(theta.shape[0], -1, -1, -1), 2,
                          i1[..., None, None].expand(-1, -1, 1, self.k_out))[:, :, 0]
        w = g0 + frac * (g1 - g0)
        dw = (g1 - g0) * self.n_bins / (2 * math.pi)
        return w, dw

    def predict_waveform(self, st: dict) -> torch.Tensor:
        """Predicted expensive-sensor sample(s) at this tick: (B, k_out)."""
        w, _ = self._lookup(st["theta"])
        return (st["amp"][..., None] * w).sum(1)

    def fisher(self, st: dict) -> torch.Tensor:
        """Analytic information of a sample now about (omega, theta): (B,)."""
        _, dw = self._lookup(st["theta"])
        g = self.group_weights()[None, :, None]
        return (g * st["amp"][..., None] ** 2 * dw ** 2).sum((1, 2))

    def macs_per_tick(self) -> int:
        return self.drive.weight.numel() + self.N * self.N + 6 * self.N
