"""Kuramoto-style latent oscillator bank with a likelihood-driven innovation step.

State per oscillator i:  phase theta_i, angular frequency omega_i (rad/tick), amplitude a_i.

Per tick (in this order):
  1. innovation (only when a burst arrives): the bank predicts the burst from its
     own waveform templates, x_hat_j = sum_i a_i w_i(theta_i + omega_i * j/k), and
     takes one gradient step of the squared residual w.r.t. (theta_i, omega_i) with
     learned per-oscillator gains.  This is a phase-locked-loop / EKF-style update
     computed from the model's own likelihood (a few hundred MACs).
  2. free-running update:
        theta_i <- theta_i + omega_i + sum_j K_ij sin(theta_j - theta_i) + g_i(u)
        omega_i <- omega_i * (1 + 0.02 tanh h_i(u))         bounded to [f_min, f_max]
        a_i     <- sigmoid(m_i(u))
     where u = [cheap-stream feature, last burst (masked), burst flag].

Read-outs: phase features, coherence (prediction coherence from the burst residual,
normalised frequency variance, 1-s smoothed Kuramoto order parameter of the task group), predicted waveform, and the analytic
Fisher information of a sample now, sum_i g_i a_i^2 w_i'(theta_i)^2.
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
        self.N, self.fs, self.n_bins, self.k = n_osc, fs, n_bins, k_out
        self.obs_dim, self.in_dim = obs_dim, in_dim
        self.w_min, self.w_max = 2 * math.pi * f_min / fs, 2 * math.pi * f_max / fs
        f0 = torch.logspace(math.log10(f_min), math.log10(f_max), n_osc)
        self.register_buffer("omega0", 2 * math.pi * f0 / fs)
        # sample offsets (in ticks) of the obs_dim burst samples and of one tick's k samples
        self.register_buffer("burst_off", torch.arange(obs_dim, dtype=torch.float32) / k_out)
        self.register_buffer("tick_off", torch.arange(k_out, dtype=torch.float32) / k_out)
        self.K = nn.Parameter(0.01 * torch.randn(n_osc, n_osc))
        self.drive = nn.Linear(in_dim + obs_dim + 1, 3 * n_osc)
        nn.init.zeros_(self.drive.bias)
        with torch.no_grad():
            self.drive.weight.mul_(0.1)
        ang = torch.linspace(0, 2 * math.pi, n_bins + 1)[:-1]
        self.template = nn.Parameter(0.5 * torch.cos(ang)[None].repeat(n_osc, 1))   # (N, n_bins)
        # innovation gains (softplus): phase ~0.5, frequency ~0.02
        self.gain_theta = nn.Parameter(torch.full((n_osc,), math.log(math.e ** 0.5 - 1)))
        self.gain_omega = nn.Parameter(torch.full((n_osc,), math.log(math.e ** 0.02 - 1)))
        self.group_logit = nn.Parameter(torch.zeros(n_osc))

    # ---------------------------------------------------------------- state
    def init_state(self, batch: int, device=None):
        theta = torch.rand(batch, self.N, device=device) * 2 * math.pi
        omega = self.omega0[None].expand(batch, -1).clone()
        return {"theta": theta, "omega": omega, "amp": torch.full((batch, self.N), 0.5, device=device),
                "om_mean": omega.clone(), "om_sq": omega.clone() ** 2,
                "r_ema": torch.full((batch, 1), 0.5, device=device),
                "res_ema": torch.ones(batch, 1, device=device)}

    def group_weights(self):
        return torch.softmax(self.group_logit, 0)

    def _lookup(self, phi: torch.Tensor):
        """Template value and phase-derivative at phi (B, N, L) -> w, dw (B, N, L)."""
        B = phi.shape[0]
        pos = phi / (2 * math.pi) * self.n_bins
        i0 = torch.floor(pos).long() % self.n_bins
        i1 = (i0 + 1) % self.n_bins
        frac = pos - torch.floor(pos)
        tpl = self.template[None].expand(B, -1, -1)
        g0 = torch.gather(tpl, 2, i0)
        g1 = torch.gather(tpl, 2, i1)
        return g0 + frac * (g1 - g0), (g1 - g0) * self.n_bins / (2 * math.pi)

    def _order_param(self, theta):
        w = self.group_weights()[None]
        return (w * torch.exp(1j * theta.to(torch.complex64))).sum(-1).abs()

    # ----------------------------------------------------------------- step
    def innovate(self, st: dict, obs: torch.Tensor, obs_flag: torch.Tensor) -> dict:
        """Likelihood gradient step on (theta, omega) from a burst.  obs (B, obs_dim), obs_flag (B, 1)."""
        theta, omega, amp = st["theta"], st["omega"], st["amp"]
        phi = theta[:, :, None] + omega[:, :, None] * self.burst_off[None, None]     # (B,N,L)
        w, dw = self._lookup(phi)
        pred = (amp[:, :, None] * w).sum(1)                                           # (B,L)
        resid = (obs - pred) * obs_flag                                               # zero when no burst
        # normalised residual energy of this burst -> EMA held between bursts (prediction coherence)
        e = (resid ** 2).mean(-1, keepdim=True) / ((obs ** 2).mean(-1, keepdim=True) + 1e-3)
        res_ema = obs_flag * (0.7 * st["res_ema"] + 0.3 * e.clamp(max=4.0)) + (1 - obs_flag) * st["res_ema"]
        # d(-0.5*resid^2)/d theta_i = a_i * sum_j resid_j * dw_ij
        s = resid[:, None, :] * dw                                                    # (B,N,L)
        d_theta = amp * s.mean(-1)
        d_omega = amp * (s * self.burst_off[None, None]).mean(-1)
        g_t, g_o = F.softplus(self.gain_theta)[None], F.softplus(self.gain_omega)[None]
        theta = theta + torch.tanh(g_t * d_theta)                                     # <= 1 rad per burst
        omega = (omega * (1 + 0.2 * torch.tanh(g_o * d_omega))).clamp(self.w_min, self.w_max)
        return {**st, "theta": theta, "omega": omega, "res_ema": res_ema}

    def step(self, st: dict, feat: torch.Tensor, obs: torch.Tensor, obs_flag: torch.Tensor) -> dict:
        st = self.innovate(st, obs, obs_flag)
        theta, omega = st["theta"], st["omega"]
        u = torch.cat([feat, obs * obs_flag, obs_flag], -1)
        d_theta, d_omega, a_logit = self.drive(u).chunk(3, -1)
        diff = theta[:, None, :] - theta[:, :, None]                                  # theta_j - theta_i
        coupling = (self.K[None] * torch.sin(diff)).sum(-1)
        theta = torch.remainder(theta + omega + coupling + 0.5 * torch.tanh(d_theta), 2 * math.pi)
        omega = (omega * (1 + 0.02 * torch.tanh(d_omega))).clamp(self.w_min, self.w_max)
        amp = torch.sigmoid(a_logit)
        alpha = 1.0 / self.fs
        om_mean = (1 - alpha) * st["om_mean"] + alpha * omega
        om_sq = (1 - alpha) * st["om_sq"] + alpha * omega ** 2
        r_ema = (1 - alpha) * st["r_ema"] + alpha * self._order_param(theta)[:, None]
        return {"theta": theta, "omega": omega, "amp": amp, "om_mean": om_mean, "om_sq": om_sq, "r_ema": r_ema,
                "res_ema": st["res_ema"]}

    # ------------------------------------------------------------- read-outs
    N_COH = 3

    def coherence(self, st: dict) -> torch.Tensor:
        """(B, 3): [prediction coherence exp(-normalised residual energy) -- drives the fallback gate,
        normalised frequency variance over 1 s, 1-s smoothed Kuramoto order parameter of the task group]."""
        w = self.group_weights()[None]
        var = (st["om_sq"] - st["om_mean"] ** 2).clamp(min=0)
        fvar = (w * var / (st["om_mean"] ** 2 + 1e-8)).sum(-1)
        r_pred = torch.exp(-st["res_ema"][:, 0])
        return torch.stack([r_pred, fvar, st["r_ema"][:, 0]], -1)

    def features(self, st: dict) -> torch.Tensor:
        return torch.cat([torch.cos(st["theta"]), torch.sin(st["theta"]),
                          st["omega"] / self.w_max, st["amp"]], -1)

    def predict_waveform(self, st: dict) -> torch.Tensor:
        """Predicted k expensive-sensor samples for this tick: (B, k)."""
        phi = st["theta"][:, :, None] + st["omega"][:, :, None] * self.tick_off[None, None]
        w, _ = self._lookup(phi)
        return (st["amp"][:, :, None] * w).sum(1)

    def fisher(self, st: dict) -> torch.Tensor:
        """Information a burst starting now carries about (theta, omega): (B,)."""
        phi = st["theta"][:, :, None] + st["omega"][:, :, None] * self.burst_off[None, None]
        _, dw = self._lookup(phi)
        g = self.group_weights()[None, :, None]
        return (g * st["amp"][:, :, None] ** 2 * dw ** 2).mean(-1).sum(-1)

    def macs_per_tick(self) -> int:
        # drive + coupling + innovation (amortised: assumes at most one burst per 4 ticks)
        return self.drive.weight.numel() + self.N * self.N + 6 * self.N + (3 * self.N * self.obs_dim) // 4
