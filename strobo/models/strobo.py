"""The stroboscopic sensing model: encoder -> oscillator bank -> sampling head -> decoder,
run as a streaming filter over a window of ticks with the expensive sensor
revealed only where the policy fires.

Modes (``sampler_mode``):
  'phase'      ours: the head sees oscillator phases + coherence + last burst + time-since
  'threshold'  learned-threshold ablation: head sees cheap feature + last burst + time-since only
  'external'   fire signal supplied by a classical baseline (decoder-only training)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvEncoder
from .oscillator import OscillatorBank
from .sampler import SamplingHead, gumbel_bernoulli
from .decoder import Decoder, gaussian_nll
from ..sim.masking import BurstMasker, fire_to_observed


@dataclass
class StroboConfig:
    cheap_ch: int = 3
    k: int = 2                    # expensive samples per tick
    burst_ticks: int = 4
    n_targets: int = 2
    fs: float = 32.0
    n_osc: int = 12
    feat_dim: int = 32
    enc_ch: int = 16
    n_buffer: int = 4
    hidden_dec: int = 64
    hidden_head: int = 32
    sampler_mode: str = "phase"   # phase | threshold | external
    use_fisher: bool = True
    use_fallback: bool = True
    f_min: float = 0.1
    f_max: float = 4.0
    init_rate: float = 0.05
    warmup_s: float = 2.0

    def to_dict(self):
        return asdict(self)


class StroboModel(nn.Module):
    def __init__(self, cfg: StroboConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg
        self.obs_dim = c.k * c.burst_ticks
        self.encoder = ConvEncoder(c.cheap_ch, c.feat_dim, c.enc_ch)
        self.osc = OscillatorBank(c.n_osc, c.feat_dim, self.obs_dim, c.fs, c.f_min, c.f_max, k_out=c.k)
        self.masker = BurstMasker(c.burst_ticks)
        osc_feat = 4 * c.n_osc
        # sampling-head input
        if c.sampler_mode == "phase":
            head_in = osc_feat + OscillatorBank.N_COH + self.obs_dim + 2   # + coherence + last burst + [tsl, tsl>1s]
        else:
            head_in = c.feat_dim + self.obs_dim + 2
        self.head = SamplingHead(head_in, c.hidden_head, c.init_rate)
        # coherence fallback threshold (learned, in order-parameter units)
        self.fallback_thr = nn.Parameter(torch.tensor(-1.5))     # sigmoid -> ~0.18
        # decoder input
        buf_dim = c.n_buffer * (self.obs_dim + 1)
        dec_in = osc_feat + OscillatorBank.N_COH + c.feat_dim + buf_dim
        self.decoder = Decoder(dec_in, c.n_targets, c.hidden_dec)
        # target normalisation
        self.register_buffer("t_mean", torch.zeros(c.n_targets))
        self.register_buffer("t_std", torch.ones(c.n_targets))

    # ----------------------------------------------------------------- utils
    def set_target_stats(self, mean, std):
        self.t_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.t_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-3))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def macs_per_tick(self) -> dict:
        d = {"encoder": self.encoder.macs_per_tick(), "oscillator": self.osc.macs_per_tick(),
             "head": self.head.macs(), "decoder": self.decoder.macs()}
        d["total"] = sum(d.values())
        return d

    # --------------------------------------------------------------- forward
    def forward(self, cheap, expensive, policy=None, tau: float = 1.0, hard: bool = True,
                force_rate: float | None = None):
        """cheap (B,T,C), expensive (B,T,k).  policy: optional (B,T) external fire signal.
        force_rate: if set, fire every round(1/force_rate) ticks (warm-start)."""
        Bt, T, _ = cheap.shape
        dev = cheap.device
        c = self.cfg
        feats = self.encoder(cheap)                                  # (B,T,F)
        st = self.osc.init_state(Bt, dev)
        buf_v = torch.zeros(Bt, c.n_buffer, self.obs_dim, device=dev)
        buf_age = torch.full((Bt, c.n_buffer), 1.0, device=dev)     # normalised age (1 = "very old")
        tsl = torch.ones(Bt, device=dev)                             # time since last, in seconds (init 1 s)
        refr = torch.zeros(Bt, device=dev)                           # refractory ticks remaining
        last_v = torch.zeros(Bt, self.obs_dim, device=dev)
        outs = {k: [] for k in ("fire", "logit", "p_policy", "gate", "coh", "fisher", "mean",
                                "logvar", "theta", "omega", "amp", "wave")}
        training = self.training
        sd_acc = torch.zeros(Bt, device=dev)                         # sigma-delta accumulator (eval)
        period = None if force_rate is None else max(1, int(round(1.0 / max(force_rate, 1e-6))))

        for t in range(T):
            f_t = feats[:, t]
            coh = self.osc.coherence(st)                             # (B,3)
            ofeat = self.osc.features(st)
            tsl_feat = torch.stack([tsl.clamp(max=5.0) / 5.0, (tsl > 1.0).float()], -1)
            if c.sampler_mode == "phase":
                hin = torch.cat([ofeat, coh, last_v, tsl_feat], -1)
            else:
                hin = torch.cat([f_t, last_v, tsl_feat], -1)
            logit = self.head(hin)
            p_pol = torch.sigmoid(logit)
            if c.use_fallback and c.sampler_mode != "external":
                gate = torch.sigmoid((torch.sigmoid(self.fallback_thr) - coh[:, 0]) / 0.05)
            else:
                gate = torch.zeros_like(p_pol)
            # decision
            if policy is not None:
                fire = policy[:, t].float()
            elif period is not None:
                fire = torch.full((Bt,), float(t % period == 0), device=dev)
            else:
                # union of policy and fallback in logit space: p = 1-(1-p_pol)(1-gate)
                p = 1 - (1 - p_pol) * (1 - gate)
                if training:
                    eff_logit = torch.logit(p.clamp(1e-5, 1 - 1e-5))
                    fire = gumbel_bernoulli(eff_logit, tau, hard, True)
                else:
                    # deterministic sigma-delta: fire when accumulated probability crosses 1.
                    # Reproduces the trained mean rate and fires where p peaks (no RNG on device).
                    sd_acc = sd_acc + p * (refr <= 0).float()
                    fire = (sd_acc >= 1.0).float()
                    sd_acc = sd_acc - fire
            fire = fire * (refr <= 0).float()
            fire_h = (fire > 0.5).float().detach()
            # observation (straight-through gradient flows through `fire`)
            v = self.masker.burst_values(expensive, t) * fire[:, None]
            last_v = fire_h[:, None] * v + (1 - fire_h[:, None]) * last_v
            # buffer shift on hard fire
            new_v = torch.cat([v[:, None], buf_v[:, :-1]], 1)
            new_age = torch.cat([torch.zeros(Bt, 1, device=dev), buf_age[:, :-1]], 1)
            m = fire_h[:, None, None]
            buf_v = m * new_v + (1 - m) * buf_v
            buf_age = (m[:, :, 0] * new_age + (1 - m[:, :, 0]) * buf_age + 1.0 / (5 * c.fs)).clamp(max=1.0)
            tsl = (1 - fire_h) * (tsl + 1.0 / c.fs)
            refr = torch.where(fire_h > 0, torch.full_like(refr, c.burst_ticks - 1), (refr - 1).clamp(min=0))
            # world model update
            st = self.osc.step(st, f_t, v, fire[:, None])
            # decoder
            din = torch.cat([ofeat, coh, f_t, buf_v.flatten(1), buf_age], -1)
            mean, logvar = self.decoder(din)
            outs["fire"].append(fire); outs["logit"].append(logit); outs["p_policy"].append(p_pol)
            outs["gate"].append(gate); outs["coh"].append(coh); outs["fisher"].append(self.osc.fisher(st))
            outs["mean"].append(mean); outs["logvar"].append(logvar)
            outs["theta"].append(st["theta"]); outs["omega"].append(st["omega"]); outs["amp"].append(st["amp"])
            outs["wave"].append(self.osc.predict_waveform(st))
        return {k: torch.stack(v, 1) for k, v in outs.items()}

    # ------------------------------------------------------------------ loss
    def loss(self, out: dict, batch: dict, lam_e: float = 0.0, lam_f: float = 0.1,
             lam_c: float = 0.01, lam_r: float = 0.1) -> dict:
        c = self.cfg
        T = out["mean"].shape[1]
        warm = int(c.warmup_s * c.fs)
        tgt = (batch["targets"] - self.t_mean) / self.t_std
        mask = torch.isfinite(tgt) & batch["valid"][..., None]
        mask[:, :warm] = False
        tgt = torch.nan_to_num(tgt)
        nll = gaussian_nll(out["mean"], out["logvar"], tgt, mask)
        fire = out["fire"]
        rate = fire.mean()
        # Fisher reward: normalised so lam_f is scale-free
        fis = out["fisher"]
        fis_n = fis / (fis.detach().mean() + 1e-6)
        fisher_term = (fire * fis_n).mean() if c.use_fisher else torch.zeros((), device=fire.device)
        # fallback regulariser: discourage the gate from being open all the time
        gate_term = out["gate"].mean() if c.use_fallback else torch.zeros((), device=fire.device)
        # template reconstruction at observed ticks (gives the templates meaning)
        obs = fire_to_observed(fire.detach(), c.burst_ticks)
        rec = ((out["wave"] - batch["expensive"]) ** 2).mean(-1)
        recon = (rec * obs).sum() / obs.sum().clamp(min=1.0)
        total = nll + lam_e * rate - lam_f * fisher_term + lam_c * gate_term + lam_r * recon
        return {"total": total, "nll": nll.detach(), "rate": rate.detach(),
                "fisher": fisher_term.detach(), "gate": gate_term.detach(), "recon": recon.detach()}

    def denorm(self, mean):
        return mean * self.t_std + self.t_mean
