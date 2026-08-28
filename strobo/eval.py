"""Inference, metrics, Pareto fronts, phase diagnostics and the fallback test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data.recording import ACT_REST, ACT_MOTION, ACT_OTHER
from .sim.energy import EnergyModel
from .sim.masking import fire_to_observed
from .train import make_loader, to_device, get_device


@torch.no_grad()
def run_inference(model, W: dict, idx: np.ndarray, policy_fn=None, device=None, batch_size: int = 128,
                  keep_states: bool = False, seed: int = 0) -> dict:
    dev = device or get_device()
    # The oscillator bank draws its initial phases uniformly at random on every forward
    # pass (OscillatorBank.init_state), so without a fixed seed the *same* trained model
    # scores differently each time it is evaluated.  Seed here so a reported number is
    # reproducible and a resumed run reproduces the rows already in results.csv.
    # (The first `warmup_s` seconds are excluded from every metric anyway, which is what
    # absorbs the lock-in transient from this initialisation.)
    torch.manual_seed(seed)
    model.to(dev).eval()
    fs, B = float(W["fs"]), model.cfg.burst_ticks
    loader = make_loader(W, idx, batch_size, False)
    cols = {k: [] for k in ("fire", "pred", "std", "gate", "coh", "fisher", "idx")}
    if keep_states:
        cols.update({"theta": [], "omega": [], "amp": [], "wave": [], "p_policy": []})
    for batch in loader:
        batch = to_device(batch, dev)
        policy = policy_fn(batch["cheap"], batch["expensive"], fs, B) if policy_fn else None
        out = model(batch["cheap"], batch["expensive"], policy=policy, tau=0.1)
        cols["fire"].append(out["fire"].cpu().numpy())
        cols["pred"].append(model.denorm(out["mean"]).cpu().numpy())
        cols["std"].append((torch.exp(0.5 * out["logvar"]) * model.t_std).cpu().numpy())
        cols["gate"].append(out["gate"].cpu().numpy())
        cols["coh"].append(out["coh"].cpu().numpy())
        cols["fisher"].append(out["fisher"].cpu().numpy())
        cols["idx"].append(np.asarray(batch["idx"]))
        if keep_states:
            for k in ("theta", "omega", "amp", "wave", "p_policy"):
                cols[k].append(out[k].cpu().numpy())
    res = {k: np.concatenate(v) for k, v in cols.items()}
    j = res["idx"]
    for k in ("targets", "beats", "activity", "ref_phase", "valid", "cheap", "expensive"):
        res[k] = W[k][j]
    res["subject"] = W["subject"][j]
    res["fs"], res["burst_ticks"], res["target_names"] = fs, B, W["target_names"]
    return res


def _tick_mask(res, mode, warm):
    m = np.ones_like(res["activity"], dtype=bool)
    m[:, :warm] = False
    m &= res["valid"]
    if mode == "rest":
        m &= res["activity"] == ACT_REST
    elif mode == "motion":
        m &= res["activity"] == ACT_MOTION
    return m


def summarise(res: dict, warmup_s: float = 2.0, energy: EnergyModel | None = None) -> dict:
    """Metrics for all / rest / motion ticks."""
    fs, B = res["fs"], res["burst_ticks"]
    warm = int(warmup_s * fs)
    energy = energy or EnergyModel()
    out = {}
    for mode in ("all", "rest", "motion"):
        m = _tick_mask(res, mode, warm)
        d = {"n_ticks": int(m.sum())}
        if m.sum() == 0:
            out[mode] = d
            continue
        nb = res["beats"][m].sum()
        nf = res["fire"][m].sum()
        d["bursts_per_beat"] = float(nf / max(nb, 1.0))
        d["bursts_per_min"] = float(nf / (m.sum() / fs) * 60.0)
        d["duty"] = float(min(1.0, nf * B / m.sum()))
        d["gate_frac"] = float((res["gate"][m] > 0.5).mean())
        d["coherence"] = float(res["coh"][..., 0][m].mean())
        d["mw"] = energy.mw(nf, m.sum(), fs)
        for j, name in enumerate(res["target_names"]):
            t = res["targets"][..., j]
            ok = m & np.isfinite(t)
            if ok.sum() == 0:
                continue
            err = np.abs(res["pred"][..., j] - t)[ok]
            d[f"mae_{name}"] = float(err.mean())
            d[f"rmse_{name}"] = float(np.sqrt((err ** 2).mean()))
            # calibration: fraction inside +-1.96 sigma
            z = np.abs(res["pred"][..., j] - t)[ok] / (res["std"][..., j][ok] + 1e-6)
            d[f"cov95_{name}"] = float((z < 1.96).mean())
        out[mode] = d
    return out


def phase_histogram(res: dict, mode: str = "all", bins: int = 24, warmup_s: float = 2.0):
    """Histogram of reference phase (from ECG R-peaks) at ticks where a burst fired."""
    warm = int(warmup_s * res["fs"])
    m = _tick_mask(res, mode, warm) & (res["fire"] > 0.5) & np.isfinite(res["ref_phase"])
    ph = res["ref_phase"][m]
    h, edges = np.histogram(ph, bins=bins, range=(0, 2 * np.pi))
    dens = h / max(h.sum(), 1) * bins / (2 * np.pi)
    # circular concentration (mean resultant length): 0 = uniform, 1 = perfectly phase-locked
    R = float(np.abs(np.exp(1j * ph).mean())) if ph.size else 0.0
    mu = float(np.angle(np.exp(1j * ph).mean())) % (2 * np.pi) if ph.size else np.nan
    return {"density": dens, "edges": edges, "R": R, "mean_phase": mu, "n": int(ph.size)}


def oscillator_locking(res: dict, mode: str = "all", warmup_s: float = 2.0) -> dict:
    """Phase-locking of each latent oscillator to the reference (ECG) phase: mean resultant
    length of (theta_i - ref_phase) over ticks, plus the best oscillator.  Needs keep_states=True."""
    if "theta" not in res:
        return {}
    warm = int(warmup_s * res["fs"])
    m = _tick_mask(res, mode, warm) & np.isfinite(res["ref_phase"])
    th = res["theta"][m]                                    # (n, N)
    ref = res["ref_phase"][m][:, None]
    z = np.exp(1j * (th - ref))
    R = np.abs(z.mean(0))
    # also allow harmonic locking (2 cycles per beat, e.g. dicrotic or gait 2:1)
    R2 = np.abs(np.exp(1j * (th - 2 * ref)).mean(0))
    best = int(np.argmax(R))
    return {"R_per_osc": R, "R2_per_osc": R2, "best_osc": best, "best_R": float(R[best]),
            "best_R_harmonic": float(R2.max()), "mean_omega_hz": res["omega"][m].mean(0) * res["fs"] / (2 * np.pi)}


def fallback_metrics(res: dict, dense_duty: float = 0.5, horizon_beats: int = 3,
                     warmup_s: float = 2.0) -> dict:
    """MIT-BIH stress test.  'irregular' ticks carry ACT_MOTION.  A tick is 'dense'
    when the trailing 1.5 s firing duty >= dense_duty."""
    fs, B = res["fs"], res["burst_ticks"]
    warm = int(warmup_s * fs)
    win = int(1.5 * fs)
    ker = np.ones(win) / win
    fire = res["fire"]
    duty = np.stack([np.convolve(f, ker, mode="full")[:f.size] for f in fire]) * B
    dense = duty >= dense_duty
    valid = np.ones_like(dense); valid[:, :warm] = False
    irr = (res["activity"] == ACT_MOTION) & valid
    normal = (res["activity"] == ACT_REST) & valid
    # onsets of irregular segments -> did we go dense within `horizon_beats` beats?
    hits, total = 0, 0
    for i in range(fire.shape[0]):
        a = irr[i].astype(int)
        onsets = np.where(np.diff(a, prepend=0) == 1)[0]
        beats = np.where(res["beats"][i] > 0)[0]
        for o in onsets:
            later = beats[beats > o]
            end = int(later[min(horizon_beats - 1, later.size - 1)]) if later.size else min(fire.shape[1] - 1, o + int(3 * fs))
            total += 1
            hits += int(dense[i, o:end + 1].any())
    return {
        "detect_within_3_beats": hits / max(total, 1), "n_onsets": total,
        "dense_frac_irregular": float(dense[irr].mean()) if irr.any() else np.nan,
        "false_fallback_rate": float(dense[normal].mean()) if normal.any() else np.nan,
        "bursts_per_beat_normal": float(fire[normal].sum() / max(res["beats"][normal].sum(), 1)),
        "bursts_per_beat_irregular": float(fire[irr].sum() / max(res["beats"][irr].sum(), 1)) if irr.any() else np.nan,
    }


def pareto_front(df: pd.DataFrame, x: str = "bursts_per_beat", y: str = "mae") -> pd.DataFrame:
    """Lower-left Pareto front: keep points with no other point that is <= in both."""
    d = df.sort_values([x, y]).reset_index(drop=True)
    keep, best = [], np.inf
    for i, r in d.iterrows():
        if r[y] < best:
            keep.append(i); best = r[y]
    return d.loc[keep]


def curve_at(df: pd.DataFrame, x_target: float, x: str = "bursts_per_beat", y: str = "mae") -> float:
    """Log-linear interpolation of a Pareto curve at ``x_target``."""
    f = pareto_front(df, x, y)
    if len(f) == 0:
        return np.nan
    xs, ys = np.log(f[x].values + 1e-6), f[y].values
    if x_target <= f[x].min():
        return float(ys[0])
    if x_target >= f[x].max():
        return float(ys[-1])
    return float(np.interp(np.log(x_target), xs, ys))


def bursts_for_mae(df: pd.DataFrame, mae_target: float, x: str = "bursts_per_beat", y: str = "mae") -> float:
    """Smallest bursts-per-beat on the Pareto front achieving mae <= target (interpolated)."""
    f = pareto_front(df, x, y).sort_values(x)
    if len(f) == 0 or f[y].min() > mae_target:
        return np.nan
    xs, ys = f[x].values, f[y].values
    for i in range(len(f)):
        if ys[i] <= mae_target:
            if i == 0:
                return float(xs[0])
            # interpolate between i-1 (above target) and i (below)
            w = (ys[i - 1] - mae_target) / max(ys[i - 1] - ys[i], 1e-9)
            return float(np.exp(np.log(xs[i - 1]) + w * (np.log(xs[i]) - np.log(xs[i - 1]))))
    return np.nan


def example_trace(res: dict, i: int, model=None) -> dict:
    """Pull one window out of an inference result for plotting."""
    fs, B = res["fs"], res["burst_ticks"]
    obs = fire_to_observed(torch.from_numpy(res["fire"][i:i + 1]), B)[0].numpy()
    d = {k: res[k][i] for k in ("fire", "pred", "std", "gate", "coh", "targets", "beats",
                                "activity", "ref_phase", "cheap", "expensive")}
    d.update(obs=obs, fs=fs, t=np.arange(res["fire"].shape[1]) / fs, target_names=res["target_names"])
    for k in ("theta", "omega", "amp", "wave", "p_policy"):
        if k in res:
            d[k] = res[k][i]
    return d
