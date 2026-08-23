"""Single training entry point.

    python -m strobo.train --config configs/default.yaml --lam_e 1.0

Programmatic use (what run_experiments.py and the notebook do):

    model = build_model(W, cfg_model, sampler_mode="phase")
    hist  = train_model(model, W, train_idx, cfg_train, lam_e=1.0, device=dev)
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.windows import WindowDataset
from .models.strobo import StroboModel, StroboConfig
from .models.sampler import anneal_tau


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_stats(W: dict, idx: np.ndarray):
    t = W["targets"][idx].reshape(-1, W["targets"].shape[-1])
    mean = np.nanmean(t, 0)
    std = np.nanstd(t, 0)
    return mean.astype(np.float32), np.maximum(std, 1e-3).astype(np.float32)


def build_model(W: dict, cfg_model: dict, sampler_mode: str = "phase", use_fisher: bool = True,
                use_fallback: bool = True, train_idx: np.ndarray | None = None) -> StroboModel:
    cfg = StroboConfig(
        cheap_ch=W["cheap"].shape[-1], k=W["expensive"].shape[-1],
        n_targets=W["targets"].shape[-1], fs=float(W["fs"]),
        sampler_mode=sampler_mode, use_fisher=use_fisher, use_fallback=use_fallback,
        **{k: v for k, v in cfg_model.items() if k in StroboConfig.__dataclass_fields__},
    )
    model = StroboModel(cfg)
    if train_idx is not None:
        mean, std = target_stats(W, train_idx)
        model.set_target_stats(mean, std)
    return model


def make_loader(W: dict, idx: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(WindowDataset(W, idx), batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, generator=g, num_workers=0)


def to_device(batch: dict, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_model(model: StroboModel, W: dict, train_idx: np.ndarray, cfg_train: dict,
                lam_e: float = 0.0, policy_fn=None, device=None, log=print, seed: int = 0,
                epochs: int | None = None, tag: str = "") -> list[dict]:
    """Train ``model``.  If ``policy_fn`` is given the sampling head is bypassed
    and only encoder/oscillator/decoder learn (classical baselines)."""
    dev = device or get_device()
    model.to(dev)
    ct = cfg_train
    n_ep = epochs if epochs is not None else int(ct["epochs"])
    warm = 0 if policy_fn is not None else int(ct.get("warm_epochs", 1))
    opt = torch.optim.AdamW(model.parameters(), lr=ct["lr"], weight_decay=ct.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_ep))
    loader = make_loader(W, train_idx, ct["batch_size"], True, seed)
    fs, B = float(W["fs"]), model.cfg.burst_ticks
    hist = []
    for ep in range(n_ep):
        model.train()
        tau = anneal_tau(max(0, ep - warm), max(1, n_ep - warm), ct.get("tau0", 1.0), ct.get("tau1", 0.1))
        force = (1.0 / fs) if ep < warm else None        # warm start: one burst per second
        agg, n, t0 = {}, 0, time.time()
        for batch in loader:
            batch = to_device(batch, dev)
            with torch.no_grad():
                policy = policy_fn(batch["cheap"], batch["expensive"], fs, B) if policy_fn else None
            out = model(batch["cheap"], batch["expensive"], policy=policy, tau=tau, force_rate=force)
            L = model.loss(out, batch, lam_e=lam_e, lam_f=ct.get("lam_f", 0.1),
                           lam_c=ct.get("lam_c", 0.01), lam_r=ct.get("lam_r", 0.1))
            opt.zero_grad(set_to_none=True)
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ct.get("grad_clip", 1.0))
            opt.step()
            for k, v in L.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            n += 1
        sched.step()
        rec = {k: v / max(n, 1) for k, v in agg.items()}
        rec.update(epoch=ep, tau=tau, secs=time.time() - t0, lam_e=lam_e)
        hist.append(rec)
        log(f"[{tag}] ep {ep:2d} tau={tau:.2f} loss={rec['total']:.3f} nll={rec['nll']:.3f} "
            f"rate={rec['rate']:.3f} recon={rec['recon']:.3f} ({rec['secs']:.0f}s)")
    return hist


def main():
    import yaml
    from .data import make_synthetic_recordings, make_windows, split_subjects_kfold
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--lam_e", type=float, default=1.0)
    ap.add_argument("--mode", default="phase")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    recs = make_synthetic_recordings(**cfg["data"]["synthetic"])
    W = make_windows(recs, cfg["data"]["window_s"], cfg["data"]["stride_s"])
    tr, te, subj = next(split_subjects_kfold(W["subject"], cfg["data"]["n_folds"], cfg["seed"]))
    model = build_model(W, cfg["model"], sampler_mode=args.mode, train_idx=tr)
    print(f"params: {model.n_params()}  macs/tick: {model.macs_per_tick()}")
    train_model(model, W, tr, cfg["train"], lam_e=args.lam_e, device=get_device(cfg["device"]))


if __name__ == "__main__":
    main()
