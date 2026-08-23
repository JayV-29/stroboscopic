"""Experiment driver: runs every method over its sweep, over leave-subjects-out
folds, and appends rows to a CSV so plots never need retraining.

    from strobo.experiments import run_suite, METHODS
    df, models = run_suite(W, cfg, out_dir="results/dalia", methods=list(METHODS))
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .data.windows import split_subjects_kfold
from .models.baselines import FixedRate, SendOnDelta, EventTriggeredKalman, IMUGated
from .train import build_model, train_model, get_device, set_seed
from .eval import run_inference, summarise, phase_histogram

METHODS = {
    "ours":              dict(kind="learned", sampler_mode="phase", use_fisher=True, use_fallback=True),
    "ours_no_fisher":    dict(kind="learned", sampler_mode="phase", use_fisher=False, use_fallback=True),
    "ours_no_fallback":  dict(kind="learned", sampler_mode="phase", use_fisher=True, use_fallback=False),
    "learned_threshold": dict(kind="learned", sampler_mode="threshold", use_fisher=False, use_fallback=False),
    "fixed_rate":        dict(kind="classical", cls=FixedRate),
    "send_on_delta":     dict(kind="classical", cls=SendOnDelta),
    "kalman":            dict(kind="classical", cls=EventTriggeredKalman),
    "imu_gated":         dict(kind="classical", cls=IMUGated),
}


class MixedFixedRate:
    """Random fixed-rate policy per batch element; used to pre-train a mask-agnostic decoder."""
    def __init__(self, periods=(0.25, 0.5, 1.0, 2.0, 4.0)):
        self.periods = periods

    def __call__(self, cheap, expensive, fs, burst_ticks):
        Bt, T, _ = cheap.shape
        dev = cheap.device
        per = torch.tensor([max(burst_ticks, int(p * fs)) for p in self.periods], device=dev)
        pick = per[torch.randint(0, len(per), (Bt, 1), device=dev)]
        off = (torch.rand(Bt, 1, device=dev) * pick).long()
        t = torch.arange(T, device=dev)[None]
        return ((t + off) % pick == 0).float()


def _setting_str(d: dict) -> str:
    return json.dumps(d, sort_keys=True)


def _rows(metrics: dict, base: dict) -> list[dict]:
    rows = []
    for mode, m in metrics.items():
        r = dict(base); r["mode"] = mode; r.update(m); rows.append(r)
    return rows


def run_suite(W: dict, cfg: dict, out_dir: str, methods: list[str] | None = None,
              n_folds: int | None = None, lam_e_list=None, max_folds: int | None = None,
              device=None, log=print, keep_models_for: tuple = ("ours",), seed: int = 0,
              classical_sweep_limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (dataframe of per-fold/per-setting/per-mode metrics, dict of kept models)."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    dev = device or get_device(cfg.get("device", "auto"))
    methods = methods or list(METHODS)
    n_folds = n_folds or cfg["data"]["n_folds"]
    lam_e_list = lam_e_list or cfg["sweep"]["lam_e"]
    ct, cm = cfg["train"], cfg["model"]
    rows, kept = [], {}
    if os.path.exists(csv_path):
        rows = pd.read_csv(csv_path).to_dict("records")
        log(f"resuming: {len(rows)} rows already in {csv_path}")
    done = {(r["method"], r["setting"], r["fold"]) for r in rows}

    folds = list(split_subjects_kfold(W["subject"], n_folds, seed))
    if max_folds:
        folds = folds[:max_folds]
    for fold, (tr, te, test_subj) in enumerate(folds):
        log(f"=== fold {fold}: test subjects {test_subj} ({tr.size} train / {te.size} test windows)")
        shared = None  # mask-agnostic pretrained model for classical baselines
        for name in methods:
            spec = METHODS[name]
            if spec["kind"] == "learned":
                settings = [{"lam_e": float(l)} for l in lam_e_list]
            else:
                settings = spec["cls"].sweep()
                if classical_sweep_limit:
                    settings = settings[:: max(1, len(settings) // classical_sweep_limit)]
            for s in settings:
                key = (name, _setting_str(s), fold)
                if key in done:
                    continue
                set_seed(seed + fold)
                t0 = time.time()
                if spec["kind"] == "learned":
                    model = build_model(W, cm, spec["sampler_mode"], spec["use_fisher"], spec["use_fallback"], tr)
                    train_model(model, W, tr, ct, lam_e=s["lam_e"], device=dev, log=log,
                                tag=f"{name} λe={s['lam_e']} f{fold}")
                    res = run_inference(model, W, te, device=dev, keep_states=(name in keep_models_for))
                else:
                    if shared is None:
                        shared = build_model(W, cm, "external", False, False, tr)
                        train_model(shared, W, tr, ct, policy_fn=MixedFixedRate(), device=dev, log=log,
                                    tag=f"shared-decoder f{fold}")
                    policy = spec["cls"](**s)
                    model = copy.deepcopy(shared)
                    if ct.get("finetune_epochs", 0) > 0:
                        train_model(model, W, tr, ct, policy_fn=policy, device=dev, log=log,
                                    epochs=int(ct["finetune_epochs"]), tag=f"{name} {s} f{fold}")
                    res = run_inference(model, W, te, policy_fn=policy, device=dev)
                metrics = summarise(res, cm.get("warmup_s", 2.0))
                ph = {m: phase_histogram(res, m) for m in ("rest", "motion", "all")}
                base = {"method": name, "setting": _setting_str(s), "fold": fold, "secs": time.time() - t0,
                        "params": model.n_params(), "phase_R_rest": ph["rest"]["R"], "phase_R_motion": ph["motion"]["R"],
                        "phase_mu_rest": ph["rest"]["mean_phase"], "phase_mu_motion": ph["motion"]["mean_phase"]}
                new = _rows(metrics, base)
                rows += new
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                a = metrics.get("all", {})
                log(f"    {name:18s} {str(s):40s} bpb={a.get('bursts_per_beat', np.nan):.2f} "
                    + " ".join(f"{k}={v:.2f}" for k, v in a.items() if k.startswith("mae_")))
                if name in keep_models_for and fold == 0:
                    kept[(name, _setting_str(s))] = {"model": model.cpu(), "res": res, "hist": ph,
                                                     "test_idx": te}
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    df = pd.DataFrame(rows)
    return df, kept


def aggregate(df: pd.DataFrame, target: str = "hr") -> pd.DataFrame:
    """Mean ± std over folds per (method, setting, mode)."""
    col = f"mae_{target}"
    g = df.groupby(["method", "setting", "mode"]).agg(
        bursts_per_beat=("bursts_per_beat", "mean"), duty=("duty", "mean"),
        mae=(col, "mean"), mae_sd=(col, "std"), n_folds=("fold", "nunique")).reset_index()
    return g


def headline(df: pd.DataFrame, target: str = "hr", at_bpb: float = 1.0, at_mae: float = 2.0) -> pd.DataFrame:
    """Success-criterion table: MAE at `at_bpb` bursts/beat and bursts needed for `at_mae`."""
    from .eval import curve_at, bursts_for_mae
    out = []
    for m in df.method.unique():
        r = {"method": m}
        for mode in ("rest", "motion"):
            sub = df[(df.method == m) & (df["mode"] == mode)]
            g = sub.groupby("setting").agg(bursts_per_beat=("bursts_per_beat", "mean"),
                                           mae=(f"mae_{target}", "mean")).reset_index()
            r[f"mae@{at_bpb}bpb_{mode}"] = curve_at(g, at_bpb)
            r[f"bpb@{at_mae}mae_{mode}"] = bursts_for_mae(g, at_mae)
        out.append(r)
    return pd.DataFrame(out).set_index("method")
