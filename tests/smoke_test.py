"""End-to-end smoke test on synthetic data (CPU, < 2 min).  Run: python tests/smoke_test.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import yaml

from strobo.data import make_synthetic_recordings, make_windows, split_subjects_kfold
from strobo.data.recording import zscore_per_recording
from strobo.models import StroboModel, StroboConfig, FixedRate, SendOnDelta, EventTriggeredKalman, IMUGated
from strobo.train import build_model, train_model, get_device
from strobo.eval import run_inference, summarise, phase_histogram, fallback_metrics, example_trace, pareto_front
from strobo.experiments import run_suite, aggregate, headline, MixedFixedRate
from strobo.export import quantize_model_int8, int8_forward_check, op_report
from strobo import viz

OUT = os.environ.get("SMOKE_OUT", "results/smoke")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

recs = [zscore_per_recording(r) for r in make_synthetic_recordings(n_subjects=4, minutes=3.0)]
for r in recs:
    print(r.summary())
W = make_windows(recs, 8.0, 4.0)
print("windows:", W["cheap"].shape, W["expensive"].shape, W["targets"].shape)
assert np.isfinite(W["targets"][:, 64:, 0]).mean() > 0.9

tr, te, subj = next(split_subjects_kfold(W["subject"], 2))
cfg = yaml.safe_load(open("configs/default.yaml"))
cfg["train"].update(epochs=2, warm_epochs=1, batch_size=32, finetune_epochs=1)

# --- model forward / backward, all modes
for mode, fi, fb in (("phase", True, True), ("threshold", False, False), ("external", False, False)):
    m = build_model(W, cfg["model"], mode, fi, fb, tr)
    print(mode, "params", m.n_params(), "macs", m.macs_per_tick())
    assert m.n_params() < 100_000
    b = {k: torch.from_numpy(W[k][tr[:4]]) for k in ("cheap", "expensive", "targets", "valid")}
    b["activity"] = torch.from_numpy(W["activity"][tr[:4]].astype(np.int64))
    pol = FixedRate(1.0)(b["cheap"], b["expensive"], W["fs"], 4) if mode == "external" else None
    out = m(b["cheap"], b["expensive"], policy=pol)
    L = m.loss(out, b, lam_e=1.0)
    L["total"].backward()
    assert torch.isfinite(L["total"]), L
    for k, v in out.items():
        assert torch.isfinite(v).all(), k
print("forward/backward ok")

# --- classical policies causal + refractory
b = {k: torch.from_numpy(W[k][tr[:3]]) for k in ("cheap", "expensive")}
for pol in (FixedRate(0.5), SendOnDelta(0.5), EventTriggeredKalman(0.5), IMUGated(2.0, 0.5), MixedFixedRate()):
    f = pol(b["cheap"], b["expensive"], W["fs"], 4)
    assert f.shape == (3, W["cheap"].shape[1]) and set(f.unique().tolist()) <= {0.0, 1.0}
    # refractory: no two fires within 4 ticks
    idx = torch.where(f[0] > 0)[0]
    if idx.numel() > 1:
        assert (idx[1:] - idx[:-1]).min() >= 4, type(pol).__name__
    print(type(pol).__name__, "mean fire", float(f.mean()))

# --- short training of ours + inference + metrics
dev = get_device("cpu")
model = build_model(W, cfg["model"], "phase", True, True, tr)
hist = train_model(model, W, tr, cfg["train"], lam_e=1.0, device=dev, tag="smoke")
res = run_inference(model, W, te, device=dev, keep_states=True)
met = summarise(res)
print({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in met.items()})
ph = phase_histogram(res, "all")
print("phase R", ph["R"], "n", ph["n"])
fb = fallback_metrics(res)
print("fallback metrics", fb)

# --- export
rep = op_report(model)
print("op report", rep)
q = quantize_model_int8(model)
print("int8 layers", len(q["layers"]), "macs", q["total_macs"], "bytes", q["weight_bytes"])
x = torch.randn(8, model.decoder.net[0].in_features)
print(int8_forward_check(model, x, "decoder.net.0"))

# --- figures
tr_ex = example_trace(res, 0)
viz.plot_trace(tr_ex, os.path.join(OUT, "trace.png"))
viz.plot_phase_histogram({"rest": phase_histogram(res, "rest"), "motion": phase_histogram(res, "motion")},
                         os.path.join(OUT, "phase.png"))
viz.architecture_figure(os.path.join(OUT, "arch.png"), model.n_params(), rep["macs_per_tick_amortised"])
viz.make_animation(tr_ex, os.path.join(OUT, "anim.gif"), max_seconds=3.0)

# --- mini suite (2 methods, 1 fold)
cfg["train"].update(epochs=1, warm_epochs=0)
df, kept = run_suite(W, cfg, os.path.join(OUT, "suite"), methods=["ours", "fixed_rate"], n_folds=2,
                     lam_e_list=[0.0, 3.0], max_folds=1, device=dev, classical_sweep_limit=3)
print(df[["method", "setting", "mode", "bursts_per_beat", "mae_hr"]].head(12))
agg = aggregate(df)
viz.plot_curves(df, "hr", path=os.path.join(OUT, "curves.png"), success=3.0)
print(headline(df))
viz.table_figure(headline(df).round(2), os.path.join(OUT, "table.png"), "headline")
viz.compose_poster([os.path.join(OUT, p) for p in ("arch.png", "curves.png", "phase.png", "trace.png")],
                   os.path.join(OUT, "poster.png"))
print(f"SMOKE TEST PASSED in {time.time()-t0:.0f}s -> {OUT}")
