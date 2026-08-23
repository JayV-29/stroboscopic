"""Longer synthetic run: does the model learn, and does phase-locking emerge?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, yaml
from strobo.data import make_synthetic_recordings, make_windows, split_subjects_kfold
from strobo.data.recording import zscore_per_recording
from strobo.models import FixedRate
from strobo.train import build_model, train_model, get_device
from strobo.eval import run_inference, summarise, phase_histogram, example_trace, oscillator_locking
from strobo import viz

OUT = "results/validate"; os.makedirs(OUT, exist_ok=True)
EP = int(os.environ.get("EP", 12))
recs = [zscore_per_recording(r) for r in make_synthetic_recordings(n_subjects=6, minutes=6.0)]
W = make_windows(recs, 8.0, 4.0)
tr, te, _ = next(split_subjects_kfold(W["subject"], 3))
cfg = yaml.safe_load(open("configs/default.yaml"))
cfg["train"].update(epochs=EP, warm_epochs=2, batch_size=32)
dev = get_device("cpu")
torch.manual_seed(0)
for lam in (0.3, 3.0):
    m = build_model(W, cfg["model"], "phase", True, True, tr)
    train_model(m, W, tr, cfg["train"], lam_e=lam, device=dev, tag=f"ours λ={lam}")
    res = run_inference(m, W, te, device=dev, keep_states=True)
    s = summarise(res)
    for mode in ("rest", "motion"):
        ph = phase_histogram(res, mode)
        print(f"OURS λ={lam} {mode}: bpb={s[mode]['bursts_per_beat']:.2f} mae_hr={s[mode]['mae_hr']:.2f} "
              f"gate={s[mode]['gate_frac']:.2f} coh={s[mode]['coherence']:.2f} phaseR={ph['R']:.2f} mu={ph['mean_phase']:.2f}")
    lk = oscillator_locking(res)
    print(f"  locking: best osc {lk['best_osc']} R={lk['best_R']:.2f} harmonicR={lk['best_R_harmonic']:.2f} "
          f"R_all={np.round(lk['R_per_osc'],2)} f_hz={np.round(lk['mean_omega_hz'],2)}")
    viz.plot_trace(example_trace(res, 3), f"{OUT}/trace_{lam}.png")
    viz.plot_phase_histogram({"rest": phase_histogram(res, "rest"), "motion": phase_histogram(res, "motion")}, f"{OUT}/phase_{lam}.png")
for per in (0.5, 1.0, 2.0):
    torch.manual_seed(0)
    m = build_model(W, cfg["model"], "external", False, False, tr)
    pol = FixedRate(per)
    train_model(m, W, tr, cfg["train"], policy_fn=pol, device=dev, tag=f"fixed {per}s")
    res = run_inference(m, W, te, policy_fn=pol, device=dev)
    s = summarise(res)
    for mode in ("rest", "motion"):
        print(f"FIXED {per}s {mode}: bpb={s[mode]['bursts_per_beat']:.2f} mae_hr={s[mode]['mae_hr']:.2f}")
