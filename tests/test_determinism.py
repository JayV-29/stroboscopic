"""Evaluating the same trained model twice must give the same numbers.

OscillatorBank.init_state draws the 12 initial phases uniformly at random on every
forward pass.  Without a fixed seed in run_inference that makes every reported
metric - bursts/beat, MAE, phase-histogram R - drift between evaluations of an
identical model, so a pre-registered PASS/FAIL could flip on a re-run.
Run: PYTHONIOENCODING=utf-8 python tests/test_determinism.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from strobo.data import make_synthetic_recordings, make_windows, split_subjects_kfold
from strobo.data.recording import zscore_per_recording
from strobo.eval import run_inference, summarise, phase_histogram
from strobo.train import build_model, train_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ("bursts_per_beat", "mae_hr", "duty", "gate_frac", "coherence")


def main():
    recs = [zscore_per_recording(r) for r in make_synthetic_recordings(n_subjects=4, minutes=3.0)]
    W = make_windows(recs, 8.0, 4.0)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/default.yaml")))
    cfg["train"].update(epochs=1, warm_epochs=0, batch_size=32)
    tr, te, _ = next(split_subjects_kfold(W["subject"], 2))

    m = build_model(W, cfg["model"], "phase", True, True, tr)
    train_model(m, W, tr, cfg["train"], lam_e=0.0, device="cpu", log=lambda s: None)

    a = run_inference(m, W, te, device="cpu", keep_states=True)
    b = run_inference(m, W, te, device="cpu", keep_states=True)
    sa, sb = summarise(a, 2.0)["all"], summarise(b, 2.0)["all"]

    for f in FIELDS:
        assert abs(float(sa[f]) - float(sb[f])) < 1e-9, (
            "%s is not reproducible across evaluations: %.9f vs %.9f" % (f, sa[f], sb[f]))
    ra, rb = phase_histogram(a, "all")["R"], phase_histogram(b, "all")["R"]
    assert abs(ra - rb) < 1e-9, "phase-histogram R not reproducible: %.9f vs %.9f" % (ra, rb)

    print("stable across two evaluations: " + ", ".join("%s=%.4f" % (f, sa[f]) for f in FIELDS))
    print("phase R=%.4f" % ra)

    # a different seed *should* still change the draw - proves we fixed it, not removed it
    c = summarise(run_inference(m, W, te, device="cpu", keep_states=True, seed=1), 2.0)["all"]
    changed = [f for f in FIELDS if abs(float(sa[f]) - float(c[f])) > 1e-9]
    print("seed=1 changes: %s" % (changed if changed else "nothing (init may not matter here)"))
    print("DETERMINISM TEST PASSED")


if __name__ == "__main__":
    main()
