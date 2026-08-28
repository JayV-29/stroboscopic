"""A resumed run_suite must still return the diagnostic models the notebook needs.

The notebook's phase-histogram (section 4), cost (section 8) and hero/summary
(section 9) cells all dereference the model kept for 'ours' on fold 0.  If a
Kaggle session dies and the run resumes from results.csv, those cells must not
crash.  Run: PYTHONIOENCODING=utf-8 python tests/test_resume.py
"""
import os, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from strobo.data import make_synthetic_recordings, make_windows
from strobo.data.recording import zscore_per_recording
from strobo.eval import summarise
from strobo.experiments import run_suite

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = ("ours", "fixed_rate")          # one learned method, one classical
METHODS = ["ours", "fixed_rate"]


def _pick_kept(kept, method, prefer_bpb=1.0):
    """Verbatim copy of the notebook's helper (notebooks/build_notebook.py)."""
    c = [(abs(summarise(v["res"])["all"]["bursts_per_beat"] - prefer_bpb), k, v)
         for k, v in kept.items() if k[0] == method]
    return sorted(c, key=lambda x: x[0])[0][1:] if c else (None, None)


def main():
    recs = [zscore_per_recording(r) for r in make_synthetic_recordings(n_subjects=4, minutes=3.0)]
    W = make_windows(recs, 8.0, 4.0)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/default.yaml")))
    cfg["train"].update(epochs=1, warm_epochs=0, batch_size=32, finetune_epochs=1)

    out = tempfile.mkdtemp(prefix="strobo_resume_")
    try:
        common = dict(methods=METHODS, n_folds=2, lam_e_list=[0.0, 3.0], max_folds=1,
                      device="cpu", classical_sweep_limit=2, keep_models_for=KEEP)

        # --- first run: trains from scratch
        df1, kept1 = run_suite(W, cfg, out, **common)
        assert len(df1) > 0, "first run produced no rows"
        for m in KEEP:
            assert any(k[0] == m for k in kept1), "first run kept no model for %r" % (m,)
        print("first run : %d rows, %d kept models" % (len(df1), len(kept1)))

        # --- second run against the same out_dir: every setting is already done
        df2, kept2 = run_suite(W, cfg, out, **common)
        assert len(df2) == len(df1), "resume changed row count: %d -> %d" % (len(df1), len(df2))
        print("resumed   : %d rows, %d kept models" % (len(df2), len(kept2)))

        for m in KEEP:
            assert any(k[0] == m for k in kept2), (
                "RESUME LOST the kept model for %r: notebook sections 4/8/9 would crash" % (m,))
        assert len(kept2) == len(kept1), (
            "resume restored %d models but the fresh run kept %d" % (len(kept2), len(kept1)))

        # --- restored weights must reproduce the run that produced the CSV rows,
        #     otherwise we would silently be reporting a differently-trained model
        for k, v in kept2.items():
            method, setting = k
            row = df1[(df1.method == method) & (df1.setting == setting) & (df1["mode"] == "all")]
            assert len(row) == 1, "expected one 'all' row for %r, got %d" % (k, len(row))
            fresh, restored = row.iloc[0], summarise(v["res"])["all"]
            for field in ("bursts_per_beat", "mae_hr", "duty"):
                a, b = float(fresh[field]), float(restored[field])
                assert abs(a - b) < 1e-4, (
                    "%s %s: %s drifted on restore, %.6f -> %.6f" % (method, setting, field, a, b))
        print("equivalence: %d restored models match their CSV rows exactly" % len(kept2))

        # --- the exact notebook expressions that used to crash
        key, K = _pick_kept(kept2, "ours")
        assert K is not None, "pick_kept returned None after resume"
        for field in ("res", "model", "test_idx"):
            assert field in K, "kept entry missing %r" % (field,)
        bpb = summarise(K["res"])["all"]["bursts_per_beat"]
        assert K["model"].n_params() < 100_000
        print("notebook  : diagnostic model %s bpb=%.2f params=%d" % (key, bpb, K["model"].n_params()))
        print("RESUME TEST PASSED")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
