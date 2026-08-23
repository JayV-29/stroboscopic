# Stroboscopic Sensing

Tiny latent-oscillator world models that decide *where in a physiological cycle* to fire an
expensive sensor (PPG / respiration belt / ECG), driven by an always-on cheap stream (wrist IMU).
The claim under test: at ≤ 1 burst per beat the phase-conditioned policy beats threshold /
event-triggered / IMU-gated sampling **during motion**, not only at rest, with < 100 k parameters
and a few thousand MACs per IMU tick.

Everything runs in one Kaggle notebook: [`notebooks/stroboscopic_kaggle.ipynb`](notebooks/stroboscopic_kaggle.ipynb).

## Layout

```
strobo/
  data/        loaders (PPG-DaLiA, WESAD, UCI-HAR, MIT-BIH), R-peak detection, per-tick targets,
               windowing, leave-subjects-out folds, synthetic generator, dataset path finder
  sim/         burst masking (differentiable in the fire signal), energy accounting
  models/      encoder.py (causal conv), oscillator.py (Kuramoto bank + templates + innovation step
               + Fisher), sampler.py (Gumbel-ST head), decoder.py, strobo.py (streaming model + loss)
               baselines/ fixed_rate, send_on_delta, kalman (event-triggered), imu_gated
  train.py     build_model / train_model (warm start -> Gumbel annealing)
  eval.py      inference, rest/motion metrics, Pareto fronts, phase histogram, MIT-BIH fallback test
  experiments.py  run_suite: every method x sweep x fold -> results.csv (resumable)
  export/      int8 fixed-point export in numpy, op counting
  viz.py       all figures (curves, phase histogram, traces, tables, architecture, GIF, poster)
  images.py    CC photos from Wikimedia Commons (+attribution) and PIL composites: hero card, device card, contact sheet
configs/default.yaml
notebooks/build_notebook.py  -> notebooks/stroboscopic_kaggle.ipynb
tests/smoke_test.py          end-to-end on synthetic data, CPU, ~20 s
tests/validate_synthetic.py  longer synthetic run: does phase-locking emerge?
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test.py                 # everything on synthetic data, writes results/smoke/*
python notebooks/build_notebook.py         # regenerate the notebook from source
STROBO_MODE=demo jupyter nbconvert --execute --to notebook notebooks/stroboscopic_kaggle.ipynb
```

## Running on Kaggle

1. Create a notebook, attach these public datasets (or upload them once, < 5 GB total):
   PPG-DaLiA (`S1/S1.pkl … S15/S15.pkl`), WESAD (`S2/S2.pkl … S17/S17.pkl`),
   UCI HAR (`UCI HAR Dataset/…/Inertial Signals`), MIT-BIH Arrhythmia (`100.dat/.hea/.atr …`).
   The path finder (`strobo/data/paths.py`) searches `/kaggle/input` recursively, so folder names don't matter.
2. Attach this repository as a dataset (or leave `REPO_URL` pointing at GitHub; the notebook clones it).
3. Turn **Settings → Internet: on** (only needed to fetch the CC-licensed technology photos from Wikimedia
   Commons; everything else works offline) and pick a GPU accelerator.
4. Set `MODE` in the first cell. Any dataset that is missing is replaced by a clearly-labelled synthetic
   stand-in so the notebook always finishes. Estimated wall-clock (the per-tick recurrent loop is python-bound,
   so a T4 and a P100 are similar; the first training log prints seconds-per-epoch — scale from that):

   | MODE | DaLiA folds | train windows / fold | epochs | est. time |
   |---|---|---|---|---|
   | `demo` (synthetic only) | 1 | 600 | 6 | ~10–15 min |
   | `quick` | 2 of 5 | 5 000 | 4 | ~3–4 h |
   | `full` | 3 of 5 | 9 000 | 5 | ~9–10 h (run as a background "Save & Run All") |

5. Everything is written to `/kaggle/working/results` and zipped into
   `/kaggle/working/stroboscopic_results_<MODE>.zip`: all CSV curves, every figure (PNG), the oscillator GIF,
   the 1920×1080 hero and device cards, poster, contact sheet, photos + attribution, the trained model, and
   `summary.json` with the PASS/FAIL of the pre-registered criterion. Runs are resumable (rows are appended
   to `results.csv` as they finish).

## How the simulation works

All streams are resampled to a 32 Hz tick grid. The expensive sensor stores `k` samples per tick
(PPG: 2); a burst reveals 4 ticks = 8 PPG samples = 125 ms, then the front-end is refractory.
Policies see only the cheap stream and bursts they fired earlier (send-on-delta is the documented
exception: it is the oracle transmitter-side threshold, an *optimistic* baseline).
Energy is reported as bursts per beat (primary), duty cycle, and a secondary mW estimate from
published MAX30101 / low-power IMU currents (`strobo/sim/energy.py`).

## Model (≈ 20 k parameters, ≈ 8 k MACs per tick amortised)

* **Encoder** – 3 causal dilated convs, 16 ch, 2.7 s receptive field, 32-d output.
* **Oscillator bank** – 12 Kuramoto oscillators (θ, ω, a) with learned coupling `K`, a neural drive from
  the IMU feature, and per-oscillator 32-bin waveform templates. When a burst arrives the bank takes one
  gradient step of its own template likelihood w.r.t. (θ, ω) with learned gains – an EKF/PLL-style
  innovation (Gauss-Newton step, ~100 MACs). Coherence = [prediction coherence `exp(−normalised burst
  residual)` (drives the fallback gate), normalised frequency variance, 1-s smoothed Kuramoto order parameter].
  At inference the fire probability is turned into decisions by a deterministic sigma-delta accumulator
  (no RNG on device; reproduces the trained rate and fires where `p` peaks).
* **Sampling head** – sees `[cos θ, sin θ, ω, a]`, coherence, the last burst and time-since-last;
  straight-through Gumbel-Bernoulli with temperature annealed 1.0 → 0.1 after a fixed-rate warm start.
  The **Fisher term** `Σ g_i a_i² w_i'(θ_i)²` is computed analytically from the templates and rewards
  sampling on steep parts of the cycle. The **coherence fallback** is a learned-threshold gate that is
  OR-ed into the fire probability.
* **Decoder** – 2 × 64 MLP on oscillator state + IMU feature + a 4-burst buffer, Gaussian head.
* **Loss** – `NLL + λe·mean(fire) − λf·mean(fire·Fisher) + λc·gate + λr·template-recon`; `λe` is swept.

## Baselines (same masking framework, each swept to a full curve)

fixed-rate · send-on-delta · event-triggered 2-state Kalman on PPG phase · IMU-gated dual-rate ·
learned-threshold sampler (our architecture, no phase input) · ours without Fisher · ours without fallback.

## Honest notes

* The plan's "few hundred MACs per tick" is not met by a 32-d feature encoder; the amortised figure is
  ≈ 8 k MACs/tick (≈ 0.2 ms on a Cortex-M4 at 48 MHz, < 1 % duty). `op_report` states both peak and
  amortised numbers.
* The synthetic generator is for pipeline checks and the demo; it is never mixed with real results and
  every figure produced from it is labelled `SYNTHETIC`.
* The success criterion is evaluated verbatim from the plan and printed PASS/FAIL; a negative result is
  reported, not hidden.
