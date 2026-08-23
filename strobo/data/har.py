"""UCI HAR raw inertial signals -> stride-cadence task.

Expected layout: <root>/**/UCI HAR Dataset/{train,test}/Inertial Signals/*.txt
Windows are 128 samples @ 50 Hz with 50 % overlap; consecutive windows with the
same subject and activity are stitched back into continuous sequences.

Cheap stream : body_acc (3 ch, 50 Hz)      -> tick rate 50 Hz
Expensive    : body_gyro (3 ch, 1 sample/tick, k=3 channels folded into k)
Target       : cadence (steps/min) from dense total_acc autocorrelation;
               NaN outside walking-type activities.
Activity     : walking / upstairs / downstairs -> motion; sitting/standing/laying -> rest
"""
from __future__ import annotations

import glob
import os

import numpy as np

from .recording import Recording, ACT_REST, ACT_MOTION
from . import rpeaks as rp

FS = 50
AXES = ("x", "y", "z")


def _find_root(root: str) -> str:
    cands = glob.glob(os.path.join(root, "**", "Inertial Signals"), recursive=True)
    if not cands:
        raise FileNotFoundError(f"UCI HAR 'Inertial Signals' not found under {root}")
    return os.path.dirname(os.path.dirname(cands[0]))


def _load_split(base: str, split: str):
    d = os.path.join(base, split, "Inertial Signals")
    acc = np.stack([np.loadtxt(os.path.join(d, f"body_acc_{a}_{split}.txt")) for a in AXES], -1)
    tot = np.stack([np.loadtxt(os.path.join(d, f"total_acc_{a}_{split}.txt")) for a in AXES], -1)
    gyr = np.stack([np.loadtxt(os.path.join(d, f"body_gyro_{a}_{split}.txt")) for a in AXES], -1)
    y = np.loadtxt(os.path.join(base, split, f"y_{split}.txt")).astype(int)
    subj = np.loadtxt(os.path.join(base, split, f"subject_{split}.txt")).astype(int)
    return acc.astype(np.float32), tot.astype(np.float32), gyr.astype(np.float32), y, subj


def _stitch(acc, tot, gyr, y, subj):
    """Stitch 50 %-overlapping windows into continuous runs of (subject, activity)."""
    runs = []
    n = acc.shape[0]
    i = 0
    while i < n:
        j = i + 1
        while j < n and y[j] == y[i] and subj[j] == subj[i] and \
                np.allclose(acc[j, :64], acc[j - 1, 64:], atol=1e-4):
            j += 1
        # windows i..j-1 form one run
        parts_a = [acc[i]] + [acc[m, 64:] for m in range(i + 1, j)]
        parts_t = [tot[i]] + [tot[m, 64:] for m in range(i + 1, j)]
        parts_g = [gyr[i]] + [gyr[m, 64:] for m in range(i + 1, j)]
        runs.append((int(subj[i]), int(y[i]), np.concatenate(parts_a), np.concatenate(parts_t),
                     np.concatenate(parts_g)))
        i = j
    return runs


def load_har(root: str, min_run_s: float = 10.0, subjects: list[int] | None = None) -> list[Recording]:
    base = _find_root(root)
    runs = []
    for split in ("train", "test"):
        runs += _stitch(*_load_split(base, split))
    # group runs per subject into one recording (concatenate with a validity gap)
    by_subj: dict[int, list] = {}
    for s, a, acc, tot, gyr in runs:
        if acc.shape[0] < min_run_s * FS:
            continue
        by_subj.setdefault(s, []).append((a, acc, tot, gyr))
    recs = []
    for s, parts in sorted(by_subj.items()):
        if subjects is not None and s not in subjects:
            continue
        cheap, exp, act, targ, beats, ph, valid = [], [], [], [], [], [], []
        for a, acc, tot, gyr in parts:
            T = acc.shape[0]
            mode = ACT_MOTION if a in (1, 2, 3) else ACT_REST
            cad = rp.cadence_from_acc(tot, FS, T) if mode == ACT_MOTION else np.full(T, np.nan, np.float32)
            ev = rp.step_events(tot, FS) if mode == ACT_MOTION else np.zeros(0)
            cheap.append(acc); exp.append(gyr); act.append(np.full(T, mode, np.int8))
            targ.append(cad[:, None]); beats.append(rp.beat_indicator(ev, T, FS))
            ph.append(rp.phase_from_events(ev, T, FS)); valid.append(np.ones(T, bool))
            # mark a boundary so windows don't straddle runs
            valid[-1][:FS] = False
        recs.append(Recording(
            subject=f"H{s:02d}", fs=FS, cheap=np.concatenate(cheap), expensive=np.concatenate(exp),
            targets=np.concatenate(targ).astype(np.float32), target_names=["cadence"],
            beats=np.concatenate(beats), activity=np.concatenate(act),
            ref_phase=np.concatenate(ph), valid=np.concatenate(valid), dataset="uci_har",
        ).check())
    if not recs:
        raise FileNotFoundError("no UCI HAR runs found")
    return recs
