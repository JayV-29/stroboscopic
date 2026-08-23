"""MIT-BIH Arrhythmia loader (PhysioNet) for the coherence-fallback stress test.

Expected layout: <root>/**/<record>.{dat,hea,atr}
There is no IMU in MIT-BIH, so the cheap stream is a *zero* channel: the
oscillator bank is driven only by its own sparse ECG observations.  This is the
hardest case for a stroboscopic sampler and exactly what the fallback exists for.

Cheap stream : zeros (1 ch, 32 Hz)
Expensive    : ECG resampled to 128 Hz -> k = 4 samples / tick
Targets      : HR (bpm) from annotated beats
activity     : ACT_REST for normal-rhythm ticks, ACT_MOTION for ticks whose
               trailing 3-beat window contains a non-normal beat (re-using the
               motion slot to mean 'irregular'), so the same rest/motion
               reporting machinery gives normal-vs-arrhythmic numbers.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from .recording import Recording, ACT_REST, ACT_MOTION
from . import rpeaks as rp

NORMAL = {"N", "L", "R", "e", "j"}          # beats treated as 'normal rhythm'
NON_BEAT = {"+", "~", "|", "s", "T", "*", "D", "=", '"', "@", "[", "]", "!", "x"}


def find_records(root: str) -> list[str]:
    heas = sorted(glob.glob(os.path.join(root, "**", "*.hea"), recursive=True))
    return [h[:-4] for h in heas if os.path.exists(h[:-4] + ".atr")]


def load_mitbih_record(rec_path: str, fs: int = 32, k: int = 4, channel: int = 0) -> Recording:
    import wfdb
    r = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")
    fs_in = float(r.fs)
    ecg = np.asarray(r.p_signal[:, channel], dtype=np.float32)
    exp = rp.resample_to(ecg, fs_in, fs * k)
    T = exp.shape[0] // k
    exp = exp[:T * k].reshape(T, k)
    sym = np.asarray(ann.symbol)
    samp = np.asarray(ann.sample)
    is_beat = ~np.isin(sym, list(NON_BEAT))
    beat_t = samp[is_beat] / fs_in
    beat_sym = sym[is_beat]
    abnormal = ~np.isin(beat_sym, list(NORMAL))

    hr = rp.rate_from_events(beat_t, T, fs, window_s=8.0)
    rmssd = rp.rmssd_from_events(beat_t, T, fs)
    # irregular flag: trailing 3-beat window contains an abnormal beat
    activity = np.full(T, ACT_REST, dtype=np.int8)
    t = np.arange(T) / fs
    idx = np.searchsorted(beat_t, t, side="right")
    for m in range(3):
        j = np.clip(idx - 1 - m, 0, beat_t.size - 1)
        activity[(idx - 1 - m >= 0) & abnormal[j]] = ACT_MOTION
    return Recording(
        subject=os.path.basename(rec_path), fs=fs, cheap=np.zeros((T, 1), np.float32),
        expensive=exp, targets=np.stack([hr, rmssd], 1).astype(np.float32),
        target_names=["hr", "rmssd"], beats=rp.beat_indicator(beat_t, T, fs), activity=activity,
        ref_phase=rp.phase_from_events(beat_t, T, fs), valid=np.ones(T, bool), dataset="mitbih",
        meta={"beat_t": beat_t, "abnormal": abnormal},
    ).check()


def load_mitbih(root: str, records: list[str] | None = None, max_records: int | None = None, **kw):
    paths = find_records(root)
    if records is not None:
        paths = [p for p in paths if os.path.basename(p) in records]
    if max_records:
        paths = paths[:max_records]
    if not paths:
        raise FileNotFoundError(f"no MIT-BIH records under {root}")
    return [load_mitbih_record(p, **kw) for p in paths]
