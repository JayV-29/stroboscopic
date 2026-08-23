"""PPG-DaLiA loader (Reiss et al. 2019, UCI).

Expected layout (any root):  <root>/**/S<i>/S<i>.pkl
Each pickle: {'signal': {'chest': {'ECG','ACC',...}, 'wrist': {'BVP','ACC',...}},
              'label': HR per 8 s window (2 s shift), 'activity': labels at 4 Hz,
              'rpeaks': R-peak indices at 700 Hz, 'subject': 'S1'}

Cheap stream : wrist ACC 32 Hz (3 channels)            -> tick rate 32 Hz
Expensive    : wrist BVP 64 Hz -> 2 samples / tick
Targets      : HR (bpm, trailing 8 s) and RMSSD (ms, trailing 30 s) from ECG R-peaks
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from .recording import Recording, ACT_OTHER, ACT_REST, ACT_MOTION
from . import rpeaks as rp

# activity ids in PPG-DaLiA
DALIA_ACT = {0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer", 4: "cycling",
             5: "driving", 6: "lunch", 7: "walking", 8: "working"}
REST_IDS = {1, 8}
MOTION_IDS = {2, 3, 4, 7}


def find_subject_files(root: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(root, "**", "S*", "S*.pkl"), recursive=True))
    files = [f for f in files if os.path.basename(f)[1:-4].isdigit()]
    return sorted(files, key=lambda f: int(os.path.basename(f)[1:-4]))


def load_dalia_subject(path: str, fs: int = 32, k: int = 2) -> Recording:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    sig = d["signal"]
    acc = np.asarray(sig["wrist"]["ACC"], dtype=np.float32)          # 32 Hz, (T,3) in 1/64 g
    bvp = np.asarray(sig["wrist"]["BVP"], dtype=np.float32).ravel()  # 64 Hz
    ecg = np.asarray(sig["chest"]["ECG"], dtype=np.float32).ravel()  # 700 Hz
    fs_ecg = 700
    acc = rp.resample_to(acc / 64.0, 32, fs)
    bvp = rp.resample_to(bvp, 64, fs * k)
    T = min(acc.shape[0], bvp.shape[0] // k)
    acc, bvp = acc[:T], bvp[:T * k].reshape(T, k)

    if "rpeaks" in d and len(d["rpeaks"]) > 10:
        beat_t = np.asarray(d["rpeaks"], dtype=np.float64) / fs_ecg
    else:
        beat_t = rp.detect_rpeaks(ecg, fs_ecg) / fs_ecg
    beat_t = rp.clean_rr(beat_t)

    act_raw = np.asarray(d["activity"]).ravel().astype(int)          # 4 Hz
    act_t = np.repeat(act_raw, fs // 4)[:T]
    act_t = np.pad(act_t, (0, T - act_t.size), constant_values=0)
    activity = np.full(T, ACT_OTHER, dtype=np.int8)
    activity[np.isin(act_t, list(REST_IDS))] = ACT_REST
    activity[np.isin(act_t, list(MOTION_IDS))] = ACT_MOTION

    hr = rp.rate_from_events(beat_t, T, fs, window_s=8.0)
    rmssd = rp.rmssd_from_events(beat_t, T, fs)
    targets = np.stack([hr, rmssd], 1).astype(np.float32)
    subject = str(d.get("subject", os.path.basename(path)[:-4]))
    return Recording(
        subject=subject, fs=fs, cheap=acc, expensive=bvp, targets=targets,
        target_names=["hr", "rmssd"], beats=rp.beat_indicator(beat_t, T, fs),
        activity=activity, ref_phase=rp.phase_from_events(beat_t, T, fs),
        valid=np.ones(T, dtype=bool), dataset="ppg_dalia",
        meta={"activity_raw": act_t.astype(np.int8)},
    ).check()


def load_dalia(root: str, subjects: list[str] | None = None, **kw) -> list[Recording]:
    recs = []
    for f in find_subject_files(root):
        sid = os.path.basename(f)[:-4]
        if subjects is not None and sid not in subjects:
            continue
        recs.append(load_dalia_subject(f, **kw))
    if not recs:
        raise FileNotFoundError(f"no PPG-DaLiA pickles found under {root}")
    return recs
