"""WESAD loader (Schmidt et al. 2018, UCI).

Expected layout:  <root>/**/S<i>/S<i>.pkl
Pickle: {'signal': {'chest': {'ACC','ECG','Resp',...} @700 Hz,
                    'wrist': {'ACC' @32 Hz, 'BVP' @64 Hz, ...}}, 'label' @700 Hz}

Two tasks are produced from the same recording:
  task='resp' : cheap = chest ACC (32 Hz), expensive = respiration belt (1 sample/tick),
                targets = RR (breaths/min, trailing 32 s)
  task='hr'   : cheap = wrist ACC, expensive = wrist BVP (2/tick), targets = HR, RMSSD
Labels: 1 baseline, 2 stress, 3 amusement, 4 meditation -> all 'rest' (WESAD is seated);
the walking transitions (label 0 / 5-7) are marked 'other'.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from .recording import Recording, ACT_OTHER, ACT_REST
from . import rpeaks as rp


def find_subject_files(root: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(root, "**", "S*", "S*.pkl"), recursive=True))
    files = [f for f in files if os.path.basename(f)[1:-4].isdigit()]
    return sorted(files, key=lambda f: int(os.path.basename(f)[1:-4]))


def load_wesad_subject(path: str, task: str = "resp", fs: int = 32) -> Recording:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    sig = d["signal"]
    fs_chest = 700
    ecg = np.asarray(sig["chest"]["ECG"], dtype=np.float32).ravel()
    label = np.asarray(d["label"]).ravel().astype(int)
    beat_t = rp.clean_rr(rp.detect_rpeaks(ecg, fs_chest) / fs_chest)

    if task == "resp":
        acc = rp.resample_to(np.asarray(sig["chest"]["ACC"], dtype=np.float32), fs_chest, fs)
        resp = np.asarray(sig["chest"]["Resp"], dtype=np.float32).ravel()
        exp = rp.resample_to(resp, fs_chest, fs)
        k = 1
        T = min(acc.shape[0], exp.shape[0])
        acc, exp = acc[:T], exp[:T].reshape(T, k)
        ev_t = rp.resp_events(resp, fs_chest)
        rr = rp.rate_from_events(ev_t, T, fs, window_s=32.0, min_events=3)
        targets = rr[:, None].astype(np.float32)
        names = ["rr"]
        event_t = ev_t
    elif task == "hr":
        acc = rp.resample_to(np.asarray(sig["wrist"]["ACC"], dtype=np.float32) / 64.0, 32, fs)
        k = 2
        bvp = rp.resample_to(np.asarray(sig["wrist"]["BVP"], dtype=np.float32).ravel(), 64, fs * k)
        T = min(acc.shape[0], bvp.shape[0] // k)
        acc, exp = acc[:T], bvp[:T * k].reshape(T, k)
        hr = rp.rate_from_events(beat_t, T, fs, window_s=8.0)
        rmssd = rp.rmssd_from_events(beat_t, T, fs)
        targets = np.stack([hr, rmssd], 1).astype(np.float32)
        names = ["hr", "rmssd"]
        event_t = beat_t
    else:
        raise ValueError(task)

    lab_t = label[::fs_chest // fs][:T]
    lab_t = np.pad(lab_t, (0, T - lab_t.size), constant_values=0)
    activity = np.full(T, ACT_OTHER, dtype=np.int8)
    activity[np.isin(lab_t, [1, 2, 3, 4])] = ACT_REST
    subject = os.path.basename(path)[:-4]
    return Recording(
        subject=subject, fs=fs, cheap=acc, expensive=exp, targets=targets, target_names=names,
        beats=rp.beat_indicator(event_t, T, fs), activity=activity,
        ref_phase=rp.phase_from_events(event_t, T, fs), valid=np.ones(T, dtype=bool),
        dataset=f"wesad_{task}", meta={"label": lab_t.astype(np.int8)},
    ).check()


def load_wesad(root: str, task: str = "resp", subjects: list[str] | None = None, **kw) -> list[Recording]:
    recs = []
    for f in find_subject_files(root):
        sid = os.path.basename(f)[:-4]
        if subjects is not None and sid not in subjects:
            continue
        recs.append(load_wesad_subject(f, task=task, **kw))
    if not recs:
        raise FileNotFoundError(f"no WESAD pickles found under {root}")
    return recs
