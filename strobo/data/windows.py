"""Cut recordings into fixed-length training windows and build folds."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .recording import Recording, ACT_REST, ACT_MOTION


def make_windows(recs: list[Recording], win_s: float = 8.0, stride_s: float = 4.0,
                 min_valid: float = 0.9) -> dict:
    """Return a dict of stacked arrays; each row is one window.

    All ticks in a window carry targets, so the loss is computed at every tick
    after a warm-up (the model runs as a streaming filter).
    """
    out = {k: [] for k in ("cheap", "expensive", "targets", "beats", "activity",
                           "ref_phase", "valid", "subject", "rec_idx", "start")}
    for ri, r in enumerate(recs):
        L, S = int(win_s * r.fs), int(stride_s * r.fs)
        for s in range(0, r.T - L + 1, S):
            sl = slice(s, s + L)
            if r.valid[sl].mean() < min_valid:
                continue
            out["cheap"].append(r.cheap[sl]); out["expensive"].append(r.expensive[sl])
            out["targets"].append(r.targets[sl]); out["beats"].append(r.beats[sl])
            out["activity"].append(r.activity[sl]); out["ref_phase"].append(r.ref_phase[sl])
            out["valid"].append(r.valid[sl]); out["subject"].append(r.subject)
            out["rec_idx"].append(ri); out["start"].append(s)
    res = {k: (np.stack(v) if k not in ("subject",) else np.asarray(v)) for k, v in out.items()}
    res["fs"] = recs[0].fs
    res["target_names"] = list(recs[0].target_names)
    return res


class WindowDataset(Dataset):
    def __init__(self, W: dict, idx: np.ndarray | None = None):
        self.W = W
        self.idx = np.arange(W["cheap"].shape[0]) if idx is None else np.asarray(idx)

    def __len__(self):
        return self.idx.size

    def __getitem__(self, i):
        j = int(self.idx[i])
        W = self.W
        return {
            "cheap": torch.from_numpy(W["cheap"][j]),
            "expensive": torch.from_numpy(W["expensive"][j]),
            "targets": torch.from_numpy(W["targets"][j]),
            "beats": torch.from_numpy(W["beats"][j]),
            "activity": torch.from_numpy(W["activity"][j].astype(np.int64)),
            "ref_phase": torch.from_numpy(W["ref_phase"][j]),
            "valid": torch.from_numpy(W["valid"][j]),
            "idx": j,
        }


def split_subjects_kfold(subjects: np.ndarray, n_folds: int = 5, seed: int = 0):
    """Leave-subjects-out folds.  Yields (train_idx, test_idx, test_subjects)."""
    uniq = np.array(sorted(set(subjects.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    parts = np.array_split(uniq, n_folds)
    for p in parts:
        test = np.isin(subjects, p)
        yield np.where(~test)[0], np.where(test)[0], p.tolist()


def window_activity_mode(W: dict) -> np.ndarray:
    """Majority activity label per window (ACT_*)."""
    a = W["activity"]
    rest = (a == ACT_REST).mean(1)
    mot = (a == ACT_MOTION).mean(1)
    mode = np.zeros(a.shape[0], dtype=np.int8)
    mode[rest > 0.5] = ACT_REST
    mode[mot > 0.5] = ACT_MOTION
    return mode
