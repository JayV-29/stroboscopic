"""The common in-memory representation every loader produces.

Everything is resampled onto a single *tick* grid (the cheap-sensor rate, 32 Hz
for wrist ACC).  The expensive sensor keeps a higher native rate by storing
``k`` samples per tick, so a burst of ``B`` ticks reveals ``k*B`` raw samples
(8 PPG samples = 4 ticks x 2 samples/tick = 125 ms at 64 Hz).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

ACT_OTHER, ACT_REST, ACT_MOTION = 0, 1, 2
ACT_NAMES = {ACT_OTHER: "other", ACT_REST: "rest", ACT_MOTION: "motion"}


@dataclass
class Recording:
    subject: str
    fs: int                          # tick rate in Hz
    cheap: np.ndarray                # (T, C) float32  always-on stream (e.g. wrist ACC)
    expensive: np.ndarray            # (T, k) float32  expensive stream, k samples per tick
    targets: np.ndarray              # (T, n_t) float32, NaN where undefined
    target_names: list[str]
    beats: np.ndarray                # (T,) float32, 1.0 at ticks containing a reference beat/cycle
    activity: np.ndarray             # (T,) int8, ACT_*
    ref_phase: np.ndarray            # (T,) float32 reference phase in [0, 2pi) (e.g. from ECG R-peaks), NaN if none
    valid: np.ndarray                # (T,) bool
    dataset: str = "unknown"
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.cheap.shape[0]

    @property
    def k(self) -> int:
        return self.expensive.shape[1]

    def check(self):
        T = self.T
        for name in ("expensive", "targets", "beats", "activity", "ref_phase", "valid"):
            a = getattr(self, name)
            assert a.shape[0] == T, f"{name} has {a.shape[0]} rows, expected {T}"
        assert self.targets.shape[1] == len(self.target_names)
        return self

    def summary(self) -> str:
        secs = self.T / self.fs
        frac_rest = float((self.activity == ACT_REST).mean())
        frac_mot = float((self.activity == ACT_MOTION).mean())
        return (f"{self.dataset}/{self.subject}: {secs/60:.1f} min @ {self.fs} Hz, "
                f"k={self.k}, beats={int(self.beats.sum())}, rest={frac_rest:.0%}, motion={frac_mot:.0%}")


def zscore_per_recording(rec: Recording, eps: float = 1e-6) -> Recording:
    """Normalise cheap and expensive streams in place (per recording, per channel)."""
    v = rec.valid
    for name in ("cheap", "expensive"):
        a = getattr(rec, name)
        mu = a[v].mean(0, keepdims=True)
        sd = a[v].std(0, keepdims=True) + eps
        setattr(rec, name, ((a - mu) / sd).astype(np.float32))
    return rec
