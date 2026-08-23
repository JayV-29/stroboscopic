"""Signal-processing helpers shared by all loaders.

* R-peak detection on ECG (Pan-Tompkins style, no external dependency)
* per-tick physiological targets from event times (HR, RMSSD, RR, cadence)
* reference cardiac phase from R-peak times (for the phase diagnostic)
"""
from __future__ import annotations

import numpy as np
from scipy import signal


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resample along axis 0 (works for 1-D or (T, C))."""
    x = np.asarray(x, dtype=np.float32)
    if abs(fs_in - fs_out) < 1e-9:
        return x
    from fractions import Fraction
    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, fr.numerator, fr.denominator, axis=0).astype(np.float32)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    lo = max(lo / nyq, 1e-4)
    hi = min(hi / nyq, 0.999)
    sos = signal.butter(order, [lo, hi], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# R-peak detection
# --------------------------------------------------------------------------- #
def detect_rpeaks(ecg: np.ndarray, fs: float, refractory_s: float = 0.25) -> np.ndarray:
    """Return R-peak sample indices.  Pan-Tompkins-like pipeline:
    band-pass 5-20 Hz -> derivative -> square -> moving integration -> adaptive peaks."""
    ecg = np.asarray(ecg, dtype=np.float64).ravel()
    if ecg.size < int(2 * fs):
        return np.zeros(0, dtype=np.int64)
    f = bandpass(ecg, fs, 5.0, 20.0)
    d = np.gradient(f)
    sq = d * d
    win = max(3, int(0.12 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    # adaptive threshold: rolling percentile-based
    thr = 0.3 * _rolling_max(integ, int(2.0 * fs))
    peaks, _ = signal.find_peaks(integ, height=thr, distance=int(refractory_s * fs))
    # refine to the true R location on the band-passed signal
    half = int(0.05 * fs)
    out = []
    for p in peaks:
        a, b = max(0, p - half), min(len(f), p + half + 1)
        out.append(a + int(np.argmax(np.abs(f[a:b]))))
    return np.unique(np.asarray(out, dtype=np.int64))


def _rolling_max(x: np.ndarray, win: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d
    m = maximum_filter1d(x, size=max(3, win), mode="nearest")
    # smooth so that isolated outliers don't dominate
    return np.maximum(m, 1e-8)


def clean_rr(times: np.ndarray, lo: float = 0.3, hi: float = 2.0) -> np.ndarray:
    """Drop beats that produce physiologically impossible RR intervals."""
    times = np.asarray(times, dtype=np.float64)
    if times.size < 2:
        return times
    keep = [times[0]]
    for t in times[1:]:
        if lo <= t - keep[-1] <= hi:
            keep.append(t)
        elif t - keep[-1] > hi:
            keep.append(t)  # a long gap: restart, keep the beat
    return np.asarray(keep)


# --------------------------------------------------------------------------- #
# per-tick targets from event times
# --------------------------------------------------------------------------- #
def rate_from_events(event_t: np.ndarray, T: int, fs: float, window_s: float,
                     stride_s: float = 0.5, min_events: int = 3) -> np.ndarray:
    """Trailing-window rate (events per minute) evaluated every ``stride_s`` and
    held between evaluations.  NaN where fewer than ``min_events`` in window."""
    out = np.full(T, np.nan, dtype=np.float32)
    if event_t.size < min_events:
        return out
    stride = max(1, int(stride_s * fs))
    ticks = np.arange(0, T, stride)
    t_end = ticks / fs
    t_start = t_end - window_s
    i0 = np.searchsorted(event_t, t_start, side="left")
    i1 = np.searchsorted(event_t, t_end, side="right")
    n = i1 - i0
    vals = np.full(ticks.size, np.nan, dtype=np.float32)
    ok = n >= min_events
    first = event_t[np.clip(i0, 0, event_t.size - 1)]
    last = event_t[np.clip(i1 - 1, 0, event_t.size - 1)]
    span = last - first
    good = ok & (span > 0)
    vals[good] = 60.0 * (n[good] - 1) / span[good]
    for j, t in enumerate(ticks):
        out[t:t + stride] = vals[j]
    return out


def rmssd_from_events(event_t: np.ndarray, T: int, fs: float, window_s: float = 30.0,
                      stride_s: float = 0.5, min_events: int = 5) -> np.ndarray:
    """Trailing-window RMSSD in milliseconds."""
    out = np.full(T, np.nan, dtype=np.float32)
    if event_t.size < min_events:
        return out
    stride = max(1, int(stride_s * fs))
    rr = np.diff(event_t)
    rr_t = event_t[1:]
    drr2 = np.diff(rr) ** 2
    d_t = rr_t[1:]
    for t in range(0, T, stride):
        t_end = t / fs
        m = (d_t > t_end - window_s) & (d_t <= t_end)
        if m.sum() >= min_events - 2:
            out[t:t + stride] = 1000.0 * np.sqrt(drr2[m].mean())
    return out


def phase_from_events(event_t: np.ndarray, T: int, fs: float) -> np.ndarray:
    """Linear phase in [0, 2pi) between consecutive events; NaN outside."""
    ph = np.full(T, np.nan, dtype=np.float32)
    if event_t.size < 2:
        return ph
    t = np.arange(T) / fs
    idx = np.searchsorted(event_t, t, side="right") - 1
    ok = (idx >= 0) & (idx < event_t.size - 1)
    a = event_t[np.clip(idx, 0, event_t.size - 2)]
    b = event_t[np.clip(idx + 1, 1, event_t.size - 1)]
    frac = (t - a) / np.maximum(b - a, 1e-6)
    ph[ok] = (2 * np.pi * frac[ok]) % (2 * np.pi)
    return ph


def beat_indicator(event_t: np.ndarray, T: int, fs: float) -> np.ndarray:
    b = np.zeros(T, dtype=np.float32)
    idx = np.round(event_t * fs).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < T)]
    b[idx] = 1.0
    return b


# --------------------------------------------------------------------------- #
# respiration and cadence events
# --------------------------------------------------------------------------- #
def resp_events(resp: np.ndarray, fs: float) -> np.ndarray:
    """Breath onset times from a respiration belt (peak of inhalation)."""
    f = bandpass(resp.astype(np.float64), fs, 0.1, 0.7)
    peaks, _ = signal.find_peaks(f, distance=int(1.5 * fs), prominence=0.3 * np.std(f))
    return peaks / fs


def step_events(acc: np.ndarray, fs: float) -> np.ndarray:
    """Step times from accelerometer magnitude (peaks of the 0.5-3 Hz band)."""
    mag = np.linalg.norm(np.asarray(acc, dtype=np.float64), axis=1)
    f = bandpass(mag, fs, 0.5, 3.0)
    peaks, _ = signal.find_peaks(f, distance=int(0.25 * fs), prominence=0.5 * np.std(f) + 1e-6)
    return peaks / fs


def cadence_from_acc(acc: np.ndarray, fs: float, T: int, window_s: float = 4.0,
                     stride_s: float = 0.5) -> np.ndarray:
    """Steps per minute from the dominant autocorrelation lag of ACC magnitude."""
    mag = np.linalg.norm(np.asarray(acc, dtype=np.float64), axis=1)
    f = bandpass(mag, fs, 0.5, 4.0)
    out = np.full(T, np.nan, dtype=np.float32)
    win = int(window_s * fs)
    stride = max(1, int(stride_s * fs))
    lag_lo, lag_hi = int(fs / 4.0), int(fs / 0.5)      # 0.5-4 Hz
    for t in range(win, T, stride):
        seg = f[t - win:t]
        seg = seg - seg.mean()
        if seg.std() < 1e-6:
            continue
        ac = np.correlate(seg, seg, mode="full")[win - 1:]
        ac = ac / (ac[0] + 1e-9)
        lag = lag_lo + int(np.argmax(ac[lag_lo:lag_hi]))
        if ac[lag] > 0.3:
            out[t:t + stride] = 60.0 * fs / lag
    return out
