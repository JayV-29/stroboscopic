"""Synthetic wrist-IMU + PPG + ECG-beat generator.

Used for (a) smoke tests without any download and (b) the notebook's demo
mode.  It reproduces the qualitative structure the model has to exploit:

* HR drifts slowly and rises during motion segments.
* PPG is a fixed pulse template evaluated at the cardiac phase, with a
  gait-locked motion artefact of comparable amplitude during motion.
* ACC carries gravity, strong gait harmonics during motion, and a very weak
  ballistocardiographic component at rest.
"""
from __future__ import annotations

import numpy as np

from .recording import Recording, ACT_REST, ACT_MOTION
from . import rpeaks as rp


def pulse_template(phase: np.ndarray) -> np.ndarray:
    """Asymmetric PPG pulse: fast systolic upstroke, slow diastolic decay, dicrotic notch."""
    p = np.mod(phase, 2 * np.pi) / (2 * np.pi)
    up = np.exp(-((p - 0.15) ** 2) / (2 * 0.05 ** 2))
    decay = np.exp(-(p - 0.15) / 0.35) * (p > 0.15)
    notch = 0.25 * np.exp(-((p - 0.45) ** 2) / (2 * 0.04 ** 2))
    return (up + 0.8 * decay + notch).astype(np.float32)


def make_synthetic_recording(subject: str, minutes: float = 10.0, fs: int = 32, k: int = 2,
                             seed: int = 0, rest_hr: float | None = None) -> Recording:
    rng = np.random.default_rng(seed)
    T = int(minutes * 60 * fs)
    fs_exp = fs * k
    t_exp = np.arange(T * k) / fs_exp

    # ---- activity schedule: alternating rest/motion segments of 45-120 s
    activity = np.zeros(T, dtype=np.int8)
    pos, mode = 0, ACT_REST
    while pos < T:
        seg = int(rng.uniform(45, 120) * fs)
        activity[pos:pos + seg] = mode
        pos += seg
        mode = ACT_MOTION if mode == ACT_REST else ACT_REST

    # ---- heart rate (bpm) as slow random walk + motion offset
    base = rest_hr if rest_hr is not None else rng.uniform(55, 80)
    drift = np.cumsum(rng.normal(0, 0.02, T)) * fs ** -0.5
    hr = base + 8 * np.sin(2 * np.pi * np.arange(T) / (fs * 300)) + 6 * drift
    motion_gain = _smooth((activity == ACT_MOTION).astype(np.float64), int(10 * fs))
    hr = hr + motion_gain * rng.uniform(35, 60)
    hr = np.clip(hr, 45, 185)
    hr_exp = np.repeat(hr, k)
    # RSA / HRV: small oscillatory modulation of instantaneous frequency
    resp_f = 0.25
    inst_f = hr_exp / 60.0 * (1 + 0.04 * np.sin(2 * np.pi * resp_f * t_exp))
    phase = np.cumsum(inst_f / fs_exp) * 2 * np.pi
    # beat times: phase crossings of multiples of 2pi
    n_beats = int(phase[-1] // (2 * np.pi))
    beat_t = np.interp(2 * np.pi * np.arange(1, n_beats + 1), phase, t_exp)
    beat_t += rng.normal(0, 0.004, beat_t.size)  # timing jitter -> RMSSD
    beat_t = np.sort(beat_t)

    # ---- gait
    cad_hz = rng.uniform(1.6, 2.0)  # steps / s during walking
    gait_phase = np.cumsum(np.full(T * k, cad_hz / fs_exp)) * 2 * np.pi
    gait_env_exp = np.repeat(motion_gain, k)

    # ---- PPG
    ppg = pulse_template(phase)
    ppg = ppg - ppg.mean()
    artefact = (1.2 * np.sin(gait_phase) + 0.5 * np.sin(2 * gait_phase + 0.7)) * gait_env_exp
    ppg = ppg + artefact + rng.normal(0, 0.08, ppg.size)
    ppg = (ppg * (1 + 0.1 * np.sin(2 * np.pi * resp_f * t_exp))).astype(np.float32)
    expensive = ppg.reshape(T, k)

    # ---- ACC (3-axis) at tick rate
    gp = gait_phase[::k]
    env = motion_gain
    t_tick = np.arange(T) / fs
    acc = np.stack([
        0.1 + env * (0.9 * np.sin(gp) + 0.3 * np.sin(2 * gp)),
        -0.98 + env * (0.7 * np.sin(gp + 1.1) + 0.2 * np.sin(3 * gp)),
        0.05 + env * (0.5 * np.sin(2 * gp + 0.3)),
    ], axis=1)
    bcg = 0.01 * np.interp(t_tick, t_exp, np.gradient(pulse_template(phase)))  # tiny cardiac trace
    acc = acc + bcg[:, None] + rng.normal(0, 0.03, acc.shape)
    cheap = acc.astype(np.float32)

    # ---- targets
    hr_t = rp.rate_from_events(beat_t, T, fs, window_s=8.0)
    rmssd_t = rp.rmssd_from_events(beat_t, T, fs)
    targets = np.stack([hr_t, rmssd_t], 1).astype(np.float32)
    rec = Recording(
        subject=subject, fs=fs, cheap=cheap, expensive=expensive,
        targets=targets, target_names=["hr", "rmssd"],
        beats=rp.beat_indicator(beat_t, T, fs), activity=activity,
        ref_phase=rp.phase_from_events(beat_t, T, fs),
        valid=np.ones(T, dtype=bool), dataset="synthetic",
        meta={"true_hr": hr.astype(np.float32), "cadence_hz": cad_hz},
    )
    return rec.check()


def make_synthetic_recordings(n_subjects: int = 6, minutes: float = 10.0, fs: int = 32,
                              k: int = 2, seed: int = 0) -> list[Recording]:
    return [make_synthetic_recording(f"syn{i:02d}", minutes, fs, k, seed=seed * 1000 + i)
            for i in range(n_subjects)]


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    ker = np.ones(win) / win
    return np.convolve(x, ker, mode="same")
