"""Energy accounting.  Primary metric is *bursts per beat*; joules are a
secondary estimate from published current figures.

Defaults (order-of-magnitude, documented sources):
  MAX30101 PPG front-end, one 125 ms burst at ~600 uA @ 1.8 V  ->  ~135 uJ
  LSM6DSx-class IMU tick at 32 Hz, low-power mode  ~ 20 uA @ 1.8 V -> 1.1 uJ / tick
  Cortex-M4 @ 48 MHz, ~300 MACs per tick (~1 us)  ~ 4 mA @ 1.8 V -> ~0.01 uJ / tick
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EnergyModel:
    uj_per_burst: float = 135.0
    uj_per_imu_tick: float = 1.1
    uj_per_inference: float = 0.01

    def joules(self, n_bursts: float, n_ticks: float) -> float:
        return 1e-6 * (self.uj_per_burst * n_bursts
                       + (self.uj_per_imu_tick + self.uj_per_inference) * n_ticks)

    def mw(self, n_bursts: float, n_ticks: float, fs: float) -> float:
        return 1e3 * self.joules(n_bursts, n_ticks) / (n_ticks / fs)


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def bursts_per_beat(fire, beats) -> float:
    fire, beats = _np(fire), _np(beats)
    nb = beats.sum()
    return float(fire.sum() / max(nb, 1.0))


def duty_cycle(fire, burst_ticks: int) -> float:
    fire = _np(fire)
    return float(min(1.0, fire.mean() * burst_ticks))
