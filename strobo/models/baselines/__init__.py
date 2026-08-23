"""Classical sampling policies.  Each is a callable

    policy(cheap: (B,T,C), expensive: (B,T,k), fs, burst_ticks) -> fire (B,T) in {0,1}

that is *causal*: a decision at tick t may only use ticks <= t of the cheap
stream and expensive-sensor bursts it fired earlier (except send-on-delta,
which is documented as an oracle transmitter-side threshold).
"""
from .fixed_rate import FixedRate
from .send_on_delta import SendOnDelta
from .kalman import EventTriggeredKalman
from .imu_gated import IMUGated

BASELINES = {"fixed_rate": FixedRate, "send_on_delta": SendOnDelta,
             "kalman": EventTriggeredKalman, "imu_gated": IMUGated}

__all__ = ["FixedRate", "SendOnDelta", "EventTriggeredKalman", "IMUGated", "BASELINES"]
