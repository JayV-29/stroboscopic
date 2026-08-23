"""Parameter and multiply-add accounting per IMU tick.

The encoder, oscillator bank and sampling head run every tick.  The decoder
only needs to run when an output is consumed (HR at 2 Hz is the wearable
convention), so its cost is amortised over ``decoder_every`` ticks.
"""
from __future__ import annotations

from ..models.strobo import StroboModel


def op_report(model: StroboModel, decoder_every: int = 16) -> dict:
    macs = model.macs_per_tick()
    n_params = model.n_params()
    c = model.cfg
    per_tick = macs["encoder"] + macs["oscillator"] + macs["head"] + macs["decoder"] / decoder_every
    # non-MAC work per tick: N sin/cos pairs, N*N sin for coupling, GELUs
    transc = 2 * c.n_osc + c.n_osc * c.n_osc + c.enc_ch * 3 + c.hidden_head
    return {
        "params": n_params,
        "params_kb_int8": n_params / 1024.0,
        "macs_per_tick_amortised": int(per_tick),
        "macs_per_tick_peak": macs["total"],
        "macs_breakdown": macs,
        "decoder_every_ticks": decoder_every,
        "transcendentals_per_tick": transc,
        "us_per_tick_m4_48mhz": per_tick / 48.0 + transc * 0.4,   # ~1 MAC/cycle, ~20 cycles per sin
        "budget_ok": per_tick < 10_000 and n_params < 100_000,
    }
