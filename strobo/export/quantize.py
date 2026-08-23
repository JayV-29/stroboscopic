"""Plain-numpy int8 fixed-point export of every Linear/Conv layer.

This is deliberately framework-free so the arithmetic budget can be verified
without TFLite Micro: weights are symmetric per-tensor int8, activations are
quantised to int8 with a per-layer scale calibrated on a few batches, and the
accumulate is int32.  ``int8_forward_check`` runs the decoder / head / encoder
matmuls in int8 and compares against float to report the quantisation error.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Int8Linear:
    def __init__(self, weight: np.ndarray, bias: np.ndarray | None, act_scale: float):
        self.w_scale = float(np.abs(weight).max() / 127.0 + 1e-12)
        self.w_q = np.clip(np.round(weight / self.w_scale), -127, 127).astype(np.int8)
        self.act_scale = float(act_scale)
        self.bias_q = None if bias is None else np.round(bias / (self.w_scale * self.act_scale)).astype(np.int32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_q = np.clip(np.round(x / self.act_scale), -127, 127).astype(np.int32)
        acc = x_q @ self.w_q.T.astype(np.int32)
        if self.bias_q is not None:
            acc = acc + self.bias_q
        return acc.astype(np.float32) * (self.w_scale * self.act_scale)

    @property
    def n_macs(self):
        return int(self.w_q.size)

    @property
    def n_bytes(self):
        return int(self.w_q.size + (0 if self.bias_q is None else 4 * self.bias_q.size))


def quantize_model_int8(model: nn.Module, calib_inputs: dict[str, np.ndarray] | None = None) -> dict:
    """Return {layer_name: Int8Linear} for every nn.Linear / nn.Conv1d in the model,
    plus a size/MAC summary.  ``calib_inputs`` maps layer name -> sample activation
    array used to set the activation scale (default: scale 1/32 for the GELU range)."""
    layers, total_macs, total_bytes = {}, 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            w = m.weight.detach().cpu().numpy()
            b = None if m.bias is None else m.bias.detach().cpu().numpy()
        elif isinstance(m, nn.Conv1d):
            w = m.weight.detach().cpu().numpy().reshape(m.out_channels, -1)
            b = None if m.bias is None else m.bias.detach().cpu().numpy()
        else:
            continue
        a_scale = 1 / 32.0
        if calib_inputs and name in calib_inputs:
            a_scale = float(np.abs(calib_inputs[name]).max() / 127.0 + 1e-12)
        q = Int8Linear(w, b, a_scale)
        layers[name] = q
        total_macs += q.n_macs
        total_bytes += q.n_bytes
    return {"layers": layers, "total_macs": total_macs, "weight_bytes": total_bytes}


def int8_forward_check(model: nn.Module, x: torch.Tensor, layer_name: str) -> dict:
    """Compare an int8 evaluation of one Linear against float on activation x."""
    lin = dict(model.named_modules())[layer_name]
    xf = x.detach().cpu().numpy().reshape(-1, x.shape[-1])
    q = Int8Linear(lin.weight.detach().cpu().numpy(), lin.bias.detach().cpu().numpy(),
                   np.abs(xf).max() / 127.0 + 1e-12)
    y_q = q(xf)
    with torch.no_grad():
        y_f = lin(x.detach()).cpu().numpy().reshape(-1, lin.out_features)
    err = np.abs(y_q - y_f)
    return {"layer": layer_name, "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
            "rel_err": float(err.mean() / (np.abs(y_f).mean() + 1e-9))}
