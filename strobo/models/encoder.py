"""Causal 1-D conv encoder for the always-on cheap stream.

3 layers x 16 channels, dilations 1/4/16, kernel 5 -> receptive field 85 ticks
(2.7 s at 32 Hz).  Output is a per-tick 32-dim feature.  ~3 k parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k=5, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

    def forward(self, x):                      # x: (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, feat_dim: int = 32, ch: int = 16, kernel: int = 5,
                 dilations=(1, 4, 16)):
        super().__init__()
        layers, c = [], in_ch
        for d in dilations:
            layers.append(CausalConv1d(c, ch, kernel, d))
            c = ch
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(ch, feat_dim)
        self.feat_dim = feat_dim
        self.receptive_field = 1 + (kernel - 1) * sum(dilations)

    def forward(self, x):                      # x: (B, T, C) -> (B, T, feat)
        h = x.transpose(1, 2)
        for l in self.layers:
            h = F.gelu(l(h))
        return self.out(h.transpose(1, 2))

    def macs_per_tick(self) -> int:
        n = 0
        for l in self.layers:
            w = l.conv.weight
            n += w.numel()
        n += self.out.weight.numel()
        return n
