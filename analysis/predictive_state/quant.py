"""Fake-quantization (QAT with straight-through estimators) used by every low-bit arm of EXP-0012.

Weight modes: fp (no quantization), int8 (per-output-channel absmax), int4 (per-output-channel, clip at 3x mean|w|),
tern (per-tensor absmean, BitNet b1.58 style, levels {-1,0,+1}). Quantized layers also fake-quantize their input
activations to INT8 per token (absmax). State quantization is a separate switch (fixed range [-1,1]).
All of this is simulation of numerics: no packed kernel exists here, so it measures quality, not speed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste(x, q):
    return x + (q - x).detach()


def fq_weight(w, mode):
    if mode == "fp":
        return w
    if mode == "int8":
        s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        return _ste(w, torch.round(w / s).clamp(-128, 127) * s)
    if mode == "int4":
        s = (3.0 * w.abs().mean(dim=1, keepdim=True)).clamp_min(1e-8) / 7.0
        return _ste(w, torch.round(w / s).clamp(-8, 7) * s)
    if mode == "tern":
        s = w.abs().mean().clamp_min(1e-8)
        return _ste(w, torch.round(w / s).clamp(-1, 1) * s)
    raise ValueError(mode)


def fq_act8(x):
    s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    return _ste(x, torch.round(x / s).clamp(-127, 127) * s)


def fq_state(h, bits):
    if bits is None:
        return h
    n = 2 ** (bits - 1) - 1
    return _ste(h, torch.round(h.clamp(-1, 1) * n) / n)


class QLinear(nn.Linear):
    def __init__(self, i, o, mode="fp", bias=True):
        super().__init__(i, o, bias=bias)
        self.mode = mode

    def forward(self, x):
        if self.mode == "fp":
            return F.linear(x, self.weight, self.bias)
        return F.linear(fq_act8(x), fq_weight(self.weight, self.mode), self.bias)


class QConv2d(nn.Conv2d):
    def __init__(self, i, o, k, stride=1, padding=0, mode="fp"):
        super().__init__(i, o, k, stride=stride, padding=padding)
        self.mode = mode

    def forward(self, x):
        if self.mode == "fp":
            return super().forward(x)
        s = x.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8) / 127.0
        xq = _ste(x, torch.round(x / s).clamp(-127, 127) * s)
        return F.conv2d(xq, fq_weight(self.weight.flatten(1), self.mode).view_as(self.weight), self.bias,
                        self.stride, self.padding)


def ternary_stats(w):
    s = w.abs().mean()
    q = torch.round(w / s).clamp(-1, 1)
    return {"zero_frac": float((q == 0).float().mean()), "pos_frac": float((q > 0).float().mean()),
            "neg_frac": float((q < 0).float().mean())}
