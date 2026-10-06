"""
Weight-only groupwise INT4 quantization of the Falcon-E output head (EXP-0004, Config A/C).

Method (round-to-nearest, asymmetric, groupwise; the standard "RTN gN" baseline used in
GPTQ/AWQ-style comparisons):
    for each output row n and each group of G consecutive input columns:
        scale = (max - min) / 15            (fp16)
        zero  = round(-min / scale) in [0, 15]   (uint8 storage; 4 bits logically)
        q     = clamp(round(w / scale) + zero, 0, 15)   (uint4 code)
    dequant: w_hat = (q - zero) * scale

Storage: codes packed two per byte along the input dimension (low nibble = even column).
Execution: the module dequantizes its codes to BF16 at every forward and calls F.linear.
So the LOGICAL compressed size is ~4.25 bits/element (g=128, fp16 scale, 4-bit zero)
while the RUNTIME creates a full-size BF16 temporary of the head on every call.
Both are reported; neither is a hardware result.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_rtn(weight: torch.Tensor, group_size: int = 128) -> dict:
    """weight: [N, K] float tensor. Returns packed codes, scales, zeros and shape metadata."""
    assert weight.dim() == 2
    n, k = weight.shape
    assert k % group_size == 0, (k, group_size)
    assert k % 2 == 0
    w = weight.float().reshape(n, k // group_size, group_size)
    wmin = w.amin(dim=-1, keepdim=True)
    wmax = w.amax(dim=-1, keepdim=True)
    scale = (wmax - wmin) / 15.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = torch.clamp(torch.round(-wmin / scale), 0, 15)
    q = torch.clamp(torch.round(w / scale) + zero, 0, 15).to(torch.uint8).reshape(n, k)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()                 # [N, K/2] uint8
    return {"packed": packed, "scale": scale.reshape(n, k // group_size).to(torch.float16),
            "zero": zero.reshape(n, k // group_size).to(torch.uint8), "shape": (n, k), "group_size": group_size, "bits": 4}


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0x0F
    hi = packed >> 4
    out = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.uint8, device=packed.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


def dequantize(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, group_size: int, dtype=torch.bfloat16) -> torch.Tensor:
    q = unpack_codes(packed)                                       # [N, K] uint8
    n, k = q.shape
    g = k // group_size
    w = (q.reshape(n, g, group_size).to(torch.float32) - zero.reshape(n, g, 1).to(torch.float32)) * scale.reshape(n, g, 1).to(torch.float32)
    return w.reshape(n, k).to(dtype)


def logical_bytes(n: int, k: int, group_size: int, zero_bits: int = 4, scale_bits: int = 16) -> dict:
    groups = n * (k // group_size)
    data = n * k // 2
    meta = groups * (scale_bits + zero_bits) // 8
    return {"data_bytes": data, "metadata_bytes": meta, "total_bytes": data + meta,
            "effective_bits_per_element": 8 * (data + meta) / (n * k), "groups": groups}


class QuantizedLinearRTN(nn.Module):
    """Drop-in replacement for a bias-free nn.Linear that stores INT4 codes and dequantizes per forward."""

    def __init__(self, qd: dict, out_dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer("packed", qd["packed"])
        self.register_buffer("scale", qd["scale"])
        self.register_buffer("zero", qd["zero"])
        self.out_features, self.in_features = qd["shape"]
        self.group_size = qd["group_size"]
        self.out_dtype = out_dtype
        self.bias = None
        self.dequant_calls = 0

    @property
    def weight(self):  # for code that introspects .weight (e.g. tied-weight checks); dequantized view
        return self.dequantized()

    def dequantized(self) -> torch.Tensor:
        return dequantize(self.packed, self.scale, self.zero, self.group_size, self.out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.dequant_calls += 1
        w = self.dequantized()           # full-size BF16 temporary (134 MB for Falcon-E 1B) every call
        return F.linear(x.to(self.out_dtype), w)

    def resident_bytes(self) -> dict:
        return {"packed_bytes": self.packed.numel() * self.packed.element_size(),
                "scale_bytes": self.scale.numel() * self.scale.element_size(),
                "zero_bytes_stored": self.zero.numel() * self.zero.element_size(),
                "total_resident_bytes": sum(t.numel() * t.element_size() for t in (self.packed, self.scale, self.zero)),
                "dequant_temporary_bytes_per_forward": self.out_features * self.in_features * 2}


def quantize_lm_head(model, group_size: int = 128):
    """Replace model.lm_head with an INT4 RTN module. Returns (original_head_on_cpu, quantized_module, stats)."""
    head = model.lm_head
    assert isinstance(head, nn.Linear) and head.bias is None, type(head)
    w = head.weight.detach()
    qd = quantize_rtn(w, group_size)
    qm = QuantizedLinearRTN(qd, out_dtype=w.dtype).to(w.device)
    w_hat = qm.dequantized().float()
    err = (w_hat - w.float())
    stats = {"group_size": group_size, "rows": qd["shape"][0], "cols": qd["shape"][1],
             "max_abs_error": float(err.abs().max()), "mean_abs_error": float(err.abs().mean()),
             "rel_fro_error": float(err.norm() / w.float().norm()),
             "max_error_over_half_scale": float((err.abs().reshape(qd["shape"][0], -1, group_size).amax(-1) / (qd["scale"].float() / 2 + 1e-12)).max()),
             "original_bytes": w.numel() * w.element_size(), **logical_bytes(*qd["shape"], group_size), **qm.resident_bytes()}
    original_cpu = nn.Linear(head.in_features, head.out_features, bias=False)
    original_cpu.weight = nn.Parameter(w.detach().to("cpu").clone(), requires_grad=False)
    model.lm_head = qm
    return original_cpu, qm, stats


def restore_lm_head(model, original_cpu: nn.Linear, device):
    model.lm_head = nn.Linear(original_cpu.in_features, original_cpu.out_features, bias=False)
    model.lm_head.weight = nn.Parameter(original_cpu.weight.detach().to(device), requires_grad=False)
    return model.lm_head
