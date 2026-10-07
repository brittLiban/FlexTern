"""
EXP-0007 workloads: canonical Falcon-E 1B/3B (MEASURED via roofline_models.Workload) and hypothetical
7B / 70B / 400B ternary models with Llama-class shapes (D11, HYPOTHESIZED; no quality claim).

Per-token quantities (one sequence, one layer unless stated), decimal bytes:
  linear params per layer = H*nH*hd (q) + 2*H*nKV*hd (k,v) + nH*hd*H (o) + 3*H*I (gate, up, down)
  kv values per token per layer = 2 * nKV * hd            (K and V)
  kv bytes per sequence at context C = L * kv_values * C * kv_bits/8
  activation bytes crossing a stage boundary = H * ACT_BYTES per token per sequence
  head params = V * H (tied or not, counted once for bytes read at decode)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from model_specs import SPECS  # noqa: E402
from roofline_models import Workload  # noqa: E402
from sources import ACT_BYTES  # noqa: E402


@dataclass(frozen=True)
class Model:
    name: str
    L: int
    H: int
    I: int
    nH: int
    nKV: int
    hd: int
    V: int
    tied: bool
    kind: str                       # "canonical (MEASURED)" | "hypothetical (A8)"
    linear_params: int              # ternary-eligible parameters (all linear projections)
    head_params: int
    max_ctx: int

    @property
    def layer_params(self) -> float:
        return self.linear_params / self.L

    @property
    def kv_values_per_token_layer(self) -> int:
        return 2 * self.nKV * self.hd

    def weight_bytes(self, bits: float) -> float:
        return self.linear_params * bits / 8

    def layer_bytes(self, bits: float) -> float:
        return self.layer_params * bits / 8

    def head_bytes(self, bits: float) -> float:
        return self.head_params * bits / 8

    def kv_bytes_per_seq(self, ctx: int, kv_bits: float) -> float:
        return self.L * self.kv_values_per_token_layer * ctx * kv_bits / 8

    def kv_bytes_per_token_layer(self, kv_bits: float) -> float:
        return self.kv_values_per_token_layer * kv_bits / 8

    def act_bytes_per_boundary(self) -> int:
        return self.H * ACT_BYTES

    def attention_flops_per_token(self, ctx: int) -> float:
        """QK^T and AV at context ctx: 2 * (2 * nH * hd * ctx) per layer (multiply-add = 2 flops)."""
        return self.L * 4 * self.nH * self.hd * ctx


def _llama_linear_params(L, H, I, nH, nKV, hd) -> int:
    return L * (H * nH * hd + 2 * H * nKV * hd + nH * hd * H + 3 * H * I)


def canonical(slug: str) -> Model:
    w = Workload.from_profile_dir(SPECS[slug].default_out)
    return Model(name=slug, L=w.L, H=w.H, I=w.I, nH=w.nH, nKV=w.nKV, hd=w.hd, V=w.V, tied=w.tied, kind="canonical (MEASURED)",
                 linear_params=w.W, head_params=w.head_params, max_ctx=w.max_ctx)


HYPOTHETICAL = {                                                   # D11 shapes, A8
    "hyp-7B":   dict(L=32,  H=4096,  I=11008, nH=32,  nKV=32, hd=128, V=32000,  max_ctx=131072),
    "hyp-70B":  dict(L=80,  H=8192,  I=28672, nH=64,  nKV=8,  hd=128, V=128256, max_ctx=131072),
    "hyp-400B": dict(L=126, H=16384, I=53248, nH=128, nKV=8,  hd=128, V=128256, max_ctx=131072),
}


def hypothetical(name: str, vocab: int | None = None) -> Model:
    s = HYPOTHETICAL[name]
    V = vocab or s["V"]
    return Model(name=name if vocab is None else f"{name}-v{V//1000}k", L=s["L"], H=s["H"], I=s["I"], nH=s["nH"], nKV=s["nKV"], hd=s["hd"], V=V, tied=False,
                 kind="hypothetical (A8)", linear_params=_llama_linear_params(s["L"], s["H"], s["I"], s["nH"], s["nKV"], s["hd"]),
                 head_params=V * s["H"], max_ctx=s["max_ctx"])


def all_models() -> list[Model]:
    return [canonical("falcon-e-1b"), canonical("falcon-e-3b"), hypothetical("hyp-7B"), hypothetical("hyp-70B"), hypothetical("hyp-400B")]
