"""
EXP-0006 capacity model: static parameter bytes from the canonical profiles at 2/4/8/16 bits per
transformer weight, head kept separate, usable-capacity margins, and hypothetical larger models.

All canonical numbers come from roofline_models.Workload (EXP-0003 loader over the EXP-0002/0002B
profiles); nothing is retyped.  Decimal units (MB = 1e6 B).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "hardware_gate"))
sys.path.insert(0, _HERE)
from model_specs import SPECS  # noqa: E402
from roofline_models import Workload  # noqa: E402
from sources import HEAD_CANDIDATE_BITS, HYPOTHETICAL_MODELS, MARGINS, NOMINAL_MARGIN  # noqa: E402

MODELS = ("falcon-e-1b", "falcon-e-3b", "bitnet-2b4t")
WEIGHT_BITS = (2, 4, 8, 16)
VOCAB_HYP = 32768


def load() -> dict[str, Workload]:
    return {m: Workload.from_profile_dir(SPECS[m].default_out) for m in MODELS}


def transformer_bytes(w: Workload, bits: int) -> int:
    return w.W * bits // 8


def head_bytes(w: Workload, bits: float = 16.0) -> float:
    """Head parameter bytes at `bits` per element (tied heads share the embedding: counted once)."""
    return w.head_params * bits / 8.0


def metadata_bytes(w: Workload) -> int:
    """BF16 norms and per-tensor scales (measured counts)."""
    return 2 * (w.norm_params + w.scale_params)


def capacity_rows(ws: dict[str, Workload]) -> list[dict]:
    rows = []
    for slug, w in ws.items():
        for bits in WEIGHT_BITS:
            tb = transformer_bytes(w, bits)
            hb16 = head_bytes(w, 16); hbc = head_bytes(w, HEAD_CANDIDATE_BITS); md = metadata_bytes(w)
            r = {"model": slug, "weight_bits": bits, "ternary_params": w.W, "transformer_weight_bytes": tb,
                 "head_params": w.head_params, "head_tied": w.tied, "head_bytes_bf16": int(hb16), "head_bytes_candidate_4p156": int(hbc),
                 "metadata_bytes": md, "static_bytes_bf16_head": int(tb + hb16 + md), "static_bytes_candidate_head": int(tb + hbc + md),
                 "transformer_MB": round(tb / 1e6, 1), "head_MB_bf16": round(hb16 / 1e6, 1), "head_MB_candidate": round(hbc / 1e6, 1)}
            for m in MARGINS:
                r[f"transformer_physical_MB_margin_{int(m*100)}pct"] = round(tb / (1 - m) / 1e6, 1)
            r["nominal_physical_transformer_MB"] = round(tb / (1 - NOMINAL_MARGIN) / 1e6, 1)
            r["label"] = "MEASURED counts (EXP-0002/0002B); bytes at other bit widths are ANALYTICAL"
            rows.append(r)
    return rows


def hypothetical_rows() -> list[dict]:
    rows = []
    for h in HYPOTHETICAL_MODELS:
        tern = h["params"] * h["ternary_fraction"]
        head_params = VOCAB_HYP * h["hidden"]
        for bits in (2, 4):
            tb = tern * bits / 8
            rows.append({"model": h["name"], "weight_bits": bits, "params_total": h["params"], "ternary_fraction": h["ternary_fraction"],
                         "ternary_params": int(tern), "transformer_weight_bytes": int(tb), "transformer_MB": round(tb / 1e6, 1),
                         "head_MB_bf16": round(head_params * 2 / 1e6, 1), "head_MB_candidate": round(head_params * HEAD_CANDIDATE_BITS / 8 / 1e6, 1),
                         "nominal_physical_transformer_MB": round(tb / (1 - NOMINAL_MARGIN) / 1e6, 1),
                         "label": "HYPOTHESIZED memory-capacity scaling only; no quality claim; shapes are conventions (A5)"})
    return rows


def all_model_points(ws: dict[str, Workload]) -> list[dict]:
    """Canonical + hypothetical points at 2 and 4 bits, as (name, bits, transformer_bytes, head_bytes_bf16, head_bytes_candidate)."""
    pts = []
    for slug, w in ws.items():
        for bits in (2, 4):
            pts.append({"model": slug, "weight_bits": bits, "transformer_bytes": transformer_bytes(w, bits),
                        "head_bytes_bf16": head_bytes(w, 16), "head_bytes_candidate": head_bytes(w, HEAD_CANDIDATE_BITS),
                        "kind": "canonical (MEASURED counts)", "layers": w.L})
    for r in hypothetical_rows():
        pts.append({"model": r["model"], "weight_bits": r["weight_bits"], "transformer_bytes": r["transformer_weight_bytes"],
                    "head_bytes_bf16": r["head_MB_bf16"] * 1e6, "head_bytes_candidate": r["head_MB_candidate"] * 1e6,
                    "kind": "hypothetical (A5)", "layers": next(h["layers"] for h in HYPOTHETICAL_MODELS if h["name"] == r["model"])})
    return pts
