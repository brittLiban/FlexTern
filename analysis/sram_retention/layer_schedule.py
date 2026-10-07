"""
EXP-0009 layer schedule: per-layer packed-weight capacities of the canonical models, derived programmatically
from the EXP-0002/0002B profiles (`weight_distribution_by_layer.csv` under each model's profile directory), and the
sequential execution order of a decoder-only transformer (layer 0 .. L-1, then the output head, which is external).

Bytes per layer at `bits` per weight = weights_in_layer x bits / 8 (decimal bytes).  The sum over layers must equal
the model's ternary linear parameter count (checked here and by the tests).  Provisioned capacity carries the EXP-0006
10% margin (A11 of EXP-0008) so that P0 reproduces EXP-0008 exactly.
"""

from __future__ import annotations

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "memory_residency"))
sys.path.insert(0, os.path.join(_HERE, "..", "sram_validation"))
from capacity import load as load_workloads  # noqa: E402
from capacity import transformer_bytes  # noqa: E402
import evidence as E8  # noqa: E402  (EXP-0008 constants: margin, LPDDR, dynamic energy)

MARGIN = E8.MARGIN                     # EXP-0008 A11 / EXP-0006 A2: 10% physical margin on provisioned capacity


def layer_weights(w) -> list[int]:
    """Ternary weights per transformer layer, in execution order, from the profile CSV."""
    path = os.path.join(w.source_dir, "weight_distribution_by_layer.csv")
    with open(path, encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["layer"]))
    ws = [int(r["weights"]) for r in rows]
    assert len(ws) == w.L, (w.slug, len(ws), w.L)
    assert sum(ws) == w.W, (w.slug, sum(ws), w.W)
    return ws


def layer_bytes(w, bits: float) -> list[float]:
    return [n * bits / 8.0 for n in layer_weights(w)]


def schedule(w, bits: float) -> dict:
    """Execution-order schedule with per-layer bytes, provisioned bytes per layer (with margin) and totals."""
    lb = layer_bytes(w, bits)
    prov = [b / (1 - MARGIN) for b in lb]
    total = transformer_bytes(w, 2) * (bits / 2.0)
    assert abs(sum(lb) - total) < 1e-6 * total
    return {"model": w.slug, "bits": bits, "L": w.L, "layer_bytes": lb, "layer_provisioned_bytes": prov,
            "weight_bytes": total, "provisioned_bytes": total / (1 - MARGIN),
            "largest_layer_bytes": max(lb), "largest_layer_provisioned_bytes": max(prov),
            "order": list(range(w.L)) + ["head (external)"]}


def schedule_rows(ws) -> list[dict]:
    rows = []
    for slug, w in ws.items():
        for bits in (2.0, 4.5):
            s = schedule(w, bits)
            for i, (b, p) in enumerate(zip(s["layer_bytes"], s["layer_provisioned_bytes"])):
                rows.append({"model": slug, "weight_bits": bits, "layer": i, "execution_order": i, "layer_bytes": int(b),
                             "layer_provisioned_bytes": int(p), "layer_MB": round(b / 1e6, 3), "share_of_model": round(b / s["weight_bytes"], 5),
                             "source": "weight_distribution_by_layer.csv (EXP-0002/0002B MEASURED shapes)"})
    return rows
