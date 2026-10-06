"""
Reconnect EXP-0004 measured representations to the EXP-0003 analytical scenarios.

Reads results/model_precision/{head_int4.json, memory.json} for the MEASURED effective
bytes per element of the INT4 head and the quantized KV cache (including scale/zero metadata
and the full-precision residual window), re-evaluates the EXP-0003 token model for Falcon-E 1B
with those values, and writes results/model_precision/roofline_reconnect.{csv,md}.

Two columns are reported:
  logical     - bytes if a runtime read the compressed representation directly (HYPOTHESIZED hardware)
  attainable  - external bytes on THIS runtime, where the head is dequantized to BF16 per forward on-chip
                (so its external read is the compressed form) but the quantized KV cache is dequantized
                into a BF16 temporary every step (resident memory shrinks; the attention read does not).

results/roofline/ (EXP-0003) is never modified.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from common import RESULTS_DIR, read_json  # noqa: E402
from model_specs import SPECS  # noqa: E402
from roofline_models import Assumptions, Workload, token_model  # noqa: E402


def main() -> int:
    w = Workload.from_profile_dir(SPECS["falcon-e-1b"].default_out)
    head = read_json("head_int4.json")
    mem = read_json("memory.json")
    head_bits = head["effective_bits_per_element"]            # logical incl. fp16 scale + 4-bit zero
    head_resident_bits = 8 * head["total_resident_bytes"] / (head["rows"] * head["cols"])  # as stored here (zero in uint8)
    # measured KV effective bits at the largest measured context for config B
    sweep = [r for r in mem["B"]["sweep"] if "effective_bits_per_kv_element" in r]
    kv_rows = {r["ctx"]: r["effective_bits_per_kv_element"] for r in sweep}
    kv_bits_max = kv_rows[max(kv_rows)]
    base = Assumptions(bw=100e9, batch=1)
    rows = []
    for ctx in (512, 2048, 4096, 8192, 16384, 32768):
        kv_bits = kv_rows.get(ctx, kv_bits_max)
        for name, a in (("EXP-0003 reference (BF16 head, BF16 KV)", replace(base, context=ctx)),
                        ("A: measured INT4 head bytes, BF16 KV", replace(base, context=ctx, head_bytes_per_elem=head_bits / 8)),
                        ("B: BF16 head, measured int4 KV bytes (logical)", replace(base, context=ctx, kv_bytes_per_elem=kv_bits / 8)),
                        ("C: measured INT4 head + int4 KV (logical)", replace(base, context=ctx, head_bytes_per_elem=head_bits / 8, kv_bytes_per_elem=kv_bits / 8)),
                        ("C attainable on this runtime (KV dequantized to BF16 before attention)", replace(base, context=ctx, head_bytes_per_elem=head_bits / 8))):
            tm = token_model(w, a)
            c = tm.ceilings()
            rows.append({"context": ctx, "scenario": name, "head_bits_per_elem": a.head_bytes_per_elem * 8, "kv_bits_per_elem": a.kv_bytes_per_elem * 8,
                         "ternary_bytes": tm.stage("ternary").memory_bytes, "head_bytes": tm.stage("head").memory_bytes,
                         "kv_bytes": tm.stage("attention_kv").memory_bytes, "total_bytes": tm.total_bytes,
                         "ternary_bytes_fraction": tm.ternary_bytes_fraction, "limiting_component": tm.limiting_component,
                         "ideal_ternary_only_ceiling_mem_bound": c["memory_bound_bytes_ratio"]})
    p = os.path.join(RESULTS_DIR, "roofline_reconnect.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wri.writeheader(); wri.writerows(rows)
    L = ["# EXP-0003 scenarios re-evaluated with EXP-0004 measured representations (Falcon-E 1B, batch 1, 100 GB/s; ANALYTICAL)", "",
         f"Measured INT4 head: {head_bits:.3f} bits/element logical (fp16 scale + 4-bit zero per group of {head['group_size']}); {head_resident_bits:.3f} bits/element as stored here (zero in uint8). "
         f"Measured int4 KV cache (hqq, g=64, axis 0, 128-token BF16 residual): effective bits/element by context: " + ", ".join(f"{k}: {v:.2f}" for k, v in sorted(kv_rows.items())) + ".", "",
         "| ctx | scenario | head bits | KV bits | ternary MB | head MB | KV MB | total MB | ternary frac | limiter | ideal ceiling (mem-bound) |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in rows:
        L.append(f"| {r['context']} | {r['scenario']} | {r['head_bits_per_elem']:.2f} | {r['kv_bits_per_elem']:.2f} | {r['ternary_bytes']/1e6:,.1f} | {r['head_bytes']/1e6:,.1f} | {r['kv_bytes']/1e6:,.1f} | {r['total_bytes']/1e6:,.1f} | {r['ternary_bytes_fraction']:.3f} | {r['limiting_component']} | {r['ideal_ternary_only_ceiling_mem_bound']:.2f}x |")
    L += ["", "Reading: 'logical' rows assume hardware that reads the compressed form directly (HYPOTHESIZED). The 'attainable' row is what this PyTorch runtime actually streams: the compressed head (dequantized on-chip per forward) but a BF16 KV temporary every step. Quality of each representation is reported separately in EXP-0004."]
    with open(os.path.join(RESULTS_DIR, "roofline_reconnect.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
