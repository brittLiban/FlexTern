"""
EXP-0005 workload mapping: real projection shapes from the canonical profiles mapped onto
the streaming engine (BYTES x ROWQ) and the banked near-memory engine (NB banks x BYTES).

Inputs: results/workload_profile/<model>/weight_distribution_by_module.csv and
model_summary.json via roofline_models.Workload (EXP-0003 loader; nothing re-typed).

Per tensor (module instance, one per layer):
    row-quads        = N / 4                      (packed rows; one byte = 4 output rows)
    packed bytes     = N * K / 4
    terms            = N * K
Streaming engine (P row streams x B bytes):
    cycles           = ceil(rowq / P) * K / B      (K % B == 0 for every canonical K and B <= 256)
    utilization      = terms / (cycles * 4 * P * B)
Near-memory (NB banks, b bytes per bank per cycle, banks own contiguous row-quad blocks):
    rows per bank    = ceil(rowq / NB)
    cycles           = max(rows_per_bank * K / b, rowq)   (banks run in parallel; the collector
                       emits one row-quad group per cycle, so a tensor can never finish in fewer
                       cycles than it has row-quads: COLLECTOR-BOUND when NB * b > K)
    utilization      = rowq / (rows_per_bank * NB)
    capacity / bank  = rows_per_bank * K bytes     (tail banks hold less; the max is what must exist)
All cycle counts are ANALYTICAL from the measured shapes; the RTL testbenches confirm the
per-tensor cycle law on the simulated cases (cycles = chunks + 2 for streaming).
"""

from __future__ import annotations

import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from model_specs import SPECS  # noqa: E402
from roofline_models import Workload  # noqa: E402

MODELS = ("falcon-e-1b", "falcon-e-3b", "bitnet-2b4t")
STREAM_CONFIGS = [(64, 1), (128, 1), (256, 1), (16, 4), (1, 64), (16, 16), (1, 256)]       # (BYTES, ROWQ)
NEARMEM_CONFIGS = [(16, 16), (64, 16), (256, 16), (1024, 16), (64, 64), (256, 64), (4096, 64)]  # (NB, BYTES)
BW_TARGETS_GBPS = (100, 250, 500)


def load_workloads() -> dict[str, Workload]:
    return {m: Workload.from_profile_dir(SPECS[m].default_out) for m in MODELS}


def tensors(w: Workload) -> list[dict]:
    """One entry per module type with the per-layer shape; `count` = layers."""
    out = []
    for m in w.modules:
        k, n = m["K"], m["N"]
        per_tensor = n * k
        count = m["weights"] // per_tensor
        assert count * per_tensor == m["weights"], (w.slug, m)
        out.append({"model": w.slug, "module": m["module"], "K": k, "N": n, "layers": count,
                    "rowq": n // 4, "packed_bytes": n * k // 4, "terms": per_tensor})
    return out


def stream_cycles(t: dict, bytes_: int, rowq_par: int) -> tuple[int, float]:
    assert t["K"] % bytes_ == 0
    cyc = math.ceil(t["rowq"] / rowq_par) * (t["K"] // bytes_)
    return cyc, t["terms"] / (cyc * 4 * rowq_par * bytes_)


def nearmem_cycles(t: dict, nb: int, bytes_: int) -> tuple[int, float, int]:
    rpb = math.ceil(t["rowq"] / nb)
    cyc = max(rpb * (t["K"] // bytes_), t["rowq"])          # collector: one result group per cycle
    return cyc, t["terms"] / (cyc * 4 * nb * bytes_), rpb * t["K"]


def mapping_rows(ws: dict[str, Workload]) -> tuple[list[dict], list[dict]]:
    rows, totals = [], []
    for slug, w in ws.items():
        ts = tensors(w)
        for b, p in STREAM_CONFIGS:
            tot_c = tot_terms = 0
            for t in ts:
                c, u = stream_cycles(t, b, p)
                rows.append({**t, "engine": "streaming", "config": f"B={b},P={p}", "terms_per_cycle": 4 * b * p,
                             "bytes_per_cycle": b * p, "banks": 1, "cycles_per_tensor": c, "utilization": round(u, 4),
                             "cycles_per_token": c * t["layers"], "capacity_bytes_per_bank": ""})
                tot_c += c * t["layers"]; tot_terms += t["terms"] * t["layers"]
            totals.append({"model": slug, "engine": "streaming", "config": f"B={b},P={p}", "terms_per_cycle": 4 * b * p,
                           "bytes_per_cycle": b * p, "banks": 1, "cycles_per_token": tot_c,
                           "utilization_token_weighted": round(tot_terms / (tot_c * 4 * b * p), 4),
                           "packed_bytes_per_token": w.W // 4, "capacity_bytes_per_bank_max": ""})
        for nb, b in NEARMEM_CONFIGS:
            tot_c = tot_terms = 0; cap = 0; cap_max = 0
            for t in ts:
                c, u, cb = nearmem_cycles(t, nb, b)
                rows.append({**t, "engine": "near-memory", "config": f"NB={nb},b={b}", "terms_per_cycle": 4 * b * nb,
                             "bytes_per_cycle": b * nb, "banks": nb, "cycles_per_tensor": c, "utilization": round(u, 4),
                             "cycles_per_token": c * t["layers"], "capacity_bytes_per_bank": cb})
                tot_c += c * t["layers"]; tot_terms += t["terms"] * t["layers"]; cap += cb * t["layers"]
            totals.append({"model": slug, "engine": "near-memory", "config": f"NB={nb},b={b}", "terms_per_cycle": 4 * b * nb,
                           "bytes_per_cycle": b * nb, "banks": nb, "cycles_per_token": tot_c,
                           "utilization_token_weighted": round(tot_terms / (tot_c * 4 * b * nb), 4),
                           "packed_bytes_per_token": w.W // 4, "capacity_bytes_per_bank_max": cap})
    return rows, totals


def bandwidth_rows() -> list[dict]:
    """required_clock = target_weight_bandwidth / packed_bytes_per_cycle.  TARGET/ANALYTICAL, never achieved."""
    out = []
    configs = [("streaming", f"B={b},P={p}", b * p) for b, p in STREAM_CONFIGS] + \
              [("near-memory aggregate", f"NB={nb},b={b}", nb * b) for nb, b in NEARMEM_CONFIGS]
    for eng, cfg, bpc in configs:
        r = {"engine": eng, "config": cfg, "packed_bytes_per_cycle": bpc, "terms_per_cycle": 4 * bpc}
        for g in BW_TARGETS_GBPS:
            r[f"required_clock_MHz_for_{g}GBps"] = round(g * 1e9 / bpc / 1e6, 1)
        r["status"] = "UNMEASURED: analytical requirement, no physical target exists" + (
            "; near-memory bytes/cycle presume every bank consuming each cycle, which the one-group-per-cycle collector prevents on tensors with K < NB*b (collector-bound)" if eng.startswith("near") else "")
        out.append(r)
    return out


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ws = load_workloads()
    rows, totals = mapping_rows(ws)
    write_csv(os.path.join("results", "hardware_gate", "workload_mapping.csv"), rows)
    write_csv(os.path.join("results", "hardware_gate", "workload_totals.csv"), totals)
    write_csv(os.path.join("results", "hardware_gate", "bandwidth_requirements.csv"), bandwidth_rows())
    print(f"{len(rows)} mapping rows, {len(totals)} totals")
