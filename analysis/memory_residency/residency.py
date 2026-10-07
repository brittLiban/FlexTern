"""
EXP-0006 residency policies, bank organization and bandwidth requirements.

Traffic levels and the collector bound are IMPORTED from EXP-0005 (analysis/hardware_gate), not
re-derived: traffic_model.per_token gives the four levels for full residency; workload_mapping gives
tensors and the collector-bound cycle law.  Decimal units.

Residency policies per token (weights W bytes, L layers):
  full       resident capacity W; global weight bytes/token = W / tokens_resident (preload amortized;
             0 in steady state); local reads W
  layer      resident capacity W/L (one layer); every token reloads every layer: global W/token,
             local W/token.  Reduces to streaming + an extra local read.
  tile(t)    resident t layers; reload per token W (same as layer); capacity t*W/L
  hot(f)     fraction f of weights resident forever, the rest streamed each token:
             global (1-f)W/token, local fW/token, capacity fW
Bandwidth: required = bytes/token x tokens/s; divided by utilization for the interface requirement.
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "hardware_gate"))
sys.path.insert(0, _HERE)
from traffic_model import per_token  # noqa: E402  (EXP-0005)
from workload_mapping import nearmem_cycles, tensors  # noqa: E402  (EXP-0005)
from sources import (HBM3E_STACK_PEAK_GBPS, LPDDR5X_GBPS_PER_64BIT, NOMINAL_UTIL, PRELOAD_AMORTIZATION_TOKENS, TARGET_TOKS,  # noqa: E402
                     UPMEM_GBPS_PER_DPU, UPMEM_MB_PER_DPU, UTILIZATIONS, VCACHE_DIE_BW_TBPS, VCACHE_DIE_MB)

BANK_COUNTS = (16, 32, 64, 128, 256)
BANK_BYTES_PER_CYCLE = 16     # EXP-0005 bank point b=16 (also 64 in the sweep)


def policy_rows(w, bits: int = 2, tokens_resident: int = PRELOAD_AMORTIZATION_TOKENS, hot_fractions=(0.25, 0.5, 0.75), tiles=(1, 2, 4, 8)) -> list[dict]:
    base = per_token(w, nb=64, weight_bits=bits)                  # EXP-0005 levels at 64 banks
    W = base["packed_weight_bytes"]; act = base["nm_D_activation_injected"]; out = base["nm_D_outputs"]
    rows = []
    def row(policy, cap, glob_w, local_w, note):
        rows.append({"model": w.slug, "weight_bits": bits, "policy": policy, "resident_capacity_bytes": int(cap),
                     "preload_bytes_once": W, "global_weight_bytes_per_token": int(glob_w), "local_weight_bytes_per_token": int(local_w),
                     "activation_injected_bytes_per_token": act, "output_bytes_per_token": out,
                     "global_total_bytes_per_token": int(glob_w + act + out),
                     "global_reduction_vs_streaming": round((W + act + out) / (glob_w + act + out), 2),
                     "exp0005_advantage_surviving": note})
    row("streaming (reference)", 0, W, 0, "none: baseline")
    row("full residency (steady state, preload excluded; EXP-0005 convention)", W, 0, W, "all: global weight stream removed; local reads remain")
    row(f"full residency (preload amortized over {tokens_resident} tokens, A10)", W, W / tokens_resident, W, "all: global weight stream removed; local reads remain")
    row("layer-at-a-time residency (reload every layer every token)", W / w.L, W, W, "none: reload traffic equals streaming, plus the local read")
    for t in tiles:
        if t <= w.L:
            row(f"tiled residency ({t} layers resident at a time)", t * W / w.L, W, W, "none per token (reload bytes unchanged); only the reload granularity changes")
    for f in hot_fractions:
        row(f"hot-set residency ({int(f*100)}% of weights resident forever)", f * W, (1 - f) * W, f * W, f"proportional: {int(f*100)}% of the global weight stream removed")
    return rows


def bank_rows(w, bits: int = 2, b: int = BANK_BYTES_PER_CYCLE) -> list[dict]:
    ts = tensors(w)
    W = w.W * bits // 8
    rows = []
    for nb in BANK_COUNTS:
        cyc = sum(nearmem_cycles(t, nb, b)[0] * t["layers"] for t in ts)             # collector-bound (EXP-0005)
        terms = sum(t["terms"] * t["layers"] for t in ts)
        util = terms / (cyc * 4 * nb * b)
        cap_bank = sum(math.ceil(t["rowq"] / nb) * t["K"] * t["layers"] for t in ts) * (bits / 2)   # tail waste included
        tr = per_token(w, nb=nb, weight_bits=bits)
        rows.append({"model": w.slug, "weight_bits": bits, "banks": nb, "bytes_per_bank_per_cycle": b,
                     "capacity_per_bank_bytes": int(cap_bank), "capacity_per_bank_MB": round(cap_bank / 1e6, 2),
                     "sum_bank_capacity_MB": round(cap_bank * nb / 1e6, 1), "whole_model_MB": round(W / 1e6, 1),
                     "tail_waste_pct": round((cap_bank * nb / W - 1) * 100, 2),
                     "local_bytes_per_token": W, "local_bytes_per_bank_per_token": int(W / nb),
                     "activation_buffer_bits_per_bank": 8 * max(t["K"] for t in ts),
                     "activation_delivered_bytes_per_token": tr["nm_D_activation_delivered"],
                     "activation_delivered_over_weight_bytes": round(tr["nm_D_activation_delivered"] / W, 4),
                     "output_bytes_per_token": tr["nm_D_outputs"], "result_groups_per_token": sum(t["rowq"] * t["layers"] for t in ts),
                     "cycles_per_token_collector_bound": cyc, "utilization": round(util, 4),
                     "collector_bound": any(nb * b > t["K"] for t in ts),
                     "label": "ANALYTICAL from measured shapes; collector law MEASURED in EXP-0005 RTL"})
    return rows


def bandwidth_rows(w, bits: int = 2) -> list[dict]:
    """Local (resident) and external (streaming) bandwidth at target token rates, with utilization."""
    W = w.W * bits // 8
    tr = per_token(w, nb=64, weight_bits=bits)
    act = tr["nm_D_activation_injected"]; out = tr["nm_D_outputs"]
    rows = []
    for tok in TARGET_TOKS:
        for u in UTILIZATIONS:
            local = W * tok; ext = (W + act + out) * tok
            rows.append({"model": w.slug, "weight_bits": bits, "target_tok_s": tok, "utilization": u,
                         "streaming_external_GBps_required": round(ext / u / 1e9, 2),
                         "resident_local_aggregate_GBps_required": round(local / u / 1e9, 2),
                         "resident_global_GBps_required_act_out": round((act + out) * tok / u / 1e9, 3),
                         **{f"local_per_bank_GBps_NB{nb}": round(local / nb / u / 1e9, 3) for nb in BANK_COUNTS},
                         **{f"activation_delivery_GBps_NB{nb}": round(per_token(w, nb=nb, weight_bits=bits)["nm_D_activation_delivered"] * tok / u / 1e9, 3) for nb in (16, 64, 256)},
                         "lpddr5x_64bit_channels_needed_streaming": math.ceil(ext / u / (LPDDR5X_GBPS_PER_64BIT * 1e9)),
                         "hbm3e_stacks_needed_streaming": math.ceil(ext / u / (HBM3E_STACK_PEAK_GBPS * 1e9)),
                         "vcache_dies_for_capacity": math.ceil(W / (VCACHE_DIE_MB * 1e6)),
                         "vcache_die_bw_TBps_each_peak": VCACHE_DIE_BW_TBPS,
                         "upmem_class_dpus_for_capacity": math.ceil(W / (UPMEM_MB_PER_DPU * 1e6)),
                         "upmem_class_per_dpu_GBps_available": UPMEM_GBPS_PER_DPU,
                         "upmem_class_per_dpu_GBps_required": round(local / math.ceil(W / (UPMEM_MB_PER_DPU * 1e6)) / u / 1e9, 3),
                         "label": "TARGET rates; peak interface figures from sources; nothing achieved"})
    return rows


def streaming_limits(w, bits: int = 2) -> dict:
    """Token rate at which each external interface class saturates (at nominal utilization)."""
    W = w.W * bits // 8
    tr = per_token(w, nb=64, weight_bits=bits)
    per_tok = W + tr["nm_D_activation_injected"] + tr["nm_D_outputs"]
    return {"model": w.slug, "weight_bits": bits, "bytes_per_token_streaming": per_tok,
            "lpddr5x_64bit_max_tok_s": round(LPDDR5X_GBPS_PER_64BIT * 1e9 * NOMINAL_UTIL / per_tok, 1),
            "lpddr5x_128bit_max_tok_s": round(2 * LPDDR5X_GBPS_PER_64BIT * 1e9 * NOMINAL_UTIL / per_tok, 1),
            "hbm3e_one_stack_max_tok_s": round(HBM3E_STACK_PEAK_GBPS * 1e9 * NOMINAL_UTIL / per_tok, 1),
            "utilization": NOMINAL_UTIL, "label": "ANALYTICAL; batch 1; peak x utilization"}
