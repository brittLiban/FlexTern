"""
EXP-0005 data-movement model: streaming vs banked near-memory, four memory levels kept apart.

LEVEL A  model load: bytes to place the weights in the target memory once (amortized over
         the tokens served while they stay resident; per-layer residency reloads every token).
LEVEL B  global fabric: bytes moved between major memory and a central compute region per token.
LEVEL C  local bank reads: bytes read inside a bank by its own compute per token.  Near-memory
         does not remove these.
LEVEL D  activations delivered to compute and results returned per token.

Streaming (central engine, weights streamed from main memory):
    A = packed once;  B = packed per token (weights) + activation injection + outputs;
    C = 0 (there is no bank-local level; the memory reads ARE the global reads);
    D = activation injected once per distinct input vector (buffer K x 8 bits) + outputs.
Near-memory (NB banks own complete rows; whole model resident):
    A = packed once (per-layer residency: packed per token, i.e. equals streaming B);
    B = 0 weight bytes; only activation injection + outputs cross the global path;
    C = packed per token (every weight byte is still read, locally);
    D = activation delivered NB x (bank-local buffers, each K x 8 bits) or, without buffers,
        re-broadcast per row-quad pass (= packed bytes, lockstep); outputs; partial sums = 0.
Activation broadcast link cost bounds (bytes x links, abstract, no wire energy):
    A ideal multicast  = injected bytes                        (one medium reaches every bank)
    B tree multicast   = injected bytes x NB                   (one inbound link per bank)
    C unicast chain    = injected bytes x NB (NB + 1) / 2      (hop-count weighted worst case)
INT4 weights (2 per byte) are run through the same model for the controlled comparison.

Activation vectors per layer (distinct quantized inputs): pre-qkv (H), pre-o (nH x hd),
pre-gate/up (H), pre-down (I) — Workload.activation_inputs_per_layer (EXP-0003).  Outputs:
Workload.outputs_per_layer, returned as 32-bit words (22-bit sums padded; also reported as
ACC_W bits).  Batch 1.  Everything here is ANALYTICAL from MEASURED shapes.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from model_specs import SPECS  # noqa: E402
from roofline_models import Assumptions, Workload, token_model  # noqa: E402
from workload_mapping import MODELS, load_workloads, tensors  # noqa: E402

ACC_W = 22
OUT_BYTES = 4
HEAD_BITS = 4.156     # EXP-0004 measured INT4 head (candidate, not frozen)
KV_BITS = 4.50        # EXP-0004 measured 4-bit KV effective bits (candidate, not frozen)
CAPACITIES = (64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20)
BANK_COUNTS = (16, 64, 256, 1024, 4096)


def per_token(w: Workload, nb: int, weight_bits: int = 2, buffered: bool = True, tokens_resident: int = 1000) -> dict:
    packed = w.W * weight_bits // 8
    act_inj = w.activation_inputs_per_layer * w.L                 # bytes, INT8, once per distinct vector
    outputs = w.outputs_per_layer * w.L * OUT_BYTES
    # lockstep re-broadcast: x re-sent once per row-quad pass -> sum over tensors of rowq * K bytes
    rebroadcast = sum(t["rowq"] * t["K"] * t["layers"] for t in tensors(w)) if weight_bits == 2 else \
        sum((t["N"] // 2) * t["K"] * t["layers"] for t in tensors(w))
    act_delivered_nm = nb * act_inj if buffered else rebroadcast
    return {
        "model": w.slug, "weight_bits": weight_bits, "banks": nb, "bank_buffers": buffered,
        "packed_weight_bytes": packed,
        # streaming
        "stream_A_load_once": packed, "stream_B_global_per_token": packed + act_inj + outputs,
        "stream_B_global_weights": packed, "stream_C_local": 0,
        "stream_D_activation_injected": act_inj, "stream_D_activation_delivered": act_inj, "stream_D_outputs": outputs,
        "stream_buffer_bits": 8 * max(t["K"] for t in tensors(w)),
        # near-memory, whole model resident
        "nm_A_load_once": packed, "nm_A_per_token_amortized": packed / tokens_resident,
        "nm_A_per_token_if_per_layer_residency": packed,
        "nm_B_global_per_token": act_inj + outputs, "nm_B_global_weights": 0,
        "nm_C_local_reads": packed,
        "nm_D_activation_injected": act_inj, "nm_D_activation_delivered": act_delivered_nm, "nm_D_outputs": outputs,
        "nm_D_partial_sums": 0,
        "nm_buffer_bits_total": nb * 8 * max(t["K"] for t in tensors(w)) if buffered else 0,
        "nm_link_bytes_A_ideal_multicast": act_inj,
        "nm_link_bytes_B_tree": act_inj * nb,
        "nm_link_bytes_C_unicast_chain": act_inj * nb * (nb + 1) // 2,
        "global_reduction_vs_streaming": round((packed + act_inj + outputs) / (act_inj + outputs), 2),
        "nm_global_over_stream_global": round((act_inj + outputs) / (packed + act_inj + outputs), 4),
        "nm_activation_delivered_over_packed": round(act_delivered_nm / packed, 4),
    }


def capacity_rows(ws: dict[str, Workload]) -> list[dict]:
    rows = []
    for slug, w in ws.items():
        ts = tensors(w)
        per_layer = sum(t["packed_bytes"] for t in ts)
        largest = max(ts, key=lambda t: t["packed_bytes"]); smallest = min(ts, key=lambda t: t["packed_bytes"])
        total = w.W // 4
        base = {"model": slug, "packed_bytes_total": total, "packed_bytes_per_layer": per_layer, "layers": w.L,
                "largest_tensor": f"{largest['module']} {largest['N']}x{largest['K']}", "largest_tensor_bytes": largest["packed_bytes"],
                "smallest_tensor": f"{smallest['module']} {smallest['N']}x{smallest['K']}", "smallest_tensor_bytes": smallest["packed_bytes"],
                "int4_bytes_total": w.W // 2}
        for cap in CAPACITIES:
            rows.append({**base, "scenario": "banks needed at capacity", "bank_capacity_bytes": cap,
                         "banks_needed_whole_model": math.ceil(total / cap), "banks_needed_one_layer": math.ceil(per_layer / cap),
                         "banks_needed_int4_whole_model": math.ceil(w.W // 2 / cap), "bank_count": "", "capacity_needed_per_bank": ""})
        for nb in BANK_COUNTS:
            need = sum(math.ceil(t["rowq"] / nb) * t["K"] * t["layers"] for t in ts)     # includes tail waste
            rows.append({**base, "scenario": "capacity per bank at bank count", "bank_capacity_bytes": "", "banks_needed_whole_model": "",
                         "banks_needed_one_layer": "", "banks_needed_int4_whole_model": "", "bank_count": nb,
                         "capacity_needed_per_bank": need})
    return rows


def reconnect_rows(ws: dict[str, Workload]) -> list[dict]:
    """EXP-0003 token model at 100 GB/s, batch 1, with the EXP-0004 candidate head/KV bits, for the
    streaming system (ternary weights cross the global path) and the near-memory system (they do not).
    Nothing in results/roofline or results/model_precision is modified."""
    rows = []
    for slug, w in ws.items():
        tm_base = per_token(w, nb=64)
        for ctx in (512, 2048, 4096, 8192, 16384, 32768):
            if ctx > w.max_ctx:
                continue
            for label, a in (("BF16 head, BF16 KV (EXP-0003 reference)", Assumptions(bw=100e9, batch=1, context=ctx)),
                             ("candidate INT4 head, 4-bit KV (EXP-0004 bits)",
                              Assumptions(bw=100e9, batch=1, context=ctx, head_bytes_per_elem=HEAD_BITS / 8, kv_bytes_per_elem=KV_BITS / 8))):
                tm = token_model(w, a)
                head = tm.stage("head").memory_bytes; kv = tm.stage("attention_kv").memory_bytes
                other = tm.stage("other").memory_bytes
                tern = tm.stage("ternary").memory_bytes
                nm_global_tern = tm_base["nm_B_global_per_token"]
                stream_total = tern + head + kv + other
                nm_total = nm_global_tern + head + kv + other
                lim_nm = max((("head", head), ("attention_kv", kv), ("activations+outputs", nm_global_tern), ("other", other)), key=lambda p: p[1])[0]
                rows.append({"model": slug, "context": ctx, "scenario": label,
                             "ternary_global_bytes_streaming": tern, "ternary_local_bytes_near_memory": tm_base["nm_C_local_reads"],
                             "activation_plus_output_global_bytes_near_memory": nm_global_tern,
                             "head_bytes": head, "kv_bytes": kv, "other_bytes": other,
                             "global_total_streaming": stream_total, "global_total_near_memory": nm_total,
                             "global_reduction_factor": round(stream_total / nm_total, 2),
                             "limiter_streaming": tm.limiting_component, "limiter_near_memory": lim_nm,
                             "ternary_fraction_of_global_streaming": round(tern / stream_total, 3),
                             "note": "ANALYTICAL; near-memory assumes whole-model residency; no speedup or energy claim"})
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wri.writeheader(); wri.writerows(rows)


if __name__ == "__main__":
    ws = load_workloads()
    traffic = []
    for slug, w in ws.items():
        for nb in (16, 64, 256, 1024):
            for bits in (2, 4):
                traffic.append(per_token(w, nb, bits, buffered=True))
            traffic.append(per_token(w, nb, 2, buffered=False))
    out = os.path.join("results", "hardware_gate")
    write_csv(os.path.join(out, "traffic_comparison.csv"), traffic)
    write_csv(os.path.join(out, "capacity_scaling.csv"), capacity_rows(ws))
    write_csv(os.path.join(out, "system_reconnect.csv"), reconnect_rows(ws))
    print("traffic, capacity, reconnect written")
