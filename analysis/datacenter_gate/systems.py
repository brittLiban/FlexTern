"""
EXP-0007 system models: per-scenario bytes, capacity gates, topology, pipeline, throughput bounds and
movement-energy proxies for

  A  conventional FP4 HBM accelerator node (tensor-parallel; pipeline-parallel across nodes)
  B  ternary streaming on the same HBM node (2-bit weights consumed natively)
  C  ternary resident tiles (SRAM weights, per-tile DRAM for KV when it does not fit)
  D  INT4 resident tiles (same topology algorithm, 4.5-bit weights)

Every scenario is (model, batch B, context C, phase, system, weight bits, KV bits, head bits, tile cap).
Bytes are per generated token per sequence (decode) or per prompt token per sequence (prefill).
Weight reuse: the group of accelerators/tiles that holds a weight shard reads it once per step and
applies it to the sequences it serves in that step (tensor-parallel group: B sequences; pipeline stage:
the microbatch b = B / stages).  KV is per sequence and never shared.  Decimal units.
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from sources import (GPU_FP4_DENSE_FLOPS, GPU_HBM_BYTES, GPU_HBM_PEAK_BPS, GPU_NVLINK_BPS_PER_DIR, GPUS_PER_NODE,  # noqa: E402
                     KV_BITS, PJ_PER_BIT, SRAM_PLANNING_PJ, TILE_DRAM_BYTES, TILE_DRAM_PEAK_BPS, TILE_LINK_BPS, TILE_LOCAL_BPS,
                     TILE_MARGIN, TILE_TERMS_PER_S, TILES_PER_BOARD, TILES_PER_PACKAGE, TILES_PER_RACK, UTIL_COMPUTE, UTIL_HBM,
                     UTIL_LINK, WEIGHT_BITS)
from workloads import Model  # noqa: E402

SYSTEMS = ("A", "B", "C", "D")
SYSTEM_DESC = {"A": "conventional FP4 HBM node (tensor-parallel)", "B": "ternary streaming on the same HBM node",
               "C": "ternary resident SRAM tiles", "D": "INT4 resident SRAM tiles (fairness control)"}


# ----------------------------------------------------------------------------- HBM node (A, B)
def hbm_node(m: Model, B: int, ctx: int, wbits: float, kvbits: float, hbits: float, phase: str, serving_util: float = 1.0) -> dict:
    """Weights sharded tensor-parallel across G GPUs in one NVLink domain (G <= 8); if weights + KV for B
    sequences exceed the domain, more nodes are added as pipeline stages (reuse falls to the microbatch)."""
    W = m.weight_bytes(wbits); Hb = m.head_bytes(hbits); kv_seq = m.kv_bytes_per_seq(ctx, kvbits)
    need = W + Hb + B * kv_seq
    gpus = max(1, math.ceil(need / (GPU_HBM_BYTES * 0.9)))
    nodes = math.ceil(gpus / GPUS_PER_NODE)
    stages = nodes                                   # pipeline across nodes
    b = max(1, B // stages)                          # microbatch per stage
    reuse = max(1.0, b * serving_util)               # sequences sharing one weight read
    tp = min(gpus, GPUS_PER_NODE)
    if phase == "decode":
        w_tok = W / reuse; h_tok = Hb / reuse; kv_r = kv_seq; kv_w = m.kv_bytes_per_token_layer(kvbits) * m.L
        ar = 2 * m.L * 2 * (tp - 1) / tp * m.act_bytes_per_boundary() if tp > 1 else 0.0   # two all-reduces per layer (D: ring)
        pp = (stages - 1) * m.act_bytes_per_boundary()
        flops = 2 * m.linear_params + 2 * m.head_params + m.attention_flops_per_token(ctx)
        hbm_bytes_step = W + Hb + B * (kv_r + kv_w)
        t_mem = hbm_bytes_step / (gpus * GPU_HBM_PEAK_BPS * UTIL_HBM)
        t_cmp = B * flops / (gpus * GPU_FP4_DENSE_FLOPS * UTIL_COMPUTE)
        t_link = B * (ar + pp) / (gpus * GPU_NVLINK_BPS_PER_DIR * UTIL_LINK)
        step = max(t_mem, t_cmp, t_link) * stages / max(1, stages)   # pipelined: throughput per step unchanged, latency x stages
        return dict(gpus=gpus, nodes=nodes, stages=stages, microbatch=b, reuse=reuse, tp=tp,
                    weight_bytes=w_tok, local_weight_bytes=0.0, kv_read=kv_r, kv_write=kv_w, head_bytes=h_tok,
                    interconnect_bytes=ar + pp, interconnect_tier="board" if tp > 1 else "none", partial_sum_bytes=ar, kv_location="hbm",
                    flops=flops, terms=m.linear_params, step_s=step, t_mem=t_mem, t_cmp=t_cmp, t_link=t_link,
                    agg_tok_s=B / step, seq_tok_s=1 / (step * stages) if stages else 1 / step, latency_s=step * stages,
                    bottleneck=max((("hbm", t_mem), ("compute", t_cmp), ("link", t_link)), key=lambda x: x[1])[0],
                    hbm_capacity_needed=need, kv_capacity_per_seq=kv_seq, kv_fits=True)
    # prefill: C prompt tokens per sequence
    P = ctx
    w_tok = W / (reuse * P); h_tok = Hb / (reuse * P)                 # head: last token only, spread over P
    kv_w = m.kv_bytes_per_token_layer(kvbits) * m.L; kv_r = 0.0
    flops_seq = 2 * m.linear_params * P + 4 * m.L * m.nH * m.hd * P * P / 2
    ar = 2 * m.L * 2 * (tp - 1) / tp * m.act_bytes_per_boundary() if tp > 1 else 0.0
    pp = (stages - 1) * m.act_bytes_per_boundary()
    hbm_bytes = W + Hb + B * P * kv_w
    t_mem = hbm_bytes / (gpus * GPU_HBM_PEAK_BPS * UTIL_HBM)
    t_cmp = B * flops_seq / (gpus * GPU_FP4_DENSE_FLOPS * UTIL_COMPUTE)
    t_link = B * P * (ar + pp) / (gpus * GPU_NVLINK_BPS_PER_DIR * UTIL_LINK)
    step = max(t_mem, t_cmp, t_link)
    return dict(gpus=gpus, nodes=nodes, stages=stages, microbatch=b, reuse=reuse, tp=tp,
                weight_bytes=w_tok, local_weight_bytes=0.0, kv_read=kv_r, kv_write=kv_w, head_bytes=h_tok,
                interconnect_bytes=ar + pp, interconnect_tier="board" if tp > 1 else "none", partial_sum_bytes=ar, kv_location="hbm",
                flops=flops_seq / P, terms=m.linear_params, step_s=step, t_mem=t_mem, t_cmp=t_cmp, t_link=t_link,
                agg_tok_s=B * P / step, seq_tok_s=P / step, latency_s=step,
                bottleneck=max((("hbm", t_mem), ("compute", t_cmp), ("link", t_link)), key=lambda x: x[1])[0],
                hbm_capacity_needed=need, kv_capacity_per_seq=kv_seq, kv_fits=True)


# ----------------------------------------------------------------------------- resident tiles (C, D)
def topology(m: Model, wbits: float, hbits: float, cap: float) -> dict:
    usable = cap * (1 - TILE_MARGIN)
    lb = m.layer_bytes(wbits)
    if lb <= usable:
        lpt = max(1, int(usable // lb)); k = 1
        stages = math.ceil(m.L / lpt)
    else:
        k = math.ceil(lb / usable); lpt = 1 / k; stages = m.L
    weight_tiles = stages * k
    head_tiles = math.ceil(m.head_bytes(hbits) / usable)
    tiles = weight_tiles + head_tiles
    layers_per_tile = m.L / stages
    return dict(cap=cap, usable=usable, layer_bytes=lb, layers_per_tile=layers_per_tile, tp_k=k, stages=stages, weight_tiles=weight_tiles,
                head_tiles=head_tiles, tiles=tiles, largest_layer_fits=lb <= usable,
                unused_capacity_bytes=tiles * usable - m.weight_bytes(wbits) - m.head_bytes(hbits),
                sram_total=tiles * cap)


def tier_for_stages(stages: int) -> str:
    if stages <= TILES_PER_PACKAGE: return "package"
    if stages <= TILES_PER_BOARD: return "board"
    return "rack"


def resident(m: Model, B: int, ctx: int, wbits: float, kvbits: float, hbits: float, phase: str, cap: float, serving_util: float = 1.0,
             mode: str = "pipelined") -> dict:
    """mode="pipelined": the batch is split into one microbatch per stage so every stage works every step
    (weight reuse = microbatch).  mode="batch_sequential": one microbatch of the whole batch walks the
    pipeline (weight reuse = B, but only one stage works at a time: bubble = 1 - 1/stages)."""
    t = topology(m, wbits, hbits, cap)
    stages, k = t["stages"], t["tp_k"]
    if mode == "batch_sequential":
        b = B; inflight = 1
    else:
        b = max(1, math.ceil(B / stages)); inflight = min(B, stages)   # every sequence served; microbatches in flight
    reuse = max(1.0, b * serving_util)
    bubble = 1 - inflight / stages                            # fraction of stage time idle
    W = m.weight_bytes(wbits); Hb = m.head_bytes(hbits); kv_seq = m.kv_bytes_per_seq(ctx, kvbits)
    kv_tile_cap = B * kv_seq / stages                         # KV for the layers on one stage, all sequences
    sram_free = t["usable"] - t["layers_per_tile"] * t["layer_bytes"] if k == 1 else 0.0
    kv_fits = kv_tile_cap <= max(0.0, sram_free)
    kv_loc = "sram" if kv_fits else "dram"
    kv_tile_dram = 0.0 if kv_fits else kv_tile_cap
    dram_stacks_per_tile = 0 if kv_fits else math.ceil(kv_tile_dram / (TILE_DRAM_BYTES * 0.9))
    act = m.act_bytes_per_boundary()
    tier = tier_for_stages(stages)
    if phase == "decode":
        w_tok = W / reuse; h_tok = Hb / reuse; kv_r = kv_seq; kv_w = m.kv_bytes_per_token_layer(kvbits) * m.L
        pp = (stages - 1) * act                                # per sequence-token hidden-state hops
        ar = 2 * m.L * 2 * (k - 1) / k * act if k > 1 else 0.0  # tensor-parallel reductions within a stage
        W_stage = W / stages; terms_stage = m.linear_params / stages
        t_local = W_stage / TILE_LOCAL_BPS
        t_cmp = b * terms_stage / TILE_TERMS_PER_S
        t_kv = b * (kv_r + kv_w) / stages / (TILE_DRAM_PEAK_BPS * UTIL_HBM if not kv_fits else TILE_LOCAL_BPS)
        t_link = b * (act + (ar / m.L) * t["layers_per_tile"]) / (TILE_LINK_BPS * UTIL_LINK)
        t_micro = max(t_local, t_cmp, t_kv, t_link)
        step = stages * t_micro                                # one token for every in-flight sequence
        return dict(**t, mode=mode, microbatch=b, reuse=reuse, inflight_microbatches=inflight, bubble_fraction=bubble,
                    weight_bytes=0.0, local_weight_bytes=w_tok, kv_read=kv_r, kv_write=kv_w, head_bytes=h_tok,
                    interconnect_bytes=pp + ar, interconnect_tier=tier, partial_sum_bytes=ar, kv_location=kv_loc,
                    kv_capacity_per_stage=kv_tile_cap, dram_stacks_per_tile=dram_stacks_per_tile,
                    flops=2 * m.linear_params + 2 * m.head_params + m.attention_flops_per_token(ctx), terms=m.linear_params,
                    step_s=step, t_micro=t_micro, t_local=t_local, t_cmp=t_cmp, t_kv=t_kv, t_link=t_link,
                    agg_tok_s=(b / t_micro) * (1 - bubble), seq_tok_s=1 / step, latency_s=step,
                    bottleneck=max((("local_sram", t_local), ("compute_target", t_cmp), ("kv_" + kv_loc, t_kv), ("link", t_link)), key=lambda x: x[1])[0],
                    kv_capacity_per_seq=kv_seq, kv_fits=kv_fits)
    P = ctx
    w_tok = W / (reuse * P); h_tok = Hb / (reuse * P)
    kv_w = m.kv_bytes_per_token_layer(kvbits) * m.L
    pp = (stages - 1) * act; ar = 2 * m.L * 2 * (k - 1) / k * act if k > 1 else 0.0
    terms_seq = m.linear_params * P + (4 * m.L * m.nH * m.hd * P * P / 2) / 2     # attention counted as terms too (ASSUMPTION)
    t_local = (W / stages) / TILE_LOCAL_BPS
    t_cmp = b * terms_seq / stages / TILE_TERMS_PER_S
    t_kv = b * P * kv_w / stages / (TILE_DRAM_PEAK_BPS * UTIL_HBM if not kv_fits else TILE_LOCAL_BPS)
    t_link = b * P * (act + (ar / m.L) * t["layers_per_tile"]) / (TILE_LINK_BPS * UTIL_LINK)
    t_micro = max(t_local, t_cmp, t_kv, t_link); step = stages * t_micro
    return dict(**t, mode=mode, microbatch=b, reuse=reuse, inflight_microbatches=inflight, bubble_fraction=bubble,
                weight_bytes=0.0, local_weight_bytes=w_tok, kv_read=0.0, kv_write=kv_w, head_bytes=h_tok,
                interconnect_bytes=pp + ar, interconnect_tier=tier, partial_sum_bytes=ar, kv_location=kv_loc,
                kv_capacity_per_stage=kv_tile_cap, dram_stacks_per_tile=dram_stacks_per_tile,
                flops=(2 * m.linear_params * P + 4 * m.L * m.nH * m.hd * P * P / 2) / P, terms=terms_seq / P,
                step_s=step, t_micro=t_micro, t_local=t_local, t_cmp=t_cmp, t_kv=t_kv, t_link=t_link,
                agg_tok_s=(b * P / t_micro) * (1 - bubble), seq_tok_s=P / step, latency_s=step,
                bottleneck=max((("local_sram", t_local), ("compute_target", t_cmp), ("kv_" + kv_loc, t_kv), ("link", t_link)), key=lambda x: x[1])[0],
                kv_capacity_per_seq=kv_seq, kv_fits=kv_fits)


# ----------------------------------------------------------------------------- energy proxy
def energy_mj(r: dict, system: str) -> tuple[float, float]:
    """Movement-energy proxy per sequence-token (mJ): weights, KV, head, interconnect.  Compute excluded."""
    lo = hi = 0.0
    def add(bytes_, tier):
        nonlocal lo, hi
        a, bb = PJ_PER_BIT[tier]; lo += bytes_ * 8 * a / 1e9; hi += bytes_ * 8 * bb / 1e9
    if system in ("A", "B"):
        add(r["weight_bytes"] + r["head_bytes"] + r["kv_read"] + r["kv_write"], "hbm")
        if r["interconnect_bytes"]:
            add(r["interconnect_bytes"], "board")
    else:
        lo += r["local_weight_bytes"] * 8 * PJ_PER_BIT["sram_local"][0] / 1e9; hi += r["local_weight_bytes"] * 8 * SRAM_PLANNING_PJ / 1e9
        lo += r["head_bytes"] * 8 * PJ_PER_BIT["sram_local"][0] / 1e9; hi += r["head_bytes"] * 8 * SRAM_PLANNING_PJ / 1e9
        add(r["kv_read"] + r["kv_write"], "hbm" if r["kv_location"] == "dram" else "sram_local")
        if r["interconnect_bytes"]:
            add(r["interconnect_bytes"], r["interconnect_tier"])
    return lo, hi


def scenario(m: Model, B: int, ctx: int, phase: str, system: str, kv: str = "fp8", head: str = "fp4", cap: float = 512e6,
             serving_util: float = 1.0, wbits_override: float | None = None, mode: str = "pipelined") -> dict:
    wbits = {"A": WEIGHT_BITS["fp4"], "B": WEIGHT_BITS["ternary"], "C": WEIGHT_BITS["ternary"], "D": WEIGHT_BITS["fp4"]}[system]
    if wbits_override is not None:
        wbits = wbits_override
    kvbits = KV_BITS[kv]
    hbits = {"bf16": 16.0, "fp4": 4.5, "candidate": 4.156}[head]
    if system in ("A", "B"):
        r = hbm_node(m, B, ctx, wbits, kvbits, hbits, phase, serving_util)
    else:
        r = resident(m, B, ctx, wbits, kvbits, hbits, phase, cap, serving_util, mode)
    lo, hi = energy_mj(r, system)
    r.update(model=m.name, kind=m.kind, batch=B, context=ctx, phase=phase, system=system, system_desc=SYSTEM_DESC[system],
             weight_bits=wbits, kv_bits=kvbits, head_bits=hbits, tile_cap=cap if system in ("C", "D") else 0, serving_util=serving_util,
             static_weight_capacity=m.weight_bytes(wbits) + m.head_bytes(hbits),
             total_bytes=r["weight_bytes"] + r["local_weight_bytes"] + r["kv_read"] + r["kv_write"] + r["head_bytes"] + r["interconnect_bytes"],
             energy_mj_min=lo, energy_mj_max=hi)
    return r
