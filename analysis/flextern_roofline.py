"""
FlexTern EXP-0003 — analytical compute-vs-memory bottleneck model (driver).

Consumes the three canonical workload profiles (ADR-0004) programmatically,
verifies invariants against EXP-0002/EXP-0002B, runs the sweeps, and writes
results/roofline/. All outputs are ANALYTICAL / INFERRED under the abstract
assumptions recorded in assumptions.json. No hardware is measured.

Usage (from repo root):
    python analysis/flextern_roofline.py [--out results/roofline]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_specs import SPECS  # noqa: E402
from roofline_models import (Assumptions, Workload, accumulator_bits, assumptions_dict,  # noqa: E402
                             batch_for_weight_eq_kv, break_even, context_crossovers, head_bytes,
                             kv_read_bytes, packed_weight_bytes, prefill_attention_equals_ternary,
                             prefill_model, residency_external_bytes, storage_facts, tile_utilization,
                             token_model, activation_bytes_per_token, norm_scale_bytes)

MODELS = ["bitnet-2b4t", "falcon-e-1b", "falcon-e-3b"]
CONTEXTS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
BATCHES = [1, 2, 4, 8, 16, 32]
BWS = [100e9, 250e9, 500e9, 1e12, 2e12, 4e12, 8e12]
TPCS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
CLOCKS = [250e6, 500e6, 1e9, 1.5e9, 2e9]
NT_RATES = [0.25e12, 0.5e12, 1e12, 2e12, 4e12, 8e12]
CAPACITIES = [0, 16e6, 32e6, 64e6, 128e6, 256e6, 512e6, 1e9]
POLICIES = ["none", "head_if_fits", "one_layer_buffer", "greedy_ternary_layers", "greedy_head_then_layers",
            "ideal_ternary_first", "ideal_nonternary_first", "ideal_including_kv"]
TILES = [128, 256, 512]

DEFAULT = Assumptions()  # bw 250 GB/s, 4096 terms/cycle @ 1 GHz, 1e12 non-ternary ops/s, BF16 head/KV, 2-bit, dense, B=1, ctx 4096

# Named analysis scenarios (abstract; not products). Each is a modification of DEFAULT.
SCENARIOS = {
    "S0_short_ctx_batch1_edge_like": dict(context=512, batch=1, bw=100e9),
    "S1_medium_ctx_batch1": dict(context=4096, batch=1, bw=250e9),
    "S2_long_ctx_batch1": dict(context=32768, batch=1, bw=250e9),
    "S3_high_batch_throughput": dict(context=2048, batch=16, bw=500e9, terms_per_cycle=16384),
    "S4_bandwidth_rich": dict(context=2048, batch=1, bw=2e12),
    "S5_wide_ternary_engine": dict(context=2048, batch=1, bw=250e9, terms_per_cycle=65536),
    "S6_hypothetical_int4_kv": dict(context=32768, batch=1, bw=250e9, kv_bytes_per_elem=0.5),
    "S7_hypothetical_4bit_act_1p6bit_weights": dict(context=2048, batch=1, bw=100e9, weight_bits=1.6, act_bits=4,
                                                    act_bitplane_scaling=True),
}


def fmt(x):
    if isinstance(x, float):
        if math.isinf(x):
            return "inf"
        return f"{x:.6g}"
    return x


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})


def tier(c):
    if c >= 3: return ">3x"
    if c >= 2: return ">2x"
    if c >= 1.5: return ">1.5x"
    if c >= 1.25: return "1.25-1.5x"
    return "<1.25x"


def tm_row(w, a, tm):
    c = tm.ceilings()
    r = {"model": w.slug, "context": a.context, "hypothetical_context": tm.hypothetical_context, "batch": a.batch,
         "bw_GBps": a.bw / 1e9, "terms_per_cycle": a.terms_per_cycle, "clock_GHz": a.clock_hz / 1e9,
         "ternary_rate_terms_per_s": a.ternary_rate, "nonternary_rate_ops_per_s": a.nonternary_rate,
         "weight_bits": a.weight_bits, "act_bits": a.act_bits, "head_bytes_per_elem": a.head_bytes_per_elem,
         "kv_bytes_per_elem": a.kv_bytes_per_elem, "zero_mode": a.zero_mode}
    for s in tm.stages:
        r[f"{s.name}_ops"] = s.compute_ops; r[f"{s.name}_bytes"] = s.memory_bytes
        r[f"{s.name}_compute_s"] = s.compute_time; r[f"{s.name}_memory_s"] = s.memory_time
        r[f"{s.name}_time_opt_s"] = s.optimistic; r[f"{s.name}_limiter"] = s.limiter
    r.update({"total_bytes": tm.total_bytes, "total_ops": tm.total_ops,
              "total_time_optimistic_s": tm.total_optimistic, "total_time_conservative_s": tm.total_conservative,
              "limiting_component": tm.limiting_component,
              "ternary_time_fraction": tm.ternary_time_fraction, "ternary_bytes_fraction": tm.ternary_bytes_fraction,
              "pct_time_ternary": 100 * tm.stage("ternary").optimistic / tm.total_optimistic,
              "pct_time_head": 100 * tm.stage("head").optimistic / tm.total_optimistic,
              "pct_time_attention_kv": 100 * tm.stage("attention_kv").optimistic / tm.total_optimistic,
              "pct_time_other": 100 * tm.stage("other").optimistic / tm.total_optimistic,
              "modeled_tokens_per_s_optimistic_ANALYTICAL": tm.tokens_per_second_optimistic()})
    r.update(c)
    r["ceiling_tier_ideal"] = tier(c["ceiling_ideal"])
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("results", "roofline"))
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    # ---- load canonical workloads and verify identity/invariants --------------
    W = {}
    for slug in MODELS:
        spec = SPECS[slug]
        w = Workload.from_profile_dir(spec.default_out)
        assert w.model_id == spec.model_id and w.revision == spec.revision, (slug, w.model_id, w.revision)
        assert sum(m["weights"] for m in w.modules) == w.W, slug
        assert abs(w.head_bytes_measured - head_bytes(w, 2.0)) < 1, slug
        assert set(m["K"] for m in w.modules) == {w.H, w.I}, slug
        W[slug] = w
    # EXP-0002B KV numbers reproduced from equations, not copied
    assert kv_read_bytes(W["bitnet-2b4t"], 4096, 2.0) == 314_572_800
    assert kv_read_bytes(W["falcon-e-1b"], 32768, 2.0) == 805_306_368
    assert kv_read_bytes(W["falcon-e-3b"], 32768, 2.0) == 1_073_741_824
    # terms per weight byte
    for w in W.values():
        assert abs(w.W / packed_weight_bytes(w, 2.0) - 4.0) < 1e-12
        assert abs(w.W / packed_weight_bytes(w, 1.6) - 5.0) < 1e-12

    # ---- per-model workload facts ------------------------------------------------
    facts = []
    for w in W.values():
        sf = storage_facts(w)
        facts.append({"model": w.slug, "model_id": w.model_id, "revision": w.revision[:12],
                      "ternary_terms_per_token": w.W, "packed_2bit_bytes": packed_weight_bytes(w, 2.0),
                      "packed_1p6bit_bytes": packed_weight_bytes(w, 1.6), "terms_per_byte_2bit": w.W / packed_weight_bytes(w, 2.0),
                      "terms_per_byte_1p6bit": w.W / packed_weight_bytes(w, 1.6), "zero_rate": w.zero_rate,
                      "useful_terms_per_token_nonzero": w.W * (1 - w.zero_rate),
                      "head_params": w.head_params, "head_bytes_bf16": head_bytes(w, 2.0), "head_macs_per_token": w.V * w.H,
                      "norm_scale_bytes": norm_scale_bytes(w), "outputs_per_layer": w.outputs_per_layer,
                      "scale_ops_per_token": w.outputs_per_layer * w.L,
                      "activation_bytes_per_token_int8_onchip": activation_bytes_per_token(w, 8),
                      "activation_bytes_per_token_int4_onchip": activation_bytes_per_token(w, 4),
                      "kv_bytes_per_token_per_ctx_bf16": 2.0 * w.nKV * w.hd * 2.0 * w.L,
                      "kv_bytes_per_token_per_layer_per_ctx_bf16": 2.0 * w.nKV * w.hd * 2.0,
                      "one_layer_packed_bytes": sf["one_layer_bytes"], "largest_tensor_bytes": sf["largest_tensor_bytes"],
                      "smallest_tensor_bytes": sf["smallest_tensor_bytes"], "all_weights_bytes": sf["all_weights_bytes"],
                      "prefill_P_attention_eq_ternary": prefill_attention_equals_ternary(w)})
    write_csv(os.path.join(out, "workload_inputs.csv"), facts)
    # per projection terms
    proj = []
    for w in W.values():
        for m in w.modules:
            proj.append({"model": w.slug, "module": m["module"], "K": m["K"], "N": m["N"], "terms_per_token": m["weights"],
                         "pct_of_ternary_terms": 100 * m["weights"] / w.W, "packed_bytes_2bit": m["weights"] / 4.0,
                         "outputs_per_token": m["N"] * w.L, "pct_zero": m["pct_zero"]})
    write_csv(os.path.join(out, "ternary_terms_by_projection.csv"), proj)

    # ---- per-model results: context x batch at DEFAULT (opportunity curve) ------
    pm = []
    for w in W.values():
        for C in CONTEXTS:
            for B in BATCHES:
                a = replace(DEFAULT, context=C, batch=B)
                pm.append(tm_row(w, a, token_model(w, a)))
    write_csv(os.path.join(out, "per_model_results.csv"), pm)

    # ---- ternary-only ceilings: model x ctx x batch x bw --------------------------
    tc = []
    for w in W.values():
        for C in CONTEXTS:
            for B in (1, 4, 16):
                for bw in (100e9, 1e12, 8e12):
                    a = replace(DEFAULT, context=C, batch=B, bw=bw)
                    tm = token_model(w, a)
                    c = tm.ceilings()
                    tc.append({"model": w.slug, "context": C, "hypothetical_context": tm.hypothetical_context, "batch": B,
                               "bw_GBps": bw / 1e9, "ternary_time_fraction": tm.ternary_time_fraction,
                               "ternary_bytes_fraction": tm.ternary_bytes_fraction, "limiting_component": tm.limiting_component,
                               **c, "tier_ideal": tier(c["ceiling_ideal"])})
    write_csv(os.path.join(out, "ternary_only_ceiling.csv"), tc)

    # ---- bottleneck map: model x ctx x batch x bw x engine width ------------------
    bm = []
    for w in W.values():
        for C in CONTEXTS:
            if C > w.max_ctx:
                continue
            for B in (1, 4, 16):
                for bw in BWS:
                    for tpc in (1024, 4096, 16384, 65536):
                        a = replace(DEFAULT, context=C, batch=B, bw=bw, terms_per_cycle=tpc)
                        tm = token_model(w, a)
                        t = tm.stage("ternary")
                        bm.append({"model": w.slug, "context": C, "batch": B, "bw_GBps": bw / 1e9, "terms_per_cycle": tpc,
                                   "ternary_limiter": t.limiter, "limiting_component": tm.limiting_component,
                                   "pct_ternary": 100 * tm.ternary_time_fraction, "ceiling_ideal": tm.ceilings()["ceiling_ideal"],
                                   "total_time_opt_s": tm.total_optimistic})
    write_csv(os.path.join(out, "bottleneck_map.csv"), bm)

    # ---- context crossovers (analytical) and numerical confirmation -------------
    cx = []
    for w in W.values():
        for B in (1, 4, 16):
            for wb in (2.0, 1.6):
                for hb in (2.0, 1.0, 0.5):
                    for kb in (2.0, 1.0, 0.5):
                        a = replace(DEFAULT, batch=B, weight_bits=wb, head_bytes_per_elem=hb, kv_bytes_per_elem=kb)
                        c = context_crossovers(w, a)
                        # numerical confirmation at low BW (memory-bound): find the first sweep ctx where ideal ceiling < 2
                        first_below_2 = None
                        for C in CONTEXTS:
                            tm = token_model(w, replace(a, context=C, bw=100e9))
                            if tm.ceilings()["ceiling_ideal"] < 2.0:
                                first_below_2 = C; break
                        cx.append({"model": w.slug, "batch": B, "weight_bits": wb, "head_bytes_per_elem": hb, "kv_bytes_per_elem": kb,
                                   **c, "sweep_first_ctx_with_ideal_ceiling_below_2x_at_100GBps": first_below_2,
                                   "max_ctx": w.max_ctx})
        # extras: batch at which weights == KV at each context
    write_csv(os.path.join(out, "context_crossovers.csv"), cx)

    # ---- batch crossovers ------------------------------------------------------------
    bc = []
    for w in W.values():
        for bw in BWS:
            for tpc in TPCS:
                a = replace(DEFAULT, bw=bw, terms_per_cycle=tpc)
                be = break_even(w, a)
                # numerical: first batch in sweep where ternary stage flips from memory- to compute-limited
                flip = None
                for B in BATCHES:
                    if token_model(w, replace(a, batch=B)).stage("ternary").limiter == "compute":
                        flip = B; break
                bc.append({"model": w.slug, "bw_GBps": bw / 1e9, "terms_per_cycle": tpc, "clock_GHz": 1.0,
                           "ternary_rate": a.ternary_rate, "batch_to_balance_ternary_analytic": be["batch_to_balance_ternary"],
                           "sweep_first_batch_compute_limited": flip})
    write_csv(os.path.join(out, "batch_crossovers.csv"), bc)
    bk = []
    for w in W.values():
        for C in CONTEXTS:
            if C <= w.max_ctx:
                bk.append({"model": w.slug, "context": C,
                           "batch_where_streamed_weight_bytes_eq_kv_read_bytes": batch_for_weight_eq_kv(w, replace(DEFAULT, context=C))})
    write_csv(os.path.join(out, "batch_weight_eq_kv.csv"), bk)

    # ---- break-even thresholds: bandwidth vs engine ----------------------------------
    be_rows = []
    for w in W.values():
        for bw in BWS:
            for f in CLOCKS:
                a = replace(DEFAULT, bw=bw, clock_hz=f)
                be = break_even(w, a)
                # numerical: first tpc where ternary stage becomes compute-limited at batch 1
                flip = None
                for tpc in TPCS:
                    if token_model(w, replace(a, terms_per_cycle=tpc)).stage("ternary").limiter == "compute":
                        flip = tpc; break
                be_rows.append({"model": w.slug, "bw_GBps": bw / 1e9, "clock_GHz": f / 1e9,
                                "terms_per_s_to_saturate_bw_batch1": be["terms_per_s_to_saturate_bw"],
                                "lanes_terms_per_cycle_to_saturate_bw_batch1": be["lanes_at_bw_and_clock"],
                                "sweep_first_terms_per_cycle_compute_limited": flip})
    write_csv(os.path.join(out, "break_even_thresholds.csv"), be_rows)
    eng = []
    for tpc in TPCS:
        for f in CLOCKS:
            a = replace(DEFAULT, terms_per_cycle=tpc, clock_hz=f)
            eng.append({"terms_per_cycle": tpc, "clock_GHz": f / 1e9, "ternary_rate_terms_per_s": a.ternary_rate,
                        "bw_GBps_to_feed_engine_batch1_2bit": break_even(W["bitnet-2b4t"], a)["bw_to_feed_engine"] / 1e9,
                        "bw_GBps_to_feed_engine_batch1_1p6bit": break_even(W["bitnet-2b4t"], replace(a, weight_bits=1.6))["bw_to_feed_engine"] / 1e9,
                        "note": "model-independent for dense packing (terms/byte fixed by packing)"})
    write_csv(os.path.join(out, "engine_bandwidth_requirements.csv"), eng)

    # ---- residency --------------------------------------------------------------------
    res = []
    for w in W.values():
        sf = storage_facts(w)
        for C in sorted({512, 4096, w.max_ctx}):
            for cap in CAPACITIES:
                for pol in POLICIES:
                    a = replace(DEFAULT, context=C, bw=100e9)
                    r = residency_external_bytes(w, a, cap, pol)
                    res.append({"model": w.slug, "context": C, "capacity_MB": cap / 1e6, **{k: v for k, v in r.items() if k != "capacity_bytes"},
                                "one_layer_MB": sf["one_layer_bytes"] / 1e6, "head_MB": sf["head_bytes_bf16"] / 1e6,
                                "full_ternary_MB": sf["full_ternary_bytes"] / 1e6, "kv_MB_at_ctx": kv_read_bytes(w, C, 2.0) / 1e6})
    write_csv(os.path.join(out, "residency.csv"), res)

    # ---- sensitivity: one-at-a-time around S0, S1, S2 ----------------------------------
    sens = []
    variations = {
        "baseline": {}, "weight_bits_1.6": dict(weight_bits=1.6), "act_bits_4_bytes_only": dict(act_bits=4),
        "act_bits_4_with_bitplane_rate_x2_HYPOTHETICAL": dict(act_bits=4, act_bitplane_scaling=True),
        "head_int8_HYPOTHETICAL": dict(head_bytes_per_elem=1.0), "head_int4_HYPOTHETICAL": dict(head_bytes_per_elem=0.5),
        "kv_int8_HYPOTHETICAL": dict(kv_bytes_per_elem=1.0), "kv_int4_HYPOTHETICAL": dict(kv_bytes_per_elem=0.5),
        "zero_ideal_skip_upper_bound": dict(zero_mode="ideal_skip"),
        "nonternary_rate_0.25T": dict(nonternary_rate=0.25e12), "nonternary_rate_8T": dict(nonternary_rate=8e12),
        "bw_x4": None, "terms_per_cycle_x4": None,
    }
    for w in W.values():
        for sname in ("S0_short_ctx_batch1_edge_like", "S1_medium_ctx_batch1", "S2_long_ctx_batch1"):
            base = replace(DEFAULT, **SCENARIOS[sname])
            for vname, mods in variations.items():
                if mods is None:
                    a = replace(base, bw=base.bw * 4) if vname == "bw_x4" else replace(base, terms_per_cycle=base.terms_per_cycle * 4)
                else:
                    a = replace(base, **mods)
                tm = token_model(w, a)
                c = tm.ceilings()
                sens.append({"model": w.slug, "scenario": sname, "variation": vname, "context": a.context, "batch": a.batch,
                             "hypothetical_context": tm.hypothetical_context,
                             "total_bytes": tm.total_bytes, "ternary_bytes": tm.stage("ternary").memory_bytes,
                             "total_time_opt_s": tm.total_optimistic, "total_time_cons_s": tm.total_conservative,
                             "limiting_component": tm.limiting_component, "pct_ternary_time": 100 * tm.ternary_time_fraction,
                             "ceiling_ideal": c["ceiling_ideal"], "ceiling_ideal_conservative": c["ceiling_ideal_conservative"],
                             "memory_bound_bytes_ratio": c["memory_bound_bytes_ratio"],
                             "modeled_tokens_per_s_ANALYTICAL": tm.tokens_per_second_optimistic()})
    write_csv(os.path.join(out, "sensitivity.csv"), sens)

    # ---- canonical scenarios ------------------------------------------------------------
    sc = []
    for sname, mods in SCENARIOS.items():
        for w in W.values():
            a = replace(DEFAULT, **mods)
            r = tm_row(w, a, token_model(w, a)); r = {"scenario": sname, **r}
            sc.append(r)
    write_csv(os.path.join(out, "canonical_scenarios.csv"), sc)

    # ---- tiles, accumulators, prefill ----------------------------------------------------
    tiles = []
    allK = sorted({m["K"] for w in W.values() for m in w.modules})
    for K in allK:
        for T in TILES:
            tiles.append({**tile_utilization(K, T), "models": "+".join(w.slug for w in W.values() if any(m["K"] == K for m in w.modules))})
    write_csv(os.path.join(out, "tiles.csv"), tiles)
    acc = [{**accumulator_bits(K, b), "models": "+".join(w.slug for w in W.values() if any(m["K"] == K for m in w.modules))}
           for K in allK for b in (8, 4)]
    write_csv(os.path.join(out, "accumulator_widths_by_act_bits.csv"), acc)
    pf = []
    for w in W.values():
        for P in (128, 512, 2048, 4096):
            if P <= w.max_ctx:
                pf.append({"model": w.slug, **prefill_model(w, replace(DEFAULT, bw=250e9), P),
                           "P_attention_eq_ternary": prefill_attention_equals_ternary(w)})
    write_csv(os.path.join(out, "prefill.csv"), pf)

    # ---- system storage comparison vs serious low-precision baselines (bytes only, INFERRED) ----
    ssc = []
    for w in W.values():
        for C in sorted({512, 4096, w.max_ctx}):
            hb = head_bytes(w, 2.0); kv = kv_read_bytes(w, C, 2.0); nb = norm_scale_bytes(w)
            tot = {bits: packed_weight_bytes(w, bits) + hb + kv + nb for bits in (16.0, 8.0, 4.0, 2.0, 1.6)}
            for bits in (16.0, 8.0, 4.0, 2.0, 1.6):
                ssc.append({"model": w.slug, "context": C, "ternary_weight_bits": bits,
                            "ternary_layer_bytes": packed_weight_bytes(w, bits), "head_bf16_bytes": hb, "kv_read_bytes_bf16": kv,
                            "norm_scale_bytes": nb, "system_bytes_per_token_batch1": tot[bits],
                            "ratio_vs_int4_weights_system": tot[4.0] / tot[bits], "ratio_vs_bf16_weights_system": tot[16.0] / tot[bits],
                            "ratio_vs_int8_weights_system": tot[8.0] / tot[bits],
                            "note": "bytes only; INT4/INT8/BF16 compute and energy not modeled; head and KV identical across rows"})
    write_csv(os.path.join(out, "system_storage_comparison.csv"), ssc)

    # ---- assumptions and environment -------------------------------------------------------
    assumptions = {
        "label": "All hardware capabilities here are ABSTRACT ASSUMPTIONS for sensitivity analysis, not FlexTern results.",
        "default": assumptions_dict(DEFAULT),
        "scenarios": {k: assumptions_dict(replace(DEFAULT, **v)) for k, v in SCENARIOS.items()},
        "sweeps": {"contexts": CONTEXTS, "batches": BATCHES, "bandwidth_bytes_per_s": BWS, "terms_per_cycle": TPCS,
                   "clocks_hz": CLOCKS, "nonternary_rates_ops_per_s": NT_RATES, "sram_capacities_bytes": CAPACITIES,
                   "residency_policies": POLICIES, "tiles": TILES},
        "accounting": {
            "per_token_normalization": "all quantities per generated token; per-step quantities divided by batch where shared",
            "weights": "ternary packed weights, head weights, norms/scales streamed once per step, shared across batch (÷B)",
            "kv": "sequence-specific; read 2*nKV*hd*b*L*ctx and write 2*nKV*hd*b*L per token; not shared across batch",
            "compute": "per token; batch does not reduce per-token compute",
            "activations": "assumed on-chip; counted, not added to external traffic",
            "attention_ops": "L*(2*nH*hd*ctx + 4*nH*ctx) op-equivalents (QK^T, AV, softmax ~4 ops/score)",
            "other_ops": "norms 3/elem, act-quant 2/elem over 4 inputs per layer, scale 1/output, RoPE 2/elem, residual, act fn, gating",
            "overlap": "optimistic = sum_i max(compute_i, memory_i); conservative = sum_i (compute_i + memory_i); no cross-stage prefetch",
            "ceilings": "compute-only: ternary stage -> its memory time; bandwidth-only: -> its compute time; ideal: -> 0; on optimistic bound",
            "zero_modes": "dense: W terms; gated: W engine cycles; ideal_skip: W*(1-z) terms; packed bytes identical in all modes",
            "residency": "explicit policies, no hit rates; external = max(0, bytes - resident)",
            "hypothetical_flags": "weight_bits 1.6, act_bits 4, head/KV INT8/INT4, bitplane scaling, kv_resident, context > max_position_embeddings",
        },
        "workloads": {w.slug: {"model_id": w.model_id, "revision": w.revision, "source_dir": w.source_dir} for w in W.values()},
    }
    with open(os.path.join(out, "assumptions.json"), "w", encoding="utf-8") as f:
        json.dump(assumptions, f, indent=2)
    import numpy
    env = [f"FlexTern EXP-0003 roofline environment", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
           f"Command: python {' '.join(sys.argv)}", f"Python: {sys.version.split()[0]} ({platform.platform()})",
           f"numpy: {numpy.__version__} (unused by the model; present in environment)",
           "Inputs: " + "; ".join(f"{w.slug}={w.model_id}@{w.revision[:8]} ({w.source_dir})" for w in W.values()),
           "Random seeds: none (deterministic)", f"Runtime: {time.time() - t0:.1f} s"]
    with open(os.path.join(out, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(env) + "\n")

    write_summary(out, W, facts, pm, tc, cx, bc, be_rows, res, sens, sc, tiles, acc, pf, ssc)
    print("\n".join(env)); print(f"wrote {out}")
    return 0


def write_summary(out, W, facts, pm, tc, cx, bc, be_rows, res, sens, sc, tiles, acc, pf, ssc):
    L = []; A = L.append
    MB = lambda b: f"{b / 1e6:,.1f} MB"
    A("# EXP-0003 roofline / bottleneck summary (generated by analysis/flextern_roofline.py; do not hand-edit)")
    A("")
    A("Everything below is ANALYTICAL / INFERRED under the abstract assumptions in `assumptions.json`. Workload inputs are MEASURED (EXP-0002/EXP-0002B). No hardware was measured; no tokens/sec figure is a measurement.")
    A("")
    A("Default abstract assumptions: 250 GB/s, 4096 terms/cycle @ 1 GHz (4.096e12 terms/s), 1e12 non-ternary op-eq/s, BF16 head and KV, 2-bit weights, dense execution, batch 1, context 4096.")
    A("")
    A("## 1. Workload inputs per token (MEASURED-derived)")
    A("")
    A("| Model | Ternary terms | Packed 2-bit bytes | terms/byte | 1.6-bit bytes | Zero rate | Head params | Head BF16 bytes | KV bytes/token/ctx (BF16) | Scale ops/token | Act bytes/token INT8 (on-chip) |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for f in facts:
        A(f"| {f['model']} | {f['ternary_terms_per_token']:,} | {MB(f['packed_2bit_bytes'])} | {f['terms_per_byte_2bit']:.1f} | {MB(f['packed_1p6bit_bytes'])} | {100*f['zero_rate']:.1f}% | {f['head_params']:,} | {MB(f['head_bytes_bf16'])} | {f['kv_bytes_per_token_per_ctx_bf16']:,.0f} B | {f['scale_ops_per_token']:,} | {MB(f['activation_bytes_per_token_int8_onchip'])} |")
    A("")
    A("## 2. Canonical scenarios (optimistic overlap; per token)")
    A("")
    A("| Scenario | Model | ctx | B | BW | Ternary % | Head % | Attn+KV % | Other % | Limiter | Bytes/token | T_opt | T_cons | Ideal ceiling | Compute-only | BW-only |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in sc:
        hyp = " (hyp ctx)" if r["hypothetical_context"] else ""
        A(f"| {r['scenario']} | {r['model']} | {r['context']}{hyp} | {r['batch']} | {r['bw_GBps']:.0f} GB/s | {r['pct_time_ternary']:.0f} | {r['pct_time_head']:.0f} | {r['pct_time_attention_kv']:.0f} | {r['pct_time_other']:.1f} | {r['limiting_component']} | {MB(r['total_bytes'])} | {r['total_time_optimistic_s']*1e3:.2f} ms | {r['total_time_conservative_s']*1e3:.2f} ms | {r['ceiling_ideal']:.2f}x | {r['ceiling_compute_only']:.2f}x | {r['ceiling_bandwidth_only']:.2f}x |")
    A("")
    A("## 3. Ternary opportunity curve: ideal ternary-only ceiling vs context (batch 1, default assumptions)")
    A("")
    A("Memory-bound identity: at 100 GB/s every stage is memory-bound, so the ideal ceiling equals total bytes / non-ternary bytes. Columns show 100 GB/s, 1 TB/s and 8 TB/s.")
    A("")
    A("| Model | ctx | ternary bytes frac | ideal @100 GB/s | ideal @1 TB/s | ideal @8 TB/s | limiter @1 TB/s | tier @100 GB/s |")
    A("|---|---:|---:|---:|---:|---:|---|---|")
    for w in W.values():
        for C in CONTEXTS:
            rows = {r["bw_GBps"]: r for r in tc if r["model"] == w.slug and r["context"] == C and r["batch"] == 1}
            hyp = " (hyp)" if rows[100.0]["hypothetical_context"] else ""
            A(f"| {w.slug} | {C}{hyp} | {rows[100.0]['ternary_bytes_fraction']:.3f} | {rows[100.0]['ceiling_ideal']:.2f}x | {rows[1000.0]['ceiling_ideal']:.2f}x | {rows[8000.0]['ceiling_ideal']:.2f}x | {rows[1000.0]['limiting_component']} | {rows[100.0]['tier_ideal']} |")
    A("")
    A("## 4. Batch effect on the ideal ceiling (100 GB/s; weights amortized ÷B, KV not)")
    A("")
    A("| Model | ctx | B=1 | B=4 | B=16 |"); A("|---|---:|---:|---:|---:|")
    for w in W.values():
        for C in sorted({512, 4096, w.max_ctx}):
            rows = {r["batch"]: r for r in tc if r["model"] == w.slug and r["context"] == C and r["bw_GBps"] == 100.0}
            A(f"| {w.slug} | {C} | {rows[1]['ceiling_ideal']:.2f}x | {rows[4]['ceiling_ideal']:.2f}x | {rows[16]['ceiling_ideal']:.2f}x |")
    A("")
    A("## 5. Context crossovers (analytical, per token, batch 1, BF16 head/KV, 2-bit weights)")
    A("")
    A("KV1 = 2·nKV·hd·b_kv·L bytes per token per unit context. ctx* = (stream bytes)/KV1. '0' means the condition holds at every context.")
    A("")
    A("| Model | KV1 (B) | KV > ternary | KV > head | KV > all weights | KV+head > ternary | ideal ceiling falls to 3x at ctx | to 2x at ctx | to 1.5x at ctx | max ctx |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in cx:
        if r["batch"] == 1 and r["weight_bits"] == 2.0 and r["head_bytes_per_elem"] == 2.0 and r["kv_bytes_per_elem"] == 2.0:
            A(f"| {r['model']} | {r['kv_bytes_per_token_per_ctx']:,.0f} | {r['ctx_kv_gt_ternary']:,.0f} | {r['ctx_kv_gt_head']:,.0f} | {r['ctx_kv_gt_all_weights']:,.0f} | {r['ctx_kv_plus_head_gt_ternary']:,.0f} | {r['ctx_ceiling_3x_memory_bound']:,.0f} | {r['ctx_nonternary_mem_gt_ternary_mem']:,.0f} | {r['ctx_ceiling_1p5x_memory_bound']:,.0f} | {r['max_ctx']:,} |")
    A("")
    A("Sensitivity of the 2x boundary context (ideal ceiling = 2x in the memory-bound regime):")
    A("")
    A("| Model | B | weights | head | KV | ctx where ideal ceiling = 2x | sweep confirms (first ctx < 2x @100 GB/s) |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for r in cx:
        if (r["batch"] in (1, 16)) and ((r["head_bytes_per_elem"], r["kv_bytes_per_elem"]) in ((2.0, 2.0), (1.0, 2.0), (2.0, 0.5), (0.5, 0.5))):
            A(f"| {r['model']} | {r['batch']} | {r['weight_bits']} | {r['head_bytes_per_elem']} | {r['kv_bytes_per_elem']} | {r['ctx_nonternary_mem_gt_ternary_mem']:,.0f} | {r['sweep_first_ctx_with_ideal_ceiling_below_2x_at_100GBps']} |")
    A("")
    A("## 6. Break-even: ternary stage compute vs weight bandwidth (batch 1, 2-bit)")
    A("")
    A("terms/s to saturate BW = BW × 4 × B; BW to feed an engine = rate/(4·B); lanes at BW and clock = BW·4·B/f; batch to balance = rate/(4·BW).")
    A("")
    A("| BW | terms/s to saturate (B=1) | lanes @1 GHz | lanes @250 MHz | lanes @2 GHz |"); A("|---:|---:|---:|---:|---:|")
    for bw in BWS:
        A(f"| {bw/1e9:.0f} GB/s | {bw*4:.3g} | {bw*4/1e9:,.0f} | {bw*4/250e6:,.0f} | {bw*4/2e9:,.0f} |")
    A("")
    A("| Engine (terms/cycle @1 GHz) | terms/s | BW to feed at B=1 | batch to balance at 100 GB/s | at 1 TB/s |"); A("|---:|---:|---:|---:|---:|")
    for tpc in TPCS:
        rate = tpc * 1e9
        A(f"| {tpc:,} | {rate:.3g} | {rate/4/1e9:,.1f} GB/s | {rate/(4*100e9):.2f} | {rate/(4*1e12):.3f} |")
    A("")
    A("These are model-independent for dense 2-bit weights (terms/byte = 4). Numerical sweep transitions agree with the analytical thresholds (see `break_even_thresholds.csv`, `batch_crossovers.csv`).")
    A("")
    A("## 7. Residency (100 GB/s, per token; external bytes after policy)")
    A("")
    A("'ideal ceiling' = external total / external non-ternary bytes after residency (memory-bound ternary-only ceiling). Residency of ternary bytes is itself a ternary optimization and lowers this ratio; residency of head/KV raises it.")
    A("")
    A("| Model | ctx | capacity | policy | ext ternary | ext head | ext KV | ext total | ideal ceiling (mem-bound) |"); A("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in res:
        show_ctx = (512, 4096) if r["model"] == "bitnet-2b4t" else (512, 32768)
        if r["capacity_MB"] in (0, 64, 256, 1000) and r["policy"] in ("none", "head_if_fits", "greedy_head_then_layers", "ideal_nonternary_first", "ideal_including_kv") and r["context"] in show_ctx:
            ceil = "∞ (no non-ternary external bytes)" if math.isinf(r["ideal_ceiling_memory_bound"]) else f"{r['ideal_ceiling_memory_bound']:.2f}x"
            A(f"| {r['model']} | {r['context']} | {r['capacity_MB']:.0f} MB | {r['policy']} | {MB(r['external_ternary_bytes'])} | {MB(r['external_head_bytes'])} | {MB(r['external_kv_bytes'])} | {MB(r['external_total_bytes'])} | {ceil} |")
    A("")
    A("Storage facts: " + "; ".join(f"{f['model']}: one layer {MB(f['one_layer_packed_bytes'])}, largest tensor {MB(f['largest_tensor_bytes'])}, smallest {MB(f['smallest_tensor_bytes'])}, head {MB(f['head_bytes_bf16'])}, full ternary {MB(f['packed_2bit_bytes'])}" for f in facts))
    A("")
    A("## 7b. System storage vs serious low-precision baselines (bytes per token, batch 1; INFERRED, bytes only)")
    A("")
    A("Same BF16 head, BF16 KV at the stated context, and norms in every row; only the ternary-layer representation changes. INT4/INT8/BF16 compute and energy are not modeled. This is the advantage already realized by 2-bit storage, as opposed to the ceiling on further ternary optimization.")
    A("")
    A("| Model | ctx | weights | system bytes/token | vs INT4-weight system | vs INT8 | vs BF16 |"); A("|---|---:|---:|---:|---:|---:|---:|")
    for r in ssc:
        if r["ternary_weight_bits"] in (2.0, 1.6):
            A(f"| {r['model']} | {r['context']} | {r['ternary_weight_bits']}-bit | {MB(r['system_bytes_per_token_batch1'])} | {r['ratio_vs_int4_weights_system']:.2f}x | {r['ratio_vs_int8_weights_system']:.2f}x | {r['ratio_vs_bf16_weights_system']:.2f}x |")
    A("")
    A("## 8. One-at-a-time sensitivity (ideal ceiling, optimistic; see sensitivity.csv for times)")
    A("")
    A("| Model | Scenario | Variation | Limiter | Ternary % time | Ideal ceiling | Bytes ratio |"); A("|---|---|---|---|---:|---:|---:|")
    for r in sens:
        A(f"| {r['model']} | {r['scenario'][:2]} | {r['variation']} | {r['limiting_component']} | {r['pct_ternary_time']:.0f} | {r['ceiling_ideal']:.2f}x | {r['memory_bound_bytes_ratio']:.2f} |")
    A("")
    A("## 9. Tiles and accumulators")
    A("")
    A("| K | tile | tiles | tail utilization | partial sums | divisible | models |"); A("|---:|---:|---:|---:|---:|---|---|")
    for t in tiles:
        A(f"| {t['K']} | {t['tile']} | {t['tiles']} | {t['tail_utilization']:.3f} | {t['partial_sums']} | {t['divisible']} | {t['models']} |")
    A("")
    A("| K | act bits | format sum max | format bits | model sum max | model bits |"); A("|---:|---:|---:|---:|---:|---:|")
    for a_ in acc:
        A(f"| {a_['K']} | {a_['act_bits']} | {a_['format_sum_max']:,} | {a_['format_bits']} | {a_['model_sum_max']:,} | {a_['model_bits']} |")
    A("")
    A("## 10. Prefill (250 GB/s, default engine; weights stream once per prompt)")
    A("")
    A("Causal attention compute ≈ nH·hd·L·P² overtakes ternary compute W·P at P* = W/(nH·hd·L).")
    A("")
    A("| Model | P | ternary bytes/token | ternary limiter | ternary compute | ternary memory | attention time | total opt | ternary % | P* |"); A("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in pf:
        A(f"| {r['model']} | {r['P']} | {MB(r['ternary_bytes_per_token'])} | {r['ternary_limiter']} | {r['ternary_compute_time']*1e3:.2f} ms | {r['ternary_memory_time']*1e3:.2f} ms | {r['attention_time']*1e3:.2f} ms | {r['total_optimistic']*1e3:.2f} ms | {100*r['ternary_fraction']:.0f}% | {r['P_attention_eq_ternary']:,.0f} |")
    A("")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
