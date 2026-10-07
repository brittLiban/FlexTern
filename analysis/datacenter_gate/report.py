"""
EXP-0007 driver: scenario matrix, targeted sweeps, crossover thresholds, dedicated 70B/400B sections,
primary founder table and summary.md.  Run from the repository root:
    python analysis/datacenter_gate/report.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, _HERE)
os.chdir(ROOT)

import dc_sources as src  # noqa: E402
import systems as sy  # noqa: E402
import workloads as wl  # noqa: E402

OUT = os.path.join("results", "datacenter_gate")
BATCHES = (1, 4, 16, 64, 128)
BATCHES_FINE = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CONTEXTS = (2048, 8192, 32768, 131072)
CONTEXTS_ALL = (128, 2048, 8192, 32768, 131072)
KEEP = ["model", "kind", "batch", "context", "phase", "system", "system_desc", "weight_bits", "kv_bits", "head_bits", "tile_cap", "serving_util",
        "static_weight_capacity", "gpus", "nodes", "tiles", "stages", "tp", "tp_k", "microbatch", "reuse", "bubble_fraction", "kv_location", "kv_fits",
        "dram_stacks_per_tile", "hbm_capacity_needed", "kv_capacity_per_seq", "weight_bytes", "local_weight_bytes", "kv_read", "kv_write", "head_bytes",
        "interconnect_bytes", "interconnect_tier", "partial_sum_bytes", "total_bytes", "flops", "terms", "step_s", "agg_tok_s", "seq_tok_s", "latency_s",
        "bottleneck", "energy_mj_min", "energy_mj_max"]


def rowify(r: dict) -> dict:
    out = {k: r.get(k, "") for k in KEEP}
    for k, v in out.items():
        if isinstance(v, float):
            out[k] = round(v, 6) if abs(v) < 1 else round(v, 3)
    return out


def write_csv(name: str, rows: list[dict]) -> None:
    os.makedirs(OUT, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def mb(b) -> str:
    b = float(b)
    return f"{b/1e9:,.2f} GB" if b >= 1e9 else (f"{b/1e6:,.1f} MB" if b >= 1e5 else f"{b/1e3:,.1f} kB")


def ratio_vs_a(rows_same_cell: dict[str, dict], key: str = "energy") -> dict[str, str]:
    a = rows_same_cell["A"]
    out = {}
    for s, r in rows_same_cell.items():
        if s == "A":
            out[s] = "1.00x"; continue
        if key == "energy":
            lo = a["energy_mj_min"] / r["energy_mj_max"] if r["energy_mj_max"] else float("inf")
            hi = a["energy_mj_max"] / r["energy_mj_min"] if r["energy_mj_min"] else float("inf")
            out[s] = f"{lo:.2f}-{hi:.2f}x"
        else:
            out[s] = f"{a['total_bytes'] / r['total_bytes']:.2f}x" if r["total_bytes"] else "inf"
    return out


def main() -> int:
    src.write(OUT)
    models = wl.all_models()
    m70v32 = wl.hypothetical("hyp-70B", vocab=32768); m400v32 = wl.hypothetical("hyp-400B", vocab=32768)

    # ---------------------------------------------------------------- scenario matrix (fp8 KV, fp4 head for A/D; candidate for B/C? no: same KV/head for all)
    matrix = []
    for m in models:
        for B in BATCHES:
            for ctx in CONTEXTS:
                if ctx > m.max_ctx:
                    continue
                for phase in ("prefill", "decode"):
                    for s in sy.SYSTEMS:
                        matrix.append(rowify(sy.scenario(m, B, ctx, phase, s, kv="fp8", head="fp4")))
    write_csv("scenario_matrix.csv", matrix)

    # ---------------------------------------------------------------- batch sweep (decode, 70B and 1B, fp8 KV; plus serving utilizations)
    bs = []
    for m in (models[0], models[3]):
        for ctx in (2048, 32768):
            for B in BATCHES_FINE:
                for u in src.SERVING_UTILS:
                    cell = {s: sy.scenario(m, B, ctx, "decode", s, kv="fp8", head="fp4", serving_util=u) for s in sy.SYSTEMS}
                    er = ratio_vs_a(cell); br = ratio_vs_a(cell, "bytes")
                    for s, r in cell.items():
                        bs.append({**rowify(r), "energy_ratio_A_over_this": er[s], "bytes_ratio_A_over_this": br[s]})
    write_csv("batch_sweep.csv", bs)

    # ---------------------------------------------------------------- context sweep (decode, batch 16, three KV precisions)
    cs = []
    for m in (models[0], models[3]):
        for ctx in CONTEXTS_ALL:
            for kv in ("bf16", "fp8", "candidate_4p5"):
                cell = {s: sy.scenario(m, 16, ctx, "decode", s, kv=kv, head="fp4") for s in sy.SYSTEMS}
                er = ratio_vs_a(cell)
                for s, r in cell.items():
                    a = cell["A"]
                    cs.append({**rowify(r), "kv_precision": kv, "kv_share_of_total_bytes": round((r["kv_read"] + r["kv_write"]) / r["total_bytes"], 3) if r["total_bytes"] else "",
                               "energy_ratio_A_over_this": er[s]})
    write_csv("context_sweep.csv", cs)
    write_csv("kv_sweep.csv", [r for r in cs if r["system"] in ("A", "C")])

    # ---------------------------------------------------------------- head sweep (decode, 70B/400B, vocab 32k vs 128k, head bf16 vs fp4 vs candidate)
    hs = []
    for m in (models[3], m70v32, models[4], m400v32):
        for head in ("bf16", "fp4", "candidate"):
            for B in (1, 16, 128):
                cell = {s: sy.scenario(m, B, 8192, "decode", s, kv="fp8", head=head) for s in sy.SYSTEMS}
                for s, r in cell.items():
                    hs.append({**rowify(r), "head_precision": head, "head_share_of_total_bytes": round(r["head_bytes"] / r["total_bytes"], 3) if r["total_bytes"] else ""})
    write_csv("head_sweep.csv", hs)

    # ---------------------------------------------------------------- model-scale and topology sweeps (decode, batch 16, ctx 8k)
    ms, ts = [], []
    for m in models:
        for s in sy.SYSTEMS:
            ms.append(rowify(sy.scenario(m, 16, 8192, "decode", s, kv="fp8", head="fp4")))
        for cap in src.TILE_CAPS:
            for s, wb in (("C", src.WEIGHT_BITS["ternary"]), ("D", src.WEIGHT_BITS["fp4"]), ("C-dense-target", src.WEIGHT_BITS["ternary_dense_target"])):
                t = sy.topology(m, wb, 4.5, cap)
                ts.append({"model": m.name, "kind": m.kind, "system": s, "weight_bits": wb, "tile_cap": cap, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()},
                           "static_MB": round((m.weight_bytes(wb) + m.head_bytes(4.5)) / 1e6, 1), "tiles_per_rack": src.TILES_PER_RACK, "racks": round(t["tiles"] / src.TILES_PER_RACK, 2),
                           "label": "ANALYTICAL placement (A7); 1.6-bit row is a TARGET, not implemented"})
    write_csv("model_scale_sweep.csv", ms); write_csv("topology_sweep.csv", ts)

    # ---------------------------------------------------------------- pipeline and interconnect detail (70B decode, ctx 8k, 512 MB tiles)
    pr, ir = [], []
    m = models[3]
    for B in BATCHES_FINE:
        for cap in src.TILE_CAPS:
            r = sy.scenario(m, B, 8192, "decode", "C", kv="fp8", head="fp4", cap=cap)
            pr.append({"model": m.name, "batch": B, "tile_cap": cap, "stages": r["stages"], "tiles": r["tiles"], "microbatch": r["microbatch"], "inflight": r["inflight_microbatches"],
                       "bubble_fraction": round(r["bubble_fraction"], 3), "t_micro_s": round(r["t_micro"], 6), "step_s": round(r["step_s"], 6),
                       "agg_tok_s": round(r["agg_tok_s"], 1), "seq_tok_s": round(r["seq_tok_s"], 2), "latency_ms_per_token": round(r["latency_s"] * 1e3, 3),
                       "bottleneck": r["bottleneck"], "t_local": round(r["t_local"], 6), "t_cmp_target": round(r["t_cmp"], 6), "t_kv": round(r["t_kv"], 6), "t_link": round(r["t_link"], 6)})
    for ctx in CONTEXTS:
        for s in ("A", "C"):
            r = sy.scenario(m, 16, ctx, "decode", s, kv="fp8", head="fp4")
            ir.append({"model": m.name, "context": ctx, "system": s, "stages": r["stages"], "tp_or_k": r.get("tp", r.get("tp_k")), "interconnect_bytes_per_seq_token": round(r["interconnect_bytes"]),
                       "partial_sum_bytes": round(r["partial_sum_bytes"]), "tier": r["interconnect_tier"], "interconnect_GBps_at_batch16_1000tok_s": round(r["interconnect_bytes"] * 16 * 1000 / 1e9, 3),
                       "activation_share_of_total_bytes": round(r["interconnect_bytes"] / r["total_bytes"], 4), "label": "ANALYTICAL"})
    # pipeline mode: pipelined (reuse = microbatch, no bubble) vs batch-sequential (reuse = B, bubble = 1 - 1/stages)
    pm = []
    for ctx in (2048, 8192, 32768):
        for B in (16, 64, 128):
            a = sy.scenario(m, B, ctx, "decode", "A", kv="fp8", head="fp4")
            for mode in ("pipelined", "batch_sequential"):
                for s in ("C", "D"):
                    r = sy.scenario(m, B, ctx, "decode", s, kv="fp8", head="fp4", mode=mode)
                    pm.append({"model": m.name, "batch": B, "context": ctx, "system": s, "mode": mode, "stages": r["stages"], "microbatch": r["microbatch"], "bubble_fraction": round(r["bubble_fraction"], 3),
                               "local_weight_bytes_per_seq_token": round(r["local_weight_bytes"]), "energy_mj_min": round(r["energy_mj_min"], 3), "energy_mj_max": round(r["energy_mj_max"], 3),
                               "A_energy_mj_min": round(a["energy_mj_min"], 3), "A_energy_mj_max": round(a["energy_mj_max"], 3),
                               "A_over_this_conservative": round(a["energy_mj_min"] / r["energy_mj_max"], 3), "agg_tok_s_bound": round(r["agg_tok_s"], 1), "A_agg_tok_s_bound": round(a["agg_tok_s"], 1),
                               "latency_ms_per_token": round(r["latency_s"] * 1e3, 2), "label": "ANALYTICAL; the two modes trade weight reuse against stage utilization"})
    write_csv("pipeline_mode_sweep.csv", pm)
    write_csv("pipeline_results.csv", pr); write_csv("interconnect_results.csv", ir)

    # ---------------------------------------------------------------- capacity and traffic comparison
    cc, tc = [], []
    for m in models:
        for s in sy.SYSTEMS:
            r = sy.scenario(m, 16, 8192, "decode", s, kv="fp8", head="fp4")
            cc.append({"model": m.name, "system": s, "weight_bits": r["weight_bits"], "static_capacity_bytes": round(r["static_weight_capacity"]), "static_capacity_GB": round(r["static_weight_capacity"] / 1e9, 3),
                       "hbm_or_sram_units": r.get("gpus", r.get("tiles")), "unit": "GPUs (180 GB HBM)" if s in ("A", "B") else "tiles (512 MB SRAM)", "kv_per_seq_8k_fp8_GB": round(r["kv_capacity_per_seq"] / 1e9, 3),
                       "kv_location": r["kv_location"], "label": "ANALYTICAL"})
            tc.append({"model": m.name, "system": s, "batch": 16, "context": 8192, "phase": "decode", "global_weight_bytes": round(r["weight_bytes"]), "local_weight_bytes": round(r["local_weight_bytes"]),
                       "kv_bytes": round(r["kv_read"] + r["kv_write"]), "head_bytes": round(r["head_bytes"]), "interconnect_bytes": round(r["interconnect_bytes"]), "partial_sum_bytes": round(r["partial_sum_bytes"]),
                       "total_bytes": round(r["total_bytes"]), "energy_mj_min": round(r["energy_mj_min"], 4), "energy_mj_max": round(r["energy_mj_max"], 4)})
    write_csv("capacity_comparison.csv", cc); write_csv("traffic_comparison.csv", tc)

    # ---------------------------------------------------------------- crossover thresholds
    xo = []
    for m in (models[0], models[2], models[3], models[4]):
        for ctx in CONTEXTS:
            for thr in (4, 3, 2, 1.5, 1.2):
                found = None
                for B in BATCHES_FINE:
                    a = sy.scenario(m, B, ctx, "decode", "A", kv="fp8", head="fp4"); c = sy.scenario(m, B, ctx, "decode", "C", kv="fp8", head="fp4")
                    adv = a["energy_mj_min"] / c["energy_mj_max"]            # conservative advantage (A low end vs C high end)
                    if adv < thr:
                        found = B; break
                xo.append({"threshold": f"batch at which C's conservative movement-energy advantage over A falls below {thr}x", "model": m.name, "context": ctx, "kv": "fp8", "value": found if found else f">{BATCHES_FINE[-1]}",
                           "note": "A at the low end of its energy range, C at the high end; compute energy excluded"})
            for B in (1, 16, 128):
                a = sy.scenario(m, B, ctx, "decode", "A", kv="fp8", head="fp4")
                xo.append({"threshold": "KV share of System A bytes per sequence-token", "model": m.name, "context": ctx, "kv": "fp8", "value": round((a["kv_read"] + a["kv_write"]) / a["total_bytes"], 3), "note": f"batch {B}"})
        for kv in ("bf16", "fp8", "candidate_4p5"):
            first = None
            for ctx in CONTEXTS_ALL:
                a = sy.scenario(m, 16, ctx, "decode", "A", kv=kv, head="fp4")
                if (a["kv_read"] + a["kv_write"]) / a["total_bytes"] > 0.5:
                    first = ctx; break
            xo.append({"threshold": "context at which KV exceeds 50% of System A bytes (batch 16)", "model": m.name, "context": "", "kv": kv, "value": first if first else ">131072", "note": ""})
        for cap in src.TILE_CAPS:
            tc_ = sy.topology(m, 2.0, 4.5, cap)["tiles"]; td = sy.topology(m, 4.5, 4.5, cap)["tiles"]
            xo.append({"threshold": "tile count ratio INT4 / ternary", "model": m.name, "context": "", "kv": "", "value": round(td / tc_, 2), "note": f"tile cap {cap/1e6:.0f} MB: {tc_} vs {td} tiles"})
        # interconnect pJ/bit at which C's activation energy equals its weight-energy saving vs A (batch 16, 8k)
        a = sy.scenario(m, 16, 8192, "decode", "A", kv="fp8", head="fp4"); c = sy.scenario(m, 16, 8192, "decode", "C", kv="fp8", head="fp4")
        saving_mj = (a["weight_bytes"] + a["head_bytes"]) * 8 * src.PJ_PER_BIT["hbm"][0] / 1e9 - (c["local_weight_bytes"] + c["head_bytes"]) * 8 * src.SRAM_PLANNING_PJ / 1e9
        pj_break = saving_mj * 1e9 / (c["interconnect_bytes"] * 8) if c["interconnect_bytes"] else float("inf")
        xo.append({"threshold": "interconnect pJ/bit at which C's activation movement equals its weight-energy saving vs A (batch 16, ctx 8k)", "model": m.name, "context": 8192, "kv": "fp8",
                   "value": round(pj_break, 2) if pj_break != float("inf") else "n/a (single stage)", "note": f"C interconnect {c['interconnect_bytes']:.0f} B/seq-token over {c['stages']} stages; tiers 1.3 (package) / 3-6 (board) / 3-8 (rack)"})
        # model scale where HBM capacity dominates: GPUs needed by capacity vs by bandwidth at 1000 agg tok/s, batch 64, 8k
        a = sy.scenario(m, 64, 8192, "decode", "A", kv="fp8", head="fp4")
        bw_gpus = (a["weight_bytes"] * a["reuse"] + a["head_bytes"] * a["reuse"] + 64 * (a["kv_read"] + a["kv_write"])) * (1000 / 64) / (src.GPU_HBM_PEAK_BPS * src.UTIL_HBM)
        xo.append({"threshold": "System A GPUs needed: by HBM capacity vs by HBM bandwidth at 1,000 aggregate tok/s (batch 64, ctx 8k)", "model": m.name, "context": 8192, "kv": "fp8",
                   "value": f"capacity {a['gpus']} vs bandwidth {bw_gpus:.2f}", "note": "capacity dominates when the first exceeds the second"})
    write_csv("crossover_thresholds.csv", xo)

    # ---------------------------------------------------------------- energy ranges and economic proxies
    en, ec = [], []
    for m in models:
        for B in (1, 16, 128):
            for ctx in (2048, 32768):
                cell = {s: sy.scenario(m, B, ctx, "decode", s, kv="fp8", head="fp4") for s in sy.SYSTEMS}
                er = ratio_vs_a(cell)
                for s, r in cell.items():
                    en.append({"model": m.name, "batch": B, "context": ctx, "system": s, "energy_mj_min": round(r["energy_mj_min"], 4), "energy_mj_max": round(r["energy_mj_max"], 4),
                               "A_over_this": er[s], "terms_per_seq_token": round(r["terms"]), "flops_per_seq_token": round(r["flops"]), "label": "INFERRED movement only; compute excluded"})
                    ec.append({"model": m.name, "batch": B, "context": ctx, "system": s, "gpus_or_tiles": r.get("gpus", r.get("tiles")), "hbm_stacks": (r.get("gpus", 0) * 8) if s in ("A", "B") else (r.get("tiles", 0) * r.get("dram_stacks_per_tile", 0)),
                               "sram_GB": 0 if s in ("A", "B") else round(r["tiles"] * r["tile_cap"] / 1e9, 2), "racks_proxy": round((r.get("gpus", 0) / 72) if s in ("A", "B") else r.get("tiles", 0) / src.TILES_PER_RACK, 3),
                               "agg_tok_s_bound": round(r["agg_tok_s"], 1), "tok_s_per_unit": round(r["agg_tok_s"] / r.get("gpus", r.get("tiles", 1)), 1), "latency_ms_per_token": round(r["latency_s"] * 1e3, 2),
                               "label": "PROXY; no prices; throughput is a bound from peak x utilization, not achieved"})
    write_csv("energy_ranges.csv", en); write_csv("economic_proxies.csv", ec)

    # ---------------------------------------------------------------- primary founder table
    prim = []
    picks = [(models[0], 1, 2048), (models[0], 16, 2048), (models[1], 16, 8192), (models[2], 16, 8192), (models[3], 1, 2048), (models[3], 16, 2048), (models[3], 64, 8192), (models[3], 128, 32768),
             (models[3], 16, 131072), (models[4], 16, 8192), (models[4], 128, 32768)]
    for m, B, ctx in picks:
        for phase in ("decode", "prefill"):
            if phase == "prefill" and ctx != 8192 and m.name not in ("hyp-70B",):
                continue
            cell = {s: sy.scenario(m, B, ctx, phase, s, kv="fp8", head="fp4") for s in sy.SYSTEMS}
            er = ratio_vs_a(cell); br = ratio_vs_a(cell, "bytes")
            for s, r in cell.items():
                prim.append({"model": m.name, "batch": B, "context": ctx, "phase": phase, "system": s, "weight_bits": r["weight_bits"], "kv_bits": r["kv_bits"],
                             "static_capacity": mb(r["static_weight_capacity"]), "hbm_bytes_per_tok": mb((r["weight_bytes"] + r["head_bytes"] if s in ("A", "B") else 0) + (r["kv_read"] + r["kv_write"] if r["kv_location"] in ("hbm", "dram") else 0)),
                             "local_bytes_per_tok": mb(r["local_weight_bytes"] + (r["head_bytes"] if s in ("C", "D") else 0) + (r["kv_read"] + r["kv_write"] if r["kv_location"] == "sram" else 0)), "kv_bytes_per_tok": mb(r["kv_read"] + r["kv_write"]),
                             "head_bytes_per_tok": mb(r["head_bytes"]), "interconnect_bytes_per_tok": mb(r["interconnect_bytes"]), "units": f"{r.get('gpus', r.get('tiles'))} {'GPUs' if s in ('A','B') else 'tiles'}",
                             "stages": r["stages"], "microbatch": r["microbatch"], "agg_tok_s_bound": round(r["agg_tok_s"]), "latency_ms_bound": round(r["latency_s"] * 1e3, 2),
                             "dominant_bottleneck": r["bottleneck"], "kv_location": r["kv_location"], "bytes_ratio_A_over_this": br[s], "energy_ratio_A_over_this": er[s],
                             "evidence": "MEASURED shapes" if m.kind.startswith("canonical") else "HYPOTHESIZED shapes", "label": "ANALYTICAL bounds; INFERRED energy; no achieved performance"})
    write_csv("primary_comparison.csv", prim)

    env = {"date": date.today().isoformat(), "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), "python": sys.version.split()[0]}
    json.dump(env, open(os.path.join(OUT, "environment.json"), "w", encoding="utf-8"), indent=1)

    # MLPerf sanity anchor: 70B, batch 128, ctx ~1.3k (1k in + 300 out), fp8 KV, 8 GPUs
    a_anchor = sy.scenario(models[3], 128, 1300, "decode", "A", kv="fp8", head="fp4")
    anchor = {"mlperf_8xB200_offline_tok_s": src.MLPERF_8xB200_OFFLINE_TOK_S, "model_A_bound_tok_s_batch128_ctx1300": round(a_anchor["agg_tok_s"]), "gpus_in_model": a_anchor["gpus"],
              "ratio_model_over_measured": round(a_anchor["agg_tok_s"] * (8 / a_anchor["gpus"]) / src.MLPERF_8xB200_OFFLINE_TOK_S, 2), "note": "sanity only (D04); model is a bound from peak x utilization, measured is FP4 Llama 2 70B offline"}
    json.dump(anchor, open(os.path.join(OUT, "mlperf_anchor.json"), "w", encoding="utf-8"), indent=1)
    summary(models, m70v32, m400v32, prim, bs, cs, xo, pr, ts, env, anchor, pm)
    print("datacenter_gate results written")
    return 0


def summary(models, m70v32, m400v32, prim, bs, cs, xo, pr, ts, env, anchor, pm) -> None:
    L = [f"# EXP-0007 — Datacenter thesis falsification (summary)", "",
         f"Generated by `analysis/datacenter_gate/report.py` at commit {env['commit']} on {env['date']}. Systems: A conventional FP4 (4.5-bit NVFP4) HBM node with tensor parallelism, native FP8 KV; B ternary streaming on the same node; C ternary resident SRAM tiles; D INT4 resident tiles (same topology). Every throughput is a BOUND from peak × utilization (A2), every energy an INFERRED movement-only range (A4; compute excluded), every 7B/70B/400B property HYPOTHESIZED (A8). Decimal units.", "",
         "Reading the energy ratios: each 'A/this' cell is a range (A low ÷ this high to A high ÷ this low). The large upper ends come from the 0.16 pJ/bit SRAM figure, which is Horowitz's 8 KB-array value and does not apply to multi-MB tiles (EXP-0006 review); the CONSERVATIVE end (A low ÷ this high, SRAM at the 1.6 pJ/bit planning figure) is the one used in every verdict and crossover. System A's GPU count is capacity-minimal; an interactive deployment would use tensor parallelism of 8 for latency, which lowers A's step time roughly 8× at small batch at negligible energy cost. Resident tiles are charged one HBM3E-class stack each for KV, which is over-provisioned for the 100 MB–few GB of KV per stage (an LPDDR-class device would do); this affects only the economic proxies.", "",
         f"MLPerf sanity anchor (D04): measured 8×B200 Llama 2 70B offline {anchor['mlperf_8xB200_offline_tok_s']:,} tok/s; this model's System A bound at batch 128, ~1.3k context: {anchor['model_A_bound_tok_s_batch128_ctx1300']:,} tok/s on {anchor['gpus_in_model']} GPUs (ratio {anchor['ratio_model_over_measured']}× per 8 GPUs). Used only to check the baseline is not undersized.", "",
         "## Primary founder table (decode unless stated; fp8 KV; 512 MB tiles)", "",
         "| model | B | ctx | phase | sys | w bits | static cap | HBM/DRAM B/tok | local B/tok | KV B/tok | head B/tok | interconnect B/tok | units | stages | µbatch | agg tok/s bound | latency ms | bottleneck | KV loc | bytes A/this | energy A/this |",
         "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|---:|---:|"]
    for r in prim:
        L.append(f"| {r['model']} | {r['batch']} | {r['context']} | {r['phase']} | {r['system']} | {r['weight_bits']} | {r['static_capacity']} | {r['hbm_bytes_per_tok']} | {r['local_bytes_per_tok']} | {r['kv_bytes_per_tok']} | {r['head_bytes_per_tok']} | {r['interconnect_bytes_per_tok']} | {r['units']} | {r['stages']} | {r['microbatch']} | {r['agg_tok_s_bound']:,} | {r['latency_ms_bound']} | {r['dominant_bottleneck']} | {r['kv_location']} | {r['bytes_ratio_A_over_this']} | {r['energy_ratio_A_over_this']} |")
    L += ["", "## Batch sweep, decode, serving utilization 100% (ideal) — energy ratio A ÷ system (movement only)", "", "| model | ctx | B | A mJ/tok | B mJ/tok | C mJ/tok | D mJ/tok | A/B | A/C | A/D | C µbatch | C KV loc |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for m in ("falcon-e-1b", "hyp-70B"):
        for ctx in (2048, 32768):
            for B in BATCHES_FINE:
                cell = {r["system"]: r for r in bs if r["model"] == m and r["context"] == ctx and r["batch"] == B and r["serving_util"] == 1.0}
                if len(cell) < 4:
                    continue
                f = lambda s: f"{cell[s]['energy_mj_min']:.2f}–{cell[s]['energy_mj_max']:.2f}"
                L.append(f"| {m} | {ctx} | {B} | {f('A')} | {f('B')} | {f('C')} | {f('D')} | {cell['B']['energy_ratio_A_over_this']} | {cell['C']['energy_ratio_A_over_this']} | {cell['D']['energy_ratio_A_over_this']} | {cell['C']['microbatch']} | {cell['C']['kv_location']} |")
    L += ["", "## Context sweep, decode, batch 16 — KV share of per-sequence bytes and energy ratio A ÷ C", "", "| model | ctx | KV | A total B/tok | A KV share | C total B/tok | C KV share | C KV loc | A/C energy |", "|---|---:|---|---:|---:|---:|---:|---|---:|"]
    for m in ("falcon-e-1b", "hyp-70B"):
        for ctx in CONTEXTS_ALL:
            for kv in ("bf16", "fp8", "candidate_4p5"):
                a = next((r for r in cs if r["model"] == m and r["context"] == ctx and r["kv_precision"] == kv and r["system"] == "A"), None)
                c = next((r for r in cs if r["model"] == m and r["context"] == ctx and r["kv_precision"] == kv and r["system"] == "C"), None)
                if a and c:
                    L.append(f"| {m} | {ctx} | {kv} | {mb(a['total_bytes'])} | {a['kv_share_of_total_bytes']} | {mb(c['total_bytes'])} | {c['kv_share_of_total_bytes']} | {c['kv_location']} | {c['energy_ratio_A_over_this']} |")
    L += ["", "## Topology (tiles by capacity; 1.6-bit row is a TARGET)", "", "| model | system | w bits | tile cap | layer bytes | layers/tile | TP split k | stages | weight tiles | head tiles | tiles | racks (128/rack) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ts:
        if r["model"] in ("hyp-7B", "hyp-70B", "hyp-400B") and r["tile_cap"] in (512e6, 2e9):
            L.append(f"| {r['model']} | {r['system']} | {r['weight_bits']} | {mb(r['tile_cap'])} | {mb(r['layer_bytes'])} | {r['layers_per_tile']} | {r['tp_k']} | {r['stages']} | {r['weight_tiles']} | {r['head_tiles']} | {r['tiles']} | {r['racks']} |")
    L += ["", "## Pipeline, 70B decode, ctx 8k, System C", "", "| B | tile cap | stages | tiles | µbatch | in flight | bubble | t_micro s | agg tok/s bound | per-seq tok/s | latency ms/token | bottleneck |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in pr:
        if r["tile_cap"] in (512e6, 2e9):
            L.append(f"| {r['batch']} | {mb(r['tile_cap'])} | {r['stages']} | {r['tiles']} | {r['microbatch']} | {r['inflight']} | {r['bubble_fraction']} | {r['t_micro_s']} | {r['agg_tok_s']:,} | {r['seq_tok_s']} | {r['latency_ms_per_token']} | {r['bottleneck']} |")
    L += ["", "## Pipeline mode, 70B decode, 512 MB tiles: reuse versus utilization", "", "| B | ctx | sys | mode | stages | µbatch | bubble | local B/seq-tok | energy mJ/tok | A mJ/tok | A/this (conservative) | agg tok/s bound | A agg tok/s bound | latency ms |", "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in pm:
        L.append(f"| {r['batch']} | {r['context']} | {r['system']} | {r['mode']} | {r['stages']} | {r['microbatch']} | {r['bubble_fraction']} | {mb(r['local_weight_bytes_per_seq_token'])} | {r['energy_mj_min']}–{r['energy_mj_max']} | {r['A_energy_mj_min']}–{r['A_energy_mj_max']} | {r['A_over_this_conservative']} | {r['agg_tok_s_bound']:,} | {r['A_agg_tok_s_bound']:,} | {r['latency_ms_per_token']} |")
    L += ["", "## Crossover thresholds", "", "| threshold | model | ctx | KV | value | note |", "|---|---|---:|---|---:|---|"]
    for r in xo:
        L.append(f"| {r['threshold']} | {r['model']} | {r['context']} | {r['kv']} | {r['value']} | {r['note']} |")
    L += ["", "Full tables: `scenario_matrix.csv` (all cells), `batch_sweep.csv` (with serving utilizations), `context_sweep.csv`, `kv_sweep.csv`, `head_sweep.csv`, `model_scale_sweep.csv`, `topology_sweep.csv`, `pipeline_results.csv`, `interconnect_results.csv`, `capacity_comparison.csv`, `traffic_comparison.csv`, `crossover_thresholds.csv`, `energy_ranges.csv`, `economic_proxies.csv`, `primary_comparison.csv`, `mlperf_anchor.json`."]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
