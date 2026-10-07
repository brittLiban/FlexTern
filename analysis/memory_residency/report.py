"""
EXP-0006 driver: runs every model, writes results/memory_residency/*.csv, sensitivity thresholds,
the EXP-0003/0004/0005 reconnection, the primary comparison table and summary.md.
Run from the repository root: python analysis/memory_residency/report.py
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
sys.path.insert(0, os.path.join(ROOT, "analysis", "hardware_gate"))
sys.path.insert(0, _HERE)
os.chdir(ROOT)

import capacity as cap  # noqa: E402
import physical as phy  # noqa: E402
import residency as res  # noqa: E402
import sources as src  # noqa: E402
from roofline_models import Assumptions, token_model  # noqa: E402
from traffic_model import per_token  # noqa: E402

OUT = os.path.join("results", "memory_residency")


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
    return f"{float(b) / 1e6:,.1f} MB"


def sensitivity_rows(ws, points) -> list[dict]:
    lo, hi = src.DENSITY_TIERS_MB_PER_MM2[src.CHIP_TIER]; clo, chi = src.DENSITY_TIERS_MB_PER_MM2[src.CACHE_DIE_TIER]
    rows = []
    for p in points:
        W = p["transformer_bytes"]
        for m in src.MARGINS:
            phys = W / (1 - m)
            rows.append({"threshold": "minimum chip-level density (MB/mm2) for one reticle", "model": p["model"], "weight_bits": p["weight_bits"], "margin": m,
                         "value": round(phys / 1e6 / src.RETICLE_MM2, 3), "note": f"chip tier range {lo}-{hi}; cache-die tier {clo}-{chi}; passes at chip tier: {'yes' if phys/1e6/src.RETICLE_MM2 <= lo else ('top of range only' if phys/1e6/src.RETICLE_MM2 <= hi else 'no')}"})
    for slug, w in ws.items():
        for bits in (2, 4):
            W = w.W * bits // 8
            for nb in res.BANK_COUNTS:
                rows.append({"threshold": "minimum per-bank local bandwidth (GB/s) for 100 tok/s at 70% utilization", "model": slug, "weight_bits": bits, "margin": "",
                             "value": round(W * 100 / nb / 0.7 / 1e9, 3), "note": f"NB={nb}; compare UPMEM-class ~1 GB/s per 64 MB bank (S20), V-Cache 2 TB/s per die (S03)"})
            lim = res.streaming_limits(w, bits)
            rows.append({"threshold": "max tok/s streaming from one 64-bit LPDDR5X channel (70% util)", "model": slug, "weight_bits": bits, "margin": "", "value": lim["lpddr5x_64bit_max_tok_s"], "note": "S13"})
            rows.append({"threshold": "max tok/s streaming from one HBM3E stack (70% util)", "model": slug, "weight_bits": bits, "margin": "", "value": lim["hbm3e_one_stack_max_tok_s"], "note": "S12"})
        ts = res.tensors(w)
        kmin = min(t["K"] for t in ts)
        rows.append({"threshold": "max bank count before the collector bound hits the smallest K (b=16)", "model": slug, "weight_bits": 2, "margin": "",
                     "value": kmin // 16, "note": f"K_min={kmin}; NB*b > K makes the K={kmin} tensors collector-bound (EXP-0005)"})
        br = res.bank_rows(w, 2)
        first_low = next((r["banks"] for r in br if r["utilization"] < 0.8), None)
        rows.append({"threshold": "bank count at which token-weighted utilization falls below 0.8 (b=16)", "model": slug, "weight_bits": 2, "margin": "",
                     "value": first_low if first_low else f">{res.BANK_COUNTS[-1]}", "note": "collector-bound cycle law"})
    for bits in (2, 4):
        for tier, (tlo, thi) in ((src.CHIP_TIER, (lo, hi)), (src.CACHE_DIE_TIER, (clo, chi))):
            cap_lo = src.RETICLE_MM2 * tlo * 1e6 * (1 - src.NOMINAL_MARGIN); cap_hi = src.RETICLE_MM2 * thi * 1e6 * (1 - src.NOMINAL_MARGIN)
            rows.append({"threshold": f"largest ternary-fraction parameter count fitting one reticle ({tier.split(' ')[0]})", "model": "any", "weight_bits": bits, "margin": src.NOMINAL_MARGIN,
                         "value": f"{cap_lo*8/bits/1e9:.2f}-{cap_hi*8/bits/1e9:.2f} B params", "note": f"capacity {cap_lo/1e6:.0f}-{cap_hi/1e6:.0f} MB of weights at {bits} bits"})
    return rows


def reconnect_rows(ws) -> list[dict]:
    rows = []
    for slug, w in ws.items():
        tr = per_token(w, nb=64)
        for ctx in (128, 2048, 8192, 32768):
            if ctx > w.max_ctx:
                continue
            for label, a in (("BF16 head + BF16 KV", Assumptions(bw=100e9, batch=1, context=ctx)),
                             ("EXP-0004 candidate head 4.156 b + KV 4.5 b", Assumptions(bw=100e9, batch=1, context=ctx, head_bytes_per_elem=src.HEAD_CANDIDATE_BITS / 8, kv_bytes_per_elem=src.KV_CANDIDATE_BITS / 8))):
                tm = token_model(w, a)
                head = tm.stage("head").memory_bytes; kv = tm.stage("attention_kv").memory_bytes; tern = tm.stage("ternary").memory_bytes
                act_out = tr["nm_B_global_per_token"]
                g_stream = tern + head + kv; g_res = act_out + head + kv
                for tok in (10, 100):
                    rows.append({"model": slug, "context": ctx, "scenario": label, "target_tok_s": tok,
                                 "global_ternary_bytes_streaming": int(tern), "local_ternary_bytes_resident": int(tr["nm_C_local_reads"]), "global_ternary_bytes_resident": 0,
                                 "head_bytes": int(head), "kv_bytes": int(kv), "activation_plus_output_bytes": act_out,
                                 "resident_static_capacity_bytes": int(tr["packed_weight_bytes"]),
                                 "global_total_streaming": int(g_stream), "global_total_resident": int(g_res),
                                 "global_GBps_streaming": round(g_stream * tok / 1e9, 2), "global_GBps_resident": round(g_res * tok / 1e9, 2),
                                 "local_GBps_resident": round(tr["nm_C_local_reads"] * tok / 1e9, 2),
                                 "dominant_global_term_resident": max((("head", head), ("kv", kv), ("act+out", act_out)), key=lambda x: x[1])[0],
                                 "head_plus_kv_share_of_resident_global": round((head + kv) / g_res, 3),
                                 "label": "ANALYTICAL; byte reductions are not token/s"})
    return rows


def primary_rows(ws, bank_rows_all, tier_rows, area_rows) -> list[dict]:
    rows = []
    for slug, w in ws.items():
        for scen, bits, resident in (("TERNARY STREAMING", 2, False), ("TERNARY RESIDENT", 2, True), ("INT4 RESIDENT", 4, True), ("INT4 STREAMING (reference)", 4, False)):
            W = w.W * bits // 8; tr = per_token(w, nb=64, weight_bits=bits)
            tier = next(t for t in tier_rows if t["model"] == slug and t["weight_bits"] == bits)
            b64 = next(b for b in bank_rows_all if b["model"] == slug and b["weight_bits"] == bits and b["banks"] == 64)
            lim = res.streaming_limits(w, bits)
            tmc = token_model(w, Assumptions(bw=100e9, batch=1, context=2048, head_bytes_per_elem=src.HEAD_CANDIDATE_BITS / 8, kv_bytes_per_elem=src.KV_CANDIDATE_BITS / 8))
            tmb = token_model(w, Assumptions(bw=100e9, batch=1, context=2048))
            rows.append({"model": slug, "scenario": scen, "weight_bits": bits,
                         "resident_capacity_MB": round(W / 1e6, 1) if resident else 0, "physical_MB_10pct_margin": round(W / 0.9 / 1e6, 1) if resident else 0,
                         "global_weight_bytes_per_token": 0 if resident else W, "local_weight_bytes_per_token": W if resident else 0,
                         "external_GBps_at_100tok_s_70pct": 0 if resident else round(W * 100 / 0.7 / 1e9, 1),
                         "local_GBps_at_100tok_s_70pct": round(W * 100 / 0.7 / 1e9, 1) if resident else 0,
                         "max_tok_s_one_lpddr5x_64bit": "" if resident else lim["lpddr5x_64bit_max_tok_s"],
                         "banks": 64 if resident else "", "capacity_per_bank_MB": b64["capacity_per_bank_MB"] if resident else "",
                         "utilization_64_banks": b64["utilization"] if resident else "",
                         "activation_delivered_MB_per_token": round(tr["nm_D_activation_delivered"] / 1e6, 2) if resident else round(tr["stream_D_activation_delivered"] / 1e6, 2),
                         "output_MB_per_token": round(tr["nm_D_outputs"] / 1e6, 2),
                         "memory_area_mm2_chip_tier": f"{tier['area_mm2_chip_tier_min']}-{tier['area_mm2_chip_tier_max']}" if resident else "n/a",
                         "memory_area_mm2_cache_die_tier": f"{tier['area_mm2_cache_die_tier_min']}-{tier['area_mm2_cache_die_tier_max']}" if resident else "n/a",
                         "memory_organization_class": tier["memory_organization_class"] if resident else "external DRAM (LPDDR5X / HBM3E)",
                         "head_plus_kv_global_MB_per_token_2k_bf16": round((tmb.stage("head").memory_bytes + tmb.stage("attention_kv").memory_bytes) / 1e6, 1),
                         "head_plus_kv_global_MB_per_token_2k_candidate": round((tmc.stage("head").memory_bytes + tmc.stage("attention_kv").memory_bytes) / 1e6, 1),
                         "major_unresolved_physical_issue": ("whether the memory tier is economical; SRAM energy at modern nodes UNMEASURED; compute area UNMEASURED" if resident else "external interface energy (5-20 pJ/bit class) and 1-2 channels of LPDDR5X per 100-250 tok/s"),
                         "label": "MEASURED counts; ANALYTICAL traffic; INFERRED area; no speedup"})
    return rows


def main() -> int:
    src.write(OUT)
    ws = cap.load()
    caprows = cap.capacity_rows(ws); write_csv("capacity_comparison.csv", caprows)
    hyp = cap.hypothetical_rows(); write_csv("model_scaling.csv", hyp)
    points = cap.all_model_points(ws)
    pol = [r for w in ws.values() for bits in (2, 4) for r in res.policy_rows(w, bits)]; write_csv("residency_policies.csv", pol)
    banks = [r for w in ws.values() for bits in (2, 4) for r in res.bank_rows(w, bits)]; write_csv("bank_sweep.csv", banks)
    bw = [r for w in ws.values() for bits in (2, 4) for r in res.bandwidth_rows(w, bits)]; write_csv("bandwidth_requirements.csv", bw)
    lims = [res.streaming_limits(w, bits) for w in ws.values() for bits in (2, 4)]; write_csv("streaming_limits.csv", lims)
    area = phy.area_rows(points); write_csv("area_ranges.csv", area)
    tiers = phy.tier_crossing_rows(points); write_csv("technology_comparison.csv", tiers)
    ao = {slug: per_token(w, nb=64)["nm_B_global_per_token"] for slug, w in ws.items()}
    en = phy.energy_rows(points, ao); write_csv("energy_ranges.csv", en)
    pk = phy.package_rows(points); write_csv("package_options.csv", pk)
    ec = phy.economics_rows(points); write_csv("economics_proxy.csv", ec)
    sens = sensitivity_rows(ws, points); write_csv("sensitivity.csv", sens)
    rec = reconnect_rows(ws); write_csv("system_reconnect.csv", rec)
    prim = primary_rows(ws, banks, tiers, area); write_csv("primary_comparison.csv", prim)
    i4 = []
    for slug, w in ws.items():
        t2 = next(t for t in tiers if t["model"] == slug and t["weight_bits"] == 2); t4 = next(t for t in tiers if t["model"] == slug and t["weight_bits"] == 4)
        b2 = next(b for b in banks if b["model"] == slug and b["weight_bits"] == 2 and b["banks"] == 64); b4 = next(b for b in banks if b["model"] == slug and b["weight_bits"] == 4 and b["banks"] == 64)
        i4.append({"model": slug, "capacity_ternary_MB": t2["physical_MB"], "capacity_int4_MB": t4["physical_MB"], "capacity_ratio": round(t4["physical_MB"] / t2["physical_MB"], 2),
                   "local_bw_ratio_int4_over_ternary": 2.0, "cells_per_term_ternary_vs_int4_exp0005": "99.5 vs 271.6 (2.7x) at 256 terms/cycle",
                   "bank_capacity_64_banks_ternary_MB": b2["capacity_per_bank_MB"], "bank_capacity_64_banks_int4_MB": b4["capacity_per_bank_MB"],
                   "vcache_dies_ternary": t2["vcache_class_dies_41mm2"], "vcache_dies_int4": t4["vcache_class_dies_41mm2"],
                   "class_ternary": t2["memory_organization_class"], "class_int4": t4["memory_organization_class"],
                   "tier_crossing": "yes" if t2["memory_organization_class"] != t4["memory_organization_class"] else "no (same class)",
                   "global_traffic_benefit_of_near_memory": "identical for both formats (EXP-0005)", "label": "same architecture, same assumptions; INFERRED areas"})
    write_csv("int4_comparison.csv", i4)
    env = {"date": date.today().isoformat(), "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "python": sys.version.split()[0], "inputs": "results/workload_profile/* via roofline_models.Workload; analysis/hardware_gate traffic_model/workload_mapping", "sources": "results/memory_residency/sources.csv + assumptions.csv"}
    json.dump(env, open(os.path.join(OUT, "environment.json"), "w", encoding="utf-8"), indent=1)
    summary(ws, caprows, hyp, pol, banks, bw, lims, tiers, en, pk, ec, sens, rec, prim, i4, env)
    print("memory_residency results written")
    return 0


def summary(ws, caprows, hyp, pol, banks, bw, lims, tiers, en, pk, ec, sens, rec, prim, i4, env) -> None:
    L = [f"# EXP-0006 — Model-resident memory feasibility (summary)", "",
         f"Generated by `analysis/memory_residency/report.py` at commit {env['commit']} on {env['date']}. Labels: MEASURED (counts from the canonical profiles, EXP-0005 RTL laws), ANALYTICAL (derived bytes, cycles, bandwidth), INFERRED (areas and energies from cited public figures), HYPOTHESIZED (larger models), TARGET (token rates), ASSUMPTION (margins, utilizations, ranges). **No FlexTern silicon exists; nothing here is an ASIC PPA or a product cost.** Decimal units throughout (1 MB = 10^6 B). Sources: `sources.csv`, `assumptions.csv`, `source_notes.md`.", "",
         "## Density tiers (never mixed)", "", "| tier | MB/mm² | includes compute? | usable for a chip estimate? |", "|---|---:|---|---|"]
    for t, (lo, hi) in src.DENSITY_TIERS_MB_PER_MM2.items():
        L.append(f"| {t} | {lo}–{hi} | {'yes' if 'incl. compute' in t else 'no'} | {'NO' if t.startswith('raw') else ('array only' if t.startswith('macro') else ('cache die only' if 'cache_only' in t or 'edram' in t else 'yes'))} |")
    L += ["", "## Capacity (MEASURED counts; bytes at 4/8/16 bits ANALYTICAL)", "", "| model | bits | transformer | head BF16 | head 4.156 b (EXP-0004 candidate) | metadata | physical @10% margin |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in caprows:
        L.append(f"| {r['model']} | {r['weight_bits']} | {r['transformer_MB']} MB | {r['head_MB_bf16']} MB | {r['head_MB_candidate']} MB | {mb(r['metadata_bytes'])} | {r['nominal_physical_transformer_MB']} MB |")
    L += ["", "## Hypothetical scaling (HYPOTHESIZED, capacity only, no quality claim)", "", "| model | bits | transformer | physical @10% | head BF16 |", "|---|---:|---:|---:|---:|"]
    for r in hyp:
        L.append(f"| {r['model']} | {r['weight_bits']} | {r['transformer_MB']} MB | {r['nominal_physical_transformer_MB']} MB | {r['head_MB_bf16']} MB |")
    L += ["", "## Residency policies, Falcon-E 1B, 2-bit (ANALYTICAL; EXP-0005 levels)", "", "| policy | resident capacity | global weight B/token | local weight B/token | global total B/token | reduction vs streaming | EXP-0005 advantage surviving |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in pol:
        if r["model"] == "falcon-e-1b" and r["weight_bits"] == 2:
            L.append(f"| {r['policy']} | {mb(r['resident_capacity_bytes'])} | {mb(r['global_weight_bytes_per_token'])} | {mb(r['local_weight_bytes_per_token'])} | {mb(r['global_total_bytes_per_token'])} | {r['global_reduction_vs_streaming']}x | {r['exp0005_advantage_surviving']} |")
    L += ["", "## Bank organization (ANALYTICAL; collector-bound cycle law MEASURED in EXP-0005), 2-bit, b = 16 B/cycle", "", "| model | banks | capacity/bank | tail waste | local B/bank/token | act. delivered/token | result groups/token | cycles/token | utilization | collector-bound |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in banks:
        if r["weight_bits"] == 2:
            L.append(f"| {r['model']} | {r['banks']} | {r['capacity_per_bank_MB']} MB | {r['tail_waste_pct']}% | {mb(r['local_bytes_per_bank_per_token'])} | {mb(r['activation_delivered_bytes_per_token'])} | {r['result_groups_per_token']:,} | {r['cycles_per_token_collector_bound']:,} | {r['utilization']} | {r['collector_bound']} |")
    L += ["", "## Bandwidth at TARGET token rates (70% utilization; peak figures from sources), 2-bit", "", "| model | tok/s | streaming external GB/s | LPDDR5X 64-bit channels | HBM3E stacks | resident local aggregate GB/s | per bank @64 | act. delivery @64 banks GB/s | global (act+out) GB/s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in bw:
        if r["weight_bits"] == 2 and r["utilization"] == src.NOMINAL_UTIL:
            L.append(f"| {r['model']} | {r['target_tok_s']} | {r['streaming_external_GBps_required']} | {r['lpddr5x_64bit_channels_needed_streaming']} | {r['hbm3e_stacks_needed_streaming']} | {r['resident_local_aggregate_GBps_required']} | {r['local_per_bank_GBps_NB64']} | {r['activation_delivery_GBps_NB64']} | {r['resident_global_GBps_required_act_out']} |")
    L += ["", "Streaming limits (70% util, batch 1): " + "; ".join(f"{r['model']} {r['weight_bits']}-bit: {r['lpddr5x_64bit_max_tok_s']} tok/s per 64-bit LPDDR5X channel, {r['hbm3e_one_stack_max_tok_s']} per HBM3E stack" for r in lims) + ".", "",
          "## Area ranges and tier crossings (INFERRED; 10% margin)", "", "| model | bits | physical | chip tier mm² (0.68–1.09 MB/mm²) | cache-die tier mm² (1.64–1.86) | V-Cache dies | organization class |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in tiers:
        L.append(f"| {r['model']} | {r['weight_bits']} | {r['physical_MB']} MB | {r['area_mm2_chip_tier_min']}–{r['area_mm2_chip_tier_max']} | {r['area_mm2_cache_die_tier_min']}–{r['area_mm2_cache_die_tier_max']} | {r['vcache_class_dies_41mm2']} | {r['memory_organization_class']} |")
    L += ["", f"Reticle limit {src.RETICLE_MM2:.0f} mm² (S21). Graphcore GC200 demonstrates 900 MB SRAM with 1,472 cores on 823 mm² at N7 (S05).", "",
          "## Energy ranges (INFERRED, A3; one movement each; NOT a chip J/token)", "", "| model | bits | weight B/token | stream LPDDR5X mJ | stream HBM-class mJ | resident SRAM-local mJ (45 nm anchor) | resident act+out crossing mJ (max) | resident-in-HBM-stack local mJ |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in en:
        if r["kind"].startswith("canonical"):
            L.append(f"| {r['model']} | {r['weight_bits']} | {mb(r['weight_bytes_per_token'])} | {r['stream_lpddr5x_mJ_per_token_min']}–{r['stream_lpddr5x_mJ_per_token_max']} | {r['stream_hbm_mJ_per_token_min']}–{r['stream_hbm_mJ_per_token_max']} | {r['resident_sram_local_mJ_per_token_min']}–{r['resident_sram_local_mJ_per_token_max']} | {r['resident_act_out_crossing_mJ_per_token_max']} | {r['resident_in_hbm_stack_local_mJ_per_token_min']}–{r['resident_in_hbm_stack_local_mJ_per_token_max']} |")
    L += ["", "## Package / 3D / chiplet organizations (ANALYTICAL), Falcon-E 1B 2-bit", "", "| organization | weights | Y-Cores | crosses die boundary | availability | capacity fit | size proxy |", "|---|---|---|---|---|---|---|"]
    for r in pk:
        if r["model"] == "falcon-e-1b" and r["weight_bits"] == 2:
            L.append(f"| {r['organization']} | {r['weights_live']} | {r['ycores_live']} | {r['data_crossing_die_boundary']} | {r['availability']} | {r['capacity_fit']} | {r['size_proxy']} |")
    L += ["", "## Economics proxy (no prices)", "", "| model | bits | physical | memory silicon mm² (chip tier) | reticle-class dies | package class |", "|---|---:|---:|---:|---:|---|"]
    for r in ec:
        L.append(f"| {r['model']} | {r['weight_bits']} | {r['physical_MB']} MB | {r['memory_silicon_mm2_min']}–{r['memory_silicon_mm2_max']} | {r['reticle_class_dies']} | {r['package_complexity_class']} |")
    L += ["", "## INT4 fairness (same architecture, same assumptions)", "", "| model | ternary MB | INT4 MB | V-Cache dies T/I | class ternary | class INT4 | tier crossing |", "|---|---:|---:|---:|---|---|---|"]
    for r in i4:
        L.append(f"| {r['model']} | {r['capacity_ternary_MB']} | {r['capacity_int4_MB']} | {r['vcache_dies_ternary']}/{r['vcache_dies_int4']} | {r['class_ternary']} | {r['class_int4']} | {r['tier_crossing']} |")
    L += ["", "## System reconnection (ANALYTICAL, 100 GB/s model; byte reductions are not token/s)", "", "| model | ctx | scenario | tok/s | global stream | global resident | local resident | head | KV | act+out | dominant global term (resident) | head+KV share |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in rec:
        if r["target_tok_s"] == 100 and r["context"] in (128, 2048, 32768):
            L.append(f"| {r['model']} | {r['context']} | {r['scenario']} | {r['target_tok_s']} | {mb(r['global_total_streaming'])} ({r['global_GBps_streaming']} GB/s) | {mb(r['global_total_resident'])} ({r['global_GBps_resident']} GB/s) | {mb(r['local_ternary_bytes_resident'])} ({r['local_GBps_resident']} GB/s) | {mb(r['head_bytes'])} | {mb(r['kv_bytes'])} | {mb(r['activation_plus_output_bytes'])} | {r['dominant_global_term_resident']} | {r['head_plus_kv_share_of_resident_global']} |")
    L += ["", "## Sensitivity thresholds", "", "| threshold | model | bits | margin | value | note |", "|---|---|---:|---:|---:|---|"]
    for r in sens:
        L.append(f"| {r['threshold']} | {r['model']} | {r['weight_bits']} | {r['margin']} | {r['value']} | {r['note']} |")
    L += ["", "## Primary comparison", "", "See `primary_comparison.csv` (TERNARY STREAMING / TERNARY RESIDENT / INT4 RESIDENT / INT4 STREAMING for the three canonical models)."]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
