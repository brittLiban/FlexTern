"""EXP-0008 driver: writes results/sram_validation/*.csv and summary.md.  Run from the repository root."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
os.chdir(ROOT)

import model as M  # noqa: E402
import evidence as S  # noqa: E402

OUT = os.path.join("results", "sram_validation")


def write_csv(name: str, rows: list[dict]) -> None:
    os.makedirs(OUT, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main() -> int:
    S.write(OUT)
    ws = M.load()
    macros = [{k: s.get(k, "") for k in ("source_id", "publication", "process", "macro_capacity", "read_width", "voltage", "frequency", "read_energy_per_access", "derived_energy_per_bit", "leakage_power", "tier", "included_energy_boundary")} for s in S.SOURCES if s["source_id"] in ("E01", "E02", "E03", "E04", "E05", "E10", "E11")]
    write_csv("modern_sram_macros.csv", macros)
    write_csv("macro_energy_scaling.csv", M.width_sweep())
    write_csv("bank_decomposition.csv", M.bank_decomposition())
    write_csv("macro_latency.csv", [{"quantity": k, "ns": v, "source": "E02/E03 (A8)"} for k, v in S.SRAM_LATENCY_NS.items()] + [{"quantity": "lpddr_" + k, "ns": v, "source": "E12/A8"} for k, v in S.LPDDR_LATENCY_NS.items()])
    write_csv("leakage_ranges.csv", [{"level": k, "mW_per_MB": v, "retention_factors": str(S.RETENTION_FACTORS), "source": "A4 (E11 T3 FinFET cell current; E05 28 nm T1 as context only); retention factors 0.2/0.1 HYPOTHESIZED"} for k, v in S.LEAK_MW_PER_MB.items()])
    write_csv("local_wire_ranges.csv", [{"distance": k, "mm": v, "pJ_per_bit_low": round(M.delivery_pj_per_bit(k, 0), 3), "pJ_per_bit_high": round(M.delivery_pj_per_bit(k, 1), 3), "source": "E04 wire x A5 distances"} for k, v in S.DISTANCES_MM.items()])
    write_csv("lpddr_energy.csv", [{"boundary": k, "pJ_per_bit": v, "source": "E06 (device IDD4R arithmetic), E07/E04 (PHY), A6"} for k, v in S.LPDDR_PJ_PER_BIT.items()] + [{"boundary": "background_" + k, "pJ_per_bit": "", "mW": v, "source": "E06 IDD6/IDD2P/IDD2N per x16 channel"} for k, v in S.LPDDR_BACKGROUND_MW.items()])
    dyn = M.model_rows(ws); write_csv("dynamic_energy.csv", dyn)
    tr = M.token_rate_sweep(ws); write_csv("token_rate_sweep.csv", tr)
    dc = M.duty_cycle_sweep(ws); write_csv("duty_cycle_sweep.csv", dc)
    be = M.break_even(ws); write_csv("break_even.csv", be)
    lat = M.latency_rows(ws); write_csv("latency_comparison.csv", lat)
    cs = M.cold_start(ws); write_csv("cold_start.csv", cs)
    sysr = M.system_rows(ws); write_csv("system_reconnect.csv", sysr)
    ovn = M.old_vs_new(ws); write_csv("exp0006_reconnect.csv", ovn)
    i4 = [r for r in dyn if r["weight_bits"] == 4.5]; write_csv("int4_comparison.csv", i4)
    prim = []
    for slug in ws:
        p = next(r for r in dyn if r["model"] == slug and r["weight_bits"] == 2.0 and r["level"] == "planning")
        lo = next(r for r in dyn if r["model"] == slug and r["weight_bits"] == 2.0 and r["level"] == "low"); hi = next(r for r in dyn if r["model"] == slug and r["weight_bits"] == 2.0 and r["level"] == "high")
        b = [r for r in be if r["model"] == slug]
        l = next(r for r in lat if r["model"] == slug and r["target_tok_s"] == 25)
        prim.append({"model": slug, "resident_capacity_MB": p["provisioned_MB"], "sram_source_range": "TSMC N5 (E01, T1), 7 nm UHD macro (E02, T1), Keckler 10 nm projection (E04, T2)",
                     "sram_macro_pj_per_bit_low_plan_high": f"{S.SRAM_MACRO_FJ_PER_BIT['low']/1000:.3f} / {S.SRAM_MACRO_FJ_PER_BIT['planning']/1000:.3f} / {S.SRAM_MACRO_FJ_PER_BIT['high']/1000:.3f}",
                     "local_delivery_pj_per_bit_medium": f"{M.delivery_pj_per_bit('medium_tile',0):.2f}-{M.delivery_pj_per_bit('medium_tile',1):.2f}",
                     "resident_dynamic_mJ_per_token_low_plan_high": f"{lo['resident_dynamic_mJ_per_token']} / {p['resident_dynamic_mJ_per_token']} / {hi['resident_dynamic_mJ_per_token']}",
                     "sram_static_mW_low_plan_high": f"{lo['sram_static_mW']} / {p['sram_static_mW']} / {hi['sram_static_mW']}",
                     "lpddr_dynamic_mJ_per_token_low_plan_high": f"{lo['lpddr_dynamic_mJ_per_token']} / {p['lpddr_dynamic_mJ_per_token']} / {hi['lpddr_dynamic_mJ_per_token']}",
                     "lpddr_background_mW": f"{S.LPDDR_BACKGROUND_MW['self_refresh_25C']} (self-refresh 25 C) - {S.LPDDR_BACKGROUND_MW['power_down']} (power-down) - {S.LPDDR_BACKGROUND_MW['idle_standby']} (idle standby)",
                     "break_even_tok_s_continuous_planning": p["break_even_tok_s_vs_power_down"],
                     "break_even_duty_cycle_25tok_s_retention_1x_0.2x_0.1x": " / ".join(str(r["value"]) for r in b if r["threshold"].startswith("duty cycle")),
                     "first_word_latency_ns_lpddr_vs_sram": f"{l['lpddr_first_access_ns']} vs {l['sram_first_access_ns']}",
                     "steady_state_latency_effect": l["verdict"] + f"; one LPDDR5X channel floor {l['lpddr_bandwidth_floor_ms_per_token_one_channel']} ms/token",
                     "dynamic_ratio_planning": p["dynamic_ratio_lpddr_over_resident"], "evidence_quality": "dynamic: T1-anchored (order of magnitude, not three figures); leakage: T3 FinFET cell statements + 28 nm T1 context (modern-node leakage per Mb UNMEASURED; retention factors HYPOTHESIZED); LPDDR: T1 device maximum, typical-device factor ASSUMED, T3 PHY"})
    write_csv("primary_comparison.csv", prim)
    env = {"date": date.today().isoformat(), "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}
    json.dump(env, open(os.path.join(OUT, "environment.json"), "w", encoding="utf-8"), indent=1)
    summary(ws, dyn, tr, dc, be, lat, cs, sysr, ovn, prim, env)
    print("sram_validation results written")
    return 0


def summary(ws, dyn, tr, dc, be, lat, cs, sysr, ovn, prim, env) -> None:
    L = [f"# EXP-0008 — Modern SRAM energy and latency validation (summary)", "",
         f"Generated by `analysis/sram_validation/report.py` at commit {env['commit']} on {env['date']}. MEMORY-SUBSYSTEM energy and power only; no chip TDP, no J/token, no battery life. Boundaries: resident = SRAM macro (array + periphery) + local wires + registers + amortized decode, with leakage reported separately; LPDDR5X = DRAM device (Micron IDD4R arithmetic, T1) + SoC PHY/controller (T2/T3 bracket). LOW / PLANNING / HIGH levels are defined in `assumptions.csv` (A1, A4, A5, A6). Decimal MB.", "",
         "## Source quality", "", "| id | what | tier | boundary |", "|---|---|---|---|"]
    for s in S.SOURCES:
        L.append(f"| {s['source_id']} | {s['publication'][:110]} | {s['tier']} | {s['included_energy_boundary']} |")
    L += ["", "## Dynamic energy per token (weights only, batch 1)", "", "| model | bits | level | SRAM total pJ/bit | LPDDR pJ/bit | resident mJ/tok | LPDDR mJ/tok | ratio | static mW (retention x1) | break-even tok/s (continuous, vs DRAM power-down) |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in dyn:
        L.append(f"| {r['model']} | {r['weight_bits']} | {r['level']} | {r['sram_total_pj_per_bit']} | {r['lpddr_pj_per_bit']} | {r['resident_dynamic_mJ_per_token']} | {r['lpddr_dynamic_mJ_per_token']} | {r['dynamic_ratio_lpddr_over_resident']}x | {r['sram_static_mW']} | {r['break_even_tok_s_vs_power_down']} |")
    L += ["", "## Token-rate sweep (planning level, 2-bit; LPDDR background = IDD2N for the channel-busy fraction, IDD2P for the rest, A13; retention 0.1 is HYPOTHESIZED)", "", "| model | tok/s | retention | resident dyn mW | resident static mW | resident total mW | LPDDR total mW | LPDDR/resident | LPDDR channels |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in tr:
        if r["weight_bits"] == 2.0 and r["retention_factor"] in (1.0, 0.1) and r["tok_s"] in (1, 5, 10, 25, 100):
            L.append(f"| {r['model']} | {r['tok_s']} | {r['retention_factor']} | {r['resident_dynamic_mW']} | {r['resident_static_mW']} | {r['resident_total_mW']} | {r['lpddr_total_mW']} | {r['lpddr_over_resident_total']}x | {r['lpddr_channels_needed']} |")
    L += ["", "## Duty-cycle sweep (planning level, 2-bit, 25 tok/s when active; idle retention 0.1 is HYPOTHESIZED)", "", "| model | duty | idle retention | resident avg mW | LPDDR avg mW | LPDDR/resident | resident static share |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in dc:
        if r["weight_bits"] == 2.0 and r["idle_retention_factor"] in (1.0, 0.1):
            L.append(f"| {r['model']} | {r['duty_cycle']} | {r['idle_retention_factor']} | {r['resident_avg_mW']} | {r['lpddr_avg_mW']} | {r['lpddr_over_resident']}x | {r['resident_static_share']} |")
    L += ["", "## Break-even thresholds (planning level)", "", "| model | threshold | value | unit |", "|---|---|---:|---|"]
    for r in be:
        L.append(f"| {r['model']} | {r['threshold']} | {r['value']} | {r['unit']} |")
    L += ["", "## Latency", "", "| model | tok/s | token budget ms | LPDDR first access ns | SRAM ns | bytes in flight to hide | hideable | LPDDR 1-channel floor ms/token | bandwidth-bound at target | verdict |", "|---|---:|---:|---:|---:|---:|---|---:|---|---|"]
    for r in lat:
        L.append(f"| {r['model']} | {r['target_tok_s']} | {r['token_budget_ms']} | {r['lpddr_first_access_ns']} | {r['sram_first_access_ns']} | {r['bytes_in_flight_to_hide_lpddr_latency']} | {r['latency_hideable']} | {r['lpddr_bandwidth_floor_ms_per_token_one_channel']} | {r['lpddr_bandwidth_bound_at_target']} | {r['verdict']} |")
    L += ["", "## Cold start", "", "| model | load bytes | load GB/s | load time s | one-time energy mJ | = tokens of LPDDR streaming |", "|---|---:|---:|---:|---:|---:|"]
    for r in cs:
        L.append(f"| {r['model']} | {r['load_bytes']:,} | {r['load_GBps']} | {r['load_time_s']} | {r['one_time_energy_mJ']} | {r['equivalent_tokens_of_lpddr_streaming']} |")
    L += ["", "## Total memory energy with head and KV external in both systems (planning, leakage excluded)", "", "| model | ctx | case | resident weight mJ | LPDDR weight mJ | external head+KV mJ (both) | resident total | streaming total | weight ratio | total ratio | head+KV share of resident total |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sysr:
        L.append(f"| {r['model']} | {r['context']} | {r['non_ternary_case']} | {r['resident_weight_mJ']} | {r['lpddr_weight_mJ']} | {r['external_head_kv_mJ_both_systems']} | {r['resident_total_memory_mJ']} | {r['streaming_total_memory_mJ']} | {r['weight_only_ratio']}x | {r['total_memory_ratio']}x | {r['head_kv_share_of_resident_total']} |")
    L += ["", "## EXP-0006 reconnection (dynamic weight movement)", "", "| model | EXP-0006 resident (1.6 pJ/bit) | EXP-0006 LPDDR 5-15 | EXP-0006 ratio | EXP-0008 resident low/plan/high | EXP-0008 LPDDR low/plan/high | planning ratio | conservative-optimistic |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ovn:
        L.append(f"| {r['model']} | {r['exp0006_resident_planning_mJ']} | {r['exp0006_lpddr_mJ_range']} | {r['exp0006_ratio_range']} | {r['exp0008_resident_mJ_low_planning_high']} | {r['exp0008_lpddr_mJ_low_planning_high']} | {r['exp0008_dynamic_ratio_planning']}x | {r['exp0008_dynamic_ratio_conservative_to_optimistic']}x |")
    L += ["", "## Primary founder table", "", "| model | capacity MB | SRAM macro pJ/bit L/P/H | delivery pJ/bit | resident mJ/tok L/P/H | static mW L/P/H | LPDDR mJ/tok L/P/H | LPDDR background mW | break-even tok/s | break-even duty (ret 1/0.2/0.1) | first word ns | steady state | dyn ratio (plan) |", "|---|---:|---|---|---|---|---|---|---:|---|---|---|---:|"]
    for r in prim:
        L.append(f"| {r['model']} | {r['resident_capacity_MB']} | {r['sram_macro_pj_per_bit_low_plan_high']} | {r['local_delivery_pj_per_bit_medium']} | {r['resident_dynamic_mJ_per_token_low_plan_high']} | {r['sram_static_mW_low_plan_high']} | {r['lpddr_dynamic_mJ_per_token_low_plan_high']} | {r['lpddr_background_mW']} | {r['break_even_tok_s_continuous_planning']} | {r['break_even_duty_cycle_25tok_s_retention_1x_0.2x_0.1x']} | {r['first_word_latency_ns_lpddr_vs_sram']} | {r['steady_state_latency_effect']} | {r['dynamic_ratio_planning']}x |")
    L += ["", "Full tables: `modern_sram_macros.csv`, `macro_energy_scaling.csv`, `bank_decomposition.csv`, `macro_latency.csv`, `leakage_ranges.csv`, `local_wire_ranges.csv`, `lpddr_energy.csv`, `dynamic_energy.csv`, `token_rate_sweep.csv`, `duty_cycle_sweep.csv`, `break_even.csv`, `latency_comparison.csv`, `cold_start.csv`, `int4_comparison.csv`, `system_reconnect.csv`, `exp0006_reconnect.csv`, `primary_comparison.csv`."]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
