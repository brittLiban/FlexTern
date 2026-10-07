"""
EXP-0008 model: resident SRAM versus external LPDDR5X memory-subsystem energy, static power and latency
for batch-1 edge decode of the canonical models.  Boundaries:

  resident:   E_weight/token = bits x (macro read fJ/bit + local delivery pJ/bit) + per-access fixed cost
              P_static       = provisioned_MB x leakage mW/MB x retention factor (always on)
  streaming:  E_weight/token = bits x LPDDR pJ/bit (device + PHY/controller per A6)
              P_background   = DRAM self-refresh / power-down / idle (A7)
Head and KV are external in both systems (reference BF16 or EXP-0004 candidate), so they add the same
LPDDR energy to both; they are reported so the weight advantage is seen against total memory energy.
Everything is MEMORY-SUBSYSTEM only; compute energy and chip totals are not modeled.
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "memory_residency"))
sys.path.insert(0, _HERE)
import evidence as S  # noqa: E402
from capacity import head_bytes, load, transformer_bytes  # noqa: E402
from roofline_models import kv_read_bytes, kv_write_bytes  # noqa: E402

CONTEXTS = (128, 2048, 8192, 32768)


# ----------------------------------------------------------------------------- energies per bit
def delivery_pj_per_bit(distance_key: str, wire_idx: int = 0) -> float:
    return S.WIRE_PJ_PER_BIT_PER_MM[wire_idx] * S.DISTANCES_MM[distance_key] + S.REGISTER_PJ_PER_BIT


def resident_pj_per_bit(level: str, distance_key: str = "medium_tile", width_bits: int = 1024) -> dict:
    macro = S.SRAM_MACRO_FJ_PER_BIT[level] / 1000.0
    wire = delivery_pj_per_bit(distance_key, 1 if level == "high" else 0)
    fixed = S.PER_ACCESS_FIXED_PJ[1 if level == "high" else 0] / width_bits
    bank = S.BANK_OVERHEAD_PJ_PER_BIT[level]
    return {"macro_pj_per_bit": macro, "delivery_pj_per_bit": wire, "fixed_pj_per_bit": fixed, "bank_overhead_pj_per_bit": bank, "total_pj_per_bit": macro + wire + fixed + bank,
            "boundary": "SRAM macro (array+periphery) + bank composition overhead (A12) + local wires + registers + amortized decode; NO leakage here"}


def width_sweep() -> list[dict]:
    rows = []
    for w in S.READ_WIDTHS:
        for level in ("low", "planning", "high"):
            r = resident_pj_per_bit(level, "medium_tile", w)
            rows.append({"read_width_bits": w, "level": level, "energy_per_access_pJ": round(r["total_pj_per_bit"] * w, 2),
                         "energy_per_bit_pJ": round(r["total_pj_per_bit"], 4), "macro_pj_per_bit": r["macro_pj_per_bit"], "bank_overhead_pj_per_bit": r["bank_overhead_pj_per_bit"],
                         "delivery_pj_per_bit": round(r["delivery_pj_per_bit"], 3), "fixed_pj_per_bit": round(r["fixed_pj_per_bit"], 4), "label": "A1/A5/A9"})
    return rows


def bank_decomposition(bank_bytes: float = 6.19e6) -> list[dict]:
    rows = []
    for m in S.MACRO_SIZES_BYTES:
        n = math.ceil(bank_bytes / m)
        rows.append({"logical_bank_MB": round(bank_bytes / 1e6, 2), "macro_bytes": int(m), "macros_per_bank": n, "provisioned_MB": round(n * m / 1e6, 2),
                     "macros_active_per_access": 1, "active_fraction": round(1 / n, 4), "capacity_conserved": n * m >= bank_bytes,
                     "note": "only the addressed macro switches; the others leak only (A9)"})
    return rows


# ----------------------------------------------------------------------------- per-model energies
def model_rows(ws) -> list[dict]:
    rows = []
    for slug, w in ws.items():
        for bits in (2.0, 4.5):
            W = transformer_bytes(w, 2) * (bits / 2.0)
            phys = W / (1 - S.MARGIN)
            for level in ("low", "planning", "high"):
                r = resident_pj_per_bit(level)
                e_res = W * 8 * r["total_pj_per_bit"] / 1e9                      # mJ/token
                e_lp = W * 8 * S.LPDDR_PJ_PER_BIT[level] / 1e9
                leak = phys / 1e6 * S.LEAK_MW_PER_MB[level]                       # mW, always on at retention factor 1
                rows.append({"model": slug, "weight_bits": bits, "level": level, "weight_bytes_per_token": int(W), "provisioned_MB": round(phys / 1e6, 1),
                             "sram_total_pj_per_bit": round(r["total_pj_per_bit"], 4), "lpddr_pj_per_bit": S.LPDDR_PJ_PER_BIT[level],
                             "resident_dynamic_mJ_per_token": round(e_res, 3), "lpddr_dynamic_mJ_per_token": round(e_lp, 3),
                             "dynamic_ratio_lpddr_over_resident": round(e_lp / e_res, 2), "sram_static_mW": round(leak, 1),
                             "lpddr_background_mW_power_down": S.LPDDR_BACKGROUND_MW["power_down"], "lpddr_background_mW_self_refresh_25C": S.LPDDR_BACKGROUND_MW["self_refresh_25C"],
                             "break_even_tok_s_vs_power_down": round((leak - S.LPDDR_BACKGROUND_MW["power_down"]) / (e_lp - e_res), 2) if e_lp > e_res else "n/a",
                             "boundary_resident": r["boundary"], "boundary_lpddr": "DRAM device (E06 IDD) + SoC PHY/controller (A6); board idle excluded"})
    return rows


def token_rate_sweep(ws, level: str = "planning") -> list[dict]:
    rows = []
    for slug, w in ws.items():
        for bits in (2.0, 4.5):
            W = transformer_bytes(w, 2) * (bits / 2.0); phys = W / (1 - S.MARGIN)
            r = resident_pj_per_bit(level); e_res = W * 8 * r["total_pj_per_bit"] / 1e9; e_lp = W * 8 * S.LPDDR_PJ_PER_BIT[level] / 1e9
            for R in S.TOKEN_RATES:
                for ret in S.RETENTION_FACTORS:
                    leak = phys / 1e6 * S.LEAK_MW_PER_MB[level] * ret
                    busy = min(1.0, W * R / (S.LPDDR_CHANNEL_GBPS * 1e9 * 0.7))
                    p_res = e_res * R + leak; p_lp = e_lp * R + busy * S.LPDDR_BACKGROUND_MW["idle_standby"] + (1 - busy) * S.LPDDR_BACKGROUND_MW["power_down"]
                    rows.append({"model": slug, "weight_bits": bits, "level": level, "tok_s": R, "retention_factor": ret,
                                 "resident_dynamic_mW": round(e_res * R, 1), "resident_static_mW": round(leak, 1), "resident_total_mW": round(p_res, 1),
                                 "lpddr_dynamic_mW": round(e_lp * R, 1), "lpddr_background_mW": round(busy * S.LPDDR_BACKGROUND_MW["idle_standby"] + (1 - busy) * S.LPDDR_BACKGROUND_MW["power_down"], 1), "lpddr_channel_busy_fraction": round(busy, 3), "lpddr_total_mW": round(p_lp, 1),
                                 "lpddr_over_resident_total": round(p_lp / p_res, 2), "local_sram_GBps": round(W * R / 1e9, 2), "lpddr_GBps": round(W * R / 1e9, 2),
                                 "lpddr_channels_needed": math.ceil(W * R / (S.LPDDR_CHANNEL_GBPS * 1e9 * 0.7)), "label": "TARGET rates; memory subsystem only"})
    return rows


def duty_cycle_sweep(ws, level: str = "planning", R_active: int = 25) -> list[dict]:
    rows = []
    for slug, w in ws.items():
        for bits in (2.0, 4.5):
            W = transformer_bytes(w, 2) * (bits / 2.0); phys = W / (1 - S.MARGIN)
            r = resident_pj_per_bit(level); e_res = W * 8 * r["total_pj_per_bit"] / 1e9; e_lp = W * 8 * S.LPDDR_PJ_PER_BIT[level] / 1e9
            for d in S.DUTY_CYCLES:
                for ret in S.RETENTION_FACTORS:
                    leak_active = phys / 1e6 * S.LEAK_MW_PER_MB[level]
                    leak_idle = leak_active * ret
                    busy = min(1.0, W * R_active / (S.LPDDR_CHANNEL_GBPS * 1e9 * 0.7))
                    p_res = d * (e_res * R_active + leak_active) + (1 - d) * leak_idle
                    p_lp = d * (e_lp * R_active + busy * S.LPDDR_BACKGROUND_MW["idle_standby"] + (1 - busy) * S.LPDDR_BACKGROUND_MW["power_down"]) + (1 - d) * S.LPDDR_BACKGROUND_MW["power_down"]
                    rows.append({"model": slug, "weight_bits": bits, "level": level, "duty_cycle": d, "active_tok_s": R_active, "idle_retention_factor": ret,
                                 "resident_avg_mW": round(p_res, 1), "lpddr_avg_mW": round(p_lp, 1), "lpddr_over_resident": round(p_lp / p_res, 2),
                                 "resident_static_share": round(((d * leak_active + (1 - d) * leak_idle) / p_res), 3), "label": "memory subsystem only; LPDDR busy fraction = IDD2N, rest = IDD2P (A13); retention factors 0.2/0.1 HYPOTHESIZED"})
    return rows


# ----------------------------------------------------------------------------- break-even
def break_even(ws, level: str = "planning") -> list[dict]:
    rows = []
    for slug, w in ws.items():
        W = transformer_bytes(w, 2); phys = W / (1 - S.MARGIN); bits = W * 8
        r = resident_pj_per_bit(level); e_res = bits * r["total_pj_per_bit"] / 1e9; e_lp = bits * S.LPDDR_PJ_PER_BIT[level] / 1e9
        leak = phys / 1e6 * S.LEAK_MW_PER_MB[level]
        sav = e_lp - e_res
        rows.append({"model": slug, "threshold": "token rate (continuous) at which SRAM dynamic savings exceed SRAM leakage minus DRAM power-down", "value": round((leak - S.LPDDR_BACKGROUND_MW["power_down"]) / sav, 2), "unit": "tok/s", "level": level})
        for ret in S.RETENTION_FACTORS:
            leak_idle = leak * ret
            # d*(e_res R + leak) + (1-d) leak_idle = d*(e_lp R + idle_standby) + (1-d) power_down, R = 25
            R = 25
            busy = min(1.0, W * R / (S.LPDDR_CHANNEL_GBPS * 1e9 * 0.7))
            num = leak_idle - S.LPDDR_BACKGROUND_MW["power_down"]
            den = (e_lp * R + busy * S.LPDDR_BACKGROUND_MW["idle_standby"] + (1 - busy) * S.LPDDR_BACKGROUND_MW["power_down"] - S.LPDDR_BACKGROUND_MW["power_down"]) - (e_res * R + leak - leak_idle)
            d = num / den if den else float("inf")
            rows.append({"model": slug, "threshold": f"duty cycle at which resident beats LPDDR in average memory power (25 tok/s active, idle retention x{ret})", "value": round(d, 4) if 0 <= d <= 1 else ("always" if d < 0 else "never"), "unit": "fraction", "level": level})
        rows.append({"model": slug, "threshold": "SRAM total pJ/bit at which residency no longer wins on dynamic energy", "value": S.LPDDR_PJ_PER_BIT[level], "unit": "pJ/bit", "level": level})
        for R in (5, 25, 100):
            rows.append({"model": slug, "threshold": f"SRAM leakage mW/MB at which residency no longer wins total memory power at {R} tok/s continuous", "value": round((sav * R + S.LPDDR_BACKGROUND_MW["power_down"]) / (phys / 1e6), 3), "unit": "mW/MB", "level": level})
        rows.append({"model": slug, "threshold": "local-wire pJ/bit that erases the dynamic advantage (macro + bank overhead + wire = LPDDR)", "value": round(S.LPDDR_PJ_PER_BIT[level] - r["macro_pj_per_bit"] - r["fixed_pj_per_bit"] - r["bank_overhead_pj_per_bit"], 3), "unit": "pJ/bit (= about %.0f mm at 0.15 pJ/bit/mm)" % ((S.LPDDR_PJ_PER_BIT[level] - r["macro_pj_per_bit"] - r["bank_overhead_pj_per_bit"]) / 0.15), "level": level})
        rows.append({"model": slug, "threshold": "LPDDR pJ/bit at which streaming becomes preferable on dynamic energy", "value": round(r["total_pj_per_bit"], 3), "unit": "pJ/bit", "level": level})
    return rows


# ----------------------------------------------------------------------------- latency
def latency_rows(ws) -> list[dict]:
    rows = []
    for slug, w in ws.items():
        W = transformer_bytes(w, 2)
        for tok in (10, 25, 100):
            token_ms = 1000 / tok
            lp_bw_ms = W / (S.LPDDR_CHANNEL_GBPS * 1e9 * 0.7) * 1e3                    # one x16 channel, 70% util
            sram_bw_ms = W / 8e12 * 1e3                                                   # EXP-0006 tile local BW 8 TB/s
            bdp = S.LPDDR_LATENCY_NS["soc_loaded"] * 1e-9 * S.LPDDR_CHANNEL_GBPS * 1e9     # bytes in flight to hide latency
            rows.append({"model": slug, "target_tok_s": tok, "token_budget_ms": round(token_ms, 2),
                         "lpddr_first_access_ns": S.LPDDR_LATENCY_NS["soc_loaded"], "sram_first_access_ns": S.SRAM_LATENCY_NS["macro_access"],
                         "first_word_advantage_ns": round(S.LPDDR_LATENCY_NS["soc_loaded"] - S.SRAM_LATENCY_NS["macro_access"], 1),
                         "first_word_advantage_as_fraction_of_token": round((S.LPDDR_LATENCY_NS["soc_loaded"] * 1e-6) / token_ms, 8),
                         "bytes_in_flight_to_hide_lpddr_latency": int(bdp), "prefetch_bytes_assumed": S.PREFETCH_BYTES, "latency_hideable": bdp <= S.PREFETCH_BYTES,
                         "lpddr_bandwidth_floor_ms_per_token_one_channel": round(lp_bw_ms, 2), "sram_bandwidth_floor_ms_per_token": round(sram_bw_ms, 4),
                         "lpddr_bandwidth_bound_at_target": lp_bw_ms > token_ms, "layers_sequential_stalls": "none after the first chunk when prefetch >= bytes in flight",
                         "verdict": "latency hidden; bandwidth decides" if bdp <= S.PREFETCH_BYTES else "latency exposed"})
    return rows


def cold_start(ws) -> list[dict]:
    rows = []
    for slug, w in ws.items():
        W = transformer_bytes(w, 2); phys = W / (1 - S.MARGIN)
        for g in S.COLD_START_GBPS:
            e = W * 8 * (S.LPDDR_PJ_PER_BIT["planning"] + S.SRAM_MACRO_FJ_PER_BIT["planning"] / 1000 * S.SRAM_WRITE_OVER_READ) / 1e9
            rows.append({"model": slug, "load_bytes": int(W), "provisioned_MB": round(phys / 1e6, 1), "load_GBps": g, "load_time_s": round(W / (g * 1e9), 3),
                         "one_time_energy_mJ": round(e, 1), "equivalent_tokens_of_lpddr_streaming": round(e / (W * 8 * S.LPDDR_PJ_PER_BIT["planning"] / 1e9), 2),
                         "note": "volatile SRAM: paid at every power-up; nonvolatile storage not chosen here (A10)"})
    return rows


# ----------------------------------------------------------------------------- head / KV and system reconnection
def system_rows(ws, level: str = "planning") -> list[dict]:
    rows = []
    for slug, w in ws.items():
        W = transformer_bytes(w, 2)
        r = resident_pj_per_bit(level); e_res = W * 8 * r["total_pj_per_bit"] / 1e9; e_lp = W * 8 * S.LPDDR_PJ_PER_BIT[level] / 1e9
        for ctx in CONTEXTS:
            if ctx > w.max_ctx:
                continue
            for label, hb, kvb in (("BF16 head + BF16 KV", 2.0, 2.0), ("EXP-0004 candidate head 4.156 b + KV 4.5 b", S.HEAD_CANDIDATE_BITS / 8, S.KV_CANDIDATE_BITS / 8)):
                head = head_bytes(w, hb * 8); kv = kv_read_bytes(w, ctx, kvb) + kv_write_bytes(w, kvb)
                e_ext = (head + kv) * 8 * S.LPDDR_PJ_PER_BIT[level] / 1e9
                rows.append({"model": slug, "context": ctx, "non_ternary_case": label, "weight_bytes": int(W), "head_bytes": int(head), "kv_bytes": int(kv),
                             "resident_weight_mJ": round(e_res, 3), "lpddr_weight_mJ": round(e_lp, 3), "external_head_kv_mJ_both_systems": round(e_ext, 3),
                             "resident_total_memory_mJ": round(e_res + e_ext, 3), "streaming_total_memory_mJ": round(e_lp + e_ext, 3),
                             "weight_only_ratio": round(e_lp / e_res, 2), "total_memory_ratio": round((e_lp + e_ext) / (e_res + e_ext), 2),
                             "head_kv_share_of_resident_total": round(e_ext / (e_res + e_ext), 3), "note": "head and KV external in both systems; leakage excluded here (see duty-cycle tables)"})
    return rows


def old_vs_new(ws) -> list[dict]:
    rows = []
    for slug, w in ws.items():
        W = transformer_bytes(w, 2); bits = W * 8
        old_res = bits * S.OLD_SRAM_PLANNING_PJ / 1e9; old_lp = (bits * S.OLD_LPDDR_RANGE_PJ[0] / 1e9, bits * S.OLD_LPDDR_RANGE_PJ[1] / 1e9)
        new = {lv: (bits * resident_pj_per_bit(lv)["total_pj_per_bit"] / 1e9, bits * S.LPDDR_PJ_PER_BIT[lv] / 1e9) for lv in ("low", "planning", "high")}
        rows.append({"model": slug, "exp0006_resident_planning_mJ": round(old_res, 2), "exp0006_lpddr_mJ_range": f"{old_lp[0]:.1f}-{old_lp[1]:.1f}",
                     "exp0006_ratio_range": f"{old_lp[0]/old_res:.1f}-{old_lp[1]/old_res:.1f}",
                     "exp0008_resident_mJ_low_planning_high": f"{new['low'][0]:.2f} / {new['planning'][0]:.2f} / {new['high'][0]:.2f}",
                     "exp0008_lpddr_mJ_low_planning_high": f"{new['low'][1]:.1f} / {new['planning'][1]:.1f} / {new['high'][1]:.1f}",
                     "exp0008_dynamic_ratio_planning": round(new["planning"][1] / new["planning"][0], 1),
                     "exp0008_dynamic_ratio_conservative_to_optimistic": f"{new['low'][1]/new['high'][0]:.1f}-{new['high'][1]/new['low'][0]:.1f}",
                     "note": "dynamic weight movement only; leakage changes the total (duty-cycle tables)"})
    return rows
