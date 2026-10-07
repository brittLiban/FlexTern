"""
EXP-0009 comparisons: policy sweeps, break-even thresholds, INT4 fairness, head/KV reconnection, primary table.
Every function returns rows for the CSVs written by report.py.  Planning level unless stated; temperatures per B2.
"""

from __future__ import annotations

import math

import retention_evidence as R
import retention as RT
import wake as WK
import evidence as E8
from policy import Scenario, lpddr_background_mw, E_RES_PJ, E_LP_PJ
from capacity import head_bytes
from roofline_models import kv_read_bytes, kv_write_bytes

LEVELS = ("low", "planning", "high")
MODES = ("DEEP_SLEEP_NOMINAL", "RETENTION_VRET")


def _r(d: dict, keys=None) -> dict:
    out = {}
    for k, v in d.items():
        if keys and k not in keys:
            continue
        out[k] = round(v, 4) if isinstance(v, float) else v
    return out


# ----------------------------------------------------------------------------- policy sweeps
def policy_rows(scheds: dict) -> list[dict]:
    rows = []
    for (slug, bits), s in scheds.items():
        for temp in R.TEMPS_C:
            for mode in MODES:
                sc = Scenario(s, "planning", temp, mode)
                for R_tok in (5, 25, 100):
                    rows.append(_r({"model": slug, "weight_bits": bits, **sc.P0(R_tok)}))
                    for k in R.WINDOWS:
                        rows.append(_r({"model": slug, "weight_bits": bits, **sc.window(k, R_tok, "paced")}))
                        rows.append(_r({"model": slug, "weight_bits": bits, **sc.window(k, R_tok, "burst")}))
                    for X in R.TIMEOUTS_S:
                        rows.append(_r({"model": slug, "weight_bits": bits, **sc.timeout(X, R_tok)}))
    return rows


def token_rate_rows(scheds: dict) -> list[dict]:
    rows = []
    for (slug, bits), s in scheds.items():
        for temp in R.TEMPS_C:
            sc = Scenario(s, "planning", temp, "RETENTION_VRET")
            for R_tok in R.TOKEN_RATES:
                for name, d in (("P0", sc.P0(R_tok)), ("P2", sc.window(2, R_tok)), ("P4", sc.window(4, R_tok)), ("P2-burst", sc.window(2, R_tok, "burst")), ("PT0", sc.timeout(0.0, R_tok)), ("PT10ms", sc.timeout(10e-3, R_tok))):
                    rows.append(_r({"model": slug, "weight_bits": bits, "temp_c": temp, "tok_s": R_tok, "policy": name,
                                    "resident_dynamic_mW": d["resident_dynamic_W"] * 1e3, "resident_leakage_mW": d["resident_leakage_W"] * 1e3, "resident_wake_mW": d["resident_wake_W"] * 1e3,
                                    "resident_total_mW": d["resident_total_W"] * 1e3, "lpddr_total_mW": d["lpddr_total_W"] * 1e3, "lpddr_over_resident": d["lpddr_over_resident"],
                                    "transition_mJ_per_token": d["transition_mJ_per_token"], "stall_us_per_token": d["stall_us_per_token"], "active_capacity_fraction": d["active_capacity_fraction"],
                                    "lpddr_channels_needed": math.ceil(s["weight_bytes"] * R_tok / (E8.LPDDR_CHANNEL_GBPS * 1e9 * 0.7))}))
    return rows


def duty_rows(scheds: dict, R_active: int = 25) -> list[dict]:
    rows = []
    for (slug, bits), s in scheds.items():
        for temp in R.TEMPS_C:
            for mode in MODES:
                sc = Scenario(s, "planning", temp, mode)
                for d in R.DUTY_CYCLES:
                    for pol, k in (("P0", 0), ("PS", 0), ("window", 2), ("window", 4)):
                        rows.append(_r({"model": slug, "weight_bits": bits, "temp_c": temp, "mode": mode, **sc.duty(d, R_active, pol, k)}))
                        if temp == 85:
                            rows.append(_r({"model": slug, "weight_bits": bits, "temp_c": temp, "mode": mode, **sc.duty(d, R_active, pol, k, idle_temp_c=R.IDLE_TEMP_C)}))
    return rows


def mode_sensitivity_rows(scheds: dict) -> list[dict]:
    """Falcon-E 1B: every retention mode x leakage level x temperature at 25 tok/s continuous (P2) and 25% duty (PS)."""
    rows = []
    s = scheds[("falcon-e-1b", 2.0)]
    for name, kw in (("RETENTION_VRET pessimistic ratio 0.23 at 25 C (B4 sensitivity)", dict(temp_c=25, ratio_override=R.PESSIMISTIC_RETENTION_RATIO_25C)),
                     ("RETENTION_VRET with IDD2N/IDD2P halved at 25 C (B15 sensitivity)", dict(temp_c=25, lpddr_bg_scale=0.5))):
        sc = Scenario(s, "planning", kw["temp_c"], "RETENTION_VRET", ratio_override=kw.get("ratio_override"), lpddr_bg_scale=kw.get("lpddr_bg_scale", 1.0))
        p0 = sc.P0(25); p2 = sc.window(2, 25); ps = sc.duty(0.25, 25, "PS")
        rows.append(_r({"model": "falcon-e-1b", "level": "planning", "temp_c": kw["temp_c"], "mode": name, "v_ret": R.V_RET_PLANNING, "active_mW_per_MB": sc.act_mw_mb, "retention_ratio": sc.ratio, "retention_mW_per_MB": sc.ret_mw_mb,
                        "wake_uJ_per_MB": sc.wake_uj_mb, "P0_resident_mW": p0["resident_total_W"] * 1e3, "P2_resident_mW": p2["resident_total_W"] * 1e3, "P2_transition_mJ_per_token": p2["transition_mJ_per_token"],
                        "lpddr_mW": p0["lpddr_total_W"] * 1e3, "P0_ratio": p0["lpddr_over_resident"], "P2_ratio": p2["lpddr_over_resident"], "PS_duty25_resident_mW": ps["resident_avg_mW"], "PS_duty25_ratio": ps["lpddr_over_resident"],
                        "PS_duty25_ratio_idle_at_%dC" % R.IDLE_TEMP_C: "", "note": "sensitivity row"}))
    for level in LEVELS:
        for temp in (25, 50, 85):
            for mode in ("LIGHT_SLEEP", "DEEP_SLEEP_NOMINAL", "RETENTION_VRET", "BODY_BIAS_BOUND"):
                for v in ((R.V_RET_PLANNING,) if mode != "RETENTION_VRET" else (0.25, 0.35, 0.45)):
                    sc = Scenario(s, level, temp, mode, v_ret=v)
                    p0 = sc.P0(25); p2 = sc.window(2, 25); ps = sc.duty(0.25, 25, "PS"); ps_cool = sc.duty(0.25, 25, "PS", idle_temp_c=R.IDLE_TEMP_C) if temp == 85 else None
                    rows.append(_r({"model": "falcon-e-1b", "level": level, "temp_c": temp, "mode": mode, "v_ret": v, "active_mW_per_MB": sc.act_mw_mb, "retention_ratio": sc.ratio, "retention_mW_per_MB": sc.ret_mw_mb,
                                    "wake_uJ_per_MB": sc.wake_uj_mb, "P0_resident_mW": p0["resident_total_W"] * 1e3, "P2_resident_mW": p2["resident_total_W"] * 1e3, "P2_transition_mJ_per_token": p2["transition_mJ_per_token"],
                                    "lpddr_mW": p0["lpddr_total_W"] * 1e3, "P0_ratio": p0["lpddr_over_resident"], "P2_ratio": p2["lpddr_over_resident"],
                                    "PS_duty25_resident_mW": ps["resident_avg_mW"], "PS_duty25_ratio": ps["lpddr_over_resident"],
                                    "PS_duty25_ratio_idle_at_%dC" % R.IDLE_TEMP_C: ps_cool["lpddr_over_resident"] if ps_cool else "",
                                    "note": "BODY_BIAS_BOUND is an FD-SOI (R05) bound, not available to FinFET; LIGHT_SLEEP is the only mode with a T1 modern-node figure"}))
    return rows


def exp0008_reproduction_rows(scheds: dict) -> list[dict]:
    """P0 with EXP-0008's flat 0.60 mW/MB input must reproduce EXP-0008's dynamic_energy / duty tables (tests check)."""
    rows = []
    for (slug, bits), s in scheds.items():
        if bits != 2.0:
            continue
        sc = Scenario(s, "planning", 25, "DEEP_SLEEP_NOMINAL", leak_mw_per_mb=R.EXP0008_LEAK_MW_PER_MB)
        p = sc.P0(25)
        be = (sc.p_all_active() - E8.LPDDR_BACKGROUND_MW["power_down"] * 1e-3) / (sc.e_lp - sc.e_dyn)
        rows.append(_r({"model": slug, "resident_dynamic_mJ_per_token": sc.e_dyn * 1e3, "lpddr_dynamic_mJ_per_token": sc.e_lp * 1e3, "dynamic_ratio": sc.e_lp / sc.e_dyn,
                        "sram_static_mW_at_0.60": sc.p_all_active() * 1e3, "break_even_tok_s": be, "resident_total_mW_25tok": p["resident_total_W"] * 1e3, "lpddr_total_mW_25tok": p["lpddr_total_W"] * 1e3,
                        "duty_be_retention_1x": sc.duty(0.85, 25, "P0", lpddr_idle="power_down")["lpddr_over_resident"], "note": "EXP-0008 A4 planning input (0.60 mW/MB flat); compare results/sram_validation/dynamic_energy.csv and break_even.csv"}))
    return rows


def session_rows(scheds: dict) -> list[dict]:
    rows = []
    for (slug, bits), s in scheds.items():
        for temp in (25, 85):
            sc = Scenario(s, "planning", temp, "RETENTION_VRET")
            for R_tok in (5, 25, 100):
                for N in R.SESSION_TOKENS:
                    for gap in R.IDLE_GAPS_S:
                        for ins in ("P0", "window"):
                            for it in ((None, R.IDLE_TEMP_C) if temp == 85 else (None,)):
                                px = sc.session(N, R_tok, gap, ins, 2, idle_temp_c=it, idle_mode="shutdown")
                                rows.append(_r({"model": slug, "weight_bits": bits, "temp_c": temp, **px, "resident_never_retain_avg_mW": ""}))
                                d = sc.session(N, R_tok, gap, ins, 2, idle_temp_c=it)
                                # reference: never retain (P0 during session and idle)
                                p0 = sc.P0(R_tok); T = N / R_tok + gap
                                never = (p0["resident_total_W"] * N / R_tok + sc.p_all_active_at(it) * gap) / T * 1e3
                                rows.append(_r({"model": slug, "weight_bits": bits, "temp_c": temp, **d, "resident_never_retain_avg_mW": never}))
    return rows


def temperature_rows(scheds: dict) -> list[dict]:
    rows = []
    for (slug, bits), s in scheds.items():
        for temp in R.TEMPS_C + (105,):
            for mode in MODES:
                sc = Scenario(s, "planning", temp, mode)
                for R_tok in (5, 25, 100):
                    p0 = sc.P0(R_tok); p2 = sc.window(2, R_tok); ps25 = sc.duty(0.25, R_tok, "PS")
                    rows.append(_r({"model": slug, "weight_bits": bits, "temp_c": temp, "mode": mode, "tok_s": R_tok, "active_mW_per_MB": sc.act_mw_mb, "retention_mW_per_MB": sc.ret_mw_mb,
                                    "P0_resident_mW": p0["resident_total_W"] * 1e3, "P2_resident_mW": p2["resident_total_W"] * 1e3, "lpddr_mW": p0["lpddr_total_W"] * 1e3,
                                    "P0_ratio": p0["lpddr_over_resident"], "P2_ratio": p2["lpddr_over_resident"], "PS_duty25_resident_mW": ps25["resident_avg_mW"], "PS_duty25_lpddr_mW": ps25["lpddr_avg_mW"], "PS_duty25_ratio": ps25["lpddr_over_resident"]}))
    return rows


def voltage_domain_rows(scheds: dict) -> list[dict]:
    """Domain = g layers switched together: the window must cover g (coarser domains keep more active)."""
    rows = []
    for (slug, bits), s in scheds.items():
        sc = Scenario(s, "planning", 85, "RETENTION_VRET")
        for g in (1, 2, 4, 8, s["L"]):
            k = min(2 * g, s["L"])       # current domain + next domain active
            dom_bytes = sum(sorted(s["layer_provisioned_bytes"], reverse=True)[:g])
            w = sc.window(k, 25)
            rows.append(_r({"model": slug, "weight_bits": bits, "domain_layers": g, "domain_MB": dom_bytes / 1e6, "domains": math.ceil(s["L"] / g), "active_layers_window": k,
                            "active_capacity_fraction": w["active_capacity_fraction"], "wake_latency_domain_us": sc.wake_t(dom_bytes) * 1e6, "wake_energy_domain_uJ": sc.wake_e(dom_bytes) * 1e6,
                            "resident_total_mW_25tok_85C": w["resident_total_W"] * 1e3, "lpddr_mW": w["lpddr_total_W"] * 1e3, "ratio": w["lpddr_over_resident"], "label": "B8: macro-granular switches grouped into layer domains"}))
    return rows


def wake_sweep_rows(scheds: dict) -> list[dict]:
    rows = []
    for (slug, bits), s in scheds.items():
        if bits != 2.0:
            continue
        for bw_name, bw in R.ENGINE_BW_BPS.items():
            t_layer_b = max(s["layer_bytes"]) / bw
            for R_tok in (25, 250):
                t_layer_p = 1 / R_tok / s["L"]
                for tw in R.WAKE_SWEEP_S:
                    rows.append(_r({"model": slug, "engine": bw_name, "tok_s": R_tok, "wake_latency_us": tw * 1e6, "layer_time_burst_us": t_layer_b * 1e6, "layer_time_paced_us": t_layer_p * 1e6,
                                    "P2_slack_burst_us": (t_layer_b - tw) * 1e6, "P2_slack_paced_us": (t_layer_p - tw) * 1e6, "P4_slack_burst_us": (3 * t_layer_b - tw) * 1e6,
                                    "P2_burst_hidden": t_layer_b >= tw, "P2_paced_hidden": t_layer_p >= tw, "P1_stall_per_token_us": s["L"] * tw * 1e6,
                                    "label": "B9 TARGET engine bandwidth; no physical clock"}))
    return rows


# ----------------------------------------------------------------------------- break-even thresholds
def break_even_rows(scheds: dict) -> list[dict]:
    rows = []
    for (slug, bits), s in scheds.items():
        if bits != 2.0:
            continue
        for temp in (25, 85):
            sc = Scenario(s, "planning", temp, "RETENTION_VRET")
            R_tok = 25; Tp = 1 / R_tok
            p_act = sc.p_all_active(); busy, bg = lpddr_background_mw(sc.W, R_tok)
            p_lp = sc.e_lp * R_tok + bg * 1e-3
            e_wake_layers = sum(sc.wake_e(b) for b in sc.layer_prov)
            sav = sc.e_lp - sc.e_dyn
            k = 2; f = k / sc.L
            # 1-2. retention ratio / absolute retention mW/MB at which P2 (paced) ties LPDDR at 25 tok/s continuous
            # p_lp = e_dyn R + f p_act + (1-f) r p_act + e_wake_layers R
            r_star = (p_lp - sc.e_dyn * R_tok - f * p_act - e_wake_layers * R_tok) / ((1 - f) * p_act)
            rows += [{"model": slug, "temp_c": temp, "threshold": "1. retention ratio (retained/active leakage) at which P2 ties LPDDR at 25 tok/s continuous", "value": round(r_star, 4) if r_star <= 1 else "any ratio <= 1 wins (solution %.2f > 1)" % r_star, "unit": "ratio (model planning ratio %.3f)" % sc.ratio},
                     {"model": slug, "temp_c": temp, "threshold": "2. absolute retention leakage at which P2 ties LPDDR at 25 tok/s continuous", "value": round(r_star * sc.act_mw_mb, 4), "unit": "mW/MB (model planning %.3f)" % sc.ret_mw_mb}]
            # 3. wake latency at which Pk starts stalling (paced at 25 tok/s and burst)
            for kk in (2, 4):
                rows.append({"model": slug, "temp_c": temp, "threshold": f"3. wake latency at which P{kk} begins stalling (paced, 25 tok/s / burst at 8 TB/s)", "value": f"{(kk - 1) / R_tok / sc.L * 1e6:.1f} / {(kk - 1) * max(sc.layer_bytes) / sc.engine_bw * 1e6:.2f}", "unit": "us (model layer wake %.2f us)" % (sc.wake_t(max(sc.layer_prov)) * 1e6)})
            # 4. wake energy per MB at which per-token layer retention stops paying (P2 vs P0) at R
            for RR in (5, 25, 100, 250):
                saved_per_token = (1 - f) * (p_act - sc.p_all_retained()) / RR
                mb = sum(sc.layer_prov) / 1e6
                rows.append({"model": slug, "temp_c": temp, "threshold": f"4. wake energy per MB at which P2 (every layer every token) stops beating P0 at {RR} tok/s", "value": round(saved_per_token / mb * 1e6, 4), "unit": "uJ/MB (model %.2f)" % sc.wake_uj_mb})
            # 5. minimum layer execution time that hides the layer wake with a P2 window
            rows.append({"model": slug, "temp_c": temp, "threshold": "5. minimum layer execution time to hide the layer wake (P2)", "value": round(sc.wake_t(max(sc.layer_prov)) * 1e6, 3), "unit": "us (= max token rate %.0f tok/s paced)" % (1 / (sc.wake_t(max(sc.layer_prov)) * sc.L))})
            # 6. minimum session length (tokens) for PS to beat P0-never-retain with a given idle gap
            for gap in (1.0, 10.0, 60.0):
                # PS: E = N e_tok + N/R p_act + E_wake_model + gap p_ret ; never: N e_tok + N/R p_act + gap p_act -> PS wins iff E_wake_model < gap (p_act - p_ret): independent of N
                rows.append({"model": slug, "temp_c": temp, "threshold": f"6. minimum session length for PS to beat never-retaining with a {gap:g} s idle gap", "value": "any N (idle saving %.3f mJ vs model wake %.3f mJ)" % (gap * (p_act - sc.p_all_retained()) * 1e3, sc.wake_e(sc.prov) * 1e3) if gap * (p_act - sc.p_all_retained()) > sc.wake_e(sc.prov) else "never", "unit": "tokens"})
            # 7. inactivity timeout break-even: X* = E_wake_model / (p_act - p_ret)
            X = sc.wake_e(sc.prov) / (p_act - sc.p_all_retained())
            rows.append({"model": slug, "temp_c": temp, "threshold": "7. idle time after which entering whole-model retention saves energy (break-even time)", "value": round(X * 1e3, 4), "unit": "ms"})
            gx = sc.e_reload / (sc.p_all_retained() - sc.p_shutdown)
            rows.append({"model": slug, "temp_c": temp, "threshold": "11. idle gap above which shutdown + reload (PX, data lost) beats retention", "value": round(gx, 3), "unit": "s (reload %.1f mJ, retained %.1f mW)" % (sc.e_reload * 1e3, sc.p_all_retained() * 1e3)})
            # 8. token-rate crossover vs LPDDR for P0, P2, PT0
            for name, fn in (("P0", lambda rr: sc.P0(rr)), ("P2", lambda rr: sc.window(2, rr)), ("PT0", lambda rr: sc.timeout(0.0, rr))):
                lo, hi = 0.01, 1000.0
                if fn(hi)["lpddr_over_resident"] < 1:
                    val = "never below 1000 tok/s"
                elif fn(lo)["lpddr_over_resident"] >= 1:
                    val = "always (>= 0.01 tok/s)"
                else:
                    for _ in range(60):
                        m = math.sqrt(lo * hi)
                        if fn(m)["lpddr_over_resident"] >= 1:
                            hi = m
                        else:
                            lo = m
                    val = round(hi, 3)
                rows.append({"model": slug, "temp_c": temp, "threshold": f"8. token rate (continuous) above which {name} beats LPDDR", "value": val, "unit": "tok/s"})
            # 9. duty-cycle crossover at 25 tok/s for P0 (no retention), PS, P2+S
            for pol, kk in (("P0", 0), ("PS", 0), ("window", 2)):
                lo, hi = 0.0, 1.0
                if sc.duty(1.0, 25, pol, kk)["lpddr_over_resident"] < 1:
                    val = "never"
                elif sc.duty(1e-4, 25, pol, kk)["lpddr_over_resident"] >= 1:
                    val = "always"
                else:
                    for _ in range(50):
                        m = (lo + hi) / 2
                        if sc.duty(m, 25, pol, kk)["lpddr_over_resident"] >= 1:
                            hi = m
                        else:
                            lo = m
                    val = round(hi, 4)
                rows.append({"model": slug, "temp_c": temp, "threshold": f"9. duty cycle at 25 tok/s above which {'P2+S' if pol == 'window' else pol} beats LPDDR", "value": val, "unit": "fraction"})
        # 10. temperature at which leakage flips the verdict (P2 at 25 tok/s continuous; PS at 25% duty)
        for name, fn in (("P2 continuous 25 tok/s", lambda t: Scenario(s, "planning", t, "RETENTION_VRET").window(2, 25)["lpddr_over_resident"]),
                         ("PS at 25% duty, 25 tok/s", lambda t: Scenario(s, "planning", t, "RETENTION_VRET").duty(0.25, 25, "PS")["lpddr_over_resident"]),
                         ("P0 continuous 25 tok/s", lambda t: Scenario(s, "planning", t, "RETENTION_VRET").P0(25)["lpddr_over_resident"])):
            lo, hi = 25.0, 105.0
            if fn(hi) >= 1:
                val = "> 105 C (residency wins at every temperature modeled)"
            elif fn(lo) < 1:
                val = "< 25 C (residency loses at every temperature modeled)"
            else:
                for _ in range(40):
                    m = (lo + hi) / 2
                    if fn(m) >= 1:
                        lo = m
                    else:
                        hi = m
                val = round(lo, 1)
            rows.append({"model": slug, "temp_c": "sweep", "threshold": f"10. temperature above which {name} loses to LPDDR", "value": val, "unit": "C"})
    return rows


# ----------------------------------------------------------------------------- INT4, head/KV, primary
def int4_rows(scheds: dict) -> list[dict]:
    rows = []
    for slug in sorted({k[0] for k in scheds}):
        s2, s4 = scheds[(slug, 2.0)], scheds[(slug, 4.5)]
        for temp in (25, 85):
            a, b = Scenario(s2, "planning", temp, "RETENTION_VRET"), Scenario(s4, "planning", temp, "RETENTION_VRET")
            for pol in ("P0", "P2", "PS25"):
                if pol == "P0":
                    da, db = a.P0(25), b.P0(25); key = "resident_total_W"; sc_ = 1e3; lp = "lpddr_total_W"
                elif pol == "P2":
                    da, db = a.window(2, 25), b.window(2, 25); key = "resident_total_W"; sc_ = 1e3; lp = "lpddr_total_W"
                else:
                    da, db = a.duty(0.25, 25, "PS"), b.duty(0.25, 25, "PS"); key = "resident_avg_mW"; sc_ = 1; lp = "lpddr_avg_mW"
                rows.append(_r({"model": slug, "temp_c": temp, "policy": pol, "ternary_provisioned_MB": s2["provisioned_bytes"] / 1e6, "int4_provisioned_MB": s4["provisioned_bytes"] / 1e6,
                                "ternary_resident_mW": da[key] * sc_, "int4_resident_mW": db[key] * sc_, "ternary_lpddr_mW": da[lp] * sc_, "int4_lpddr_mW": db[lp] * sc_,
                                "ternary_ratio": da["lpddr_over_resident"], "int4_ratio": db["lpddr_over_resident"], "int4_over_ternary_resident_power": db[key] / da[key],
                                "ternary_layer_wake_us": a.wake_t(max(a.layer_prov)) * 1e6, "int4_layer_wake_us": b.wake_t(max(b.layer_prov)) * 1e6,
                                "ternary_transition_mJ_per_token_P2": a.window(2, 25)["transition_mJ_per_token"], "int4_transition_mJ_per_token_P2": b.window(2, 25)["transition_mJ_per_token"],
                                "note": "identical policy, mode, voltage, temperature, wake model; differences follow from 2.25x capacity"}))
    return rows


def system_rows(scheds: dict, ws: dict) -> list[dict]:
    rows = []
    for (slug, bits), s in scheds.items():
        if bits != 2.0:
            continue
        w = ws[slug]
        for temp in (25, 85):
            sc = Scenario(s, "planning", temp, "RETENTION_VRET")
            for ctx in (2048, 8192):
                for label, hb, kvb in (("BF16 head + BF16 KV", 2.0, 2.0), ("EXP-0004 candidate head 4.156 b + KV 4.5 b", E8.HEAD_CANDIDATE_BITS / 8, E8.KV_CANDIDATE_BITS / 8)):
                    head = head_bytes(w, hb * 8); kv = kv_read_bytes(w, ctx, kvb) + kv_write_bytes(w, kvb)
                    e_ext = (head + kv) * 8 * E_LP_PJ * 1e-12
                    p0 = sc.P0(25); p2 = sc.window(2, 25); ps = sc.duty(0.25, 25, "PS"); p0d = sc.duty(0.25, 25, "P0")
                    rows.append(_r({"model": slug, "temp_c": temp, "context": ctx, "case": label, "external_head_kv_mJ_per_token_both": e_ext * 1e3,
                                    "weights_only_ratio_P0_25tok": p0["lpddr_over_resident"], "weights_only_ratio_P2_25tok": p2["lpddr_over_resident"],
                                    "total_ratio_P0_25tok": (p0["lpddr_total_W"] + e_ext * 25) / (p0["resident_total_W"] + e_ext * 25),
                                    "total_ratio_P2_25tok": (p2["lpddr_total_W"] + e_ext * 25) / (p2["resident_total_W"] + e_ext * 25),
                                    "total_ratio_P0_duty25": (p0d["lpddr_avg_mW"] + e_ext * 25 * 0.25 * 1e3) / (p0d["resident_avg_mW"] + e_ext * 25 * 0.25 * 1e3),
                                    "total_ratio_PS_duty25": (ps["lpddr_avg_mW"] + e_ext * 25 * 0.25 * 1e3) / (ps["resident_avg_mW"] + e_ext * 25 * 0.25 * 1e3),
                                    "note": "head and KV external at LPDDR energy in both systems (EXP-0008 convention); candidate remains CANDIDATE"}))
    return rows


def primary_rows(scheds: dict) -> list[dict]:
    rows = []
    for (slug, bits), s in scheds.items():
        if bits != 2.0:
            continue
        for temp in (25, 85):
            sc = Scenario(s, "planning", temp, "RETENTION_VRET")
            p0 = sc.P0(25); p2 = sc.window(2, 25); ps = sc.duty(0.25, 25, "PS")
            be = next(r for r in break_even_rows({(slug, 2.0): s}) if r["temp_c"] == temp and r["threshold"].startswith("8. token rate (continuous) above which P2"))
            bd = next(r for r in break_even_rows({(slug, 2.0): s}) if r["temp_c"] == temp and r["threshold"].startswith("9. duty cycle at 25 tok/s above which PS"))
            verdict = ("wins continuous and 25% duty" if (p2["lpddr_over_resident"] > 1.0 and ps["lpddr_over_resident"] > 1.0) else ("loses both" if p2["lpddr_over_resident"] < 1.0 else "wins continuous only"))
            ps_cool = sc.duty(0.25, 25, "PS", idle_temp_c=R.IDLE_TEMP_C)
            rows.append({"model": slug, "temp_c": temp, "provisioned_MB": round(sc.prov / 1e6, 1), "active_leakage_mW_per_MB": round(sc.act_mw_mb, 3), "retention_leakage_mW_per_MB": round(sc.ret_mw_mb, 4),
                         "retention_ratio": round(sc.ratio, 3), "retention_voltage_V": sc.v_ret, "wake_latency_layer_us": round(sc.wake_t(max(sc.layer_prov)) * 1e6, 2), "wake_energy_layer_uJ": round(sc.wake_e(max(sc.layer_prov)) * 1e6, 2),
                         "active_layer_window": "P2 (current + next)", "resident_dynamic_mJ_per_token": round(sc.e_dyn * 1e3, 3), "resident_static_mW_P0": round(p0["resident_leakage_W"] * 1e3, 1), "resident_static_mW_P2": round(p2["resident_leakage_W"] * 1e3, 1),
                         "transition_mJ_per_token_P2": round(p2["transition_mJ_per_token"], 3), "resident_total_mW_P2_25tok": round(p2["resident_total_W"] * 1e3, 1), "lpddr_total_mW_25tok": round(p2["lpddr_total_W"] * 1e3, 1),
                         "ratio_P2_25tok": round(p2["lpddr_over_resident"], 2), "resident_avg_mW_PS_duty25": round(ps["resident_avg_mW"], 1), "lpddr_avg_mW_duty25": round(ps["lpddr_avg_mW"], 1), "ratio_PS_duty25": round(ps["lpddr_over_resident"], 2), "ratio_PS_duty25_idle_at_%dC" % R.IDLE_TEMP_C: round(ps_cool["lpddr_over_resident"], 2),
                         "break_even_tok_s_P2": be["value"], "break_even_duty_PS_25tok": bd["value"], "heuristic_flag_not_a_verdict": verdict,
                         "evidence_quality": "active leakage T1 (25 C) x T2 temperature; retention ratio T2 (ASAP7) + T3 modes; wake B6/B7 ASSUMPTION; dynamic T1-anchored (EXP-0008)"})
    return rows


def walkthrough_rows(scheds: dict) -> list[dict]:
    rows = []
    s = scheds[("falcon-e-1b", 2.0)]
    for temp in (25, 85):
        sc = Scenario(s, "planning", temp, "RETENTION_VRET")
        for R_tok in (5, 25, 100):
            items = [("P0 all active", sc.P0(R_tok)), ("P1 one layer", sc.window(1, R_tok)), ("P2 current + next", sc.window(2, R_tok)), ("P4 window", sc.window(4, R_tok)),
                     ("P2 burst (8 TB/s engine)", sc.window(2, R_tok, "burst")), ("PT0 race-to-idle (whole model sleeps between tokens)", sc.timeout(0.0, R_tok))]
            for name, d in items:
                rows.append(_r({"temp_c": temp, "tok_s": R_tok, "policy": name, "dynamic_mW": d["resident_dynamic_W"] * 1e3, "active_leakage_mW": d["active_capacity_fraction"] * sc.p_all_active() * 1e3,
                                "retention_leakage_mW": (1 - d["active_capacity_fraction"]) * sc.p_all_retained() * 1e3, "wake_mW": d["resident_wake_W"] * 1e3, "resident_total_mW": d["resident_total_W"] * 1e3,
                                "stall_us_per_token": d["stall_us_per_token"], "lpddr_dynamic_mW": d["lpddr_dynamic_W"] * 1e3, "lpddr_background_mW": d["lpddr_background_W"] * 1e3, "lpddr_total_mW": d["lpddr_total_W"] * 1e3, "lpddr_over_resident": d["lpddr_over_resident"]}))
            ps = sc.session(256, R_tok, 10.0, "P0"); px = sc.session(256, R_tok, 10.0, "P0", idle_mode="shutdown")
            rows.append(_r({"temp_c": temp, "tok_s": R_tok, "policy": "PS session 256 tokens + 10 s idle retained (avg over session+idle)", "dynamic_mW": "", "active_leakage_mW": "", "retention_leakage_mW": "", "wake_mW": "",
                            "resident_total_mW": ps["resident_avg_mW"], "stall_us_per_token": ps["first_token_stall_us"], "lpddr_dynamic_mW": "", "lpddr_background_mW": "", "lpddr_total_mW": ps["lpddr_avg_mW"], "lpddr_over_resident": ps["lpddr_over_resident"]}))
            rows.append(_r({"temp_c": temp, "tok_s": R_tok, "policy": "PX session 256 tokens + 10 s idle shut down, reload 13 mJ (data lost; not retention)", "dynamic_mW": "", "active_leakage_mW": "", "retention_leakage_mW": "", "wake_mW": "",
                            "resident_total_mW": px["resident_avg_mW"], "stall_us_per_token": px["first_token_stall_us"], "lpddr_dynamic_mW": "", "lpddr_background_mW": "", "lpddr_total_mW": px["lpddr_avg_mW"], "lpddr_over_resident": px["lpddr_over_resident"]}))
    return rows
