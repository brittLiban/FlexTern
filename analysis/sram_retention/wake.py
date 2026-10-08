"""
EXP-0009 wake model: latency and energy of bringing a retained SRAM domain back to active state.
CORRECTED 2026-10-08 (EXP-0009A): the wake energy is built from physical rail capacitances in SI units.

  t_wake(domain)  = max(t_floor, t_macro x ceil(macros / parallel_wake))      (B6: in-rush-limited staggered macro wake)
  E_wake(domain)  = MB x E_mb[mode, level] + macros x E_sequencer               (B7)
  E_mb            = sum over switched rails of  C_rail x V_active x dV          (B7; supply energy of a hard-switched rail)
      rails: (a) cell array rail, dV = V_active - V_ret          [RETENTION_VRET exit only]
             (b) bitlines, dV (follow the array rail) or V_active (discharged, HIGH)
             (c) periphery virtual rail, dV = V_active            [periphery-off modes]
             (d) decoupling + grid share of the switched capacitance

Sleep entry is charged nothing (the stored rail charge is dissipated, not drawn from the supply); every wake is charged
exactly once (tests check the count).  No measured modern-node macro wake energy exists; the per-MB figures are
physics-based estimates with a range (retention_evidence.ASSUMPTIONS B7).  The EXP-0009 (899976e) values, which
applied 1.2 uJ/MB where the stated formula gives 1.2 nJ/MB, are kept as `exp0009_original_j_per_mb` for reproduction only.
"""

from __future__ import annotations

import math

import retention_evidence as R

CELLS_PER_MB = 8e6                      # decimal MB of 2-bit weights = 8e6 bitcells
RAIL_COMPONENTS = {                     # which rails swing on exit from each mode (B7)
    "RETENTION_VRET": ("cell_rail", "bitline", "periphery", "decap"),
    "BODY_BIAS_BOUND": ("bitline", "periphery", "decap"),          # bound only: well charge not modeled
    "DEEP_SLEEP_NOMINAL": ("bitline", "periphery", "decap"),
    "LIGHT_SLEEP": ("bitline",),
    "ACTIVE_STANDBY": (),
}


def rail_capacitances_f_per_mb(level: str = "planning") -> dict:
    """Switched capacitance per decimal MB by rail, in farads (B7 a-c; d is a share, applied in components)."""
    return {"cell_rail": CELLS_PER_MB * R.CELL_RAIL_FF_PER_CELL[level] * 1e-15,
            "bitline": CELLS_PER_MB * 2 * R.BITLINE_FF_PER_CELL_PER_LINE[level] * 1e-15,
            "periphery": R.PERIPHERY_NF_PER_MB[level] * 1e-9}


def rail_energy_j(c_f: float, v_active: float, dv: float) -> float:
    """Supply energy to recharge capacitance c_f from (v_active - dv) to v_active through a switch: E = C V_active dV (KNOWN)."""
    return c_f * v_active * dv


def wake_components_j_per_mb(mode: str = "RETENTION_VRET", level: str = "planning", v_active: float = R.V_ACTIVE, v_ret: float = R.V_RET_PLANNING) -> dict:
    """Per-MB wake energy by component (J) for exiting `mode`; 'total' is the sum (B7, corrected)."""
    c = rail_capacitances_f_per_mb(level)
    dv_arr = v_active - v_ret
    comps = {}
    rails = RAIL_COMPONENTS[mode]
    if "cell_rail" in rails:
        comps["cell_rail"] = rail_energy_j(c["cell_rail"], v_active, dv_arr)
    if "bitline" in rails:
        # bitlines follow the array rail (swing dV) unless the level assumes they were discharged (full V_active swing);
        # on exit from a periphery-only mode the array rail did not move, so only a discharged bitline costs anything
        if R.BITLINE_FULL_SWING[level]:
            comps["bitline"] = rail_energy_j(c["bitline"], v_active, v_active)
        elif "cell_rail" in rails:
            comps["bitline"] = rail_energy_j(c["bitline"], v_active, dv_arr)
        else:
            comps["bitline"] = 0.0
    if "periphery" in rails:
        comps["periphery"] = rail_energy_j(c["periphery"], v_active, v_active)
    if "decap" in rails:
        comps["decap_grid"] = R.DECAP_GRID_SHARE[level] * sum(comps.values())
    comps["total"] = sum(v for k, v in comps.items() if k != "total")
    return comps


def wake_j_per_mb(mode: str = "RETENTION_VRET", level: str = "planning", v_active: float = R.V_ACTIVE, v_ret: float = R.V_RET_PLANNING) -> float:
    return wake_components_j_per_mb(mode, level, v_active, v_ret)["total"]


def exp0009_original_j_per_mb(mode: str, level: str) -> float:
    """SUPERSEDED EXP-0009 (899976e) per-MB wake energy, J, as the original Scenario applied it (reproduction only)."""
    if mode == "RETENTION_VRET":
        return {"low": R.EXP0009_ORIGINAL_WAKE_UJ_PER_MB["planning"], "planning": R.EXP0009_ORIGINAL_ARRAY_SWING_UJ_PER_MB,
                "high": R.EXP0009_ORIGINAL_ARRAY_SWING_UJ_PER_MB * 2}[level] * 1e-6
    return R.EXP0009_ORIGINAL_WAKE_UJ_PER_MB[level] * 1e-6


def macros_in(domain_bytes: float, macro_bytes: float = R.MACRO_BYTES) -> int:
    return max(1, math.ceil(domain_bytes / macro_bytes))


def wake_latency_s(domain_bytes: float, level: str = "planning", macro_bytes: float = R.MACRO_BYTES, t_macro_s: float | None = None,
                   parallel: int | None = None) -> float:
    t_macro = R.WAKE_T_MACRO_S[level] if t_macro_s is None else t_macro_s
    par = R.WAKE_PARALLEL_MACROS[level] if parallel is None else parallel
    n = macros_in(domain_bytes, macro_bytes)
    return max(R.WAKE_T_FLOOR_S, t_macro * math.ceil(n / par))


def wake_energy_j(domain_bytes: float, level: str = "planning", j_per_mb: float | None = None, macro_bytes: float = R.MACRO_BYTES,
                  mode: str = "RETENTION_VRET", v_active: float = R.V_ACTIVE, v_ret: float = R.V_RET_PLANNING) -> float:
    """Wake energy of a domain (J): per-MB rail recharge (B7) plus the sequencer per macro.  j_per_mb overrides the model."""
    per_mb = wake_j_per_mb(mode, level, v_active, v_ret) if j_per_mb is None else j_per_mb
    return domain_bytes / 1e6 * per_mb + macros_in(domain_bytes, macro_bytes) * R.WAKE_SEQUENCER_J


def inrush_a(level: str = "planning", mode: str = "RETENTION_VRET", macro_bytes: float = R.MACRO_BYTES, v_active: float = R.V_ACTIVE,
             v_ret: float = R.V_RET_PLANNING, t_macro_s: float | None = None) -> dict:
    """Average in-rush current implied by B7's charge and B6's wake time: per macro and per parallel group (consistency check)."""
    c = rail_capacitances_f_per_mb(level)
    rails = RAIL_COMPONENTS[mode]
    q = 0.0
    if "cell_rail" in rails:
        q += c["cell_rail"] * (v_active - v_ret)
    if "bitline" in rails:
        q += c["bitline"] * (v_active if R.BITLINE_FULL_SWING[level] else ((v_active - v_ret) if "cell_rail" in rails else 0.0))
    if "periphery" in rails:
        q += c["periphery"] * v_active
    if "decap" in rails:
        q *= 1 + R.DECAP_GRID_SHARE[level]
    q_macro = q * macro_bytes / 1e6
    t = R.WAKE_T_MACRO_S[level] if t_macro_s is None else t_macro_s
    return {"charge_per_macro_C": q_macro, "i_per_macro_A": q_macro / t, "i_per_group_A": q_macro / t * R.WAKE_PARALLEL_MACROS[level], "t_macro_s": t}


def wake_rows(schedules: dict) -> list[dict]:
    rows = []
    for (slug, bits), s in schedules.items():
        for level in ("low", "planning", "high"):
            for name, b in (("one macro", R.MACRO_BYTES), ("one layer (largest)", s["largest_layer_provisioned_bytes"]), ("whole model", s["provisioned_bytes"])):
                e = wake_energy_j(b, level)
                e_old = wake_energy_j(b, level, j_per_mb=exp0009_original_j_per_mb("RETENTION_VRET", level))
                rows.append({"model": slug, "weight_bits": bits, "domain": name, "domain_MB": round(b / 1e6, 3), "level": level,
                             "macros": macros_in(b), "wake_latency_us": round(wake_latency_s(b, level) * 1e6, 3),
                             "wake_energy_uJ": round(e * 1e6, 6), "wake_energy_nJ_per_MB": round(wake_j_per_mb("RETENTION_VRET", level) * 1e9, 4),
                             "per_token_if_every_layer_wakes_uJ": round(s["L"] * wake_energy_j(s["largest_layer_provisioned_bytes"], level) * 1e6, 4) if name.startswith("one layer") else "",
                             "superseded_exp0009_wake_energy_uJ": round(e_old * 1e6, 4),
                             "superseded_exp0009_per_token_mJ": round(s["L"] * wake_energy_j(s["largest_layer_provisioned_bytes"], level, j_per_mb=exp0009_original_j_per_mb("RETENTION_VRET", level)) * 1e3, 4) if name.startswith("one layer") else "",
                             "inrush_per_macro_mA": round(inrush_a(level)["i_per_macro_A"] * 1e3, 3),
                             "label": "B6/B7 ASSUMPTION ranges (B7 corrected 2026-10-08, EXP-0009A); R05 (200 ns per 128 KiB macro, 22 nm FD-SOI) and R03 (4-10 cycles, 7 nm PDK) anchor latency; R11 anchors the nJ/mm2 class"})
    return rows


def wake_component_rows() -> list[dict]:
    rows = []
    for mode in ("RETENTION_VRET", "DEEP_SLEEP_NOMINAL", "LIGHT_SLEEP"):
        for level in ("low", "planning", "high"):
            c = rail_capacitances_f_per_mb(level); comps = wake_components_j_per_mb(mode, level); ir = inrush_a(level, mode)
            rows.append({"mode_exited": mode, "level": level, "v_active_V": R.V_ACTIVE, "v_ret_V": R.V_RET_PLANNING, "cell_rail_nF_per_MB": round(c["cell_rail"] * 1e9, 3), "bitline_nF_per_MB": round(c["bitline"] * 1e9, 3),
                         "periphery_nF_per_MB": round(c["periphery"] * 1e9, 3), "decap_grid_share": R.DECAP_GRID_SHARE[level],
                         "cell_rail_nJ_per_MB": round(comps.get("cell_rail", 0.0) * 1e9, 4), "bitline_nJ_per_MB": round(comps.get("bitline", 0.0) * 1e9, 4),
                         "periphery_nJ_per_MB": round(comps.get("periphery", 0.0) * 1e9, 4), "decap_grid_nJ_per_MB": round(comps.get("decap_grid", 0.0) * 1e9, 4),
                         "total_nJ_per_MB": round(comps["total"] * 1e9, 4), "superseded_exp0009_nJ_per_MB": round(exp0009_original_j_per_mb(mode, level) * 1e9, 1),
                         "ratio_superseded_over_corrected": round(exp0009_original_j_per_mb(mode, level) / comps["total"], 1) if comps["total"] > 0 else "",
                         "inrush_per_macro_mA": round(ir["i_per_macro_A"] * 1e3, 3), "inrush_per_group_mA": round(ir["i_per_group_A"] * 1e3, 2),
                         "formula": "E = C x V_active x dV per hard-switched rail (supply energy); sequencer 10 pJ/macro added per domain", "label": "B7 CORRECTED (EXP-0009A); ASSUMPTION ranges, R11 T2 class anchor; no measured modern figure"})
    return rows
