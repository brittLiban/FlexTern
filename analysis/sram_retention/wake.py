"""
EXP-0009 wake model: latency and energy of bringing a retained SRAM domain back to active state.

  t_wake(domain)  = max(t_floor, t_macro x ceil(macros / parallel_wake))      (B6: in-rush-limited staggered macro wake)
  E_wake(domain)  = MB x uJ_per_MB[level] + macros x E_sequencer               (B7: rail recharge + periphery restore)

Sleep entry is charged nothing (the stored rail energy is dissipated, not drawn from the supply); every wake is charged
exactly once (tests check the count).  No measured modern-node macro wake energy exists; the per-MB figures are
physics-based estimates with a range (see retention_evidence.ASSUMPTIONS B7).
"""

from __future__ import annotations

import math

import retention_evidence as R


def macros_in(domain_bytes: float, macro_bytes: float = R.MACRO_BYTES) -> int:
    return max(1, math.ceil(domain_bytes / macro_bytes))


def wake_latency_s(domain_bytes: float, level: str = "planning", macro_bytes: float = R.MACRO_BYTES, t_macro_s: float | None = None,
                   parallel: int | None = None) -> float:
    t_macro = R.WAKE_T_MACRO_S[level] if t_macro_s is None else t_macro_s
    par = R.WAKE_PARALLEL_MACROS[level] if parallel is None else parallel
    n = macros_in(domain_bytes, macro_bytes)
    return max(R.WAKE_T_FLOOR_S, t_macro * math.ceil(n / par))


def wake_energy_j(domain_bytes: float, level: str = "planning", uj_per_mb: float | None = None, macro_bytes: float = R.MACRO_BYTES) -> float:
    per_mb = R.WAKE_UJ_PER_MB[level] if uj_per_mb is None else uj_per_mb
    return domain_bytes / 1e6 * per_mb * 1e-6 + macros_in(domain_bytes, macro_bytes) * R.WAKE_SEQUENCER_J


def wake_rows(schedules: dict) -> list[dict]:
    rows = []
    for (slug, bits), s in schedules.items():
        for level in ("low", "planning", "high"):
            for name, b in (("one macro", R.MACRO_BYTES), ("one layer (largest)", s["largest_layer_provisioned_bytes"]), ("whole model", s["provisioned_bytes"])):
                rows.append({"model": slug, "weight_bits": bits, "domain": name, "domain_MB": round(b / 1e6, 3), "level": level,
                             "macros": macros_in(b), "wake_latency_us": round(wake_latency_s(b, level) * 1e6, 3),
                             "wake_energy_uJ": round(wake_energy_j(b, level) * 1e6, 4),
                             "wake_energy_uJ_per_MB": round(wake_energy_j(b, level) / (b / 1e6) * 1e6, 4),
                             "per_token_if_every_layer_wakes_mJ": round(s["L"] * wake_energy_j(s["largest_layer_provisioned_bytes"], level) * 1e3, 4) if name.startswith("one layer") else "",
                             "label": "B6/B7 ASSUMPTION ranges; R05 (200 ns per 128 KiB macro, 22 nm FD-SOI) and R03 (4-10 cycles, 7 nm PDK) are the anchors"})
    return rows
