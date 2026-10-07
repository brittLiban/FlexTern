"""
EXP-0006 physical models: area ranges by density tier, reticle/die-count tier crossings, energy ranges,
and package organizations.  Every constant comes from sources.py (S..) or assumptions (A..).

Area:   area_mm2 = physical_bytes / (density_MB_per_mm2 * 1e6), per tier, low and high density.
        physical_bytes = transformer_bytes / (1 - margin).
Tier crossing: fits one reticle (858 mm2, S21)?  dies of 41 mm2 (V-Cache class, S03)?  needs
        wafer-scale / DRAM-class capacity?
Energy: bytes x 8 x pJ/bit range -> mJ/token contribution of ONE movement, never a chip total.
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from sources import (CACHE_DIE_TIER, CHIP_TIER, DENSITY_TIERS_MB_PER_MM2, ENERGY_PJ_PER_BIT, GRAPHCORE_DIE_MB,  # noqa: E402
                     HBM3E_STACK_GB, MARGINS, MULTI_DIE_SRAM_CEILING_BYTES, NOMINAL_MARGIN, RETICLE_MM2, VCACHE_DIE_MB, VCACHE_DIE_MM2)


def area_rows(points: list[dict]) -> list[dict]:
    rows = []
    for p in points:
        phys = p["transformer_bytes"] / (1 - NOMINAL_MARGIN)
        for tier, (lo, hi) in DENSITY_TIERS_MB_PER_MM2.items():
            a_hi_density = phys / (hi * 1e6); a_lo_density = phys / (lo * 1e6)
            usable = "NO (bit-cell only)" if tier.startswith("raw") else ("array only, no compute/TSV/routing" if tier.startswith("macro") else ("no compute" if "cache_only" in tier or "edram" in tier else "yes (includes compute)"))
            rows.append({"model": p["model"], "weight_bits": p["weight_bits"], "kind": p["kind"], "physical_MB_at_10pct_margin": round(phys / 1e6, 1),
                         "density_tier": tier, "density_MB_per_mm2_low": lo, "density_MB_per_mm2_high": hi,
                         "area_mm2_min": round(a_hi_density, 1), "area_mm2_max": round(a_lo_density, 1),
                         "fits_one_reticle_858mm2": "yes" if a_lo_density <= RETICLE_MM2 else ("at high density only" if a_hi_density <= RETICLE_MM2 else "no"),
                         "reticles_needed_min": math.ceil(a_hi_density / RETICLE_MM2), "reticles_needed_max": math.ceil(a_lo_density / RETICLE_MM2),
                         "usable_for_chip_estimate": usable, "label": "INFERRED from cited densities; not ASIC area of any FlexTern design"})
    return rows


def tier_crossing_rows(points: list[dict]) -> list[dict]:
    """Founder-readable: at the chip tier (A1) and the cache-die tier, which memory organization class each point lands in."""
    lo, hi = DENSITY_TIERS_MB_PER_MM2[CHIP_TIER]; clo, chi = DENSITY_TIERS_MB_PER_MM2[CACHE_DIE_TIER]
    rows = []
    for p in points:
        phys = p["transformer_bytes"] / (1 - NOMINAL_MARGIN)
        a_chip = (phys / (hi * 1e6), phys / (lo * 1e6)); a_cache = (phys / (chi * 1e6), phys / (clo * 1e6))
        vdies = math.ceil(phys / (VCACHE_DIE_MB * 1e6))
        if a_chip[1] <= RETICLE_MM2:
            cls = "one monolithic SRAM-heavy die (Graphcore/Dojo class)"
        elif a_chip[0] <= RETICLE_MM2:
            cls = "one reticle only at the top of the density range (marginal)"
        elif a_cache[1] <= RETICLE_MM2:
            cls = "exceeds one reticle at any whole-die density; fits only as compute die + cache-only die(s) or multiple dies"
        elif phys <= MULTI_DIE_SRAM_CEILING_BYTES:                                   # A9
            cls = "multi-die SRAM (chiplets or stacked) required"
        else:
            cls = "wafer-scale or DRAM-class (HBM/LPDDR) capacity required"
        rows.append({"model": p["model"], "weight_bits": p["weight_bits"], "kind": p["kind"], "physical_MB": round(phys / 1e6, 1),
                     "area_mm2_chip_tier_min": round(a_chip[0]), "area_mm2_chip_tier_max": round(a_chip[1]),
                     "area_mm2_cache_die_tier_min": round(a_cache[0]), "area_mm2_cache_die_tier_max": round(a_cache[1]),
                     "vcache_class_dies_41mm2": vdies, "graphcore_class_dies_900MB": math.ceil(phys / (GRAPHCORE_DIE_MB * 1e6)),
                     "hbm3e_stacks_36GB_for_capacity": math.ceil(phys / (HBM3E_STACK_GB * 1e9)),
                     "memory_organization_class": cls, "label": "INFERRED tier crossing; densities A1; margin 10%"})
    return rows


def energy_rows(points: list[dict], act_out_bytes: dict[str, int]) -> list[dict]:
    """mJ/token for the WEIGHT movement only, per class; activation+output movement added for resident."""
    rows = []
    for p in points:
        W = p["transformer_bytes"]; ao = act_out_bytes.get(p["model"], 0)
        def mj(bytes_, key):
            lo, hi = ENERGY_PJ_PER_BIT[key]; return (bytes_ * 8 * lo / 1e9, bytes_ * 8 * hi / 1e9)
        s_l = mj(W, "lpddr5x_class"); s_h = mj(W, "hbm_class"); s_d = mj(W, "ddr_class_io")
        r_l = mj(W, "sram_local_read_45nm"); r_c = mj(ao, "hybrid_bond_crossing"); r_h = mj(W, "hbm_class")
        rows.append({"model": p["model"], "weight_bits": p["weight_bits"], "kind": p["kind"], "weight_bytes_per_token": W,
                     "stream_lpddr5x_mJ_per_token_min": round(s_l[0], 2), "stream_lpddr5x_mJ_per_token_max": round(s_l[1], 2),
                     "stream_hbm_mJ_per_token_min": round(s_h[0], 2), "stream_hbm_mJ_per_token_max": round(s_h[1], 2),
                     "stream_ddr_io_mJ_per_token": round(s_d[0], 2),
                     "resident_sram_local_mJ_per_token_min": round(r_l[0], 2), "resident_sram_local_mJ_per_token_max": round(r_l[1], 2),
                     "resident_act_out_crossing_mJ_per_token_max": round(r_c[1], 4),
                     "resident_in_hbm_stack_local_mJ_per_token_min": round(r_h[0], 2), "resident_in_hbm_stack_local_mJ_per_token_max": round(r_h[1], 2),
                     "label": "INFERRED ranges (A3): one movement each, SRAM figure from a 45 nm tutorial; NOT a chip J/token"})
    return rows


def package_rows(points: list[dict]) -> list[dict]:
    """Candidate organizations for each canonical point: where weights live, what crosses dies, capacity and bandwidth fit."""
    lo, hi = DENSITY_TIERS_MB_PER_MM2[CHIP_TIER]
    orgs = [
        ("P1 monolithic SRAM + Y-Core banks (Graphcore/Dojo class, N7-class)", "on die", "on die", "nothing (activations/results on-die fabric)", "commercial (S05, S07)",
         lambda phys: phys / (lo * 1e6) <= RETICLE_MM2, lambda phys: f"{phys/(hi*1e6):.0f}-{phys/(lo*1e6):.0f} mm2 one die"),
        ("P2 compute die + hybrid-bonded SRAM cache die(s) (V-Cache class)", "stacked cache die(s)", "base die", "every weight byte crosses the bond (2-2.5 TB/s per 64 MB die, S03/S04); activations stay on base die", "commercial for ONE die; multi-die SRAM stacks not shipping (A6)",
         lambda phys: math.ceil(phys / (VCACHE_DIE_MB * 1e6)) <= 2, lambda phys: f"{math.ceil(phys/(VCACHE_DIE_MB*1e6))} x 41 mm2 cache dies"),
        ("P3 SRAM chiplets + fabric (d-Matrix class, N6)", "per chiplet", "per chiplet", "activations broadcast and results collected over the chiplet fabric; weights never cross", "commercial (S09)",
         lambda phys: phys <= 2e9, lambda phys: f"{math.ceil(phys/256e6)} x 256 MB-class chiplets"),
        ("P4 HBM stack with base-die/PIM compute (Samsung HBM-PIM class)", "DRAM dies in stack", "base die or bank-level PIM units", "every weight byte crosses DRAM-to-logic inside the stack (HBM-class energy, S14)", "research prototype (S18); capacity trivially sufficient (36 GB, S12)",
         lambda phys: True, lambda phys: f"{math.ceil(phys/36e9)} stack(s), {phys/36e9*100:.1f}% of one stack"),
        ("P5 DRAM-PIM DIMM (UPMEM class)", "64 MB DRAM bank per DPU", "DRAM-process logic per bank", "weights local; ~1 GB/s per bank (S20) limits token rate", "commercial (S20) but DRAM-process logic, not a ternary engine",
         lambda phys: True, lambda phys: f"{math.ceil(phys/67.1e6)} DPU banks"),
        ("P6 hybrid: hot set on-die SRAM + remainder streamed (LPDDR5X/HBM)", "split", "on die", "the streamed fraction crosses the external interface every token", "composed of commercial parts; benefit proportional to the resident fraction (residency.py hot-set rows)",
         lambda phys: True, lambda phys: "resident fraction chosen by die budget"),
        ("P7 external streaming (LPDDR5X / HBM3E) — baseline", "external DRAM", "on die", "every weight byte crosses the external interface every token", "commercial (S12, S13)",
         lambda phys: True, lambda phys: "n/a"),
    ]
    rows = []
    for p in points:
        if not p["kind"].startswith("canonical"):
            continue
        phys = p["transformer_bytes"] / (1 - NOMINAL_MARGIN)
        for name, wloc, cloc, crossing, status, fits, size in orgs:
            rows.append({"model": p["model"], "weight_bits": p["weight_bits"], "organization": name, "weights_live": wloc, "ycores_live": cloc,
                         "data_crossing_die_boundary": crossing, "availability": status, "capacity_fit": "yes" if fits(phys) else "no / marginal",
                         "size_proxy": size(phys), "physical_MB": round(phys / 1e6, 1), "label": "ANALYTICAL organization; nothing designed"})
    return rows


def economics_rows(points: list[dict]) -> list[dict]:
    """Parametric cost proxies (no prices): memory area range at the chip tier, die count, stacks, package class."""
    lo, hi = DENSITY_TIERS_MB_PER_MM2[CHIP_TIER]
    rows = []
    for p in points:
        phys = p["transformer_bytes"] / (1 - NOMINAL_MARGIN)
        a = (phys / (hi * 1e6), phys / (lo * 1e6))
        if a[1] <= RETICLE_MM2:
            pkg = "monolithic die (standard flip-chip)"
        elif a[0] <= RETICLE_MM2:
            pkg = "monolithic only at best-case density; otherwise 2.5D multi-chiplet"
        elif phys <= MULTI_DIE_SRAM_CEILING_BYTES:                                   # A9
            pkg = "2.5D multi-chiplet (interposer/bridge) or 3D stacked SRAM"
        else:
            pkg = "HBM 2.5D (CoWoS-class) or wafer-scale"
        rows.append({"model": p["model"], "weight_bits": p["weight_bits"], "kind": p["kind"], "physical_MB": round(phys / 1e6, 1),
                     "memory_silicon_mm2_min": round(a[0]), "memory_silicon_mm2_max": round(a[1]),
                     "logic_silicon_mm2": "UNMEASURED (A7)", "reticle_class_dies": math.ceil(a[1] / RETICLE_MM2),
                     "hbm_stacks_if_dram_class": math.ceil(phys / (HBM3E_STACK_GB * 1e9)), "package_complexity_class": pkg,
                     "label": "cost PROXY only; no price, no yield model, no NRE"})
    return rows
