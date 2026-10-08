"""
EXP-0009A validation tests: B7 wake-energy dimensional correction and B14 backing-memory accounting.  Run from the repository root:
    python -m unittest tests.test_sram_retention_correction -v
"""

from __future__ import annotations

import csv
import math
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for p in ("analysis", "analysis/memory_residency", "analysis/sram_validation", "analysis/sram_retention"):
    sys.path.insert(0, os.path.join(_ROOT, p))
os.chdir(_ROOT)

import retention_evidence as R  # noqa: E402
import retention as RT  # noqa: E402
import wake as WK  # noqa: E402
import layer_schedule as LS  # noqa: E402
import compare as C  # noqa: E402
import evidence as E8  # noqa: E402
from policy import Scenario, lpddr_idle_mw, E_LP_PJ  # noqa: E402

WS = LS.load_workloads()
S1 = LS.schedule(WS["falcon-e-1b"], 2.0)
S4 = LS.schedule(WS["falcon-e-1b"], 4.5)
SUPERSEDED = os.path.join("results", "sram_retention", "superseded_899976e")


def _csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestB7DimensionalCorrection(unittest.TestCase):
    def test_1_supply_energy_formula_in_si_units(self):
        # E = C x V_active x dV: 4 nF/MB x 0.75 V x 0.40 V = 1.2e-9 J (nJ, not uJ)
        self.assertAlmostEqual(WK.rail_energy_j(4e-9, 0.75, 0.40), 1.2e-9, places=15)
        c = WK.rail_capacitances_f_per_mb("planning")
        self.assertAlmostEqual(c["cell_rail"], 8e6 * 0.5e-15, places=18)                 # 4 nF per MB
        comps = WK.wake_components_j_per_mb("RETENTION_VRET", "planning")
        self.assertAlmostEqual(comps["cell_rail"], 4e-9 * R.V_ACTIVE * (R.V_ACTIVE - R.V_RET_PLANNING), places=15)
        self.assertAlmostEqual(comps["total"], 2.844e-9, delta=0.01e-9)
        # the superseded figure is exactly 1000x the formula's cell-rail term... times the components it never had
        self.assertAlmostEqual(WK.exp0009_original_j_per_mb("RETENTION_VRET", "planning") / comps["cell_rail"], 1000.0, places=6)

    def test_2_capacitance_and_voltage_unit_conversions(self):
        for level in ("low", "planning", "high"):
            c = WK.rail_capacitances_f_per_mb(level)
            self.assertAlmostEqual(c["cell_rail"] * 1e9, 8 * R.CELL_RAIL_FF_PER_CELL[level], places=9)       # fF/cell x 8e6 cells = nF x 8
            self.assertAlmostEqual(c["bitline"] * 1e9, 16 * R.BITLINE_FF_PER_CELL_PER_LINE[level], places=9)
            self.assertAlmostEqual(c["periphery"] * 1e9, R.PERIPHERY_NF_PER_MB[level], places=12)
            comps = WK.wake_components_j_per_mb("RETENTION_VRET", level)
            self.assertTrue(1e-9 <= comps["total"] <= 1e-8, (level, comps["total"]))                           # nJ class for every level
        # HIGH assumes a full-V_active bitline swing: E = C V^2
        self.assertAlmostEqual(WK.wake_components_j_per_mb("RETENTION_VRET", "high")["bitline"], 3.2e-9 * R.V_ACTIVE ** 2, places=15)
        # per-mode component sets: a periphery-only exit does not move the array rail; light sleep only restores bitlines
        self.assertNotIn("cell_rail", WK.wake_components_j_per_mb("DEEP_SLEEP_NOMINAL", "planning"))
        self.assertEqual(set(WK.wake_components_j_per_mb("LIGHT_SLEEP", "planning")) - {"total"}, {"bitline"})
        self.assertEqual(WK.wake_components_j_per_mb("ACTIVE_STANDBY", "planning")["total"], 0.0)

    def test_3_wake_energy_scales_linearly_with_capacity(self):
        for level in ("low", "planning", "high"):
            e1 = WK.wake_energy_j(1e6, level); e10 = WK.wake_energy_j(10e6, level); e100 = WK.wake_energy_j(100e6, level)
            self.assertAlmostEqual(e10 / e1, 10.0, delta=0.02)                      # sequencer term is per macro, same ratio
            self.assertAlmostEqual(e100 / e10, 10.0, delta=0.02)
        a, b = Scenario(S1, "planning", 25, "RETENTION_VRET"), Scenario(S4, "planning", 25, "RETENTION_VRET")
        self.assertAlmostEqual(b.wake_e(b.prov) / a.wake_e(a.prov), 2.25, delta=0.01)
        self.assertAlmostEqual(a.wake_e(a.prov), a.prov / 1e6 * a.wake_j_mb + WK.macros_in(a.prov) * R.WAKE_SEQUENCER_J, places=15)

    def test_4_inrush_consistent_with_b6_wake_time(self):
        ir = WK.inrush_a("planning")
        self.assertLess(ir["i_per_macro_A"], 0.01)                                   # < 10 mA per 256 KB macro at 200 ns
        self.assertLess(ir["i_per_group_A"], 0.1)                                    # < 100 mA per group of 8
        # the superseded figure would have needed ~2 A per macro
        c_super = R.EXP0009_ORIGINAL_ARRAY_SWING_UJ_PER_MB * 1e-6 / (R.V_ACTIVE * (R.V_ACTIVE - R.V_RET_PLANNING))   # F per MB
        i_super = c_super * R.MACRO_BYTES / 1e6 * (R.V_ACTIVE - R.V_RET_PLANNING) / R.WAKE_T_MACRO_S["planning"]
        self.assertGreater(i_super, 1.0)
        # LOW latency level is recorded as not in-rush-consistent for 256 KB macros (documented in B7)
        self.assertGreater(WK.inrush_a("low")["i_per_group_A"], 1.0)
        b7 = next(a for a in R.ASSUMPTIONS if a["source_id"] == "B7")
        self.assertIn("not in-rush-consistent", b7["value"]); self.assertIn("SUPERSEDED", b7["value"]); self.assertIn("1.2 nJ/MB", b7["value"])

    def test_5_transitions_counted_once(self):
        sc = Scenario(S1, "planning", 25, "RETENTION_VRET")
        w = sc.window(2, 25)
        self.assertEqual(w["wakes_per_token"], S1["L"])
        self.assertAlmostEqual(w["transition_mJ_per_token"] * 1e-3, sum(sc.wake_e(b) for b in sc.layer_prov), places=15)
        # window in-session: the per-token layer wakes already cover the first token; no extra whole-model wake per session
        ws = sc.session(64, 25, 10.0, "window", 2)
        self.assertEqual(ws["session_wake_mJ"], 0.0)
        self.assertAlmostEqual(ws["resident_session_mJ"] * 1e-3, sc.window(2, 25)["resident_total_W"] * 64 / 25, places=12)
        # P0 in-session: exactly one whole-model wake per session
        ps = sc.session(64, 25, 10.0, "P0")
        self.assertAlmostEqual(ps["session_wake_mJ"] * 1e-3, sc.wake_e(sc.prov), places=15)
        self.assertAlmostEqual(ps["resident_session_mJ"] * 1e-3, sc.P0(25)["resident_total_W"] * 64 / 25 + sc.wake_e(sc.prov), places=12)
        # the superseded model double-counted: it charged a model wake on top of the window's per-token wakes
        old = Scenario(S1, "planning", 25, "RETENTION_VRET", wake_model="exp0009_original")
        self.assertEqual(old.session(64, 25, 10.0, "window", 2)["session_wake_mJ"], 0.0)   # fixed in both models now
        # shutdown: reload energy charged once per session and never together with a wake
        px = sc.session(64, 25, 10.0, "P0", idle_mode="shutdown", backing="flash")
        self.assertAlmostEqual(px["session_wake_mJ"] * 1e-3, sc.e_reload_flash, places=15)
        self.assertAlmostEqual(px["resident_session_mJ"] * 1e-3, sc.P0(25)["resident_total_W"] * 64 / 25 + sc.e_reload_flash, places=12)

    def test_6_energy_conservation_of_session_average(self):
        sc = Scenario(S1, "planning", 85, "RETENTION_VRET")
        for backing in ("dram", "flash"):
            d = sc.session(64, 25, 10.0, "P0", idle_mode="shutdown", backing=backing)
            T = 64 / 25 + 10.0
            self.assertAlmostEqual(d["resident_avg_mW"], (d["resident_session_mJ"] + d["resident_idle_mJ"]) / T, places=9)
            self.assertAlmostEqual(d["lpddr_avg_mW"], (d["lpddr_session_mJ"] + d["lpddr_idle_mJ"]) / T, places=9)
            rl = sc.reload(backing, 10.0)
            self.assertAlmostEqual(d["resident_idle_mJ"] * 1e-3, (sc.p_shutdown + rl["p_backing_idle"]) * 10.0, places=12)


class TestB14BackingMemory(unittest.TestCase):
    def test_7_backing_memory_idle_cost_is_charged(self):
        sc = Scenario(S1, "planning", 85, "RETENTION_VRET")
        d = sc.session(64, 25, 60.0, "P0", idle_mode="shutdown", backing="dram")
        f = sc.session(64, 25, 60.0, "P0", idle_mode="shutdown", backing="flash")
        u = sc.session(64, 25, 60.0, "P0", idle_mode="shutdown", backing="unbacked_superseded")
        self.assertAlmostEqual(d["backing_idle_mW"], lpddr_idle_mw(60.0, 85), places=9)                 # self-refresh at 85 C (9.8 mW)
        self.assertAlmostEqual(f["backing_idle_mW"], R.FLASH_STANDBY_MW["planning"], places=9)
        self.assertEqual(u["backing_idle_mW"], 0.0); self.assertIn("SUPERSEDED", u["backing_idle_mode"])
        self.assertGreater(d["resident_idle_mJ"], u["resident_idle_mJ"]); self.assertGreater(f["resident_idle_mJ"], u["resident_idle_mJ"])
        # weights-only view: PX-DRAM idle power >= the streaming baseline's idle power (same die, same retaining mode)
        self.assertGreaterEqual(d["resident_idle_mJ"], d["lpddr_idle_mJ"])
        # system view drops the session-time DRAM term only
        dv = sc.session(64, 25, 60.0, "P0", idle_mode="shutdown", backing="dram", view="system")
        self.assertAlmostEqual(dv["resident_idle_mJ"], d["resident_idle_mJ"], places=9)
        self.assertAlmostEqual(d["resident_session_mJ"] - dv["resident_session_mJ"], E8.LPDDR_BACKGROUND_MW["power_down"] * 64 / 25, places=9)

    def test_8_reload_energy_and_latency(self):
        sc = Scenario(S1, "planning", 25, "RETENTION_VRET")
        bits = S1["weight_bytes"] * 8
        self.assertAlmostEqual(sc.e_reload_dram, bits * (E_LP_PJ + R.SRAM_WRITE_PATH_PJ_PER_BIT) * 1e-12, places=12)
        self.assertAlmostEqual(sc.e_reload_flash, bits * (R.FLASH_READ_PJ_PER_BIT["planning"] + R.SRAM_WRITE_PATH_PJ_PER_BIT) * 1e-12, places=12)
        self.assertAlmostEqual(sc.t_reload_dram, S1["weight_bytes"] / (17.07e9 * 0.7), places=9)
        self.assertAlmostEqual(sc.t_reload_flash, S1["weight_bytes"] / 2.0e9, places=9)
        self.assertAlmostEqual(sc.e_reload_dram * 1e3, 14.3, delta=0.1); self.assertAlmostEqual(sc.e_reload_flash * 1e3, 296.5, delta=0.5)
        self.assertAlmostEqual(sc.e_reload_superseded * 1e3, 13.2, delta=0.1)           # EXP-0008 cold start reproduced as the superseded input
        self.assertGreater(sc.e_reload_flash / sc.e_reload_dram, 15)                        # flash reload is ~20x the DRAM reload
        # the SRAM write path is the EXP-0008 read path with the macro write swapped in
        self.assertAlmostEqual(R.SRAM_WRITE_PATH_PJ_PER_BIT, 0.4405 - 0.090 + 0.090 * E8.SRAM_WRITE_OVER_READ, places=9)
        # the flash figure reproduces the datasheet arithmetic: (330 mA x 2.5 V + 550 mA x 1.2 V) / 2,000 MB/s
        self.assertAlmostEqual((0.33 * 2.5 + 0.55 * 1.2) / (2.0e9 * 8) * 1e12, R.FLASH_READ_PJ_PER_BIT["planning"], delta=0.3)
        for level in ("low", "planning", "high"):
            s = Scenario(S1, "planning", 25, "RETENTION_VRET", backing_level=level)
            self.assertAlmostEqual(s.t_reload_flash, S1["weight_bytes"] / R.FLASH_RELOAD_BPS[level], places=9)
            self.assertTrue(0.09 <= s.t_reload_flash <= 0.41)

    def test_9_break_even_equations(self):
        for temp in (25, 85):
            sc = Scenario(S1, "planning", temp, "RETENTION_VRET")
            for backing in ("dram", "flash"):
                g = sc.shutdown_break_even_gap_s(backing)
                ps = sc.session(64, 25, g, "P0")["resident_avg_mW"]; px = sc.session(64, 25, g, "P0", idle_mode="shutdown", backing=backing)["resident_avg_mW"]
                self.assertAlmostEqual(ps / px, 1.0, places=4)
            # closed form equals the numeric solution when the backing has no session-time term (flash)
            self.assertAlmostEqual(sc.shutdown_break_even_gap_closed_form_s("flash"), sc.shutdown_break_even_gap_s("flash"), delta=1e-3 * sc.shutdown_break_even_gap_s("flash"))
            gf = sc.shutdown_break_even_gap_closed_form_s("flash")
            self.assertAlmostEqual(gf, (sc.e_reload_flash - sc.wake_e(sc.prov)) / (sc.p_all_retained() - sc.p_shutdown - sc.p_flash_standby), places=9)
            # ordering: the flash reload costs ~20x more, so its break-even gap is longer than DRAM's, and both exceed the superseded figure
            self.assertGreater(sc.shutdown_break_even_gap_s("flash"), sc.shutdown_break_even_gap_s("dram"))
            self.assertGreater(sc.shutdown_break_even_gap_s("dram"), sc.shutdown_break_even_gap_s("unbacked_superseded"))
        # threshold rows exist for both backings and the superseded one is labeled
        rows = C.break_even_rows({("falcon-e-1b", 2.0): S1})
        labels = [r["threshold"] for r in rows if r["temp_c"] == 25]
        self.assertTrue(any(l.startswith("11a.") for l in labels)); self.assertTrue(any(l.startswith("11b.") for l in labels)); self.assertTrue(any("SUPERSEDED" in l for l in labels))

    def test_10_limiting_cases(self):
        sc = Scenario(S1, "planning", 85, "RETENTION_VRET")
        # gap -> 0: shutdown pays the reload for nothing, so PX >= PS
        for backing in ("dram", "flash"):
            self.assertGreater(sc.session(64, 25, 1e-3, "P0", idle_mode="shutdown", backing=backing)["resident_avg_mW"], sc.session(64, 25, 1e-3, "P0")["resident_avg_mW"])
        # gap -> infinity: PX-FLASH average -> flash standby + SRAM residual; PS -> retained array power; baseline -> self-refresh
        big = 1e6
        f = sc.session(64, 25, big, "P0", idle_mode="shutdown", backing="flash")
        self.assertAlmostEqual(f["resident_avg_mW"], (sc.p_flash_standby + sc.p_shutdown) * 1e3, delta=0.01)
        self.assertAlmostEqual(sc.session(64, 25, big, "P0")["resident_avg_mW"], sc.p_all_retained() * 1e3, delta=0.01)
        self.assertAlmostEqual(f["lpddr_avg_mW"], lpddr_idle_mw(big, 85), delta=0.01)
        # gap -> infinity, PX-DRAM: average -> self-refresh + residual, i.e. >= the baseline's self-refresh: DRAM backing can never win at idle
        d = sc.session(64, 25, big, "P0", idle_mode="shutdown", backing="dram")
        self.assertGreater(d["resident_avg_mW"], d["lpddr_avg_mW"])
        # symmetric baseline (DRAM off, flash reload) converges to flash standby alone
        sym = sc.session(64, 25, big, "P0", idle_mode="shutdown", backing="flash", baseline_idle="shutdown_flash")
        self.assertAlmostEqual(sym["lpddr_avg_mW"], sc.p_flash_standby * 1e3, delta=0.01)
        # wake energy -> 0 is the P0-equivalent lower bound on transitions: P2 total < P0 total at both temperatures
        self.assertLess(sc.window(2, 25)["resident_total_W"], sc.P0(25)["resident_total_W"])

    def test_11_vret_supply_sensitivity_is_current_ratio(self):
        a = Scenario(S1, "planning", 85, "RETENTION_VRET"); b = Scenario(S1, "planning", 85, "RETENTION_VRET", vret_supply="linear")
        self.assertAlmostEqual(b.ratio / a.ratio, R.ASAP7_VNOM / R.V_RET_PLANNING, places=9)
        self.assertAlmostEqual(b.ratio, RT.array_power_ratio(0.35, 85) * (1 - R.PERIPHERY_SHARE["planning"]) * 0.70 / 0.35, places=9)
        self.assertEqual(Scenario(S1, "planning", 85, "DEEP_SLEEP_NOMINAL", vret_supply="linear").ratio, Scenario(S1, "planning", 85, "DEEP_SLEEP_NOMINAL").ratio)
        with self.assertRaises(ValueError):
            Scenario(S1, "planning", 85, "RETENTION_VRET", vret_supply="magic")


class TestReproduction(unittest.TestCase):
    def test_12_superseded_model_reproduces_the_899976e_artifacts(self):
        old = Scenario(S1, "planning", 25, "RETENTION_VRET", wake_model="exp0009_original")
        self.assertAlmostEqual(old.wake_uj_mb, 1.2, places=9)
        self.assertAlmostEqual(old.window(2, 25)["transition_mJ_per_token"], 0.5285, places=3)
        self.assertAlmostEqual(old.window(2, 25)["resident_wake_W"] * 1e3, 13.21, delta=0.01)
        walk = _csv(os.path.join(SUPERSEDED, "falcon_e_1b_walkthrough.csv"))
        for temp in (25, 85):
            o = Scenario(S1, "planning", temp, "RETENTION_VRET", wake_model="exp0009_original")
            row = next(r for r in walk if r["temp_c"] == str(temp) and r["tok_s"] == "25" and r["policy"].startswith("P2 current"))
            self.assertAlmostEqual(o.window(2, 25)["resident_total_W"] * 1e3, float(row["resident_total_mW"]), delta=0.05)
            self.assertAlmostEqual(o.window(2, 25)["lpddr_over_resident"], float(row["lpddr_over_resident"]), delta=0.005)
            px = next(r for r in walk if r["temp_c"] == str(temp) and r["tok_s"] == "25" and r["policy"].startswith("PX session"))
            self.assertAlmostEqual(o.session(256, 25, 10.0, "P0", idle_mode="shutdown", backing="unbacked_superseded")["resident_avg_mW"], float(px["resident_total_mW"]), delta=0.05)
        be = _csv(os.path.join(SUPERSEDED, "break_even.csv"))
        r11 = next(r for r in be if r["model"] == "falcon-e-1b" and r["temp_c"] == "25" and r["threshold"].startswith("11."))
        sc = Scenario(S1, "planning", 25, "RETENTION_VRET")
        self.assertAlmostEqual(sc.e_reload_superseded / (sc.p_all_retained() - sc.p_shutdown), float(r11["value"]), delta=0.002)   # the 899976e formula
        orig = {r["threshold"]: r["value"] for r in C._thresholds_with_model(S1, 25, "exp0009_original")}
        self.assertAlmostEqual(orig["11. idle gap above which shutdown + reload beats retention (s)"], float(r11["value"]), delta=0.002)

    def test_13_unaffected_results_unchanged(self):
        # P0 (no wake, no reload) and the leakage table are byte-identical to the 899976e artifacts
        for name in ("leakage_temperature.csv", "retention_voltage.csv", "layer_schedule.csv", "exp0008_reconnect.csv"):
            with open(os.path.join("results", "sram_retention", name), encoding="utf-8") as a, open(os.path.join(SUPERSEDED, name), encoding="utf-8") as b:
                self.assertEqual(a.read(), b.read(), name)
        new = _csv(os.path.join("results", "sram_retention", "falcon_e_1b_walkthrough.csv")); old = _csv(os.path.join(SUPERSEDED, "falcon_e_1b_walkthrough.csv"))
        for temp in ("25", "85"):
            for rate in ("5", "25", "100"):
                a = next(r for r in new if r["temp_c"] == temp and r["tok_s"] == rate and r["policy"].startswith("P0"))
                b = next(r for r in old if r["temp_c"] == temp and r["tok_s"] == rate and r["policy"].startswith("P0"))
                self.assertEqual(a["resident_total_mW"], b["resident_total_mW"]); self.assertEqual(a["lpddr_total_mW"], b["lpddr_total_mW"])
        # EXP-0008 inputs untouched
        self.assertEqual(E8.LEAK_MW_PER_MB["planning"], 0.60); self.assertEqual(E8.LPDDR_PJ_PER_BIT["planning"], 4.0)

    def test_14_corrected_wake_is_immaterial_at_every_level_and_rate(self):
        for temp in (25, 85):
            for level in ("low", "planning", "high"):
                sc = Scenario(S1, "planning", temp, "RETENTION_VRET", wake_level=level)
                for R_tok in (5, 25, 100, 250):
                    w = sc.window(2, R_tok)
                    self.assertLess(w["transition_mJ_per_token"] / (sc.e_dyn * 1e3), 0.01)      # < 1% of the read energy at every level
        rows = C.correction_rows({("falcon-e-1b", 2.0): S1})
        r = next(x for x in rows if x["quantity"].startswith("P2 wake share") and "25 tok/s" in x["quantity"] and x["temp_c"] == 25)
        self.assertAlmostEqual(r["exp0009_original"], 0.378, delta=0.002); self.assertLess(r["corrected"], 0.001)

    def test_15_provenance_of_new_constants(self):
        ids = {s["source_id"] for s in R.SOURCES}
        for sid in ("R19", "R20", "R21"):
            self.assertIn(sid, ids)
        for bid in ("B16", "B17", "B18", "B19"):
            self.assertTrue(any(a["source_id"] == bid for a in R.ASSUMPTIONS), bid)
        r19 = next(s for s in R.SOURCES if s["source_id"] == "R19")
        self.assertTrue(r19["tier"].startswith("T1")); self.assertIn("92.8 pJ/bit", r19["active_leakage"])
        r21 = next(s for s in R.SOURCES if s["source_id"] == "R21")
        self.assertIn("not read", r21["measured_or_simulated"])                     # the cited primary was not opened; labeled
        # artifacts carry the correction and the superseded snapshot exists
        self.assertTrue(os.path.exists(os.path.join(SUPERSEDED, "summary.md")))
        with open(os.path.join("results", "sram_retention", "summary.md"), encoding="utf-8") as f:
            self.assertIn("CORRECTION NOTICE", f.read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
