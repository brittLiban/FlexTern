"""
EXP-0009 validation tests (prompt items 1-22).  Run from the repository root:
    python -m unittest tests.test_sram_retention -v
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
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
from policy import Scenario, E_RES_PJ, E_LP_PJ  # noqa: E402
from model import resident_pj_per_bit  # noqa: E402

WS = LS.load_workloads()
S1 = LS.schedule(WS["falcon-e-1b"], 2.0)
S4 = LS.schedule(WS["falcon-e-1b"], 4.5)
SCHEDS = {(slug, b): LS.schedule(w, b) for slug, w in WS.items() for b in (2.0, 4.5)}


def _csv(name):
    with open(os.path.join("results", "sram_validation", name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestRepositoryAndReproduction(unittest.TestCase):
    def test_1_head_is_a_known_commit(self):
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
        self.assertEqual(len(h), 40)

    def test_2_exp0008_dynamic_pj_per_bit_reproduce_with_retention_disabled(self):
        self.assertAlmostEqual(E_RES_PJ, 0.4405, places=4)
        self.assertEqual(E_LP_PJ, 4.0)
        sc = Scenario(S1, "planning", 25, "DEEP_SLEEP_NOMINAL", leak_mw_per_mb=R.EXP0008_LEAK_MW_PER_MB)
        row = next(r for r in _csv("dynamic_energy.csv") if r["model"] == "falcon-e-1b" and r["weight_bits"] == "2.0" and r["level"] == "planning")
        self.assertAlmostEqual(sc.e_dyn * 1e3, float(row["resident_dynamic_mJ_per_token"]), places=2)
        self.assertAlmostEqual(sc.e_lp * 1e3, float(row["lpddr_dynamic_mJ_per_token"]), places=2)
        self.assertAlmostEqual(sc.p_all_active() * 1e3, float(row["sram_static_mW"]), delta=0.2)

    def test_3_canonical_capacities_reproduce(self):
        self.assertEqual(int(S1["weight_bytes"]), 396361728)
        self.assertAlmostEqual(S1["provisioned_bytes"] / 1e6, 440.4, places=1)
        self.assertEqual(int(SCHEDS[("falcon-e-3b", 2.0)]["weight_bytes"]), 729808896)
        self.assertEqual(int(SCHEDS[("bitnet-2b4t", 2.0)]["weight_bytes"]), 521011200)
        self.assertAlmostEqual(S4["weight_bytes"] / S1["weight_bytes"], 2.25, places=9)

    def test_4_layer_counts_and_capacities_from_profiles(self):
        self.assertEqual(S1["L"], 24); self.assertEqual(SCHEDS[("falcon-e-3b", 2.0)]["L"], 32); self.assertEqual(SCHEDS[("bitnet-2b4t", 2.0)]["L"], 30)
        self.assertAlmostEqual(sum(S1["layer_bytes"]), S1["weight_bytes"], places=3)
        self.assertEqual(LS.layer_weights(WS["falcon-e-1b"])[0], 66060288)
        self.assertEqual(LS.schedule(WS["falcon-e-1b"], 2.0)["order"][-1], "head (external)")

    def test_12_duty_cycle_arithmetic_reproduces_exp0008_P0(self):
        sc = Scenario(S1, "planning", 25, "DEEP_SLEEP_NOMINAL", leak_mw_per_mb=R.EXP0008_LEAK_MW_PER_MB)
        for row in _csv("duty_cycle_sweep.csv"):
            if row["model"] == "falcon-e-1b" and row["weight_bits"] == "2.0" and row["idle_retention_factor"] == "1.0":
                d = sc.duty(float(row["duty_cycle"]), 25, "P0", lpddr_idle="power_down")      # EXP-0008 charged power-down when idle
                self.assertAlmostEqual(d["resident_avg_mW"], float(row["resident_avg_mW"]), delta=0.15)
                self.assertAlmostEqual(d["lpddr_avg_mW"], float(row["lpddr_avg_mW"]), delta=0.15)
        be = next(r for r in _csv("break_even.csv") if r["model"] == "falcon-e-1b" and r["threshold"].startswith("token rate"))
        rep = C.exp0008_reproduction_rows(SCHEDS)[0]
        self.assertAlmostEqual(rep["break_even_tok_s"], float(be["value"]), places=1)


class TestLeakageModel(unittest.TestCase):
    def test_5_6_active_plus_retained_equals_provisioned_and_no_double_count(self):
        sc = Scenario(S1, "planning", 85, "RETENTION_VRET")
        for k in (1, 2, 4, 8):
            w = sc.window(k, 25)
            f = w["active_capacity_fraction"]
            self.assertAlmostEqual(f, k / S1["L"], places=9)
            self.assertLessEqual(f, 1.0)
            self.assertAlmostEqual(w["resident_leakage_W"], f * sc.p_all_active() + (1 - f) * sc.p_all_retained(), places=12)
        self.assertEqual(sc.P0(25)["active_capacity_fraction"], 1.0)

    def test_7_leakage_scales_with_capacity_and_policy(self):
        a, b = Scenario(S1, "planning", 25, "RETENTION_VRET"), Scenario(S4, "planning", 25, "RETENTION_VRET")
        self.assertAlmostEqual(b.p_all_active() / a.p_all_active(), 2.25, places=6)
        self.assertLess(a.window(2, 25)["resident_leakage_W"], a.window(4, 25)["resident_leakage_W"])
        self.assertLess(a.window(4, 25)["resident_leakage_W"], a.P0(25)["resident_leakage_W"])

    def test_8_retention_uses_voltage_and_temperature_point(self):
        r25 = RT.mode_ratio("RETENTION_VRET", 25, "planning", 0.35); r85 = RT.mode_ratio("RETENTION_VRET", 85, "planning", 0.35)
        self.assertLess(r25, r85)                                               # retention saves less when hot (subthreshold dominates)
        self.assertAlmostEqual(RT.array_power_ratio(0.35, 25), (5.63e-12 * 0.35) / (2.592e-11 * 0.70), places=3)
        self.assertLess(RT.mode_ratio("RETENTION_VRET", 25, "planning", 0.25), RT.mode_ratio("RETENTION_VRET", 25, "planning", 0.45))
        self.assertAlmostEqual(RT.temp_factor(85), 1.265e-10 / 2.592e-11, places=2)
        self.assertEqual(RT.mode_ratio("ACTIVE_STANDBY", 85), 1.0)

    def test_16_temperature_scaling_is_labeled_modeled(self):
        for row in RT.leakage_temperature_rows():
            self.assertIn("MODELED", row["label"])
        b2 = next(a for a in R.ASSUMPTIONS if a["source_id"] == "B2")
        self.assertIn("MODELED", b2["value"]); self.assertIn("T2", b2["label"])

    def test_17_no_data_loss_mode_used_as_retention(self):
        with self.assertRaises(ValueError):
            RT.mode_ratio("SHUTDOWN", 25)
        for m in C.MODES:
            self.assertTrue(RT.RETAINS[m])
        self.assertFalse(RT.RETAINS["SHUTDOWN"])
        for s in R.SOURCES:
            if "NOT retained" in s["data_retention_guarantee"] or "lost" in s["data_retention_guarantee"]:
                self.assertTrue(s["deep_sleep_leakage"] or "lose" in s["data_retention_guarantee"] or "lost" in s["data_retention_guarantee"])


class TestWakeAndPolicies(unittest.TestCase):
    def test_9_wake_energy_charged_once_per_transition(self):
        sc = Scenario(S1, "planning", 25, "RETENTION_VRET")
        w = sc.window(2, 25)
        self.assertEqual(w["wakes_per_token"], S1["L"])
        self.assertAlmostEqual(w["transition_mJ_per_token"], sum(sc.wake_e(b) for b in sc.layer_prov) * 1e3, places=9)
        self.assertAlmostEqual(w["resident_wake_W"], w["transition_mJ_per_token"] * 1e-3 * 25, places=12)
        self.assertEqual(sc.P0(25)["resident_wake_W"], 0.0)
        t = sc.timeout(0.0, 25); self.assertEqual(t["wakes_per_token"], 1)
        self.assertAlmostEqual(t["transition_mJ_per_token"], sc.wake_e(sc.prov) * 1e3, places=9)
        self.assertEqual(sc.timeout(1.0, 25)["wakes_per_token"], 0)           # gap 40 ms < 1 s timeout: never sleeps

    def test_10_wake_latency_vs_slack(self):
        sc = Scenario(S1, "planning", 25, "RETENTION_VRET")
        p1 = sc.window(1, 25); p2 = sc.window(2, 25)
        self.assertGreater(p1["stall_us_per_token"], 0.0)                     # no lead time
        self.assertAlmostEqual(p1["stall_us_per_token"], sum(sc.wake_t(b) for b in sc.layer_prov) * 1e6, places=6)
        self.assertEqual(p2["stall_us_per_token"], 0.0)                       # 1.67 ms of lead at 25 tok/s
        slow = Scenario(S1, "high", 25, "RETENTION_VRET", engine_bw=8e12)
        b = slow.window(2, 25, "burst")
        self.assertGreater(b["stall_us_per_token"], 0.0)                      # high-case 72 us wake > 2 us burst layer
        self.assertAlmostEqual(WK.wake_latency_s(256e3, "planning"), 200e-9, places=12)
        self.assertAlmostEqual(WK.wake_latency_s(S1["provisioned_bytes"], "planning"), 200e-9 * -(-WK.macros_in(S1["provisioned_bytes"]) // 8), places=12)

    def test_11_transition_frequency_per_policy(self):
        sc = Scenario(S1, "planning", 25, "RETENTION_VRET")
        self.assertEqual(sc.window(4, 100)["wakes_per_token"], 24)
        self.assertEqual(sc.timeout(0.0, 100)["wakes_per_token"], 1)
        self.assertEqual(sc.timeout(1e-3, 100)["wakes_per_token"], 1)          # 10 ms gap > 1 ms timeout
        self.assertEqual(sc.timeout(100e-3, 100)["wakes_per_token"], 0)
        s = sc.session(64, 25, 10.0)
        self.assertAlmostEqual(s["session_wake_mJ"], sc.wake_e(sc.prov) * 1e3, places=9)

    def test_13_session_length_amortization(self):
        sc = Scenario(S1, "planning", 25, "RETENTION_VRET")
        a = sc.session(16, 25, 10.0); b = sc.session(1024, 25, 10.0)
        # one wake per session: its share of average power shrinks with session length
        self.assertGreater(a["session_wake_mJ"] / (a["resident_session_mJ"]), b["session_wake_mJ"] / (b["resident_session_mJ"]))
        T_a = 16 / 25 + 10.0
        self.assertAlmostEqual(a["resident_avg_mW"], (a["resident_session_mJ"] + a["resident_idle_mJ"]) / T_a, places=6)
        self.assertEqual(sc.session(16, 25, 0.5)["lpddr_idle_mode"], "power-down")
        self.assertEqual(sc.session(16, 25, 60.0)["lpddr_idle_mode"], "self-refresh")

    def test_14_timeout_policy_energy(self):
        sc = Scenario(S1, "planning", 85, "RETENTION_VRET")
        Tp = 1 / 25; tc = sc.t_compute(); X = 10e-3
        t = sc.timeout(X, 25)
        f = (tc + X) / Tp
        self.assertAlmostEqual(t["active_capacity_fraction"], f, places=9)
        self.assertAlmostEqual(t["resident_leakage_W"], f * sc.p_all_active() + (1 - f) * sc.p_all_retained(), places=12)
        bet = sc.wake_e(sc.prov) / (sc.p_all_active() - sc.p_all_retained())
        be = next(r for r in C.break_even_rows({("falcon-e-1b", 2.0): S1}) if r["temp_c"] == 85 and r["threshold"].startswith("7."))
        self.assertAlmostEqual(be["value"], bet * 1e3, places=3)

    def test_15_int4_identical_policy_algorithm(self):
        a, b = Scenario(S1, "planning", 85, "RETENTION_VRET"), Scenario(S4, "planning", 85, "RETENTION_VRET")
        self.assertEqual(a.ratio, b.ratio); self.assertEqual(a.act_mw_mb, b.act_mw_mb); self.assertEqual(a.wake_uj_mb, b.wake_uj_mb)
        wa, wb = a.window(2, 25), b.window(2, 25)
        self.assertEqual(wa["wakes_per_token"], wb["wakes_per_token"])
        self.assertAlmostEqual(wb["resident_leakage_W"] / wa["resident_leakage_W"], 2.25, places=6)
        self.assertAlmostEqual(wb["resident_dynamic_W"] / wa["resident_dynamic_W"], 2.25, places=6)

    def test_18_head_kv_included_identically(self):
        rows = C.system_rows({("falcon-e-1b", 2.0): S1}, WS)
        r = next(x for x in rows if x["context"] == 2048 and x["case"].startswith("BF16") and x["temp_c"] == 25)
        self.assertGreater(r["external_head_kv_mJ_per_token_both"], 0)
        self.assertLess(r["total_ratio_P2_25tok"], r["weights_only_ratio_P2_25tok"])
        ref = next(x for x in _csv("system_reconnect.csv") if x["model"] == "falcon-e-1b" and x["context"] == "2048" and x["non_ternary_case"].startswith("BF16"))
        self.assertAlmostEqual(r["external_head_kv_mJ_per_token_both"], float(ref["external_head_kv_mJ_both_systems"]), places=2)

    def test_19_break_even_equations(self):
        sc = Scenario(S1, "planning", 25, "RETENTION_VRET")
        rows = C.break_even_rows({("falcon-e-1b", 2.0): S1})
        r8 = next(r for r in rows if r["temp_c"] == 25 and r["threshold"].startswith("8. token rate (continuous) above which P2"))
        R_star = r8["value"]
        self.assertAlmostEqual(sc.window(2, R_star)["lpddr_over_resident"], 1.0, places=2)
        r9 = next(r for r in rows if r["temp_c"] == 25 and r["threshold"].startswith("9. duty cycle at 25 tok/s above which PS"))
        self.assertAlmostEqual(sc.duty(r9["value"], 25, "PS")["lpddr_over_resident"], 1.0, places=2)
        r1 = next(r for r in rows if r["temp_c"] == 85 and r["threshold"].startswith("1."))
        sc85 = Scenario(S1, "planning", 85, "RETENTION_VRET")
        f = 2 / S1["L"]; p_act = sc85.p_all_active(); R_tok = 25
        e_w = sum(sc85.wake_e(b) for b in sc85.layer_prov)
        p_res = sc85.e_dyn * R_tok + f * p_act + (1 - f) * r1["value"] * p_act + e_w * R_tok
        self.assertAlmostEqual(p_res, sc85.window(2, 25)["lpddr_total_W"], delta=1e-4)


class TestProvenanceAndUnits(unittest.TestCase):
    def test_20_every_constant_has_provenance(self):
        ids = {s["source_id"] for s in R.SOURCES} | {a["source_id"] for a in R.ASSUMPTIONS}
        with open(os.path.join("analysis", "sram_retention", "retention_evidence.py"), encoding="utf-8") as f:
            txt = f.read()
        block = txt.split("# ----------------------------------------------------------------------------- constants (every number tagged)")[1].split("def write")[0]
        for line in block.splitlines():
            if line.strip().startswith("(") or line.strip().startswith("}") or line.strip().startswith("#") or not line.strip():
                continue
            unq = re.sub(r'"[^"]*"', '""', line)
            if "=" in line and re.search(r"\d", unq):
                tags = re.findall(r"\b([RBA]\d{1,2})\b", line)
                self.assertTrue(tags or "EXP-000" in line, f"no provenance on: {line.strip()[:80]}")
                for t in tags:
                    if t.startswith(("R", "B")):
                        self.assertIn(t, ids)
        for s in R.SOURCES:
            self.assertTrue(s["tier"].startswith("T"), s["source_id"])

    def test_21_units_explicit(self):
        self.assertEqual(R.MB_BITS, 8e6); self.assertEqual(R.MBIT, 2 ** 20)
        r01 = next(s for s in R.SOURCES if s["source_id"] == "R01")
        self.assertIn("mW/MB", r01["leakage_per_MB"])
        self.assertAlmostEqual(float(r01["leakage_per_MB"].split()[0]), 27.5e-6 * 8e6 / 2 ** 20 * 1e3, places=3)
        self.assertAlmostEqual(R.ACTIVE_LEAK_25C_MW_PER_MB["low"], 0.21, places=2)

    def test_22_existing_regressions_untouched(self):
        self.assertAlmostEqual(resident_pj_per_bit("planning")["total_pj_per_bit"], 0.4405, places=4)
        self.assertEqual(E8.LEAK_MW_PER_MB["planning"], 0.60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
