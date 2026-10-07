"""EXP-0008 validation tests (prompt items 1-19).  python -m unittest tests.test_sram_validation -v"""

from __future__ import annotations

import os
import re
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "analysis"))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "memory_residency"))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "sram_validation"))
os.chdir(_ROOT)

import model as M  # noqa: E402
import evidence as S  # noqa: E402
from capacity import transformer_bytes  # noqa: E402

WS = M.load()
F1B = WS["falcon-e-1b"]


class TestReconciliation(unittest.TestCase):
    def test_1_capacities_reproduce_exp0006(self):
        self.assertEqual(transformer_bytes(F1B, 2), 396361728)
        self.assertEqual(transformer_bytes(WS["falcon-e-3b"], 2), 729808896)
        self.assertEqual(transformer_bytes(WS["bitnet-2b4t"], 2), 521011200)
        r = next(x for x in M.model_rows(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2.0 and x["level"] == "planning")
        self.assertEqual(r["provisioned_MB"], 440.4)

    def test_2_old_planning_energy_reproduces(self):
        r = next(x for x in M.old_vs_new(WS) if x["model"] == "falcon-e-1b")
        self.assertAlmostEqual(r["exp0006_resident_planning_mJ"], 396361728 * 8 * 1.6 / 1e9, places=1)       # 5.07 mJ (EXP-0006)
        self.assertEqual(r["exp0006_lpddr_mJ_range"], "15.9-47.6")

    def test_3_17_provenance_and_node_labels(self):
        ids = {s["source_id"] for s in S.SOURCES} | {a["source_id"] for a in S.ASSUMPTIONS}
        txt = open(os.path.join("analysis", "sram_validation", "evidence.py"), encoding="utf-8").read()
        block = txt.split("# ----------------------------------------------------------------------------- constants (tagged)")[1]
        for line in block.splitlines():
            unq = re.sub(r'"[^"]*"', '""', line)
            if "=" in line and not line.strip().startswith("#") and re.search(r"\d", unq):
                tags = re.findall(r"\b([EA]\d{1,2})\b", line)
                self.assertTrue(tags or "EXP-000" in line, f"no provenance on: {line.strip()}")
        for s in S.SOURCES:
            self.assertTrue(s["tier"].startswith(("T1", "T2", "T3", "T4")), s["source_id"])
        e05 = next(s for s in S.SOURCES if s["source_id"] == "E05"); e10 = next(s for s in S.SOURCES if s["source_id"] == "E10")
        self.assertIn("NOT modern-node", e05["tier"]); self.assertIn("historical", e10["tier"])

    def test_13_14_boundaries_tagged(self):
        r = M.model_rows(WS)[0]
        self.assertIn("SRAM macro", r["boundary_resident"]); self.assertIn("NO leakage", r["boundary_resident"])
        self.assertIn("DRAM device", r["boundary_lpddr"]); self.assertIn("PHY", r["boundary_lpddr"])


class TestConversions(unittest.TestCase):
    def test_4_energy_per_access_to_per_bit(self):
        self.assertAlmostEqual(5.8 / 64 * 1000, 90.6, places=1)                 # E02 at 64 bits -> fJ/bit
        self.assertAlmostEqual(5.8 / 256 * 1000, 22.7, places=1)
        lp = (16 * 1.80 + 382 * 1.05 + 0.2 * 0.90 + 121 * 0.50) / (17.07 * 8)   # E06 mW / Gbit/s = pJ/bit
        self.assertAlmostEqual(lp, 3.59, places=2)

    def test_5_read_width_normalization(self):
        rows = M.width_sweep()
        p128 = next(r for r in rows if r["read_width_bits"] == 128 and r["level"] == "planning"); p2048 = next(r for r in rows if r["read_width_bits"] == 2048 and r["level"] == "planning")
        self.assertGreater(p128["energy_per_bit_pJ"], p2048["energy_per_bit_pJ"])      # fixed cost amortized
        self.assertAlmostEqual(p2048["energy_per_access_pJ"] / 2048, p2048["energy_per_bit_pJ"], places=3)

    def test_16_macro_vs_cell(self):
        self.assertGreater(S.SRAM_MACRO_FJ_PER_BIT["low"], 10)                   # a cell-only figure would be single-digit fJ; macro boundary enforced
        self.assertLess(S.SRAM_MACRO_FJ_PER_BIT["high"] / 1000, S.LPDDR_PJ_PER_BIT["low"])

    def test_18_decimal_units(self):
        self.assertEqual(round(396361728 / 1e6, 1), 396.4)
        self.assertNotEqual(round(396361728 / 1e6, 1), round(396361728 / 2**20, 1))


class TestBankAndLeakage(unittest.TestCase):
    def test_6_7_bank_composition(self):
        for r in M.bank_decomposition():
            self.assertTrue(r["capacity_conserved"]); self.assertEqual(r["macros_active_per_access"], 1)
            self.assertAlmostEqual(r["active_fraction"], 1 / r["macros_per_bank"], places=4)

    def test_8_leakage_scales_with_provisioned_capacity(self):
        a = next(x for x in M.model_rows(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2.0 and x["level"] == "planning")
        b = next(x for x in M.model_rows(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 4.5 and x["level"] == "planning")
        self.assertAlmostEqual(b["sram_static_mW"] / a["sram_static_mW"], 2.25, places=2)
        self.assertAlmostEqual(a["sram_static_mW"], 440.4 * 0.60, delta=0.3)

    def test_9_int4_same_organization(self):
        a = next(x for x in M.model_rows(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2.0 and x["level"] == "planning")
        b = next(x for x in M.model_rows(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 4.5 and x["level"] == "planning")
        self.assertEqual(a["sram_total_pj_per_bit"], b["sram_total_pj_per_bit"])
        self.assertAlmostEqual(b["resident_dynamic_mJ_per_token"] / a["resident_dynamic_mJ_per_token"], 2.25, places=2)


class TestSweeps(unittest.TestCase):
    def test_10_duty_cycle_arithmetic(self):
        r = next(x for x in M.duty_cycle_sweep(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2.0 and x["duty_cycle"] == 1.0 and x["idle_retention_factor"] == 1.0)
        t = next(x for x in M.token_rate_sweep(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2.0 and x["tok_s"] == 25 and x["retention_factor"] == 1.0)
        self.assertAlmostEqual(r["resident_avg_mW"], t["resident_total_mW"], delta=0.2)                     # duty 1.0 == continuous 25 tok/s
        r0 = next(x for x in M.duty_cycle_sweep(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2.0 and x["duty_cycle"] == 0.01 and x["idle_retention_factor"] == 1.0)
        self.assertGreater(r0["resident_static_share"], 0.9)                                                  # at 1% duty leakage dominates

    def test_11_token_rate_power(self):
        d = next(x for x in M.model_rows(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2.0 and x["level"] == "planning")
        t = next(x for x in M.token_rate_sweep(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2.0 and x["tok_s"] == 10 and x["retention_factor"] == 1.0)
        self.assertAlmostEqual(t["resident_dynamic_mW"], d["resident_dynamic_mJ_per_token"] * 10, delta=0.2)
        self.assertAlmostEqual(t["lpddr_dynamic_mW"], d["lpddr_dynamic_mJ_per_token"] * 10, delta=0.2)

    def test_12_break_even_equations(self):
        be = [r for r in M.break_even(WS) if r["model"] == "falcon-e-1b"]
        tok = next(r for r in be if r["threshold"].startswith("token rate"))
        d = next(x for x in M.model_rows(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2.0 and x["level"] == "planning")
        sav = d["lpddr_dynamic_mJ_per_token"] - d["resident_dynamic_mJ_per_token"]
        self.assertAlmostEqual(tok["value"], (d["sram_static_mW"] - 5.4) / sav, places=1)
        bo = next(r for r in be if r["threshold"].startswith("local-wire"))
        self.assertLess(bo["value"], S.LPDDR_PJ_PER_BIT["planning"])
        pj = next(r for r in be if r["threshold"].startswith("SRAM total pJ/bit"))
        self.assertEqual(pj["value"], S.LPDDR_PJ_PER_BIT["planning"])

    def test_15_latency_first_vs_steady(self):
        r = next(x for x in M.latency_rows(WS) if x["model"] == "falcon-e-1b" and x["target_tok_s"] == 25)
        self.assertGreater(r["first_word_advantage_ns"], 50)
        self.assertLess(r["first_word_advantage_as_fraction_of_token"], 1e-5)
        self.assertTrue(r["latency_hideable"])
        self.assertFalse(r["lpddr_bandwidth_bound_at_target"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
