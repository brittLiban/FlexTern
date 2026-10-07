"""
EXP-0006 validation tests (prompt items 1-17).  Run from the repository root:
    python -m unittest tests.test_memory_residency -v
Pure Python; reads the committed canonical profiles.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "analysis"))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "hardware_gate"))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "memory_residency"))
os.chdir(_ROOT)

import capacity as cap  # noqa: E402
import physical as phy  # noqa: E402
import residency as res  # noqa: E402
import sources as src  # noqa: E402
from traffic_model import per_token  # noqa: E402
from workload_mapping import nearmem_cycles  # noqa: E402

WS = cap.load()
F1B = WS["falcon-e-1b"]


class TestCapacity(unittest.TestCase):
    def test_1_2_canonical_sizes_match_exp0005(self):
        self.assertEqual(cap.transformer_bytes(F1B, 2), 396361728)                    # EXP-0005 / EXP-0003
        self.assertEqual(cap.transformer_bytes(WS["falcon-e-3b"], 2), 729808896)
        self.assertEqual(cap.transformer_bytes(WS["bitnet-2b4t"], 2), 521011200)
        hg = list(csv.DictReader(open(os.path.join("results", "hardware_gate", "traffic_comparison.csv"), encoding="utf-8")))
        self.assertEqual(int(next(r for r in hg if r["model"] == "falcon-e-1b")["packed_weight_bytes"]), cap.transformer_bytes(F1B, 2))

    def test_3_int4_exactly_comparable(self):
        for w in WS.values():
            self.assertEqual(cap.transformer_bytes(w, 4), 2 * cap.transformer_bytes(w, 2))
            self.assertEqual(cap.transformer_bytes(w, 8), 4 * cap.transformer_bytes(w, 2))
            self.assertEqual(cap.transformer_bytes(w, 16), 8 * cap.transformer_bytes(w, 2))
            self.assertEqual(cap.head_bytes(w, 16), w.head_bytes_measured)          # measured BF16 head bytes

    def test_13_decimal_units_explicit(self):
        rows = cap.capacity_rows(WS)
        r = next(x for x in rows if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2)
        self.assertEqual(r["transformer_MB"], 396.4)                                   # 396,361,728 / 1e6, not / 2^20 (= 378.0)
        self.assertNotEqual(r["transformer_MB"], round(396361728 / 2**20, 1))
        self.assertAlmostEqual(src.LPDDR5X_GBPS_PER_64BIT, 68.264, places=3)

    def test_margins(self):
        r = next(x for x in cap.capacity_rows(WS) if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2)
        self.assertAlmostEqual(r["transformer_physical_MB_margin_20pct"], 396.4 / 0.8, delta=0.1)

    def test_15_16_hypothetical_labeled(self):
        for r in cap.hypothetical_rows():
            self.assertIn("HYPOTHESIZED", r["label"])
        r7 = next(r for r in cap.hypothetical_rows() if r["model"] == "hyp-7B" and r["weight_bits"] == 2)
        self.assertAlmostEqual(r7["transformer_MB"], 7e9 * 0.95 * 2 / 8 / 1e6, delta=0.1)


class TestResidency(unittest.TestCase):
    def test_4_bank_capacities_sum(self):
        for r in res.bank_rows(F1B, 2):
            self.assertGreaterEqual(r["sum_bank_capacity_MB"], r["whole_model_MB"])
            self.assertLess(r["tail_waste_pct"], 25)
        r16 = next(r for r in res.bank_rows(F1B, 2) if r["banks"] == 16)
        self.assertEqual(r16["tail_waste_pct"], 0.0)

    def test_5_9_levels_match_exp0005(self):
        tr = per_token(F1B, nb=64)
        rows = res.policy_rows(F1B, 2)
        full = next(r for r in rows if r["policy"].startswith("full"))
        self.assertEqual(full["local_weight_bytes_per_token"], tr["nm_C_local_reads"])
        self.assertEqual(full["activation_injected_bytes_per_token"], tr["nm_D_activation_injected"])
        b64 = next(r for r in res.bank_rows(F1B, 2) if r["banks"] == 64)
        self.assertEqual(b64["activation_delivered_bytes_per_token"], tr["nm_D_activation_delivered"])
        self.assertEqual(b64["activation_delivered_bytes_per_token"], 64 * tr["nm_D_activation_injected"])

    def test_6_full_residency_zero_global_weights(self):
        rows = res.policy_rows(F1B, 2, tokens_resident=10**9)
        full = next(r for r in rows if r["policy"].startswith("full"))
        self.assertEqual(full["global_weight_bytes_per_token"], 0)

    def test_7_8_partial_and_layer_reintroduce_traffic(self):
        rows = res.policy_rows(F1B, 2)
        W = cap.transformer_bytes(F1B, 2)
        layer = next(r for r in rows if r["policy"].startswith("layer"))
        self.assertEqual(layer["global_weight_bytes_per_token"], W)                  # streaming-like
        self.assertEqual(layer["global_reduction_vs_streaming"], 1.0)
        hot = next(r for r in rows if r["policy"].startswith("hot-set residency (50%"))
        self.assertEqual(hot["global_weight_bytes_per_token"], W // 2)
        tiles = [r for r in rows if r["policy"].startswith("tiled")]
        self.assertTrue(all(r["global_weight_bytes_per_token"] == W for r in tiles))
        full1 = next(r for r in res.policy_rows(F1B, 2, tokens_resident=1) if "amortized" in r["policy"])
        self.assertEqual(full1["global_weight_bytes_per_token"], W)                  # reload every token = streaming

    def test_10_collector_preserved(self):
        t = {"K": 2048, "N": 256, "rowq": 64, "terms": 256 * 2048}
        self.assertEqual(nearmem_cycles(t, 256, 16)[0], 128)
        b = res.bank_rows(F1B, 2)
        self.assertTrue(next(r for r in b if r["banks"] == 256)["collector_bound"])
        self.assertFalse(next(r for r in b if r["banks"] == 64)["collector_bound"])
        self.assertLess(next(r for r in b if r["banks"] == 256)["utilization"], next(r for r in b if r["banks"] == 64)["utilization"])

    def test_bandwidth_formula(self):
        r = next(x for x in res.bandwidth_rows(F1B, 2) if x["target_tok_s"] == 100 and x["utilization"] == 0.7)
        self.assertAlmostEqual(r["resident_local_aggregate_GBps_required"], 396361728 * 100 / 0.7 / 1e9, places=2)
        self.assertAlmostEqual(r["local_per_bank_GBps_NB64"], 396361728 * 100 / 64 / 0.7 / 1e9, places=3)


class TestPhysical(unittest.TestCase):
    def test_11_every_constant_has_provenance(self):
        ids = {s["source_id"] for s in src.SOURCES} | {a["source_id"] for a in src.ASSUMPTIONS}
        txt = open(os.path.join("analysis", "memory_residency", "sources.py"), encoding="utf-8").read()
        const_block = txt.split("# ----------------------------------------------------------------------------- constants the models import")[1]
        for line in const_block.splitlines():
            unquoted = re.sub(r'"[^"]*"', '""', line)          # digits inside tier-name strings are not constants
            if "=" in line and not line.strip().startswith("#") and re.search(r"\d", unquoted):
                tags = re.findall(r"\b([SA]\d{1,2})\b", line)
                self.assertTrue(tags or "EXP-0004" in line or line.strip().startswith(('"', "}", "]", "dict(")), f"no provenance tag on: {line.strip()}")
                for t in tags:
                    self.assertIn(t, ids)
        # models import constants only from sources.py
        for mod in ("capacity.py", "residency.py", "physical.py"):
            body = open(os.path.join("analysis", "memory_residency", mod), encoding="utf-8").read()
            self.assertNotIn("pJ", body.split('"""')[2] if body.count('"""') >= 2 else body, mod)

    def test_12_unit_conversions(self):
        self.assertAlmostEqual(src.LPDDR5X_GBPS_PER_64BIT, 8533e6 * 8 / 1e9, places=6)   # 64 bits = 8 bytes per transfer
        lo, hi = src.ENERGY_PJ_PER_BIT["hbm_class"]
        self.assertAlmostEqual(396361728 * 8 * lo / 1e9, 12.37, places=1)              # mJ per token at 3.9 pJ/bit

    def test_14_energy_rows_ranges_labeled(self):
        pts = cap.all_model_points(WS)
        rows = phy.energy_rows(pts, {"falcon-e-1b": 2777088})
        r = next(x for x in rows if x["model"] == "falcon-e-1b" and x["weight_bits"] == 2)
        self.assertAlmostEqual(r["stream_hbm_mJ_per_token_min"], 12.37, places=1)
        self.assertLess(r["resident_sram_local_mJ_per_token_max"], r["stream_hbm_mJ_per_token_min"])
        self.assertIn("NOT a chip J/token", r["label"])

    def test_15_area_tiers_distinguished(self):
        rows = phy.area_rows(cap.all_model_points(WS))
        f = [r for r in rows if r["model"] == "falcon-e-1b" and r["weight_bits"] == 2]
        raw = next(r for r in f if r["density_tier"].startswith("raw")); chip = next(r for r in f if r["density_tier"] == src.CHIP_TIER)
        self.assertLess(raw["area_mm2_min"], chip["area_mm2_min"])
        self.assertEqual(raw["usable_for_chip_estimate"], "NO (bit-cell only)")
        self.assertEqual(chip["usable_for_chip_estimate"], "yes (includes compute)")
        self.assertAlmostEqual(chip["area_mm2_max"], 396361728 / 0.9 / 0.68e6, delta=1)

    def test_tier_crossings(self):
        tiers = {(r["model"], r["weight_bits"]): r for r in phy.tier_crossing_rows(cap.all_model_points(WS))}
        self.assertTrue(tiers[("falcon-e-1b", 2)]["memory_organization_class"].startswith("one monolithic"))
        self.assertTrue(tiers[("falcon-e-3b", 4)]["memory_organization_class"].startswith(("exceeds one reticle", "multi-die SRAM")))
        self.assertTrue(tiers[("bitnet-2b4t", 4)]["memory_organization_class"].startswith("exceeds one reticle"))
        self.assertTrue(tiers[("hyp-7B", 2)]["memory_organization_class"].startswith("multi-die SRAM"))
        self.assertIn("DRAM-class", tiers[("hyp-30B", 2)]["memory_organization_class"])
        self.assertEqual(tiers[("falcon-e-1b", 4)]["physical_MB"], round(2 * tiers[("falcon-e-1b", 2)]["physical_MB"], 1))

    def test_16_extrapolations_labeled(self):
        for r in phy.area_rows(cap.all_model_points(WS)) + phy.economics_rows(cap.all_model_points(WS)):
            self.assertTrue(any(k in r["label"] for k in ("INFERRED", "PROXY")), r["label"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
