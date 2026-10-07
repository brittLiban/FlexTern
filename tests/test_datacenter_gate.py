"""
EXP-0007 validation tests (prompt items 1-24).  Run from the repository root:
    python -m unittest tests.test_datacenter_gate -v
"""

from __future__ import annotations

import os
import re
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "analysis"))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "hardware_gate"))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "memory_residency"))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "datacenter_gate"))
os.chdir(_ROOT)

import dc_sources as src  # noqa: E402
import systems as sy  # noqa: E402
import workloads as wl  # noqa: E402
from model_specs import SPECS  # noqa: E402
from roofline_models import Assumptions, Workload, token_model  # noqa: E402
from traffic_model import per_token  # noqa: E402

F1B = wl.canonical("falcon-e-1b"); F3B = wl.canonical("falcon-e-3b"); M70 = wl.hypothetical("hyp-70B"); M400 = wl.hypothetical("hyp-400B")
W1B = Workload.from_profile_dir(SPECS["falcon-e-1b"].default_out)


class TestWorkloads(unittest.TestCase):
    def test_1_canonical_constants_load(self):
        self.assertEqual(F1B.linear_params, 1585446912); self.assertEqual(F1B.L, 24); self.assertEqual(F1B.V, 32768)
        self.assertEqual(F3B.linear_params, 2919235584)

    def test_hypothetical_shapes_from_d11(self):
        self.assertAlmostEqual(M70.linear_params / 1e9, 68.4, delta=0.5)       # Llama-3.1-70B linear projections
        self.assertAlmostEqual(M400.linear_params / 1e9, 402, delta=3)
        self.assertEqual(M70.kv_values_per_token_layer, 2 * 8 * 128)

    def test_2_5_batch1_matches_exp0003_stream_once(self):
        """System B at batch 1, BF16 KV, BF16 head reproduces EXP-0003's total bytes per token (ternary + head + KV)."""
        r = sy.scenario(F1B, 1, 2048, "decode", "B", kv="bf16", head="bf16")
        tm = token_model(W1B, Assumptions(bw=100e9, batch=1, context=2048))
        self.assertAlmostEqual(r["weight_bytes"], tm.stage("ternary").memory_bytes, delta=1)
        self.assertAlmostEqual(r["head_bytes"], tm.stage("head").memory_bytes, delta=1)
        self.assertAlmostEqual(r["kv_read"] + r["kv_write"], tm.stage("attention_kv").memory_bytes, delta=1)

    def test_3_4_exp0005_exp0006_reconcile(self):
        r = sy.scenario(F1B, 1, 2048, "decode", "C", kv="fp8", head="fp4")
        self.assertAlmostEqual(r["local_weight_bytes"], per_token(W1B, nb=64)["nm_C_local_reads"], delta=1)      # EXP-0005 local reads
        self.assertEqual(F1B.weight_bytes(2.0), 396361728)                                                           # EXP-0006 capacity
        self.assertEqual(F1B.weight_bytes(4.5) / F1B.weight_bytes(2.0), 2.25)                                        # NVFP4 vs 2-bit, exact


class TestBatching(unittest.TestCase):
    def test_6_7_ideal_batch_monotone_and_limit(self):
        prev = None
        for B in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            r = sy.scenario(F1B, B, 2048, "decode", "A")
            if prev is not None:
                self.assertLessEqual(r["weight_bytes"], prev + 1e-6)
            prev = r["weight_bytes"]
        big = sy.scenario(F1B, 100000, 2048, "decode", "A")
        self.assertLess(big["weight_bytes"], F1B.weight_bytes(4.5) / 1000)

    def test_serving_utilization_reduces_reuse(self):
        r1 = sy.scenario(F1B, 64, 2048, "decode", "A", serving_util=1.0); r2 = sy.scenario(F1B, 64, 2048, "decode", "A", serving_util=0.5)
        self.assertAlmostEqual(r2["weight_bytes"], 2 * r1["weight_bytes"], delta=1)

    def test_pipeline_reuse_is_microbatch(self):
        r = sy.scenario(M70, 64, 8192, "decode", "C", cap=512e6)
        self.assertGreater(r["stages"], 1)
        self.assertEqual(r["microbatch"], max(1, -(-64 // r["stages"])))                        # ceil: every sequence is served
        self.assertAlmostEqual(r["local_weight_bytes"], M70.weight_bytes(2.0) / r["microbatch"], delta=1)


class TestKV(unittest.TestCase):
    def test_8_9_kv_per_sequence_and_context(self):
        r1 = sy.scenario(F1B, 1, 2048, "decode", "A"); r16 = sy.scenario(F1B, 16, 2048, "decode", "A")
        self.assertAlmostEqual(r1["kv_read"], r16["kv_read"])                                     # not batch-shared
        r4k = sy.scenario(F1B, 1, 4096, "decode", "A")
        self.assertAlmostEqual(r4k["kv_read"], 2 * r1["kv_read"])
        self.assertAlmostEqual(r1["kv_read"], 24 * 2 * 2 * 128 * 2048 * 1, delta=1)                # L x 2 x nKV x hd x ctx x 1 B (fp8)

    def test_kv_capacity_gate(self):
        r = sy.scenario(M70, 64, 131072, "decode", "A")
        self.assertGreater(r["gpus"], 8)                                                           # 64 x 128k fp8 KV does not fit one node
        c = sy.scenario(M70, 64, 8192, "decode", "C", cap=512e6)
        self.assertEqual(c["kv_location"], "dram")                                                 # resident tiles cannot hold KV
        c1 = sy.scenario(F1B, 1, 128, "decode", "C", cap=2e9)
        self.assertEqual(c1["kv_location"], "sram")

    def test_10_head_consistent(self):
        a = sy.scenario(M70, 1, 2048, "decode", "A", head="bf16"); b = sy.scenario(M70, 1, 2048, "decode", "A", head="fp4")
        self.assertAlmostEqual(a["head_bytes"] / b["head_bytes"], 16 / 4.5, places=3)
        self.assertAlmostEqual(a["head_bytes"], 128256 * 8192 * 2, delta=1)


class TestResident(unittest.TestCase):
    def test_11_12_13_capacity_and_traffic(self):
        self.assertEqual(sy.topology(M70, 4.5, 4.5, 512e6)["tiles"] >= sy.topology(M70, 2.0, 4.5, 512e6)["tiles"], True)
        c = sy.scenario(F1B, 1, 2048, "decode", "C")
        self.assertEqual(c["weight_bytes"], 0.0)                                                   # no global weight bytes after preload
        self.assertGreater(c["local_weight_bytes"], 0)

    def test_14_15_pipeline_fill_drain_and_throughput_vs_latency(self):
        r = sy.scenario(M70, 1, 8192, "decode", "C", cap=512e6)
        self.assertAlmostEqual(r["bubble_fraction"], 1 - 1 / r["stages"], places=6)                # one microbatch in flight
        r2 = sy.scenario(M70, 1000, 8192, "decode", "C", cap=512e6)
        self.assertEqual(r2["bubble_fraction"], 0.0)
        self.assertAlmostEqual(r2["latency_s"], r2["stages"] * r2["t_micro"], places=9)
        self.assertGreater(r2["agg_tok_s"], r2["seq_tok_s"])                                        # throughput != single-sequence rate

    def test_16_17_tiles_reconcile_same_algorithm(self):
        for cap in src.TILE_CAPS:
            tc = sy.topology(M70, 2.0, 4.5, cap); td = sy.topology(M70, 4.5, 4.5, cap)
            self.assertGreaterEqual(tc["weight_tiles"] * tc["usable"], M70.weight_bytes(2.0))
            self.assertGreaterEqual(td["weight_tiles"] * td["usable"], M70.weight_bytes(4.5))
            self.assertEqual(set(tc.keys()), set(td.keys()))

    def test_18_19_interconnect_conserved_and_partial_sums(self):
        r = sy.scenario(M70, 16, 8192, "decode", "C", cap=512e6)
        self.assertAlmostEqual(r["interconnect_bytes"], (r["stages"] - 1) * M70.H * src.ACT_BYTES + r["partial_sum_bytes"], delta=1)
        small = sy.scenario(M400, 16, 8192, "decode", "C", cap=256e6)
        self.assertGreater(small["tp_k"], 1); self.assertGreater(small["partial_sum_bytes"], 0)


class TestProvenance(unittest.TestCase):
    def test_20_23_constants_tagged_and_peaks_labeled(self):
        ids = {s["source_id"] for s in src.SOURCES} | {a["source_id"] for a in src.ASSUMPTIONS}
        txt = open(os.path.join("analysis", "datacenter_gate", "dc_sources.py"), encoding="utf-8").read()
        block = txt.split("# ----------------------------------------------------------------------------- constants (every number tagged)")[1]
        for line in block.splitlines():
            unq = re.sub(r'"[^"]*"', '""', line)
            if "=" in line and not line.strip().startswith("#") and re.search(r"\d", unq):
                tags = re.findall(r"\b([DA]\d{1,2})\b", line)
                self.assertTrue(tags or "EXP-000" in line, f"no provenance on: {line.strip()}")
                for t in tags:
                    self.assertIn(t, ids)
        for s in src.SOURCES:
            if "PFLOPS" in s["precision"] or "TB/s" in s["bandwidth"]:
                self.assertIn("peak", s["measured_or_peak"])

    def test_21_22_units(self):
        self.assertEqual(src.GPU_HBM_BYTES, 180e9)                                                 # decimal GB
        self.assertAlmostEqual(F1B.weight_bytes(2.0) / 1e6, 396.36, places=1)
        self.assertNotEqual(round(F1B.weight_bytes(2.0) / 1e6, 1), round(F1B.weight_bytes(2.0) / 2**20, 1))
        lo, hi = sy.energy_mj(sy.scenario(F1B, 1, 2048, "decode", "B", kv="bf16", head="bf16"), "B")
        self.assertAlmostEqual(lo, (396361728 + 134217728 + 50356224) * 8 * 3.9 / 1e9, delta=0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
