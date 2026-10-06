"""
Validation tests for the EXP-0003 analytical bottleneck model.

Run from the repository root:
    python -m unittest tests.test_roofline -v

Reads the committed canonical profiles under results/workload_profile/.
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, os.path.join(_ROOT, "analysis"))
os.chdir(_ROOT)

from model_specs import SPECS  # noqa: E402
from roofline_models import (Assumptions, Workload, accumulator_bits, attention_ops, batch_for_weight_eq_kv,  # noqa: E402
                             break_even, context_crossovers, head_bytes, kv_read_bytes, kv_write_bytes,
                             packed_weight_bytes, residency_external_bytes, token_model, tile_utilization,
                             prefill_attention_equals_ternary)

MODELS = ["bitnet-2b4t", "falcon-e-1b", "falcon-e-3b"]
W = {s: Workload.from_profile_dir(SPECS[s].default_out) for s in MODELS}
D = Assumptions()


class TestInputs(unittest.TestCase):
    def test_1_all_three_summaries_load_with_expected_identity(self):
        for s in MODELS:
            self.assertEqual(W[s].model_id, SPECS[s].model_id)
            self.assertEqual(W[s].revision, SPECS[s].revision)

    def test_2_totals_reproduce_exp0002_and_0002b(self):
        self.assertEqual(W["bitnet-2b4t"].W, 2_084_044_800)
        self.assertEqual(W["falcon-e-1b"].W, 1_585_446_912)
        self.assertEqual(W["falcon-e-3b"].W, 2_919_235_584)
        self.assertEqual(kv_read_bytes(W["bitnet-2b4t"], 4096, 2.0), 314_572_800)
        self.assertEqual(kv_read_bytes(W["falcon-e-1b"], 32768, 2.0), 805_306_368)
        self.assertEqual(kv_read_bytes(W["falcon-e-3b"], 32768, 2.0), 1_073_741_824)
        self.assertEqual(head_bytes(W["bitnet-2b4t"], 2.0), 656_670_720)
        self.assertEqual(head_bytes(W["falcon-e-1b"], 2.0), 134_217_728)
        self.assertEqual(W["bitnet-2b4t"].head_bytes_measured, 656_670_720)

    def test_20_no_model_specific_constants(self):
        # Every workload quantity used by the model differs across the three inputs; nothing is shared by accident.
        self.assertEqual(len({W[s].W for s in MODELS}), 3)
        self.assertEqual(len({W[s].L for s in MODELS}), 3)
        self.assertEqual(len({kv_read_bytes(W[s], 1024, 2.0) for s in MODELS}), 3)


class TestTermsPerByte(unittest.TestCase):
    def test_3_dense_2bit_gives_4_terms_per_weight_byte(self):
        for w in W.values():
            self.assertAlmostEqual(w.W / packed_weight_bytes(w, 2.0), 4.0, places=12)
            tm = token_model(w, replace(D, batch=1))
            t = tm.stage("ternary")
            self.assertAlmostEqual(t.compute_ops / t.memory_bytes, 4.0, places=9)
            tm = token_model(w, replace(D, batch=8))
            t = tm.stage("ternary")
            self.assertAlmostEqual(t.compute_ops / t.memory_bytes, 32.0, places=9)  # 4B at batch B

    def test_4_five_trit_gives_5_terms_per_byte(self):
        for w in W.values():
            self.assertAlmostEqual(w.W / packed_weight_bytes(w, 1.6), 5.0, places=12)


class TestMonotonicity(unittest.TestCase):
    def test_5_more_bandwidth_never_increases_memory_time(self):
        for w in W.values():
            prev = None
            for bw in (100e9, 250e9, 500e9, 1e12, 2e12, 4e12, 8e12):
                tm = token_model(w, replace(D, bw=bw))
                mt = sum(s.memory_time for s in tm.stages)
                if prev is not None:
                    self.assertLessEqual(mt, prev + 1e-15)
                prev = mt

    def test_6_more_compute_never_increases_compute_time(self):
        for w in W.values():
            prev = None
            for tpc in (128, 512, 2048, 8192, 32768, 131072):
                tm = token_model(w, replace(D, terms_per_cycle=tpc))
                ct = sum(s.compute_time for s in tm.stages)
                if prev is not None:
                    self.assertLessEqual(ct, prev + 1e-15)
                prev = ct

    def test_7_context_monotonically_increases_kv_read(self):
        for w in W.values():
            prev = -1
            for C in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
                kv = kv_read_bytes(w, C, 2.0)
                self.assertGreater(kv, prev); prev = kv
            self.assertEqual(kv_read_bytes(w, 2048, 2.0), 2 * kv_read_bytes(w, 1024, 2.0))

    def test_8_batch_amortizes_reusable_weights_only(self):
        for w in W.values():
            a1 = token_model(w, replace(D, batch=1)); a8 = token_model(w, replace(D, batch=8))
            self.assertAlmostEqual(a8.stage("ternary").memory_bytes * 8, a1.stage("ternary").memory_bytes)
            self.assertAlmostEqual(a8.stage("head").memory_bytes * 8, a1.stage("head").memory_bytes)
            self.assertEqual(a8.stage("attention_kv").memory_bytes, a1.stage("attention_kv").memory_bytes)  # KV not shared
            self.assertEqual(a8.stage("ternary").compute_ops, a1.stage("ternary").compute_ops)             # per-token compute unchanged
            # ceiling falls with batch at fixed context (weights shrink, KV does not)
            self.assertLess(a8.ceilings()["ceiling_ideal"], a1.ceilings()["ceiling_ideal"])

    def test_9_zeros_do_not_reduce_dense_packed_bytes(self):
        for w in W.values():
            b = [token_model(w, replace(D, zero_mode=m)).stage("ternary").memory_bytes for m in ("dense", "gated", "ideal_skip")]
            self.assertEqual(b[0], b[1]); self.assertEqual(b[1], b[2])
            self.assertLess(token_model(w, replace(D, zero_mode="ideal_skip")).stage("ternary").compute_ops, w.W)

    def test_10_head_precision_scaling(self):
        for w in W.values():
            b16 = token_model(w, replace(D, head_bytes_per_elem=2.0)).stage("head").memory_bytes
            b8 = token_model(w, replace(D, head_bytes_per_elem=1.0)).stage("head").memory_bytes
            b4 = token_model(w, replace(D, head_bytes_per_elem=0.5)).stage("head").memory_bytes
            self.assertAlmostEqual(b16 / b8, 2.0); self.assertAlmostEqual(b8 / b4, 2.0)

    def test_11_kv_precision_scaling(self):
        for w in W.values():
            self.assertAlmostEqual(kv_read_bytes(w, 4096, 2.0) / kv_read_bytes(w, 4096, 1.0), 2.0)
            self.assertAlmostEqual(kv_read_bytes(w, 4096, 1.0) / kv_read_bytes(w, 4096, 0.5), 2.0)
            self.assertAlmostEqual(kv_write_bytes(w, 2.0) * 4096, kv_read_bytes(w, 4096, 2.0))


class TestReconciliation(unittest.TestCase):
    def test_12_accumulator_widths_match_exp0002b(self):
        exp = {2048: (20, 19), 2560: (20, 20), 6912: (21, 21), 9216: (22, 22), 13312: (22, 22)}
        for K, (f, m) in exp.items():
            r = accumulator_bits(K, 8)
            self.assertEqual((r["format_bits"], r["model_bits"]), (f, m), K)
        self.assertEqual(accumulator_bits(16, 8)["format_bits"], 13)      # Phase 1 Y-Dot16
        self.assertEqual(accumulator_bits(2048, 4)["format_bits"], 16)    # 8*2048 = 2^14 exactly -> needs 16 bits signed
        self.assertEqual(accumulator_bits(2048, 4)["model_bits"], 15)     # 7*2048 = 14336 < 2^14 -> 15 bits

    def test_13_component_bytes_sum_to_total(self):
        for w in W.values():
            tm = token_model(w, D)
            self.assertAlmostEqual(sum(s.memory_bytes for s in tm.stages), tm.total_bytes)
            self.assertAlmostEqual(tm.nonternary_bytes + tm.stage("ternary").memory_bytes, tm.total_bytes)

    def test_14_component_times_sum_under_overlap_assumptions(self):
        for w in W.values():
            tm = token_model(w, D)
            self.assertAlmostEqual(sum(max(s.compute_time, s.memory_time) for s in tm.stages), tm.total_optimistic)
            self.assertAlmostEqual(sum(s.compute_time + s.memory_time for s in tm.stages), tm.total_conservative)
            self.assertLessEqual(tm.total_optimistic, tm.total_conservative)
            self.assertGreaterEqual(tm.total_conservative, 0.5 * tm.total_optimistic)  # cons <= 2*opt per stage

    def test_19_units_consistent(self):
        w = W["falcon-e-1b"]
        a = replace(D, bw=1e9, terms_per_cycle=1, clock_hz=1.0)  # 1 byte/ns bandwidth, 1 term/s engine
        tm = token_model(w, a)
        self.assertAlmostEqual(tm.stage("ternary").memory_time, packed_weight_bytes(w, 2.0) / 1e9)
        self.assertAlmostEqual(tm.stage("ternary").compute_time, w.W)


class TestBreakEven(unittest.TestCase):
    def test_15_break_even_equations_agree_with_sweep_transitions(self):
        tpcs = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        for w in W.values():
            for bw in (100e9, 1e12, 8e12):
                a = replace(D, bw=bw, clock_hz=1e9)
                lanes = break_even(w, a)["lanes_at_bw_and_clock"]
                limiters = [token_model(w, replace(a, terms_per_cycle=t)).stage("ternary").limiter for t in tpcs]
                # memory-limited exactly when terms/cycle >= lanes
                for t, lim in zip(tpcs, limiters):
                    self.assertEqual(lim, "memory" if t >= lanes else "compute", (w.slug, bw, t))
                # batch break-even
                B_star = break_even(w, a)["batch_to_balance_ternary"]
                for B in (1, 2, 4, 8, 16, 32):
                    lim = token_model(w, replace(a, batch=B)).stage("ternary").limiter
                    self.assertEqual(lim, "compute" if B >= B_star else "memory", (w.slug, bw, B))

    def test_context_crossover_matches_direct_comparison(self):
        for w in W.values():
            c = context_crossovers(w, D)
            C = c["ctx_kv_gt_ternary"]
            self.assertAlmostEqual(kv_read_bytes(w, C, 2.0), packed_weight_bytes(w, 2.0))
            C2 = c["ctx_nonternary_mem_gt_ternary_mem"]
            if C2 == 0.0:
                # 'always': non-ternary bytes exceed ternary bytes even at tiny context (BitNet's head)
                tm = token_model(w, replace(D, context=128, bw=100e9))
                self.assertLess(tm.ceilings()["memory_bound_bytes_ratio"], 2.0)
            else:
                tm = token_model(w, replace(D, context=int(round(C2)), bw=100e9))
                ratio = tm.ceilings()["memory_bound_bytes_ratio"]
                self.assertLess(abs(ratio - 2.0) / 2.0, 0.01)   # KV write bytes (1 context unit) explain the residual
                self.assertLess(abs(tm.ceilings()["ceiling_ideal"] - 2.0) / 2.0, 0.01)

    def test_batch_weight_eq_kv(self):
        for w in W.values():
            a = replace(D, context=4096)
            B = batch_for_weight_eq_kv(w, a)
            weights = packed_weight_bytes(w, 2.0) + head_bytes(w, 2.0) + 2 * (w.norm_params + w.scale_params)
            self.assertAlmostEqual(weights / B, kv_read_bytes(w, 4096, 2.0))


class TestCeilings(unittest.TestCase):
    def test_16_ideal_never_below_partial(self):
        for w in W.values():
            for C in (128, 4096, 32768):
                for bw in (100e9, 1e12, 8e12):
                    for tpc in (512, 4096, 65536):
                        c = token_model(w, replace(D, context=C, bw=bw, terms_per_cycle=tpc)).ceilings()
                        self.assertGreaterEqual(c["ceiling_ideal"] + 1e-12, c["ceiling_compute_only"])
                        self.assertGreaterEqual(c["ceiling_ideal"] + 1e-12, c["ceiling_bandwidth_only"])
                        self.assertGreaterEqual(c["ceiling_compute_only"], 1.0)
                        self.assertGreaterEqual(c["ceiling_bandwidth_only"], 1.0)

    def test_17_ideal_ceiling_approaches_1_when_nonternary_dominates(self):
        for w in W.values():
            c_small = token_model(w, replace(D, context=128, bw=100e9)).ceilings()["ceiling_ideal"]
            c_huge = token_model(w, replace(D, context=1 << 22, bw=100e9)).ceilings()["ceiling_ideal"]  # hypothetical 4M context
            self.assertLess(c_huge, c_small)
            self.assertLess(c_huge, 1.05)

    def test_memory_bound_identity(self):
        # At very low BW every stage is memory-bound and the ideal ceiling equals the byte ratio.
        for w in W.values():
            for C in (512, 4096):
                tm = token_model(w, replace(D, context=C, bw=1e9))
                c = tm.ceilings()
                self.assertAlmostEqual(c["ceiling_ideal"], c["memory_bound_bytes_ratio"], places=9)


class TestResidency(unittest.TestCase):
    def test_18_residency_never_increases_external_traffic(self):
        for w in W.values():
            a = replace(D, context=4096)
            base = residency_external_bytes(w, a, 0, "none")["external_total_bytes"]
            for cap in (0, 16e6, 64e6, 256e6, 1e9, 4e9):
                for pol in ("none", "head_if_fits", "one_layer_buffer", "greedy_ternary_layers", "greedy_head_then_layers",
                            "ideal_ternary_first", "ideal_nonternary_first", "ideal_including_kv"):
                    r = residency_external_bytes(w, a, cap, pol)
                    self.assertLessEqual(r["external_total_bytes"], base + 1e-6, (w.slug, cap, pol))
                    self.assertGreaterEqual(r["external_total_bytes"], 0.0)
            full = residency_external_bytes(w, a, 1e12, "ideal_including_kv")
            self.assertEqual(full["external_total_bytes"], 0.0)
            # storage facts: one layer is the model's packed bytes divided by L
            from roofline_models import storage_facts
            sf = storage_facts(w)
            self.assertAlmostEqual(sf["one_layer_bytes"] * w.L, sf["full_ternary_bytes"])


class TestMisc(unittest.TestCase):
    def test_tiles(self):
        self.assertEqual(tile_utilization(6912, 512)["tiles"], 14)
        self.assertAlmostEqual(tile_utilization(6912, 512)["tail_utilization"], 6912 / (14 * 512))
        self.assertTrue(tile_utilization(6912, 256)["divisible"])
        for K in (2048, 2560, 9216, 13312):
            self.assertTrue(tile_utilization(K, 512)["divisible"])

    def test_attention_ops_linear_in_context(self):
        w = W["bitnet-2b4t"]
        self.assertAlmostEqual(attention_ops(w, 4096) / attention_ops(w, 2048), 2.0)
        # QK^T + AV MACs at 4096 match EXP-0002's 629,145,600 plus softmax ops
        self.assertEqual(w.L * 2 * w.nH * w.hd * 4096, 629_145_600)

    def test_prefill_crossover(self):
        w = W["bitnet-2b4t"]
        self.assertAlmostEqual(prefill_attention_equals_ternary(w), w.W / (w.nH * w.hd * w.L))


if __name__ == "__main__":
    unittest.main(verbosity=2)
