"""Unit tests for the EXP-0011 Stage 1 analysis code (parsers, comparator logic, preregistered classification). No model is run."""
import json
import math
import unittest

from analysis.ternary_vs_int4 import config as C
from analysis.ternary_vs_int4 import perf, quality, summarize


class TestConfig(unittest.TestCase):
    def test_p_eff_respects_trained_context(self):
        self.assertEqual(C.p_eff(512, 128), 512)
        self.assertEqual(C.p_eff(4096, 128), 3968)
        self.assertEqual(C.p_eff(4096, 256), 3840)
        for g in C.OUTPUTS:
            for p in C.PROMPTS:
                self.assertLessEqual(C.p_eff(p, g) + g, C.CTX_MAX)


class TestParsers(unittest.TestCase):
    def test_parse_bench_json_with_log_prefix(self):
        arr = [{"n_prompt": 512, "n_gen": 1, "n_depth": 0, "samples_ns": [10, 12, 11], "avg_ns": 11, "model_size": 5, "model_n_params": 4,
                "build_commit": "x", "cpu_info": "c", "backends": "CPU"}]
        r = perf.parse_bench_json("load: foo [bar]\n" + json.dumps(arr, indent=2) + "\n")
        self.assertEqual(r["samples_ns"], [10, 12, 11])
        self.assertEqual(r["n_prompt"], 512)

    def test_parse_time_v(self):
        t = "\tCommand being timed: x\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:03.21\n\tMaximum resident set size (kbytes): 123456\n"
        r = perf.parse_time_v(t)
        self.assertEqual(r["max_rss_kb"], 123456)
        self.assertEqual(r["wall_str"], "0:03.21")

    def test_derive_decode_tps(self):
        d = perf.derive([2_000_000_000] * 3, [4_000_000_000] * 3, 512, 101)
        self.assertEqual(d["ttft_s"], 2.0)
        self.assertEqual(d["total_s"], 4.0)
        self.assertAlmostEqual(d["decode_tps"], 50.0)
        self.assertAlmostEqual(d["prefill_tps"], 256.0)

    def test_quality_parsers(self):
        hs = "x\n5\t20.00000000%\t[3.6%, 62.4%]\n8\t25.00000000%\t[7.1%, 59.0%]\n"
        self.assertEqual(quality.parse_score("hellaswag", hs), {"acc": 25.0, "n": 8})
        old = "399\t61.40350877\n400\t61.50000000\n\n"          # bitnet.cpp 765741d format (no percent sign)
        self.assertEqual(quality.parse_score("hellaswag", old), {"acc": 61.5, "n": 400})
        wg = "1\t0.0000\t -4.36   -4.33  2  1\n8\t50.0000\t -2.76   -2.63  2  2\n\n"
        self.assertEqual(quality.parse_score("winogrande", wg), {"acc": 50.0, "n": 8})
        mc = "0.11.724.050 I Final result: 25.0000 +/- 16.3663\n0.11.724.051 I Random chance: 25.0000 +/- 16.3663\n"
        self.assertEqual(quality.parse_score("arc_easy", mc)["acc"], 25.0)
        with self.assertRaises(ValueError):
            quality.parse_score("hellaswag", "no rows here")

    def test_stderr_points(self):
        self.assertAlmostEqual(quality.stderr_points(50.0, 400), 2.5)


class TestDecision(unittest.TestCase):
    def test_matched_set_tolerance_boundaries(self):
        q = {C.TERNARY: 60.0, "QWEN25_1P5B_Q4KM": 62.0, "SMOLLM2_1P7B_Q4KM": 57.0, "LLAMA32_1B_Q4KM": 56.9}
        self.assertEqual(set(summarize.matched_set(q, 3.0)), {"QWEN25_1P5B_Q4KM", "SMOLLM2_1P7B_Q4KM"})
        self.assertEqual(set(summarize.matched_set(q, 0.0)), {"QWEN25_1P5B_Q4KM"})
        self.assertEqual(set(summarize.matched_set(q, 5.0)), {"QWEN25_1P5B_Q4KM", "SMOLLM2_1P7B_Q4KM", "LLAMA32_1B_Q4KM"})

    @staticmethod
    def _m(ttft, total, dec, rss, i="x"):
        return {"ttft_s": ttft, "total_s": total, "decode_tps": dec, "rss_kb": rss, "id": i}

    def test_cell_ratios_use_best_conventional_alternative_per_metric(self):
        tern = self._m(2.0, 10.0, 20.0, 1000)
        cands = [self._m(1.0, 12.0, 10.0, 2000, "a"), self._m(1.5, 11.0, 12.0, 1500, "b")]
        r = summarize.cell_ratios(tern, cands)
        self.assertAlmostEqual(r["decode"], 20 / 12)
        self.assertEqual(r["decode_comparator"], "b")
        self.assertAlmostEqual(r["ttft"], 0.5)
        self.assertEqual(r["ttft_comparator"], "a")
        self.assertAlmostEqual(r["total"], 1.1)
        self.assertEqual(r["total_comparator"], "b")
        self.assertAlmostEqual(r["memory"], 1.5)

    @staticmethod
    def _cells(total, dec, ttft, bytes_ratio=3.0, comp="QWEN25_1P5B_Q4KM|bitnet|t8"):
        return [{"total": total, "decode": dec, "ttft": ttft, "bytes_ratio": bytes_ratio, "total_comparator": comp} for _ in range(8)]

    Q = {C.TERNARY: 60.0, "QWEN25_1P5B_Q4KM": 61.0}

    def test_classification_labels_and_flags(self):
        c = summarize.classify
        self.assertEqual(c(self._cells(2.2, 2.4, 1.1), 3.0, 60.0, self.Q)["label"], "SPEED_PASS")
        self.assertEqual(c(self._cells(1.5, 1.7, 1.2), 3.0, 60.0, self.Q)["label"], "SPEED_MIDDLE")
        self.assertEqual(c(self._cells(1.1, 1.2, 1.0), 3.0, 60.0, self.Q)["label"], "SPEED_LOW")
        self.assertIn("MIXED_DECODE_WIN_PREFILL_LOSS", c(self._cells(1.4, 1.8, 0.6), 3.0, 60.0, self.Q)["flags"])
        self.assertIn("AUDIT_DECODE_EXCEEDS_BANDWIDTH_IMPLIED_RATIO", c(self._cells(2.5, 1.5, 1.5, bytes_ratio=0.8), 3.0, 60.0, self.Q)["flags"])
        self.assertNotIn("AUDIT_DECODE_EXCEEDS_BANDWIDTH_IMPLIED_RATIO", c(self._cells(1.0, 0.7, 1.0, bytes_ratio=0.8), 3.0, 60.0, self.Q)["flags"])
        mixed = self._cells(1.6, 1.8, 1.2)
        mixed[0]["total"] = 1.0
        self.assertIn("WIN_NOT_UNIFORM_ACROSS_CELLS", c(mixed, 3.0, 60.0, self.Q)["flags"])
        far = {C.TERNARY: 60.0, "QWEN25_1P5B_Q4KM": 70.0}
        self.assertIn("COMPARATOR_ABOVE_TOLERANCE_TERNARY_DISADVANTAGED", c(self._cells(1.0, 1.0, 1.0), 3.0, 60.0, far)["flags"])

    def test_empty_pool_gives_no_ratios(self):
        q = {C.TERNARY: 70.0, "QWEN25_1P5B_Q4KM": 60.0}
        self.assertEqual(summarize.matched_set(q, 3.0), [])
        self.assertEqual(summarize.no_matched()["label"], "NO_MATCHED_COMPARATOR")

    def test_bytes_ratio(self):
        from analysis.ternary_vs_int4 import bytes_model as bm
        a = {"weight_bytes_streamed": 1000, "kv_bytes_per_token": 10}
        b = {"weight_bytes_streamed": 500, "kv_bytes_per_token": 20}
        self.assertAlmostEqual(bm.bytes_ratio(a, b, 0), 2.0)
        self.assertAlmostEqual(bm.bytes_ratio(a, b, 100), (1000 + 1000) / (500 + 2000))

    def test_regime(self):
        self.assertEqual(C.regime(512), "short")
        self.assertEqual(C.regime(1024), "short")
        self.assertEqual(C.regime(2048), "long")

    def test_geomean(self):
        self.assertAlmostEqual(summarize.geomean([1, 4]), 2.0)
        self.assertTrue(math.isnan(summarize.geomean([])))


if __name__ == "__main__":
    unittest.main()
