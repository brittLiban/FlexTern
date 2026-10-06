"""
Validation tests for the generalized (multi-model) profiler logic, EXP-0002B.

Run from the repository root:
    python -m unittest tests.test_analysis_cross_model -v

No model download is needed. The regression tests read the committed
Microsoft outputs under results/workload_profile/ and pin their totals so a
future refactor cannot silently change the EXP-0002 canonical numbers.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, os.path.join(_ROOT, "analysis"))

from model_specs import SPECS, get_spec  # noqa: E402
from ternary_model_profile import (accumulator_width, classify_tensors, divisibility,  # noqa: E402
                                   expected_projection_dims, expected_tensor_count, expected_ternary_total)

BITNET_CFG = {"num_hidden_layers": 30, "hidden_size": 2560, "intermediate_size": 6912, "num_attention_heads": 20,
              "num_key_value_heads": 5, "vocab_size": 128256}
FALCON_1B_CFG = {"num_hidden_layers": 24, "hidden_size": 2048, "intermediate_size": 9216, "num_attention_heads": 16,
                 "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 32768}
FALCON_3B_CFG = {"num_hidden_layers": 32, "hidden_size": 2048, "intermediate_size": 13312, "num_attention_heads": 16,
                 "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 32768}


def synthetic_names(cfg, norm_names, tied):
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    if not tied:
        names.append("lm_head.weight")
    for layer in range(cfg["num_hidden_layers"]):
        for grp, mods in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")), ("mlp", ("gate_proj", "up_proj", "down_proj"))):
            for m in mods:
                names.append(f"model.layers.{layer}.{grp}.{m}.weight")
                names.append(f"model.layers.{layer}.{grp}.{m}.weight_scale")
        for n in norm_names:
            names.append(f"model.layers.{layer}.{n}.weight")
    return names


class TestSpecs(unittest.TestCase):
    def test_all_specs_well_formed(self):
        for slug, sp in SPECS.items():
            self.assertEqual(sp.slug, slug)
            self.assertIn(sp.scale_convention, ("multiply", "divide"))
            self.assertTrue(sp.norm_names)
            self.assertTrue(sp.revision and len(sp.revision) == 40)
            self.assertGreater(sp.expected_file_size, 0)
            self.assertTrue(sp.default_out.startswith("results/workload_profile"))

    def test_bitnet_default_out_is_canonical_location(self):
        # EXP-0002's committed outputs must stay where they are.
        self.assertEqual(get_spec("bitnet-2b4t").default_out, "results/workload_profile")

    def test_falcon_outputs_do_not_collide(self):
        outs = [sp.default_out for sp in SPECS.values()]
        self.assertEqual(len(outs), len(set(outs)))

    def test_unknown_spec_exits(self):
        with self.assertRaises(SystemExit):
            get_spec("no-such-model")


class TestClassification(unittest.TestCase):
    def _check(self, cfg, slug, expected_total):
        sp = get_spec(slug)
        names = synthetic_names(cfg, sp.norm_names, sp.tied_head)
        classes = classify_tensors(names, sp)
        L = cfg["num_hidden_layers"]
        self.assertEqual(classes.get("unclassified", []), [])
        self.assertEqual(len(classes["packed_ternary"]), 7 * L)
        self.assertEqual(len(classes["weight_scale"]), 7 * L)
        self.assertEqual(len(classes["norm"]), len(sp.norm_names) * L + 1)
        self.assertEqual(len(classes["embedding"]), 1)
        self.assertEqual(len(classes.get("lm_head", [])), 0 if sp.tied_head else 1)
        self.assertEqual(len(names), expected_tensor_count(L, len(sp.norm_names), sp.tied_head))
        self.assertEqual(len(names), expected_total)

    def test_bitnet_542(self):
        self._check(BITNET_CFG, "bitnet-2b4t", 542)

    def test_falcon_1b_387(self):
        self._check(FALCON_1B_CFG, "falcon-e-1b", 387)

    def test_falcon_3b_515(self):
        self._check(FALCON_3B_CFG, "falcon-e-3b", 515)

    def test_untied_head_is_unclassified_for_tied_spec(self):
        # A stray lm_head in a tied model must surface, not be silently absorbed.
        names = synthetic_names(BITNET_CFG, get_spec("bitnet-2b4t").norm_names, tied=False)
        classes = classify_tensors(names, get_spec("bitnet-2b4t"))
        self.assertEqual(classes["unclassified"], ["lm_head.weight"])

    def test_subnorm_tensor_is_unclassified_for_falcon_spec(self):
        names = ["model.layers.0.self_attn.attn_sub_norm.weight"]
        classes = classify_tensors(names, get_spec("falcon-e-1b"))
        self.assertEqual(classes["unclassified"], names)


class TestShapes(unittest.TestCase):
    def test_projection_dims_bitnet(self):
        out, inn = expected_projection_dims(BITNET_CFG)
        self.assertEqual((out["q_proj"], inn["q_proj"]), (2560, 2560))
        self.assertEqual((out["k_proj"], inn["k_proj"]), (640, 2560))
        self.assertEqual((out["down_proj"], inn["down_proj"]), (2560, 6912))

    def test_projection_dims_falcon_use_head_dim(self):
        out, inn = expected_projection_dims(FALCON_1B_CFG)
        self.assertEqual((out["q_proj"], inn["q_proj"]), (2048, 2048))
        self.assertEqual((out["k_proj"], inn["k_proj"]), (256, 2048))   # 2 KV heads x 128
        self.assertEqual((out["gate_proj"], inn["gate_proj"]), (9216, 2048))
        self.assertEqual((out["down_proj"], inn["down_proj"]), (2048, 9216))

    def test_ternary_totals(self):
        self.assertEqual(expected_ternary_total(BITNET_CFG), 2_084_044_800)
        self.assertEqual(expected_ternary_total(FALCON_1B_CFG), 1_585_446_912)
        self.assertEqual(expected_ternary_total(FALCON_3B_CFG), 2_919_235_584)

    def test_divisibility(self):
        self.assertEqual(divisibility(2048), {"div_64": True, "div_128": True, "div_256": True, "div_512": True})
        self.assertEqual(divisibility(6912), {"div_64": True, "div_128": True, "div_256": True, "div_512": False})
        self.assertEqual(divisibility(13312), {"div_64": True, "div_128": True, "div_256": True, "div_512": True})
        self.assertEqual(divisibility(2560), {"div_64": True, "div_128": True, "div_256": True, "div_512": True})
        self.assertEqual(divisibility(9216), {"div_64": True, "div_128": True, "div_256": True, "div_512": True})


class TestAccumulatorsFalconK(unittest.TestCase):
    def test_k2048_format_vs_model_range_differ(self):
        # 128*2048 = 2^18 exactly -> needs 20 bits; 127*2048 fits in 19.
        self.assertEqual(accumulator_width(2048, -128, 127)["signed_bits"], 20)
        self.assertEqual(accumulator_width(2048, -127, 127)["signed_bits"], 19)

    def test_k9216_and_k13312(self):
        self.assertEqual(accumulator_width(9216, -128, 127)["signed_bits"], 22)
        self.assertEqual(accumulator_width(9216, -127, 127)["signed_bits"], 22)
        self.assertEqual(accumulator_width(13312, -128, 127)["signed_bits"], 22)
        self.assertEqual(accumulator_width(13312, -127, 127)["signed_bits"], 22)

    def test_microsoft_k_unchanged(self):
        self.assertEqual(accumulator_width(2560, -128, 127)["signed_bits"], 20)
        self.assertEqual(accumulator_width(6912, -128, 127)["signed_bits"], 21)


class TestMicrosoftRegression(unittest.TestCase):
    """Pin the committed EXP-0002 canonical numbers. If these change, the profiler refactor broke something."""

    def _rows(self, name):
        with open(os.path.join(_ROOT, "results", "workload_profile", name), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_module_totals(self):
        rows = {r["module"]: r for r in self._rows("weight_distribution_by_module.csv")}
        self.assertEqual(int(rows["q_proj"]["weights"]), 196_608_000)
        self.assertEqual(int(rows["down_proj"]["weights"]), 530_841_600)
        self.assertEqual(int(rows["down_proj"]["K"]), 6912)
        self.assertAlmostEqual(float(rows["q_proj"]["pct_zero"]), 46.75, places=1)
        self.assertAlmostEqual(float(rows["v_proj"]["pct_zero"]), 37.64, places=1)

    def test_global_distribution(self):
        with open(os.path.join(_ROOT, "results", "workload_profile", "model_summary.json"), encoding="utf-8") as f:
            ms = json.load(f)
        g = ms["global_ternary_distribution"]
        self.assertEqual(g["total_ternary_weights"], 2_084_044_800)
        self.assertEqual(g["count_neg"], 602_163_685)
        self.assertEqual(g["count_zero"], 879_693_294)
        self.assertEqual(g["count_pos"], 602_187_821)
        self.assertEqual(ms["dead_channels"]["mlp_intermediate_channels_dead_total"], 9_870)
        self.assertEqual(ms["file_size_bytes"], 1_178_623_988)

    def test_dot_lengths(self):
        rows = self._rows("dot_length_summary.csv")
        self.assertEqual(sorted(int(r["K"]) for r in rows), [2560, 6912])
        self.assertEqual(int(next(r for r in rows if r["K"] == "2560")["weights"]), 1_553_203_200)


class TestFalconRegression(unittest.TestCase):
    """Pin the EXP-0002B Falcon-E numbers once generated (skipped if the outputs are absent)."""

    def _summary(self, slug):
        p = os.path.join(_ROOT, SPECS[slug].default_out, "model_summary.json")
        if not os.path.exists(p):
            self.skipTest(f"{p} not generated")
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def test_falcon_1b(self):
        ms = self._summary("falcon-e-1b")
        g = ms["global_ternary_distribution"]
        self.assertEqual(g["total_ternary_weights"], 1_585_446_912)
        self.assertEqual(g["count_neg"] + g["count_zero"] + g["count_pos"], g["total_ternary_weights"])
        self.assertEqual(g["reserved_code3_count"], 0)
        self.assertEqual(ms["dead_channels"]["mlp_intermediate_channels_dead_total"], 0)
        self.assertEqual(sorted(d["K"] for d in ms["dot_lengths"]), [2048, 9216])
        self.assertFalse(ms["tie_word_embeddings"])
        self.assertEqual(ms["head"]["tensor"], "lm_head.weight")
        self.assertEqual(ms["file_size_bytes"], 665_041_488)

    def test_falcon_3b(self):
        ms = self._summary("falcon-e-3b")
        g = ms["global_ternary_distribution"]
        self.assertEqual(g["total_ternary_weights"], 2_919_235_584)
        self.assertEqual(g["reserved_code3_count"], 0)
        self.assertEqual(sorted(d["K"] for d in ms["dot_lengths"]), [2048, 13312])
        self.assertEqual(ms["file_size_bytes"], 998_568_704)


if __name__ == "__main__":
    unittest.main(verbosity=2)
