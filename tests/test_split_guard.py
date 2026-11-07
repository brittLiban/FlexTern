"""Reserved-split guard (EXP-0013 Phase 1.1 closeout). Static check: the string literal "test" (the name of the reserved layout split) may
appear in analysis/predictive_state/*.py only in files listed in split_manifest.json. This is about the simulation data split, not about
test-suite files named test_*.py."""
import ast
import json
import unittest
from pathlib import Path

from analysis.predictive_state import env as E

PKG = Path(__file__).resolve().parent.parent / "analysis" / "predictive_state"
MANIFEST = json.loads((PKG / "split_manifest.json").read_text())


def string_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc_nodes = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                 and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_nodes]


class TestSplitGuard(unittest.TestCase):
    def test_manifest_matches_env_ranges(self):
        self.assertEqual({k: list(v) for k, v in E.LAYOUT_RANGES.items()}, MANIFEST["layout_ranges"])

    def test_only_listed_files_reference_the_reserved_split(self):
        offenders = [p.name for p in sorted(PKG.glob("*.py"))
                     if MANIFEST["reserved_split"] in string_literals(p) and p.name not in MANIFEST["allowed_files"]]
        self.assertEqual(offenders, [], f"{offenders} reference the reserved 'test' split; use train/dev or justify in split_manifest.json")

    def test_listed_files_exist_and_really_reference_it(self):
        for name in MANIFEST["allowed_files"]:
            p = PKG / name
            self.assertTrue(p.exists(), name)
            self.assertIn(MANIFEST["reserved_split"], string_literals(p), f"{name} is listed but does not use the split: remove it from the manifest")

    def test_closeout_and_phase11_benchmark_scripts_are_test_free(self):
        for name in ("closeout_paired.py", "closeout_repro_v1.py", "run_phase11_benchmark.py", "run_la_benchmark.py", "controllers_la.py",
                     "controllers_obs.py", "planning_diag.py", "env_la.py"):
            self.assertNotIn("test", string_literals(PKG / name), name)

    def test_timing_trace_defaults_to_train(self):
        import os
        src = (PKG / "phase11_timing.py").read_text()
        self.assertIn("FLEXTERN_ALLOW_TEST_SPLIT", src)
        self.assertNotIn("FLEXTERN_ALLOW_TEST_SPLIT", os.environ)       # opt-in must not be set in the test environment


if __name__ == "__main__":
    unittest.main()
