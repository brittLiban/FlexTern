"""Reserved-split guard (EXP-0013 Phase 1.1 closeout). Static check: the string literal "test" (the name of the reserved layout split) may
appear in analysis/predictive_state/*.py only in files listed in split_manifest.json. This is about the simulation data split, not about
test-suite files named test_*.py."""
import ast
import json
import os
import unittest
from pathlib import Path

from analysis.predictive_state import env as E

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:       # pragma: no cover
    HAVE_TORCH = False

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

    def test_split_policy_default_and_opt_in(self):
        from analysis.predictive_state import split_policy as SP
        old = os.environ.pop(SP.OPT_IN, None)
        try:
            self.assertEqual(SP.timing_split(), "train")
            self.assertEqual(SP.resolve(None), "train")
            self.assertEqual(SP.resolve("dev"), "dev")
            with self.assertRaises(PermissionError):
                SP.resolve("test")
            os.environ[SP.OPT_IN] = "1"
            self.assertEqual(SP.timing_split(), "test")
            self.assertEqual(SP.resolve("test"), "test")
            os.environ[SP.OPT_IN] = "0"
            self.assertEqual(SP.timing_split(), "train")
        finally:
            os.environ.pop(SP.OPT_IN, None)
            if old is not None:
                os.environ[SP.OPT_IN] = old

    def test_opt_in_not_set_in_test_environment(self):
        self.assertNotIn("FLEXTERN_ALLOW_TEST_SPLIT", os.environ)

    def test_timing_entrypoints_have_no_reserved_literal(self):
        for name in ("latency.py", "phase11_timing.py", "analyze_state.py"):
            self.assertNotIn("test", string_literals(PKG / name), name)


class _Stop(Exception):
    pass


@unittest.skipUnless(HAVE_TORCH, "needs torch to import the timing entrypoints (the static checks above do not)")
class TestActualTimingDefaults(unittest.TestCase):
    """Executes the entrypoints with the simulator replaced by a recorder, and checks the split they ACTUALLY ask for."""

    def _captured_split(self, call):
        seen = []

        def fake_env(n, split, cond, seed, **kw):
            seen.append(split)
            raise _Stop()
        orig, old = E.BatchEnv, os.environ.pop("FLEXTERN_ALLOW_TEST_SPLIT", None)
        E.BatchEnv = fake_env
        try:
            with self.assertRaises(_Stop):
                call()
        finally:
            E.BatchEnv = orig
            if old is not None:
                os.environ["FLEXTERN_ALLOW_TEST_SPLIT"] = old
        return seen

    def test_latency_run_defaults_to_train(self):
        from analysis.predictive_state import latency
        self.assertEqual(self._captured_split(lambda: latency.run(None, "full", 1, episodes=1)), ["train"])

    def test_frozen_exp0012_latency_copy_defaults_to_train(self):
        from analysis.predictive_state import _latency_exp0012 as legacy
        self.assertEqual(self._captured_split(lambda: legacy.run(None, "full", 1, episodes=1)), ["train"])

    def test_phase11_timing_record_defaults_to_train(self):
        from analysis.predictive_state import phase11_timing
        self.assertEqual(self._captured_split(lambda: phase11_timing.record(None, "full", episodes=1)), ["train"])

    def test_historical_trace_needs_explicit_opt_in(self):
        from analysis.predictive_state import latency
        os.environ["FLEXTERN_ALLOW_TEST_SPLIT"] = "1"
        try:
            seen = []

            def fake_env(n, split, cond, seed, **kw):
                seen.append(split)
                raise _Stop()
            orig, E.BatchEnv = E.BatchEnv, fake_env
            try:
                with self.assertRaises(_Stop):
                    latency.run(None, "full", 1, episodes=1)
            finally:
                E.BatchEnv = orig
            self.assertEqual(seen, ["test"])
        finally:
            os.environ.pop("FLEXTERN_ALLOW_TEST_SPLIT", None)


if __name__ == "__main__":
    unittest.main()
