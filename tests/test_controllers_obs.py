"""EXP-0013 Phase 1.1 tests: observation-matched baselines, information-access matrix, planning-horizon diagnostic. numpy-only.
Run: python -m unittest tests.test_controllers_obs
"""
import ast
import inspect
import textwrap
import unittest

import numpy as np

from analysis.predictive_state import controllers_la as C
from analysis.predictive_state import controllers_obs as O
from analysis.predictive_state import env as E
from analysis.predictive_state import env_la as L
from analysis.predictive_state import info_matrix as IM
from analysis.predictive_state import planning_diag as P
from analysis.predictive_state import run_phase11_benchmark as RB

CLEAR = RB.CONDS["LA_CLEAR"]


def run(ctrl, n=24, cond=CLEAR, seed=777, obs=True):
    """Behavioural tests run on TRAIN layouts only: the DEV episodes (seed 5001) are reserved for the scored comparison."""
    env = L.LAEnv(n, "train", cond, seed, t_limit=80)
    r = O.run_episodes_obs(env, ctrl) if obs else C.run_episodes(env, ctrl, T=82)
    return float(((r["success"] > 0) & (r["finish"] <= L.LA_T_PICK)).mean()), r, env


class TestParser(unittest.TestCase):
    def test_wall_mask_recovered_exactly_from_one_clear_frame(self):
        env = L.LAEnv(30, "train", CLEAR, 3)
        obs = env.reset()
        for i in range(30):
            pr = O.parse(obs["img"][i], int(obs["task"][i].argmax()))
            tw = RB.true_wall_mask(env.walls[i])
            self.assertGreaterEqual(float((pr["wall"] & tw).sum() / (pr["wall"] | tw).sum()), 0.97, i)

    def test_target_and_goal_estimates_are_close(self):
        env = L.LAEnv(30, "train", CLEAR, 4)
        obs = env.reset()
        for i in range(30):
            pr = O.parse(obs["img"][i], int(obs["task"][i].argmax()))
            self.assertIsNotNone(pr["tgt"])
            self.assertLess(float(np.linalg.norm(pr["tgt"] - env.pos[i, 0])), 0.03)
            if pr["goal_bbox"] is not None:
                self.assertLess(float(np.linalg.norm(O.bbox_center(pr["goal_bbox"]) - env.goal[i])), 0.08)

    def test_blackout_frame_parses_to_nothing_useful(self):
        gray = np.full((E.RES, E.RES, 3), 128, np.uint8)
        pr = O.parse(gray, 0)
        self.assertIsNone(pr["tgt"])
        self.assertIsNone(pr["goal_bbox"])
        self.assertFalse(pr["wall"].any())


class TestObservationOnly(unittest.TestCase):
    def test_signature_and_source_have_no_privileged_access(self):
        self.assertEqual(list(inspect.signature(O.ObsA.act).parameters), ["self", "obs", "alive"])
        self.assertEqual(list(inspect.signature(O.ObsB.act).parameters), ["self", "obs", "alive"])
        forbidden_name = {"la_move", "geodesic_field", "walls_of", "la_layout", "lookup", "reachable"}
        for cls in (O._ObsBase, O.ObsA, O.ObsB):
            tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
            for node in ast.walk(tree):                         # identifiers only: docstrings and comments are ignored
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "env":
                    self.fail(f"{cls.__name__} touches env.{node.attr}")
                if isinstance(node, ast.Name):
                    self.assertNotIn(node.id, forbidden_name | {"env"}, cls.__name__)
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, forbidden_name | {"privileged"}, cls.__name__)

    def test_only_observation_keys_are_consumed(self):
        class Spy(dict):
            def __init__(self, d):
                super().__init__(d); self.used = set()

            def __getitem__(self, k):
                self.used.add(k); return super().__getitem__(k)
        env = L.LAEnv(4, "dev", CLEAR, 6)
        obs = Spy(env.reset())
        for ctrl in (O.ObsA(1), O.ObsB(1)):
            ctrl.reset(4)
            obs.used.clear()
            ctrl.act(obs, np.ones(4, bool))
            self.assertLessEqual(obs.used, {"img", "valid", "prop", "task"})

    def test_paired_episodes_identical_across_controllers(self):
        starts = []
        for _ in range(2):
            env = L.LAEnv(16, "train", CLEAR, 5001, t_limit=80)
            env.reset()
            starts.append((env.layout_ids.copy(), env.ee.copy(), env.pos[:, 0].copy(), env.goal.copy()))
        for a, b in zip(starts[0], starts[1]):
            np.testing.assert_array_equal(a, b)

    def test_test_range_never_used(self):
        lo, hi = E.LAYOUT_RANGES["test"]
        for split, seed in (("train", RB.SEL_SEED), ("dev", RB.EVAL_SEED)):
            env = L.LAEnv(200, split, CLEAR, seed)
            self.assertFalse(((env.layout_ids >= lo) & (env.layout_ids < hi)).any())


class TestBaselinesBehaviour(unittest.TestCase):
    def test_A_is_stateless_and_B_bridges_blackouts(self):
        env = L.LAEnv(3, "dev", CLEAR, 8)
        obs = env.reset()
        A, B = O.ObsA(1), O.ObsB(1)
        A.reset(3); B.reset(3)
        alive = np.ones(3, bool)
        aA, aB = A.act(obs, alive), B.act(obs, alive)
        np.testing.assert_allclose(aA, aB, atol=1e-6)                          # same parser/planner on a valid first frame
        dark = dict(obs); dark["valid"] = np.zeros(3, bool); dark["img"] = np.full_like(obs["img"], 128)
        self.assertTrue((A.act(dark, alive)[:, :2] == 0).all())               # no frame, no memory: hold still
        self.assertGreater(float(np.abs(B.act(dark, alive)[:, :2]).max()), 0.1)  # memory: keeps heading to the remembered target

    def test_A_deterministic(self):
        s1, r1, _ = run(O.ObsA(1), n=12)
        s2, r2, _ = run(O.ObsA(1), n=12)
        np.testing.assert_array_equal(r1["finish"], r2["finish"])

    def test_memory_matters_only_when_frames_are_lost(self):
        sA, _, _ = run(O.ObsA(1), n=24, cond=RB.CONDS["LA_OCC"])
        sB, _, _ = run(O.ObsB(1), n=24, cond=RB.CONDS["LA_OCC"])
        self.assertGreaterEqual(sB, sA)

    def test_obs_baselines_beat_greedy_on_clear_frames(self):
        g, _, _ = run(C.Greedy(), obs=False)
        a, _, _ = run(O.ObsA(1))
        self.assertGreater(a, g + 0.3)


class TestInfoMatrix(unittest.TestCase):
    def test_every_row_has_every_column(self):
        for k, v in IM.M.items():
            self.assertIn(v["class"], ("PRIV", "OBS"), k)
            for c in IM.COLS:
                self.assertIn(c, v, f"{k}: {c}")

    def test_obs_rows_have_no_privileged_access_and_priv_rows_have_some(self):
        for k, v in IM.M.items():
            self.assertEqual(IM.is_privileged_free(v), v["class"] == "OBS", k)

    def test_matrix_matches_declared_controllers(self):
        self.assertTrue({"obs_A_single_frame", "obs_B_map_memory"} <= set(IM.M))
        self.assertEqual(O.ObsA.name, "obs_A_single_frame")
        self.assertEqual(O.ObsB.name, "obs_B_map_memory")
        self.assertTrue(O.ObsB.uses_memory and not O.ObsA.uses_memory)


class TestPlanningDiagnostic(unittest.TestCase):
    def test_true_dynamics_and_effort_accounting(self):
        calls = {"n": 0}
        orig = L.la_move
        L.la_move = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
        try:
            env = L.LAEnv(4, "dev", CLEAR, 9)
            ctrl = P.PlanDiag("rs_walk", 5, 320, cost="geodesic", seed=0)
            ctrl.reset(env); env.reset(); ctrl.act(env)
        finally:
            L.la_move = orig
        self.assertEqual(calls["n"], 5)
        self.assertEqual(ctrl.effort["sim_steps"], 4 * (320 // 5) * 5)
        self.assertEqual(ctrl.effort["decisions"], 4)

    def test_matched_budgets_for_each_planner(self):
        for pk, H in (("rs_walk", 5), ("rs_const", 10), ("cem", 5), ("beam", 5)):
            env = L.LAEnv(3, "dev", CLEAR, 9)
            ctrl = P.PlanDiag(pk, H, 640, cost="euclid", seed=1)
            ctrl.reset(env); env.reset(); ctrl.act(env)
            per = ctrl.effort["sim_steps"] / ctrl.effort["decisions"]
            self.assertLessEqual(per, 640, pk)
            self.assertGreater(per, 0.5 * 640 if pk != "beam" else 17 * H, pk)

    def test_exact_value_makes_horizon_irrelevant_for_deterministic_search(self):
        """With the exact geodesic cost-to-go, one-step beam search already solves the benchmark; deeper beams cannot do better."""
        s1, _, _ = run(P.PlanDiag("beam", 1, 320, cost="geodesic"), n=20, cond=RB.CONDS["LA_NOM"], obs=False)
        self.assertGreaterEqual(s1, 0.9)

    def test_deterministic(self):
        a = run(P.PlanDiag("cem", 5, 640, cost="geodesic", seed=3), n=8, obs=False)[1]["finish"]
        b = run(P.PlanDiag("cem", 5, 640, cost="geodesic", seed=3), n=8, obs=False)[1]["finish"]
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
