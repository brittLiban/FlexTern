"""EXP-0013 Phase 1 tests for the OccPick2D-LA benchmark and its privileged controls. numpy-only (runs on the Windows host)."""
import unittest

import numpy as np

from analysis.predictive_state import controllers_la as C
from analysis.predictive_state import env as E
from analysis.predictive_state import env_la as L

NOM = L.LA_CONDS["LA_NOM"]


class TestLayout(unittest.TestCase):
    def test_deterministic(self):
        a, b = L.la_layout(7), L.la_layout(7)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertFalse(np.array_equal(L.la_layout(7)[0], L.la_layout(8)[0]))

    def test_geometry_all_splits(self):
        for lo, hi in E.LAYOUT_RANGES.values():
            for i in range(lo, min(hi, lo + 60)):
                walls, goal, info = L.la_layout(i)
                g = np.stack(np.meshgrid(np.linspace(goal[0] - .09, goal[0] + .09, 7), np.linspace(goal[1] - .09, goal[1] + .09, 7)), -1)
                self.assertFalse(L.inside(g, walls).any(), f"goal zone overlaps a wall (layout {i})")
                self.assertGreaterEqual(abs(info["open"] - info["decoy"]), 0.27)
                self.assertTrue(L.reachable(i, [0.1, 0.5], goal), f"goal unreachable (layout {i})")
                self.assertTrue(((walls[:, 2] == 0) | (walls[:, 2] >= 0.079)).all())

    def test_decoy_is_a_closed_pocket(self):
        for i in range(30):
            walls, goal, info = L.la_layout(i)
            p = np.array([info["xw"] + 0.10, info["decoy"]], np.float32)           # inside the pocket
            self.assertFalse(L.inside(p, walls))
            for u in ([1, 0], [0, 1], [0, -1]):
                self.assertLess(C.ray_len(walls, p, np.array(u, np.float32), 0.5), 0.3, f"pocket open in direction {u} (layout {i})")


class TestDynamics(unittest.TestCase):
    def test_blocked_axis_slides(self):
        walls, goal, info = L.la_layout(3)
        face = info["xw"] - L.WALL_T / 2
        ee = np.array([face - 0.005, 0.1], np.float32)                           # just left of the lower divider piece
        new = L.la_move(ee, np.array([0.04, 0.03], np.float32), walls)
        self.assertAlmostEqual(float(new[0]), float(ee[0]), places=6)
        self.assertAlmostEqual(float(new[1]), 0.13, places=5)

    def test_never_inside_wall_random_actions(self):
        env = L.LAEnv(20, "dev", NOM, 3)
        rng = np.random.default_rng(0)
        for _ in range(250):
            a = np.concatenate([rng.uniform(-1, 1, (20, 2)), np.zeros((20, 1))], 1).astype(np.float32)
            env.step(a)
            self.assertFalse(L.inside(env.ee, env.walls).any())

    def test_planner_dynamics_equal_env_dynamics(self):
        """H oracle-predicted steps (la_move) == H executed env steps. Same check Phase 2 reuses for analytical rollouts."""
        env = L.LAEnv(16, "dev", NOM, 4)
        rng = np.random.default_rng(1)
        pred = env.ee.copy()
        for _ in range(60):
            a = np.concatenate([rng.uniform(-1, 1, (16, 2)) * 1.4, np.zeros((16, 1))], 1).astype(np.float32)   # includes over-length actions
            pred = L.la_move(pred, L.cap_step(a[:, :2]), env.walls)
            env.step(a)
            np.testing.assert_allclose(pred, env.ee, atol=1e-6)

    def test_displacement_events_rejected(self):
        with self.assertRaises(ValueError):
            L.LAEnv(2, "dev", E.Cond("x", 0.0, (5, 10), 0.5, 2), 0)

    def test_spawns_left_of_divider_and_outside_walls(self):
        env = L.LAEnv(40, "dev", NOM, 5)
        for i in range(40):
            xw = env.la_info[i]["xw"]
            pts = [env.ee[i]] + [env.pos[i, j] for j in range(env.k[i] + 1)]
            for p in pts:
                self.assertLess(p[0], xw - 0.09)
                self.assertFalse(L.inside(p, env.walls[i], 0.03))
            self.assertGreater(env.goal[i][0], 0.8)

    def test_timeout_at_budget(self):
        env = L.LAEnv(3, "dev", NOM, 6, t_limit=5)
        for _ in range(6):
            env.step(np.zeros((3, 3), np.float32))
        self.assertTrue(env.done.all())
        self.assertTrue((env.fail_kind == "timeout").all())


class TestObservations(unittest.TestCase):
    def test_walls_are_visible_and_separable_at_32px(self):
        # separable colours: wall colour is >= 60 (10 sigma of the pixel noise) from every other drawn class
        others = [E.BG, E.GOAL_COL, E.OCC_COL, np.array([255.0, 255, 255], np.float32)] + list(E.COLORS)
        self.assertGreater(min(np.linalg.norm(L.WALL_COL - o) for o in others), 60.0)
        env = L.LAEnv(12, "dev", E.Cond("v", 0.0, (5, 10), 0.0, 2), 7)
        obs = env.observe()
        img = obs["img"].astype(np.float32)
        total_px = 0
        for i in range(12):
            wm = np.zeros((E.RES, E.RES), bool)
            for k in range(L.NWALL):
                w = env.walls[i, k]
                wm |= (np.abs(E._X - w[0]) < w[2] / 2) & (np.abs(E._Y - w[1]) < w[3] / 2) & (w[2] > 0)
            # the end-effector disc may overdraw a wall pixel; exclude it
            em = (E._X - env.ee[i, 0]) ** 2 + (E._Y - env.ee[i, 1]) ** 2 < E.EE_R ** 2
            m = wm & ~em
            total_px += int(m.sum())
            self.assertGreater(int(m.sum()), 40)
            self.assertLess(float(np.abs(img[i][m].mean(0) - L.WALL_COL).max()), 4.0)
            cols = wm.any(0)
            runs = [len(r) for r in np.split(np.arange(E.RES), np.nonzero(np.diff(cols.astype(int)))[0] + 1)]
            self.assertGreaterEqual(int(cols.sum()), 2)
        self.assertGreater(total_px, 12 * 40)

    def test_observation_has_no_privileged_fields(self):
        env = L.LAEnv(2, "dev", NOM, 8)
        obs = env.observe()
        self.assertEqual(set(obs), {"img", "valid", "prop", "task"})
        self.assertEqual(obs["prop"].shape, (2, 4))
        pv = env.privileged()
        self.assertIn("walls", pv)
        self.assertIn("target", pv)


class TestControlsValidateBenchmark(unittest.TestCase):
    N = 40

    def run_ctrl(self, ctrl):
        env = L.LAEnv(self.N, "dev", NOM, 5001, t_limit=80)
        r = C.run_episodes(env, ctrl, T=82)
        return float(((r["success"] > 0) & (r["finish"] <= L.LA_T_PICK)).mean()), r

    def test_oracle_solves_and_greedy_fails(self):
        o, _ = self.run_ctrl(C.Oracle("descent"))
        g, _ = self.run_ctrl(C.Greedy())
        self.assertGreaterEqual(o, 0.97)
        self.assertLess(g, 0.5)

    def test_local_reactive_controllers_do_not_solve_within_budget(self):
        o, _ = self.run_ctrl(C.Oracle("descent"))
        for retreat in (False, True):
            s, _ = self.run_ctrl(C.Reactive(retreat))
            self.assertLess(s, o - 0.4, f"reactive(retreat={retreat}) too close to the oracle: benchmark would not need lookahead")

    def test_euclid_shooting_does_not_beat_greedy_but_geodesic_value_does(self):
        g, _ = self.run_ctrl(C.Greedy())
        e, _ = self.run_ctrl(C.Oracle("shoot", H=5, N=64, cost="euclid", seed=5))
        v, _ = self.run_ctrl(C.Oracle("shoot", H=1, N=64, cost="geodesic", seed=1))
        self.assertLess(e, g + 0.15)
        self.assertGreaterEqual(v, 0.97)

    def test_oracle_calls_true_dynamics_and_counts_effort(self):
        calls = {"n": 0}
        orig = L.la_move
        L.la_move = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
        try:
            env = L.LAEnv(4, "dev", NOM, 9)
            ctrl = C.Oracle("shoot", H=3, N=16, cost="geodesic", seed=0)
            ctrl.reset(env)
            env.reset()
            ctrl.act(env)
        finally:
            L.la_move = orig
        self.assertEqual(calls["n"], 3)                          # H planner steps through la_move
        self.assertEqual(ctrl.effort["sim_steps"], 4 * 16 * 3)
        self.assertEqual(ctrl.effort["decisions"], 4)

    def test_controllers_are_deterministic(self):
        a, ra = self.run_ctrl(C.Oracle("descent"))
        b, rb = self.run_ctrl(C.Oracle("descent"))
        np.testing.assert_array_equal(ra["finish"], rb["finish"])
        np.testing.assert_array_equal(ra["success"], rb["success"])


if __name__ == "__main__":
    unittest.main()
