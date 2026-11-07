"""EXP-0012 tests: benchmark semantics, expert solvability, quantizers, model shapes, cost model.
env/expert tests are numpy-only; the rest need torch (WSL venv) and are skipped on the Windows host (KI-005)."""
import unittest

import numpy as np

from analysis.predictive_state import env as E
from analysis.predictive_state import expert as X

try:
    import torch
    from analysis.predictive_state import models as M
    from analysis.predictive_state import quant as Q
    from analysis.predictive_state import cost_model as C
    HAVE_TORCH = True
except Exception:       # pragma: no cover
    HAVE_TORCH = False


class TestEnv(unittest.TestCase):
    def test_layout_ranges_disjoint(self):
        r = list(E.LAYOUT_RANGES.values())
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                self.assertTrue(r[i][1] <= r[j][0] or r[j][1] <= r[i][0])

    def test_layout_deterministic(self):
        a, b = E.layout_from_id(1234), E.layout_from_id(1234)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_target_visible_at_reset(self):
        e = E.BatchEnv(64, "test", E.TEST_CONDS["HARD"], seed=0)
        self.assertFalse(E.under_occluder(e.pos[:, 0], e.occ, 0.0).any())
        self.assertTrue(e.observe()["valid"].all())

    def test_blackout_gives_gray_and_invalid(self):
        e = E.BatchEnv(8, "test", E.TEST_CONDS["OCC"], seed=0)
        e.reset()
        e.black_left[:] = 5
        o = e.observe()
        self.assertFalse(o["valid"].any())
        self.assertTrue((o["img"] == 128).all())

    def test_occluder_hides_object(self):
        e = E.BatchEnv(1, "test", E.TEST_CONDS["NOM"], seed=0)
        e.reset()
        e.occ[0, 0] = [0.5, 0.5, 0.3, 0.3]
        e.pos[0, 1] = [0.5, 0.5]
        e.ee[0] = [0.1, 0.9]
        e.active[0, 1] = True
        img = e.render()[0]
        col = E.COLORS[e.col[0, 1]]
        near = np.abs(img - col).sum(-1) < 40
        mid = np.zeros((E.RES, E.RES), bool)
        mid[12:20, 12:20] = True
        self.assertFalse((near & mid).any())

    def test_pick_wrong_object_fails(self):
        e = E.BatchEnv(1, "test", E.Cond("t", 0, (5, 6), 0.0, 2), seed=0)
        e.reset()
        e.ee[0] = e.pos[0, 1]
        e.step(np.array([[0, 0, 1.0]], np.float32))
        self.assertTrue(e.done[0] and e.failed[0])
        self.assertEqual(e.fail_kind[0], "wrong_pick")

    def test_displacement_drops_held(self):
        c = E.Cond("t", 0, (5, 6), 1.0, 1)
        e = E.BatchEnv(1, "test", c, seed=0)
        e.reset()
        e.disp_at[0] = 3
        e.ee[0] = e.pos[0, 0]
        e.step(np.array([[0, 0, 1.0]], np.float32))
        self.assertTrue(e.held[0])
        before = e.pos[0, 0].copy()
        for _ in range(3):
            e.step(np.array([[0, 0, 1.0]], np.float32))
        self.assertFalse(e.held[0])
        self.assertGreater(np.linalg.norm(e.pos[0, 0] - before), 0.2)

    def test_expert_solves_every_condition(self):
        for name, c in E.TEST_CONDS.items():
            e = E.BatchEnv(60, "dev", c, seed=5)
            s = X.rollout_expert(e, 80 * c.n_picks + 5)
            self.assertGreaterEqual(float(s["success"].mean()), 0.95, name)

    def test_passive_policy_fails(self):
        e = E.BatchEnv(20, "dev", E.TEST_CONDS["NOM"], seed=0)
        e.reset()
        for _ in range(85):
            e.step(np.zeros((20, 3), np.float32))
        self.assertEqual(float(X.success_summary(e)["success"].mean()), 0.0)

    def test_seeded_reproducible(self):
        a = E.BatchEnv(4, "dev", E.TEST_CONDS["DIST"], seed=9).reset()["img"]
        b = E.BatchEnv(4, "dev", E.TEST_CONDS["DIST"], seed=9).reset()["img"]
        np.testing.assert_array_equal(a, b)


@unittest.skipUnless(HAVE_TORCH, "torch not importable on this host")
class TestQuantModels(unittest.TestCase):
    def test_ternary_levels(self):
        w = torch.randn(32, 64)
        q = Q.fq_weight(w, "tern")
        self.assertLessEqual(len(torch.unique(q.detach().round(decimals=6))), 3)

    def test_int4_levels(self):
        q = Q.fq_weight(torch.randn(16, 64), "int4")
        for r in q.detach():
            self.assertLessEqual(len(torch.unique(r.round(decimals=6))), 16)

    def test_ste_gradient_flows(self):
        w = torch.randn(8, 8, requires_grad=True)
        Q.fq_weight(w, "tern").sum().backward()
        self.assertTrue(torch.all(w.grad == 1))

    def test_state_quant_range(self):
        h = Q.fq_state(torch.linspace(-2, 2, 101), 8)
        self.assertLessEqual(float(h.abs().max()), 1.0)

    def test_all_arms_forward_and_loss(self):
        B, T = 3, 7
        b = {"img": torch.randint(0, 255, (B, T, 32, 32, 3), dtype=torch.uint8), "prop": torch.rand(B, T, 4),
             "task": torch.eye(6)[torch.randint(0, 6, (B, T))], "a_exec": torch.rand(B, T, 3), "a_label": torch.rand(B, T, 3).round(),
             "mask": torch.ones(B, T, dtype=torch.bool), "valid_env": torch.ones(B, T, dtype=torch.bool),
             "valid_in": torch.ones(B, T, dtype=torch.bool)}
        for arm in M.ARMS:
            net = M.build(arm)
            out = net.forward_seq(b)
            self.assertTrue(torch.isfinite(out["loss"]), arm)
            out["loss"].backward()

    def test_ema_not_trained(self):
        net = M.build("G")
        self.assertTrue(all(not p.requires_grad for p in net.ema.parameters()))

    def test_cost_model_orders(self):
        g, t, i4 = (C.step_cost(a, 1.0)["bytes"] for a in ("G", "G-Ta", "G-I4a"))
        self.assertLess(t, i4)
        self.assertLess(i4, g)
        self.assertEqual(C.step_cost("G", 1.0)["macs"], C.step_cost("G-Ta", 1.0)["macs"])    # ternary does not change MAC count
        half = C.step_cost("G-Ta", 0.5)
        full = C.step_cost("G-Ta", 1.0)
        self.assertLess(half["macs"], full["macs"])


if __name__ == "__main__":
    unittest.main()
