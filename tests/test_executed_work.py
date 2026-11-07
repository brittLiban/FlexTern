"""EXP-0013 Phase 1.1 tests: executed-work accounting. Every runner counter is checked against an independent spy on the module
that actually executes (encoder, GRU cell, policy MLP, trigger head, pixel pooling), not against logical masks.
Needs torch (WSL venv); skipped on the Windows host (KI-005). Run: python -m unittest tests.test_executed_work
"""
import unittest

try:
    import torch
    from analysis.predictive_state import models as M
    from analysis.predictive_state import infer as I
    from tests.test_inference_skip import make_net, obs, ones, N
    HAVE_TORCH = True
except Exception:       # pragma: no cover
    HAVE_TORCH = False


class RowSpy:
    """Counts rows entering a module's forward (first tensor argument)."""
    def __init__(self, mod):
        self.rows, self.calls = 0, 0
        self.orig = mod.forward
        mod.forward = self.fwd

    def fwd(self, x, *a, **k):
        self.rows += int(x.shape[0]); self.calls += 1
        return self.orig(x, *a, **k)


def alive_mixed(t):
    return torch.tensor([1, 1, 0, 1, 0, 0], dtype=torch.bool) if t >= 2 else ones()


@unittest.skipUnless(HAVE_TORCH, "needs torch (WSL)")
class TestExecutedWorkAccounting(unittest.TestCase):
    def spied(self, trigger, steps, alive_fn=None, valid_fn=None, select=True, n=N, net=None):
        net = net or make_net()
        sp = {"enc": RowSpy(net.enc), "cell": RowSpy(net.cell), "pol": RowSpy(net.pol), "trig": RowSpy(net.trig)}
        pool = {"rows": 0}
        orig_pool = M.pool_feat
        M.pool_feat = lambda im: (pool.__setitem__("rows", pool["rows"] + int(im.shape[0])), orig_pool(im))[1]
        try:
            r = I.RecurrentRunner(net, trigger, n, "cpu", select_alive=select)
            outs = []
            for t in range(steps):
                img, prop, task = obs(n, seed=t)
                v = valid_fn(t) if valid_fn else ones(n)
                a = alive_fn(t) if alive_fn else ones(n)
                act, m = r.step(img, v, prop, task, a)
                outs.append((act, m, a.clone()))
        finally:
            M.pool_feat = orig_pool
        return r, sp, pool, outs

    def test_counters_equal_spies_mixed_active_finished(self):
        valid = lambda t: torch.tensor([1, 0, 1, 1, 1, 1], dtype=torch.bool) if t % 2 else ones()
        for trig in ("full", "fixed:0.5", "pix:0.02", "learned:0.0"):
            r, sp, pool, _ = self.spied(trig, 6, alive_mixed, valid)
            c = r.counters
            self.assertEqual(c.enc_samples, sp["enc"].rows, trig)
            self.assertEqual(c.enc_calls, sp["enc"].calls, trig)
            self.assertEqual(c.rec_updates, sp["cell"].rows, trig)
            self.assertEqual(c.policy_rows, sp["pol"].rows, trig)
            self.assertEqual(c.trigger_rows, pool["rows"], trig)
            self.assertEqual(c.rec_updates, c.active_steps, trig)               # selective: executed == useful
            self.assertEqual(c.finished_rows_processed, 0, trig)
            self.assertEqual(c.active_steps, 2 * N + 4 * 3, trig)

    def test_full_batch_mode_counts_finished_rows_as_executed(self):
        r, sp, _, _ = self.spied("full", 6, alive_mixed, select=False)
        c = r.counters
        self.assertEqual(c.rec_updates, sp["cell"].rows)
        self.assertEqual(c.rec_updates, 6 * N)
        self.assertEqual(c.active_steps, 2 * N + 4 * 3)
        self.assertEqual(c.finished_rows_processed, 4 * 3)
        self.assertEqual(sp["enc"].rows, c.enc_samples)                          # the encoder is gated either way

    def test_all_finished_does_nothing(self):
        r, sp, pool, outs = self.spied("fixed:0.5", 3, alive_fn=lambda t: torch.zeros(N, dtype=torch.bool))
        for k in ("enc", "cell", "pol", "trig"):
            self.assertEqual(sp[k].calls, 0, k)
        self.assertEqual(pool["rows"], 0)
        self.assertEqual(r.counters.rec_updates, 0)
        self.assertEqual(r.counters.finished_rows, 3 * N)
        self.assertTrue(bool((r.h == 0).all()))
        for act, m, _ in outs:
            self.assertFalse(bool(m.any()))

    def test_all_skipped_vision_no_encoder_but_cell_and_policy_run(self):
        r, sp, _, _ = self.spied("full", 3, valid_fn=lambda t: torch.zeros(N, dtype=torch.bool))
        self.assertEqual(sp["enc"].calls, 0)
        self.assertEqual(sp["cell"].rows, 3 * N)
        self.assertEqual(sp["pol"].rows, 3 * N)
        self.assertEqual(r.counters.blackout_rows, 3 * N)

    def test_trigger_skipped_all_rows_no_encoder_call(self):
        r, sp, _, outs = self.spied("learned:1e9", 5, n=1)
        self.assertEqual(sp["enc"].calls, 1)                  # only the forced first frame
        self.assertEqual(sp["cell"].rows, 5)

    def test_selective_equals_full_batch_for_active_rows(self):
        alive = lambda t: torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool) if t >= 2 else ones()
        for trig in ("full", "fixed:0.4", "learned:0.0"):
            net = make_net()
            rs, _, _, os_ = self.spied(trig, 8, alive, net=net)
            rf, _, _, of_ = self.spied(trig, 8, alive, select=False, net=net)
            for t, ((a1, m1, al), (a2, m2, _)) in enumerate(zip(os_, of_)):
                self.assertTrue(torch.equal(m1, m2), (trig, t))
                self.assertTrue(torch.allclose(a1[al], a2[al], atol=1e-5), (trig, t))
            rows = alive(7)
            self.assertTrue(torch.allclose(rs.h[rows], rf.h[rows], atol=1e-5), trig)

    def test_finished_rows_cannot_change_active_outputs(self):
        """Arbitrary garbage in the finished rows' inputs leaves the active rows' actions, states and decisions bit-identical."""
        alive_v = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.bool)
        for trig in ("full", "fixed:0.5", "pix:0.02", "learned:0.0"):
            net = make_net()
            runs = []
            for garbage in (False, True):
                r = I.RecurrentRunner(net, trig, N, "cpu")
                outs = []
                for t in range(8):
                    img, prop, task = obs(seed=t)
                    if garbage:
                        g = torch.Generator().manual_seed(77 + t)
                        k = int((~alive_v).sum())
                        img = img.clone(); prop = prop.clone()
                        img[~alive_v] = torch.randint(0, 256, (k, 32, 32, 3), generator=g, dtype=torch.uint8)
                        prop[~alive_v] = torch.rand(k, 4, generator=g) * 50
                    a, m = r.step(img, ones(), prop, task, alive_v)
                    outs.append((a[alive_v].clone(), m[alive_v].clone()))
                runs.append((outs, r.h[alive_v].clone()))
            for (a1, m1), (a2, m2) in zip(runs[0][0], runs[1][0]):
                self.assertTrue(torch.equal(a1, a2), trig)
                self.assertTrue(torch.equal(m1, m2), trig)
            self.assertTrue(torch.equal(runs[0][1], runs[1][1]), trig)

    def test_row_finishing_midway_is_frozen(self):
        net = make_net()
        r = I.RecurrentRunner(net, "full", N, "cpu")
        h_at_finish = None
        for t in range(6):
            alive = ones() if t < 3 else torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.bool)
            img, prop, task = obs(seed=t)
            r.step(img, ones(), prop, task, alive)
            if t == 2:
                h_at_finish = r.h[2].clone()
        self.assertTrue(torch.equal(r.h[2], h_at_finish))

    def test_state_advances_from_action_and_proprio_when_vision_skipped(self):
        net = make_net()
        hs = []
        for flip, dprop in ((False, 0.0), (True, 0.0), (False, 0.5)):
            r = I.RecurrentRunner(net, "full", 2, "cpu")
            img, prop, task = obs(2, seed=3)
            r.step(img, torch.zeros(2, dtype=torch.bool), prop, task, ones(2))
            if flip:
                r.a_prev = -r.a_prev + 0.3
            r.step(img, torch.zeros(2, dtype=torch.bool), prop + dprop, task, ones(2))
            hs.append(r.h.clone())
        self.assertFalse(torch.allclose(hs[0], hs[1]))
        self.assertFalse(torch.allclose(hs[0], hs[2]))

    def test_blackout_rows_never_reach_encoder_all_triggers(self):
        valid = lambda t: torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.bool)
        for trig in ("full", "fixed:1.0", "pix:0.0", "learned:-1e9"):
            r, sp, _, outs = self.spied(trig, 3, valid_fn=valid)
            self.assertEqual(sp["enc"].rows, r.counters.enc_samples, trig)
            for _, m, _ in outs:
                self.assertFalse(bool(m[1]) or bool(m[3]), trig)


if __name__ == "__main__":
    unittest.main()
