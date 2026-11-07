"""EXP-0013 Phase 1 regression tests: true encoder skipping, trigger semantics, evaluation/latency equivalence.

Torch is needed (WSL venv); the module is skipped on the Windows host (KI-005). All tests use an *untrained* build("G") on CPU, so
they do not depend on any checkpoint. A spy on `net.enc.forward` records the exact batches the encoder receives.
Run: python -m unittest tests.test_inference_skip
"""
import unittest

import numpy as np

try:
    import torch
    from analysis.predictive_state import env as E
    from analysis.predictive_state import models as M
    from analysis.predictive_state import infer as I
    from analysis.predictive_state import evaluate as V
    HAVE_TORCH = True
except Exception:       # pragma: no cover
    HAVE_TORCH = False

N = 6


def make_net(seed=0):
    torch.manual_seed(seed)
    net = M.build("G")
    net.eval()
    return net


class Spy:
    def __init__(self, net):
        self.batches = []
        self.orig = net.enc.forward
        net.enc.forward = self.fwd

    def fwd(self, x, task):
        self.batches.append((x.detach().clone(), task.detach().clone()))
        return self.orig(x, task)

    @property
    def calls(self):
        return len(self.batches)

    @property
    def samples(self):
        return sum(b[0].shape[0] for b in self.batches)


def obs(n=N, seed=0, held=0.0, grip=0.0, task_id=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.randint(0, 256, (n, 32, 32, 3), generator=g, dtype=torch.uint8)
    prop = torch.rand(n, 4, generator=g)
    prop[:, 2], prop[:, 3] = held, grip
    task = torch.zeros(n, E.N_COLORS)
    task[:, task_id] = 1
    return img, prop, task


def ones(n=N):
    return torch.ones(n, dtype=torch.bool)


@unittest.skipUnless(HAVE_TORCH, "needs torch (WSL)")
class TestGatedEncode(unittest.TestCase):
    def test_none_needed_never_calls_encoder(self):
        net = make_net(); spy = Spy(net); c = I.Counters()
        img, _, task = obs()
        z = I.gated_encode(net.enc, img, task, torch.zeros(N, dtype=torch.bool), c)
        self.assertEqual(spy.calls, 0)
        self.assertEqual(tuple(z.shape), (N, M.Z))
        self.assertTrue(bool((z == 0).all()))
        self.assertEqual((c.enc_calls, c.enc_samples), (0, 0))

    def test_mixed_batch_encodes_only_selected_rows(self):
        net = make_net(); spy = Spy(net); c = I.Counters()
        img, _, task = obs()
        need = torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.bool)
        z = I.gated_encode(net.enc, img, task, need, c)
        self.assertEqual((spy.calls, spy.samples), (1, 3))
        self.assertEqual((c.enc_calls, c.enc_samples), (1, 3))
        self.assertTrue(torch.equal(spy.batches[0][0], M.to_x(img[need])))
        self.assertTrue(bool((z[~need] == 0).all()))
        full = net.enc(M.to_x(img), task)
        self.assertTrue(torch.allclose(z[need], full[need], atol=1e-5))

    def test_all_needed(self):
        net = make_net(); spy = Spy(net); c = I.Counters()
        img, _, task = obs()
        z = I.gated_encode(net.enc, img, task, ones(), c)
        self.assertEqual((spy.calls, spy.samples), (1, N))
        self.assertTrue(bool(torch.isfinite(z).all()))


@unittest.skipUnless(HAVE_TORCH, "needs torch (WSL)")
class TestRunnerSemantics(unittest.TestCase):
    def run_seq(self, trigger, steps, valid=None, alive=None, tasks=None, imgs=None, props=None, n=N, net=None):
        net = net or make_net(); spy = Spy(net)
        r = I.RecurrentRunner(net, trigger, n, "cpu")
        masks = []
        for t in range(steps):
            img, prop, task = obs(n, seed=t)
            if imgs is not None:
                img = imgs(t, img)
            if props is not None:
                prop = props(t, prop)
            if tasks is not None:
                task = tasks(t, task)
            v = valid(t) if valid else ones(n)
            a = alive(t) if alive else ones(n)
            _, m = r.step(img, v, prop, task, a)
            masks.append(m.clone())
        return r, spy, masks

    def test_all_samples_encoded_full(self):
        r, spy, _ = self.run_seq("full", 4)
        self.assertEqual((spy.calls, spy.samples), (4, 4 * N))
        self.assertEqual((r.counters.enc_calls, r.counters.enc_samples), (4, 4 * N))

    def test_no_samples_encoded_but_state_advances(self):
        net = make_net(); spy = Spy(net)
        r = I.RecurrentRunner(net, "full", N, "cpu")
        hs = []
        for t in range(3):
            img, prop, task = obs(seed=t)
            r.step(img, torch.zeros(N, dtype=torch.bool), prop, task, ones())
            hs.append(r.h.clone())
        self.assertEqual(spy.calls, 0)
        self.assertEqual(r.counters.enc_samples, 0)
        self.assertEqual(r.counters.rec_updates, 3 * N)
        self.assertFalse(torch.allclose(hs[0], hs[1]), "recurrent state must keep evolving without vision")
        self.assertFalse(torch.allclose(hs[1], hs[2]))
        self.assertTrue(all(bool(torch.isfinite(h).all()) for h in hs))

    def test_blackout_rows_never_encoded_even_when_forced(self):
        valid = lambda t: torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.bool)
        for trig in ("full", "fixed:1.0", "pix:0.0", "learned:-1e9"):
            r, spy, masks = self.run_seq(trig, 3, valid=valid)
            for m in masks:
                self.assertFalse(bool(m[1]) or bool(m[3]), trig)
            exp = sum(int(m.sum()) for m in masks)
            self.assertEqual(spy.samples, exp, trig)
            for xb, _ in spy.batches:
                self.assertLessEqual(xb.shape[0], 4)
        self.assertEqual(r.counters.blackout_rows, 3 * 2)

    def test_finished_rows_never_encoded(self):
        alive = lambda t: torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
        r, spy, masks = self.run_seq("full", 3, alive=alive)
        self.assertEqual(spy.samples, 3 * 3)
        self.assertEqual(r.counters.finished_rows, 3 * 3)
        self.assertEqual(r.counters.active_steps, 3 * 3)
        for m in masks:
            self.assertFalse(bool(m[2]) or bool(m[3]) or bool(m[5]))

    def test_mixed_batch_inputs_are_exactly_eligible_rows(self):
        valid = lambda t: torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool)
        alive = lambda t: torch.tensor([1, 1, 1, 0, 1, 1], dtype=torch.bool)
        r, spy, masks = self.run_seq("full", 1, valid=valid, alive=alive)
        elig = valid(0) & alive(0)
        img, _, task = obs(seed=0)
        self.assertTrue(torch.equal(spy.batches[0][0], M.to_x(img[elig])))
        self.assertTrue(torch.equal(spy.batches[0][1], task[elig]))

    def test_fixed_rate_schedule_exact(self):
        # fixed:0.25 -> forced at t=0, accumulator reaches 1 after 4 additions: t=3, then every 4 steps (<= KMAX_SKIP apart)
        r, spy, masks = self.run_seq("fixed:0.25", 16, n=1)
        enc_t = [t for t, m in enumerate(masks) if bool(m[0])]
        self.assertEqual(enc_t, [0, 3, 7, 11, 15])
        self.assertEqual(spy.samples, 5)
        self.assertEqual(spy.calls, 5)
        self.assertEqual(r.counters.trigger_skipped_rows, 11)

    def test_first_frame_forced_for_every_trigger(self):
        for trig in ("fixed:0.001", "pix:1e9", "learned:1e9"):
            _, _, masks = self.run_seq(trig, 1)
            self.assertTrue(bool(masks[0].all()), trig)

    def test_max_skip_forced(self):
        _, spy, masks = self.run_seq("learned:1e9", 20, n=1)
        enc_t = [t for t, m in enumerate(masks) if bool(m[0])]
        self.assertEqual(enc_t, [0, 9, 18])            # since >= KMAX_SKIP(8) after 8 skipped steps
        self.assertEqual(spy.samples, 3)
        for t in (1, 5, 8):
            self.assertFalse(bool(masks[t][0]))

    def test_learned_trigger_threshold_controls_rows(self):
        net = make_net()
        # deterministic stub: rows 0,2,4 report high surprise, rows 1,3,5 low
        class Stub(torch.nn.Module):
            def forward(self, x):
                return torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])
        net.trig = Stub()
        _, spy, masks = self.run_seq("learned:0.5", 3, net=net)
        for t in (1, 2):
            self.assertEqual([bool(b) for b in masks[t]], [True, False, True, False, True, False])
        self.assertEqual([bool(b) for b in masks[0]], [True] * N)      # t=0 forced
        self.assertEqual(spy.samples, N + 3 + 3)

    def test_pixel_change_trigger(self):
        base = lambda t, img: obs(1, seed=123)[0]
        changed = lambda t, img: (obs(1, seed=123)[0] if t < 5 else obs(1, seed=999)[0])
        _, _, m_static = self.run_seq("pix:0.05", 8, imgs=base, n=1)
        _, _, m_change = self.run_seq("pix:0.05", 8, imgs=changed, n=1)
        self.assertEqual([t for t, m in enumerate(m_static) if bool(m[0])], [0])
        self.assertEqual([t for t, m in enumerate(m_change) if bool(m[0])], [0, 5])

    def test_task_change_forced(self):
        tasks = lambda t, task: obs(N, seed=t, task_id=(1 if t >= 3 else 0))[2]
        _, _, masks = self.run_seq("learned:1e9", 5, tasks=tasks)
        self.assertTrue(bool(masks[3].all()))
        self.assertFalse(bool(masks[2].any()) or bool(masks[4].any()))

    def test_slip_forced(self):
        def props(t, prop):                    # held at t=1..2 with grip closed, then dropped (held=0) while grip stays closed
            prop = prop.clone(); prop[:, 3] = 1.0; prop[:, 2] = 1.0 if t in (1, 2) else 0.0
            return prop
        _, _, masks = self.run_seq("learned:1e9", 6, props=props, n=1)
        self.assertTrue(bool(masks[3][0]))
        self.assertFalse(bool(masks[2][0]) or bool(masks[4][0]))

    def test_runner_batched_equals_per_row_runner(self):
        """latency (n=1) and evaluation (batched) share the runner: identical decisions and states row by row."""
        for trig in ("fixed:0.4", "pix:0.02", "learned:0.0"):
            net = make_net()
            torch.use_deterministic_algorithms(True)
            rb = I.RecurrentRunner(net, trig, N, "cpu")
            rows = [I.RecurrentRunner(net, trig, 1, "cpu") for _ in range(N)]
            for t in range(14):
                img, prop, task = obs(seed=t)
                valid = torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.bool) if t % 5 == 2 else ones()
                _, mb = rb.step(img, valid, prop, task, ones())
                for i, ri in enumerate(rows):
                    _, mi = ri.step(img[i:i + 1], valid[i:i + 1], prop[i:i + 1], task[i:i + 1], ones(1))
                    self.assertEqual(bool(mb[i]), bool(mi[0]), f"{trig} t={t} row={i}")
                    self.assertTrue(torch.allclose(rb.h[i], ri.h[0], atol=1e-5), f"{trig} t={t} row={i}")
            torch.use_deterministic_algorithms(False)

    def test_full_perception_matches_original_open_loop(self):
        """Gated runner reproduces the EXP-0012 computation (encode everything, mask by validity) step by step."""
        net = make_net()
        r = I.RecurrentRunner(net, "full", N, "cpu")
        h = net.init_state(N, "cpu"); a_prev = torch.zeros(N, 3)
        for t in range(10):
            img, prop, task = obs(seed=t)
            valid = torch.tensor([1, 1, 0, 1, 0, 1], dtype=torch.bool) if t % 3 == 1 else ones()
            alive = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool) if t > 6 else ones()
            act, _ = r.step(img, valid, prop, task, alive)
            with torch.no_grad():
                z = net.enc(M.to_x(img), task)
                vf = (valid & alive).float()
                h = net.cell_step(h, z, vf, prop, task, a_prev)
                lg = net.policy(h, prop, task, z * vf[:, None])
                a_prev = torch.cat([torch.tanh(lg[:, :2]), (lg[:, 2:3] > 0).float()], -1)
            self.assertTrue(torch.allclose(r.h, h, atol=1e-5), t)
            self.assertTrue(torch.allclose(act[:, :2], a_prev[:, :2], atol=1e-5), t)
            self.assertTrue(bool(torch.isfinite(r.h).all()))


@unittest.skipUnless(HAVE_TORCH, "needs torch (WSL)")
class TestRolloutEquivalence(unittest.TestCase):
    """New closed-loop rollout vs the unmodified EXP-0012 rollout (rollout_legacy), CPU, untrained G."""

    def setUp(self):
        self.net = make_net()
        self.cond = E.TEST_CONDS["NOM"]

    def _both(self, trigger, n=6):
        spy = Spy(self.net)
        new = V.rollout(self.net, "dev", self.cond, 11, n, trigger, "cpu")
        s_new = (spy.calls, spy.samples)
        spy.batches.clear()
        old = V.rollout_legacy(self.net, "dev", self.cond, 11, n, trigger, "cpu")
        s_old = (spy.calls, spy.samples)
        return new, old, s_new, s_old

    def test_decisions_outcomes_and_true_encoder_work(self):
        for trig in ("full", "fixed:0.25", "pix:0.03"):
            new, old, s_new, s_old = self._both(trig)
            np.testing.assert_array_equal(new["enc_count"], old["enc_count"], trig)
            np.testing.assert_array_equal(new["success"], old["success"], trig)
            np.testing.assert_array_equal(new["alive_steps"], old["alive_steps"], trig)
            # executed work: new == logical request count; legacy encodes every row at every batch step
            self.assertEqual(new["counters"]["enc_samples"], int(old["enc_count"].sum()), trig)
            self.assertEqual(s_new[1], new["counters"]["enc_samples"], trig)
            self.assertGreater(s_old[1], s_new[1], trig)

    def test_skipping_reduces_executed_samples(self):
        full = V.rollout(self.net, "dev", self.cond, 11, 6, "full", "cpu")["counters"]
        skip = V.rollout(self.net, "dev", self.cond, 11, 6, "fixed:0.25", "cpu")["counters"]
        self.assertLess(skip["enc_samples"], 0.6 * full["enc_samples"])
        self.assertEqual(skip["rec_updates"], skip["active_steps"])

    def test_deterministic_under_fixed_seed(self):
        a = V.rollout(self.net, "dev", self.cond, 11, 6, "fixed:0.25", "cpu")
        b = V.rollout(self.net, "dev", self.cond, 11, 6, "fixed:0.25", "cpu")
        np.testing.assert_array_equal(a["enc_count"], b["enc_count"])
        np.testing.assert_array_equal(a["success"], b["success"])
        self.assertEqual(a["counters"], b["counters"])

    def test_non_recurrent_arm_falls_back_to_legacy(self):
        torch.manual_seed(0)
        net = M.build("R"); net.eval()
        r = V.rollout(net, "dev", self.cond, 11, 4, "full", "cpu", T=5)
        self.assertNotIn("counters", r)


@unittest.skipUnless(HAVE_TORCH, "needs torch (WSL)")
class TestLatencyPath(unittest.TestCase):
    def test_latency_uses_same_runner_and_supports_fixed(self):
        from analysis.predictive_state import latency as L
        net = make_net(); spy = Spy(net)
        out = L.run(net, "fixed:0.25", 1, episodes=2, seed=31)
        c = out["counters"]
        self.assertEqual(c["enc_samples"], spy.samples)
        self.assertEqual(c["enc_calls"], spy.calls)
        self.assertEqual(c["rec_updates"], c["steps"])
        self.assertLess(out["enc_frac"], 0.5)
        self.assertGreater(c["trigger_skipped_rows"], 0)
        for k in ("trigger", "encoder", "update"):
            self.assertGreaterEqual(out["component_mean_ms"][k], 0.0)

    def test_latency_full_has_no_trigger_cost_and_encodes_valid_frames_only(self):
        from analysis.predictive_state import latency as L
        net = make_net(); spy = Spy(net)
        out = L.run(net, "full", 1, episodes=2, seed=31)
        c = out["counters"]
        self.assertEqual(c["enc_samples"], spy.samples)
        self.assertEqual(c["enc_samples"] + c["blackout_rows"], c["active_steps"])


@unittest.skipUnless(HAVE_TORCH, "needs torch (WSL)")
class TestDagger(unittest.TestCase):
    def test_gated_collection_matches_encode_everything(self):
        from analysis.predictive_state import dagger as D
        net = make_net()
        new = D.collect(net, 4, 3, Tn=6, batch=4, dev="cpu")
        orig = I.gated_encode
        D.I.gated_encode = lambda enc, img, task, need, counters=None, z_dim=M.Z: enc(M.to_x(img), task)   # old behaviour
        try:
            old = D.collect(net, 4, 3, Tn=6, batch=4, dev="cpu")
        finally:
            D.I.gated_encode = orig
        for k in new:
            np.testing.assert_allclose(new[k].astype(np.float64), old[k].astype(np.float64), atol=1e-5, err_msg=k)


if __name__ == "__main__":
    unittest.main()
