"""EXP-0013 Phase 1.1 closeout: tests of the paired-replay environment (numpy only, fast).

The point of these tests is that exogenous events are independent of the ACTIONS taken, of the batch composition and of the batch order,
not merely that initial seeds match. `test_v1_shared_rng_is_not_paired` documents the defect in the v1 environment."""
import hashlib
import unittest

import numpy as np

from analysis.predictive_state import env as E
from analysis.predictive_state import env_la as L
from analysis.predictive_state import env_paired as P

COND = L.LA_CONDS["LA_OCC"]                      # frequent, long blackouts
T = 60


def rollout(env, policy, T=T):
    obs = env.reset()
    valids, imgs = [obs["valid"].copy()], [obs["img"].copy()]
    for t in range(T):
        obs, _ = env.step(policy(env, t))
        valids.append(obs["valid"].copy())
        imgs.append(obs["img"].copy())
    return np.stack(valids, 1), np.stack(imgs, 1)


def stand_still(env, t):
    return np.zeros((env.n, 3), np.float32)


def random_fast(seed):
    g = np.random.default_rng(seed)
    return lambda env, t: np.concatenate([g.uniform(-1.5, 1.5, (env.n, 2)), np.zeros((env.n, 1))], 1).astype(np.float32)


def drive_right(env, t):
    return np.tile(np.array([1.0, 0.0, 0.0], np.float32), (env.n, 1))


class PairedEnvTests(unittest.TestCase):
    def test_divergent_policies_share_every_exogenous_event(self):
        va = vb = None
        res = {}
        for name, pol in (("still", stand_still), ("random", random_fast(1)), ("right", drive_right)):
            env = P.PairedLAEnv(24, "dev", COND, seed=11)
            sd = env.spawn_digest()
            v, img = rollout(env, pol)
            res[name] = (sd, v, img, env.layout_ids.copy())
        base = res["still"]
        for name in ("random", "right"):
            self.assertEqual(base[0], res[name][0], "initial layout/spawn/goal/target differ")
            self.assertTrue((base[3] == res[name][3]).all())
            self.assertTrue((base[1] == res[name][1]).all(), f"blackout schedule of '{name}' differs from 'still'")
        # the schedule is non-trivial (otherwise the test is vacuous) and the controllers really did diverge
        self.assertGreater((~base[1]).sum(), 100)
        self.assertGreater(np.abs(res["random"][2].astype(int) - base[2].astype(int)).sum(), 0)

    def test_realized_valid_equals_closed_form_schedule(self):
        env = P.PairedLAEnv(24, "dev", COND, seed=3, t_limit=30)      # every episode times out at step 30 < T = 60
        v, _ = rollout(env, random_fast(2))
        self.assertTrue(env.done.all())
        # a done episode's t_pick freezes, but the schedule must keep running regardless
        self.assertTrue((v == env.scheduled_valid(T)).all())

    def test_noise_is_a_pure_function_of_episode_and_time(self):
        a = P.PairedLAEnv(6, "dev", COND, seed=5)
        b = P.PairedLAEnv(6, "dev", COND, seed=5)
        rollout(a, stand_still, 10)
        rollout(b, random_fast(4), 10)
        self.assertEqual(a.t, b.t)
        self.assertTrue((a.last_noise == b.last_noise).all())
        self.assertTrue((a.last_noise[2] == P.noise_field(5, 2, 10)).all())
        # different time step -> different field; different episode -> different field
        self.assertFalse((P.noise_field(5, 2, 10) == P.noise_field(5, 2, 11)).all())
        self.assertFalse((P.noise_field(5, 2, 10) == P.noise_field(5, 3, 10)).all())
        # far from both end-effectors the frames are bit identical (same background noise), near them they differ
        a1 = P.PairedLAEnv(4, "dev", COND, seed=9); b1 = P.PairedLAEnv(4, "dev", COND, seed=9)
        oa, ob = a1.reset(), b1.reset()
        oa, _ = a1.step(np.tile(np.array([1.0, 0, 0], np.float32), (4, 1)))
        ob, _ = b1.step(np.tile(np.array([0, 1.0, 0], np.float32), (4, 1)))
        same = (oa["img"] == ob["img"]).all(-1)
        self.assertGreater(same.mean(), 0.5)

    def test_event_streams_independent_of_batch_size_and_order(self):
        big = P.PairedLAEnv(20, "dev", COND, seed=21)
        lay = big.layout_ids.copy()
        sd_big = big.spawn_digest()
        vb, ib = rollout(big, random_fast(7))
        for sel in ([7], [13, 2, 19], list(range(19, -1, -1))):
            sub = P.PairedLAEnv(len(sel), "dev", COND, seed=21, episode_ids=sel, layout_ids=lay[sel])
            self.assertEqual(sub.spawn_digest(), [sd_big[i] for i in sel], "spawn depends on batch composition/order")
            vs, is_ = rollout(sub, stand_still)
            self.assertTrue((vs == vb[sel]).all(), f"blackouts depend on batch composition/order for {sel}")
        # noise for episode e at time t is the same alone or in a batch
        solo = P.PairedLAEnv(1, "dev", COND, seed=21, episode_ids=[7], layout_ids=lay[[7]])
        rollout(solo, stand_still, 5)
        self.assertTrue((solo.last_noise[0] == P.noise_field(21, 7, 5)).all())

    def test_early_termination_does_not_shift_other_episodes(self):
        """One episode finishing early (forced done) must not change any other episode's schedule or noise (the v1 coupling)."""
        ref = P.PairedLAEnv(8, "dev", COND, seed=4)
        v_ref, _ = rollout(ref, stand_still)
        env = P.PairedLAEnv(8, "dev", COND, seed=4)
        vs = [env.reset()["valid"].copy()]
        env.done[0] = True                                    # episode 0 ends immediately
        for t in range(T):
            o, _ = env.step(stand_still(env, t))
            vs.append(o["valid"].copy())
        self.assertTrue((np.stack(vs, 1)[1:] == v_ref[1:]).all())

    def test_v1_shared_rng_is_not_paired(self):
        """Documents the defect: in v1, the same layouts/spawn state but different actions give different blackout schedules."""
        lay = np.arange(1000, 1016)
        outs = []
        for pol in (stand_still, drive_right):
            env = L.LAEnv(16, "dev", COND, seed=2, layout_ids=lay, t_limit=80)
            v, _ = rollout(env, pol)
            outs.append(v)
        self.assertFalse((outs[0] == outs[1]).all())

    def test_transition_equivalence_with_v1(self):
        """Same state + same actions: the paired env's physics (ee, objects, held, done, fail_kind) equal v1's. Blackouts/noise don't act on state."""
        pe = P.PairedLAEnv(16, "dev", L.LA_CONDS["LA_NOM"], seed=8, t_limit=80)
        v1 = L.LAEnv(16, "dev", L.LA_CONDS["LA_NOM"], seed=8, layout_ids=pe.layout_ids, t_limit=80)
        for k in ("ee", "pos", "col", "active", "held", "grip", "task", "t_pick", "picks_done", "done", "failed", "fail_kind", "goal"):
            setattr(v1, k, getattr(pe, k).copy())
        g = np.random.default_rng(0)
        # include grasping so pick / release / wrong_pick paths are exercised
        for t in range(80):
            a = np.concatenate([g.uniform(-1.5, 1.5, (16, 2)), g.uniform(0, 1, (16, 1))], 1).astype(np.float32)
            pe.step(a); v1.step(a)
            for k in ("ee", "pos", "held", "grip", "done", "failed", "t_pick", "picks_done"):
                self.assertTrue((getattr(pe, k) == getattr(v1, k)).all(), f"{k} diverged at t={t}")
            self.assertTrue((pe.fail_kind == v1.fail_kind).all())

    def test_shared_generator_use_is_refused_and_test_split_blocked(self):
        env = P.PairedLAEnv(2, "dev", COND, seed=1)
        with self.assertRaises(RuntimeError):
            env.rng.uniform(0, 1)
        with self.assertRaises(ValueError):
            P.PairedLAEnv(2, "test", COND, seed=1)

    def test_blackout_marginals_match_cond(self):
        env = P.PairedLAEnv(300, "dev", L.LA_CONDS["LA_NOM"], seed=77)
        v = env.scheduled_valid(65)
        dark = (~v[:, 1:]).mean()
        self.assertTrue(0.03 < dark < 0.5)                    # p=0.02/step, 5-15 long: roughly 0.15 of steps are dark

    def test_clear_condition_has_no_blackouts(self):
        env = P.PairedLAEnv(20, "dev", E.Cond("LA_CLEAR", 0.0, (5, 15), 0.0, 2), seed=1)
        self.assertTrue(env.scheduled_valid(82).all())

    # ------------------------------------------------------------------ hardening round: episode id <-> layout id
    def test_custom_episode_ids_require_explicit_layout_ids(self):
        with self.assertRaises(ValueError):
            P.PairedLAEnv(2, "dev", COND, seed=21, episode_ids=[7, 13])
        full = P.PairedLAEnv(20, "dev", COND, seed=21)                       # default full batch: episode i is id i (locked protocol, unchanged)
        np.testing.assert_array_equal(full.episode_ids, np.arange(20))
        np.testing.assert_array_equal(full.layout_ids, np.random.default_rng([21, 777]).integers(1000, 1100, size=20))

    def test_episode_keeps_its_layout_when_its_batch_index_changes(self):
        full = P.PairedLAEnv(20, "dev", COND, seed=21)
        sd = full.spawn_digest()
        for sel in ([7], [13, 7], [7, 13], [19, 7, 0]):
            sub = P.PairedLAEnv(len(sel), "dev", COND, seed=21, episode_ids=sel, layout_ids=full.layout_ids[sel])
            for pos, e in enumerate(sel):
                self.assertEqual(sub.layout_ids[pos], full.layout_ids[e], f"episode {e} received another layout at batch index {pos}")
                self.assertEqual(sub.spawn_digest()[pos], sd[e])
        # the silent failure mode being removed: with the batch-position default, episode 7 at index 0 would get layout_ids[0], not layout_ids[7]
        silent = P.PairedLAEnv(1, "dev", COND, seed=21)
        self.assertNotEqual(silent.layout_ids[0], full.layout_ids[7])

    def test_ambiguous_episode_identities_are_rejected(self):
        lay = np.arange(1000, 1003)
        for bad_ids, bad_lay in (([1, 1, 2], lay), ([1, 2], lay), ([-1, 2, 3], lay), ([[1, 2, 3]], lay), ([1, 2, 3], lay[:2]),
                                 ([1, 2, 3], np.array([1000, 1001, 2000])), ([1, 2, 3], np.array([1000, 1001, 5]))):
            with self.assertRaises(ValueError, msg=str((bad_ids, bad_lay))):
                P.PairedLAEnv(3, "dev", COND, seed=1, episode_ids=bad_ids, layout_ids=bad_lay)
        with self.assertRaises(ValueError):
            P.PairedLAEnv(3, "dev", COND, seed=1, layout_ids=lay[:2])

    def test_explicit_layout_ids_must_belong_to_the_requested_split(self):
        """Every initialization path (episode_ids None or custom): positive cases per split, negative cases incl. reserved TEST geometry."""
        ranges = E.LAYOUT_RANGES
        for split in ("train", "dev"):
            lo, hi = ranges[split]
            for ids in (None, [0, 1]):
                kw = {} if ids is None else {"episode_ids": ids}
                env = P.PairedLAEnv(2, split, COND, seed=1, layout_ids=np.array([lo, hi - 1]), **kw)
                np.testing.assert_array_equal(env.layout_ids, [lo, hi - 1])
            others = [r for k, r in ranges.items() if k != split]
            for bad in ([ranges["test"][0], lo], [lo, ranges["test"][1] - 1], [lo - 1, lo], [lo, hi], [others[0][0], lo]):
                for kw in ({}, {"episode_ids": [0, 1]}):
                    with self.assertRaises(ValueError, msg=f"{split} {bad} {kw}"):
                        P.PairedLAEnv(2, split, COND, seed=1, layout_ids=np.array(bad), **kw)
        with self.assertRaises(ValueError):
            P.PairedLAEnv(1, "dev", COND, seed=1, layout_ids=[2000])                    # the reported defect: dev + reserved id, no episode_ids
        with self.assertRaises(ValueError):
            P.PairedLAEnv(1, "dev", COND, seed=1, layout_ids=[1000.0])                  # non-integer ids
        with self.assertRaises(ValueError):
            P.PairedLAEnv(1, "test", COND, seed=1, layout_ids=[2000])                   # reserved split itself stays refused
        # default generation stays inside the split and is unchanged
        for split in ("train", "dev"):
            e = P.PairedLAEnv(50, split, COND, seed=9)
            lo, hi = ranges[split]
            self.assertTrue(((e.layout_ids >= lo) & (e.layout_ids < hi)).all())
            np.testing.assert_array_equal(e.layout_ids, np.random.default_rng([9, 777]).integers(lo, hi, size=50))

    # ------------------------------------------------------------------ hardening round: unsupported random draws fail visibly
    def test_unsupported_random_draws_fail_visibly(self):
        env = P.PairedLAEnv(4, "dev", COND, seed=1)
        rng = env.rng
        for call in (lambda: rng.normal(0, E.PIX_NOISE, (4, 2)),          # Gaussian in the dynamics (outside render): must NOT return zeros
                     lambda: rng.normal(0, 1.0, (4, 2)),
                     lambda: rng.random(3),                              # wrong shape
                     lambda: rng.integers(0, 5),                         # not the blackout duration draw
                     lambda: rng.uniform(0, 1), lambda: rng.permutation(3), lambda: rng.choice(3)):
            with self.assertRaises(RuntimeError):
                call()
        # inside render only the zero-mean PIX_NOISE placeholder is served; any other Gaussian still fails
        rng.render_scope = True
        try:
            self.assertEqual(float(np.abs(rng.normal(0, E.PIX_NOISE, (3, 3))).sum()), 0.0)
            for call in (lambda: rng.normal(0, 2 * E.PIX_NOISE, (3, 3)), lambda: rng.normal(0.5, E.PIX_NOISE, (3, 3))):
                with self.assertRaises(RuntimeError):
                    call()
        finally:
            rng.render_scope = False

    def test_gaussian_dynamics_draw_in_step_is_caught(self):
        """Negative control: a (hypothetical) future dynamics change that draws Gaussian noise inside step() must raise, not silently get zeros."""
        class Rogue(P.PairedLAEnv):
            def step(self, action):
                self.rng.normal(0.0, E.PIX_NOISE, (self.n, 2))
                return super().step(action)
        env = Rogue(2, "dev", COND, seed=1)
        env.reset()
        with self.assertRaises(RuntimeError):
            env.step(np.zeros((2, 3), np.float32))

    def test_render_noise_distribution_preserved(self):
        n = np.stack([P.noise_field(3, e, 0) for e in (0, 1, 2, 3)])
        self.assertAlmostEqual(float(n.std()), E.PIX_NOISE, delta=0.1)
        self.assertAlmostEqual(float(n.mean()), 0.0, delta=0.4)
        # pinned realization: any change to the noise recipe (a protocol change) must be deliberate and must bump PROTOCOL
        self.assertEqual(hashlib.sha1(P.noise_field(3, 0, 0).tobytes()).hexdigest(), "d8142868718398564bf4b16a53fc6e963a0d22c3")
        env = P.PairedLAEnv(2, "dev", COND, seed=3)
        env.reset()
        self.assertTrue((env.last_noise[1] == P.noise_field(3, 1, 0)).all())

    def test_locked_protocol_reproduces_committed_digests(self):
        """The hardening must not change the locked la-paired-replay-1.1 protocol: spawn and schedule digests of the committed DEV runs are reproduced."""
        import json
        from pathlib import Path
        base = Path(__file__).resolve().parent.parent / "results" / "predictive_state_phase1_1_closeout"
        for cn in ("LA_CLEAR", "LA_NOM", "LA_OCC"):
            f = base / f"paired_replay_{cn}.json"
            if not f.exists():
                self.skipTest("committed paired-replay results not present")
            rec = json.loads(f.read_text())
            cond = __import__("analysis.predictive_state.run_phase11_benchmark", fromlist=["CONDS"]).CONDS[cn]
            env = P.PairedLAEnv(rec["n"], "dev", cond, rec["seed"], t_limit=80)
            ref = rec["controllers"]["greedy"]
            self.assertEqual(rec["protocol"], P.PROTOCOL)
            self.assertEqual(env.layout_ids.tolist(), ref["layout_id"])
            self.assertEqual(env.spawn_digest(), ref["spawn_digest"])
            self.assertEqual(env.schedule_digest(82), ref["schedule_digest"])

if __name__ == "__main__":
    unittest.main()
