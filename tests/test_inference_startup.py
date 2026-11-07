"""EXP-0013 Phase 1.1 closeout (hardening round): CHARACTERIZATION of how the current inference runner treats an invalid first observation.

These tests pin down present behaviour so that the limitation is documented and cannot change silently; they do not assert that the behaviour is
desirable. v1/paired-replay never produce a blackout before step 3 (`t_pick >= 3` in the blackout recurrence), so none of this affects any reported
EXP-0012 / EXP-0013 number. It matters for any v2 environment with startup blackouts. See docs/experiments/EXP-0013-phase1-1-closeout.md.
Torch needed (host Python 3.12 imports it since the closeout; otherwise the WSL venv). Run: python -m unittest tests.test_inference_startup"""
import unittest

try:
    import torch
    from analysis.predictive_state import env as E
    from analysis.predictive_state import infer as I
    from analysis.predictive_state import models as M
    HAVE_TORCH = True
except Exception:       # pragma: no cover
    HAVE_TORCH = False


def frames(n=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.randint(0, 256, (n, 32, 32, 3), generator=g, dtype=torch.uint8)
    prop = torch.zeros(n, 4)
    task = torch.zeros(n, E.N_COLORS)
    task[:, 0] = 1
    return img, prop, task


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class StartupBlackout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = M.build("G")
        self.net.eval()

    def encode_flags(self, trigger, valid_seq):
        """Per-step enc_mask for a stream of distinct random frames whose validity follows valid_seq."""
        r = I.RecurrentRunner(self.net, trigger, 1, "cpu")
        out = []
        with torch.no_grad():
            for t, v in enumerate(valid_seq):
                img, prop, task = frames(seed=t)
                _, m = r.step(img, torch.tensor([v]), prop, task, torch.ones(1, dtype=torch.bool))
                out.append(bool(m[0]))
        return out, r

    def test_forced_first_observation_is_consumed_by_an_invalid_first_frame(self):
        """`fixed:0.0` never asks for a frame; only the forced first observation (t == 0) and the KMAX_SKIP safety bound can encode."""
        ok, _ = self.encode_flags("fixed:0.0", [True] * 12)
        late, _ = self.encode_flags("fixed:0.0", [False] + [True] * 11)
        self.assertTrue(ok[0], "a valid first frame is encoded (forced)")
        self.assertFalse(any(late[:I.KMAX_SKIP]), "after an invalid first frame the force is NOT re-armed ...")
        self.assertTrue(late[I.KMAX_SKIP], "... the first embedding only arrives at the KMAX_SKIP safety bound")

    def test_runner_does_not_distinguish_startup_blackout_from_intentional_skip(self):
        """The cell receives `valid = enc_mask`, never `valid_env`: a blackout frame and a trigger-skipped frame are identical inputs to the network."""
        img, prop, task = frames()
        alive = torch.ones(1, dtype=torch.bool)
        seen = []
        orig = self.net.cell_step

        def spy(h, z, valid, *a):
            seen.append((valid.clone(), z.abs().sum().item()))
            return orig(h, z, valid, *a)
        self.net.cell_step = spy
        with torch.no_grad():
            r_black = I.RecurrentRunner(self.net, "fixed:0.0", 1, "cpu")
            r_black.step(img, torch.tensor([False]), prop, task, alive)           # blackout at t=0
            r_skip = I.RecurrentRunner(self.net, "fixed:0.0", 1, "cpu")
            r_skip.step(img, torch.tensor([True]), prop, task, alive)             # forced encode at t=0 ...
            for t in range(1, 3):
                r_skip.step(img, torch.tensor([True]), prop, task, alive)         # ... then intentional skips (valid frame, trigger says no)
        self.net.cell_step = orig
        blackout_in, skip_in = seen[0], seen[2]
        self.assertEqual(float(blackout_in[0]), 0.0)
        self.assertEqual(float(skip_in[0]), 0.0)
        self.assertEqual(blackout_in[1], 0.0)
        self.assertEqual(skip_in[1], 0.0)
        # the ONLY difference is in the instrumentation counters, which the network never sees
        self.assertEqual(r_black.counters.blackout_rows, 1)
        self.assertEqual(r_skip.counters.trigger_skipped_rows, 2)

    def test_pixel_trigger_reference_is_initialised_from_the_blackout_frame(self):
        """`last_pool` is set from the first frame whatever its validity, so a pix trigger compares the next valid frame against a gray frame."""
        r = I.RecurrentRunner(self.net, "pix:0.0199", 1, "cpu")
        gray = torch.full((1, 32, 32, 3), 128, dtype=torch.uint8)
        _, prop, task = frames()
        with torch.no_grad():
            r.step(gray, torch.tensor([False]), prop, task, torch.ones(1, dtype=torch.bool))
        self.assertTrue(torch.allclose(r.last_pool, M.pool_feat(gray)), "last_pool should be the (invalid) first frame's pooled features")


if __name__ == "__main__":
    unittest.main()
