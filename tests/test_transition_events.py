"""EXP-0013 Phase 1.1 closeout (hardening round): deterministic transition fixtures for the paired-replay environment vs OccPick2D-LA v1.

Random actions do not prove branch coverage, so each fixture scripts the state and the action sequence, ASSERTS THAT THE INTENDED EVENT OCCURRED in
the paired environment, and compares the full physical state with an `env_la.LAEnv` started from the identical state and fed the identical actions
at every step. Blackouts and noise do not act on the physics, so a different blackout stream in v1 does not affect the comparison (the blackout
fixture compares against the closed-form schedule instead). Neither env_la.py nor env.py is modified: the fixtures only overwrite state arrays.
numpy only. Run: python -m unittest tests.test_transition_events"""
import unittest

import numpy as np

from analysis.predictive_state import env as E
from analysis.predictive_state import env_la as L
from analysis.predictive_state import env_paired as P

CLEAR = E.Cond("T_CLEAR", 0.0, (5, 15), 0.0, 2)
PHYS = ("ee", "pos", "col", "active", "held", "grip", "task", "t_pick", "picks_done", "done", "failed", "fail_kind", "failed_grasps", "goal")
GOAL = np.array([0.80, 0.50], np.float32)
FAR = np.array([0.30, 0.90], np.float32)
NO_WALLS = np.zeros((L.NWALL_ALL, 4), np.float32)


def a(dx=0.0, dy=0.0, g=0.0):
    return np.array([dx, dy, g], np.float32)


def pt(x, y):
    return np.array([x, y], np.float32)


def place(env, i, ee, target, distractor=None, walls=NO_WALLS, goal=GOAL):
    """Overwrite episode i's state (the same call is applied to the paired env and to v1)."""
    env.ee[i] = ee
    env.pos[i] = 0.5
    env.pos[i, 0] = target
    env.pos[i, 1] = FAR if distractor is None else distractor
    env.col[i] = np.arange(E.KMAX + 1)                  # colour 0 is the task colour (target); every other slot is inactive
    env.task[i] = 0
    env.active[i] = False
    env.active[i, :2] = True
    env.held[i] = False
    env.grip[i] = False
    env.goal[i] = goal
    env.walls[i] = walls


def twin(cond, setups, t_limit=80, seed=3):
    n = len(setups)
    pe = P.PairedLAEnv(n, "dev", cond, seed=seed, t_limit=t_limit)
    v1 = L.LAEnv(n, "dev", cond, seed=seed, layout_ids=pe.layout_ids, t_limit=t_limit)
    for env in (pe, v1):
        for i, s in enumerate(setups):
            place(env, i, **s)
    return pe, v1


def run(pe, v1, actions):
    """Step both envs with identical actions; assert identical physics after every step; return the paired env's per-step trajectory."""
    traj = []
    for t, act in enumerate(actions):
        act = np.asarray(act, np.float32).reshape(pe.n, 3)
        pe.step(act.copy())
        v1.step(act.copy())
        for k in PHYS:
            assert (getattr(pe, k) == getattr(v1, k)).all(), f"{k} diverged between paired and v1 at step {t}"
        traj.append({k: getattr(pe, k).copy() for k in ("ee", "held", "done", "failed", "fail_kind", "picks_done", "t_pick", "failed_grasps", "black_left")})
        traj[-1]["pos0"] = pe.pos[:, 0].copy()
    return traj


class TransitionEvents(unittest.TestCase):
    def test_successful_acquisition_and_release_at_goal(self):
        start = pt(0.20, 0.50)
        pe, v1 = twin(CLEAR, [dict(ee=start, target=pt(0.22, 0.50))])
        tr = run(pe, v1, [a(g=1.0)] + [a(1.0, 0.0, 1.0)] * 12 + [a(0, 0, 0.0)])
        self.assertTrue(tr[0]["held"][0], "target not acquired by the closing gripper")
        self.assertTrue((tr[0]["pos0"][0] == tr[0]["ee"][0]).all(), "held object does not follow the end-effector")
        self.assertFalse(tr[0]["done"][0])
        self.assertTrue(tr[-2]["held"][0] and not tr[-2]["done"][0], "still carrying before the release step")
        self.assertLess(np.abs(tr[-2]["ee"][0] - GOAL).max(), E.GOAL_R, "fixture error: the carry did not reach the goal zone")
        last = tr[-1]
        self.assertTrue(last["done"][0] and not last["failed"][0] and not last["held"][0])
        self.assertEqual(last["picks_done"][0], 1)
        self.assertEqual(last["fail_kind"][0], "")

    def test_release_outside_goal_drops_without_success(self):
        pe, v1 = twin(CLEAR, [dict(ee=pt(0.20, 0.50), target=pt(0.20, 0.50))])
        tr = run(pe, v1, [a(g=1.0), a(1.0, 0.0, 1.0), a(0, 0, 0.0)])
        self.assertTrue(tr[1]["held"][0])
        self.assertFalse(tr[2]["held"][0])
        self.assertFalse(tr[2]["done"][0])
        self.assertEqual(tr[2]["picks_done"][0], 0)

    def test_wrong_object_acquisition_fails(self):
        pe, v1 = twin(CLEAR, [dict(ee=pt(0.30, 0.50), target=pt(0.60, 0.50), distractor=pt(0.31, 0.50))])
        tr = run(pe, v1, [a(g=1.0), a(1.0, 0.0, 0.0)])
        self.assertTrue(tr[0]["done"][0] and tr[0]["failed"][0])
        self.assertEqual(tr[0]["fail_kind"][0], "wrong_pick")
        self.assertFalse(tr[0]["held"][0])
        np.testing.assert_array_equal(tr[1]["ee"][0], tr[0]["ee"][0], err_msg="a finished episode must be frozen")

    def test_failed_grasp_in_empty_space_is_counted_not_terminal(self):
        pe, v1 = twin(CLEAR, [dict(ee=pt(0.20, 0.20), target=pt(0.60, 0.60))])
        tr = run(pe, v1, [a(g=1.0)])
        self.assertEqual(tr[0]["failed_grasps"][0], 1)
        self.assertFalse(tr[0]["done"][0] or tr[0]["held"][0])

    def test_wall_collision_blocks_and_slides(self):
        walls = NO_WALLS.copy()
        walls[0] = [0.50, 0.50, 0.08, 0.60]                      # x in [0.46, 0.54], y in [0.20, 0.80]
        start = pt(0.44, 0.50)
        pe, v1 = twin(CLEAR, [dict(ee=start, target=FAR, walls=walls)])
        # the unobstructed move WOULD enter the wall, so the block is real and the fixture is not vacuous
        self.assertTrue(L.inside(pt(start[0] + 0.05 / np.sqrt(2), start[1]), walls))
        tr = run(pe, v1, [a(1.0, 1.0)] * 4)
        for k in range(4):
            self.assertEqual(tr[k]["ee"][0][0], start[0], "x must stay blocked at the wall face")
        ys = [start[1]] + [tr[k]["ee"][0][1] for k in range(4)]
        self.assertTrue(all(y1 > y0 for y0, y1 in zip(ys, ys[1:])), "the end-effector must slide along the wall (y keeps increasing)")
        pe2, v12 = twin(CLEAR, [dict(ee=start, target=FAR, walls=walls)])         # head-on: the blocked axis is the only commanded one
        tr2 = run(pe2, v12, [a(1.0, 0.0)] * 3)
        np.testing.assert_array_equal(tr2[-1]["ee"][0], start)

    def test_timeout_at_t_limit(self):
        pe, v1 = twin(CLEAR, [dict(ee=pt(0.2, 0.5), target=FAR)], t_limit=5)
        tr = run(pe, v1, [a()] * 6)
        self.assertFalse(tr[3]["done"][0])
        self.assertTrue(tr[4]["done"][0] and tr[4]["failed"][0])
        self.assertEqual(tr[4]["fail_kind"][0], "timeout")
        self.assertEqual(tr[5]["t_pick"][0], tr[4]["t_pick"][0], "t_pick must freeze after termination")

    def test_base_simulator_cap_timeout(self):
        pe, v1 = twin(CLEAR, [dict(ee=pt(0.2, 0.5), target=FAR)], t_limit=500)
        tr = run(pe, v1, [a()] * E.T_PICK)
        self.assertFalse(tr[E.T_PICK - 2]["done"][0])
        self.assertTrue(tr[E.T_PICK - 1]["done"][0])
        self.assertEqual(tr[E.T_PICK - 1]["fail_kind"][0], "timeout")

    def test_early_termination_freezes_one_episode_and_leaves_the_other_alive(self):
        pe, v1 = twin(CLEAR, [dict(ee=pt(0.30, 0.50), target=pt(0.60, 0.50), distractor=pt(0.31, 0.50)), dict(ee=pt(0.20, 0.30), target=pt(0.20, 0.30))])
        acts = [np.stack([a(g=1.0), a(g=1.0)])] + [np.stack([a(1.0, 0.0), a(1.0, 0.0, 1.0)])] * 3
        tr = run(pe, v1, acts)
        self.assertTrue(tr[0]["done"][0] and tr[0]["fail_kind"][0] == "wrong_pick")
        self.assertFalse(tr[0]["done"][1])
        for k in range(1, 4):
            np.testing.assert_array_equal(tr[k]["ee"][0], tr[0]["ee"][0])
            self.assertGreater(tr[k]["ee"][1][0], tr[k - 1]["ee"][1][0], "the surviving episode must keep moving")
        self.assertTrue(tr[3]["held"][1])
        self.assertEqual(tr[3]["t_pick"][0], tr[0]["t_pick"][0])
        self.assertGreater(tr[3]["t_pick"][1], tr[0]["t_pick"][1])

    def test_blackout_spans_termination(self):
        """p=1 blackouts of fixed length 20 start at the first allowed step; the episode ends INSIDE the dark window; the schedule keeps running afterwards."""
        cond = E.Cond("T_DARK", 1.0, (20, 20), 0.0, 2)
        pe, v1 = twin(cond, [dict(ee=pt(0.20, 0.50), target=pt(0.20, 0.50))])
        T = 40
        acts = [a(g=1.0)] + [a(1.0, 0.0, 1.0)] * 12 + [a(0.0, 0.0, 0.0)] + [a()] * (T - 14)
        valids = [pe.observe()["valid"].copy()]
        done_at, dark_at_done = None, None
        for t, act in enumerate(acts):
            act = np.asarray(act, np.float32).reshape(1, 3)
            ob, _ = pe.step(act.copy())
            v1.step(act.copy())
            valids.append(ob["valid"].copy())
            if pe.done[0] and done_at is None:
                done_at, dark_at_done = t, int(pe.black_left[0])
        self.assertEqual(pe.picks_done[0], 1, "fixture error: the pick-and-place must succeed")
        self.assertIsNotNone(done_at)
        self.assertGreater(dark_at_done, 0, "termination must happen while a blackout is in progress")
        self.assertFalse(valids[done_at + 1][0], "the observation returned with the terminal transition must be a blackout frame")
        got = np.concatenate(valids)
        # the closed form assumes the episode stays alive (t_pick == k+1); after termination t_pick freezes, which only matters for the
        # `t_pick >= 3` start condition, already satisfied here, so the closed form must hold for the whole horizon
        np.testing.assert_array_equal(got[1:], pe.scheduled_valid(T)[0][1:], "realized validity must equal the closed-form schedule through and after termination")
        self.assertGreater((~got).sum(), 20, "the schedule keeps producing blackouts after the episode ended")


if __name__ == "__main__":
    unittest.main()
