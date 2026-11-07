"""EXP-0012 benchmark: batched, partially observable 2D pick-and-place ("OccPick2D").

A point end-effector moves in the unit square, picks the *target* object (identified only by colour, given as a
task one-hot) among distractors and carries it to a fixed goal zone. Observations are 32x32 RGB top-down
renders with pixel noise plus proprioception. Partial observability comes from static occluders, camera
blackouts (frames are lost, flagged by obs_valid=0) and an unexpected displacement of the target (which drops
it if held). Layouts (occluders + goal) are drawn from disjoint integer id ranges for train/dev/test.

numpy-only, deterministic given seeds. No physics engine: this is a controlled stand-in, see the EXP-0012 record
for what it cannot establish.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RES = 32
MAX_STEP = 0.05          # max end-effector displacement per step (arena units)
PICK_R = 0.06            # pick radius
GOAL_R = 0.10            # goal half-size (square); release inside counts as success
OBJ_R = 0.05             # rendered object radius
EE_R = 0.035             # rendered end-effector radius
T_PICK = 80              # step limit per pick
N_COLORS = 6
KMAX = 5                 # max distractors
NOCC = 3
PIX_NOISE = 6.0
COLORS = np.array([[255, 60, 60], [60, 255, 60], [255, 200, 40], [200, 60, 255], [60, 200, 255], [255, 120, 0]],
                  dtype=np.float32)
BG = np.array([20, 20, 20], np.float32)
GOAL_COL = np.array([40, 70, 140], np.float32)
OCC_COL = np.array([110, 110, 110], np.float32)
GRAY = 128.0

LAYOUT_RANGES = {"train": (0, 400), "dev": (1000, 1100), "test": (2000, 2200)}


@dataclass(frozen=True)
class Cond:
    """Episode-distribution parameters."""
    name: str
    p_blackout: float      # per-step blackout start probability
    dur: tuple             # blackout duration range (inclusive low, high)
    p_disp: float          # probability a pick has a displacement event
    k: int                 # number of distractors
    n_picks: int = 1


TEST_CONDS = {
    "NOM": Cond("NOM", 0.02, (5, 15), 0.0, 2),
    "OCC": Cond("OCC", 0.05, (10, 30), 0.0, 3),
    "DIST": Cond("DIST", 0.02, (5, 15), 1.0, 2),
    "HARD": Cond("HARD", 0.05, (10, 30), 1.0, 4),          # K=4 distractors never seen in training
    "LONG": Cond("LONG", 0.03, (5, 20), 0.5, 2, n_picks=10),  # 10 sequential picks (~400 steps, 2.5x the 160-step training length), no state reset (amendment 3: was 5)
}

# Exploratory (post-hoc, added after dev showed persistent state gave no advantage under the locked conditions): longer
# blackouts so that remembering the target matters. Not part of any locked gate.
EXPLORATORY_CONDS = {
    "BLK40": Cond("BLK40", 0.04, (30, 50), 0.0, 2),
    "BLK40D": Cond("BLK40D", 0.04, (30, 50), 1.0, 2),
}
TEST_CONDS_ALL = {**TEST_CONDS, **EXPLORATORY_CONDS}


def layout_from_id(layout_id: int):
    """Deterministic layout: occluder rectangles (cx, cy, w, h) x NOCC (some may be empty) and goal centre."""
    rng = np.random.default_rng([int(layout_id), 12345])
    n = rng.integers(2, NOCC + 1)
    occ = np.zeros((NOCC, 4), np.float32)
    for i in range(n):
        occ[i] = [rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85), rng.uniform(0.18, 0.32), rng.uniform(0.18, 0.32)]
    goal = rng.uniform(0.15, 0.85, size=2).astype(np.float32)
    return occ, goal


def _grid():
    c = (np.arange(RES, dtype=np.float32) + 0.5) / RES
    X, Y = np.meshgrid(c, c)          # X[row, col] = x (col), Y = y (row)
    return X, Y


_X, _Y = _grid()


def under_occluder(pos, occ, margin=0.0):
    """pos (...,2), occ (...,NOCC,4) broadcast-compatible -> bool (...). Empty occluders (w=0) never cover."""
    dx = np.abs(pos[..., None, 0] - occ[..., :, 0])
    dy = np.abs(pos[..., None, 1] - occ[..., :, 1])
    return ((dx < occ[..., :, 2] / 2 + margin) & (dy < occ[..., :, 3] / 2 + margin) & (occ[..., :, 2] > 0)).any(-1)


class BatchEnv:
    def __init__(self, n, split, cond: Cond | list, seed, layout_ids=None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        lo, hi = LAYOUT_RANGES[split]
        self.split = split
        if layout_ids is None:
            layout_ids = self.rng.integers(lo, hi, size=n)
        self.layout_ids = np.asarray(layout_ids)
        lays = [layout_from_id(i) for i in self.layout_ids]
        self.occ = np.stack([l[0] for l in lays])
        self.goal = np.stack([l[1] for l in lays])
        conds = cond if isinstance(cond, list) else [cond] * n
        self.conds = conds
        self.p_blackout = np.array([c.p_blackout for c in conds], np.float32)
        self.dur_lo = np.array([c.dur[0] for c in conds])
        self.dur_hi = np.array([c.dur[1] for c in conds])
        self.p_disp = np.array([c.p_disp for c in conds], np.float32)
        self.k = np.array([c.k for c in conds])
        self.n_picks = np.array([c.n_picks for c in conds])
        self.reset()

    # ------------------------------------------------------------------ spawning
    def _free_point(self, i, others, min_sep=0.12, avoid_occ=True):
        for _ in range(200):
            p = self.rng.uniform(0.08, 0.92, size=2).astype(np.float32)
            if avoid_occ and under_occluder(p, self.occ[i], 0.02):
                continue
            if np.abs(p - self.goal[i]).max() < GOAL_R + 0.06:
                continue
            if all(np.linalg.norm(p - o) > min_sep for o in others):
                return p
        return p

    def reset(self):
        n = self.n
        self.ee = np.zeros((n, 2), np.float32)
        self.pos = np.zeros((n, KMAX + 1, 2), np.float32)       # index 0 = target
        self.col = np.zeros((n, KMAX + 1), np.int64)
        self.active = np.zeros((n, KMAX + 1), bool)
        self.held = np.zeros(n, bool)
        self.grip = np.zeros(n, bool)
        self.task = np.zeros(n, np.int64)
        self.t_pick = np.zeros(n, np.int64)
        self.t = 0
        self.picks_done = np.zeros(n, np.int64)
        self.done = np.zeros(n, bool)
        self.failed = np.zeros(n, bool)
        self.fail_kind = np.array([""] * n, dtype=object)
        self.failed_grasps = np.zeros(n, np.int64)
        self.black_left = np.zeros(n, np.int64)
        self.disp_at = -np.ones(n, np.int64)
        for i in range(n):
            cols = self.rng.permutation(N_COLORS)[: self.k[i] + 1]
            placed = []
            self.ee[i] = self._free_point(i, [], avoid_occ=False)
            placed.append(self.ee[i])
            for j in range(self.k[i] + 1):
                p = self._free_point(i, placed, avoid_occ=(j == 0))   # target always visible at t=0
                self.pos[i, j] = p
                self.col[i, j] = cols[j]
                self.active[i, j] = True
                placed.append(p)
            self.task[i] = self.col[i, 0]
            self._schedule_disp(i)
        return self.observe()

    def _schedule_disp(self, i):
        self.disp_at[i] = int(self.rng.integers(12, 40)) if self.rng.random() < self.p_disp[i] else -1

    def _respawn_target(self, i):
        used = set(self.col[i, 1: self.k[i] + 1].tolist())
        free = [c for c in range(N_COLORS) if c not in used]
        self.col[i, 0] = free[int(self.rng.integers(len(free)))]
        others = [self.ee[i]] + [self.pos[i, j] for j in range(1, self.k[i] + 1)]
        self.pos[i, 0] = self._free_point(i, others)
        self.task[i] = self.col[i, 0]
        self.held[i] = False
        self.t_pick[i] = 0
        self.black_left[i] = 0          # camera restored when a new target appears
        self._schedule_disp(i)

    # ------------------------------------------------------------------ dynamics
    def step(self, action):
        """action (n,3): dx, dy in [-1,1] (scaled by MAX_STEP), grip logit-like value (>0.5 = closed)."""
        a = np.asarray(action, np.float32)
        live = ~self.done
        d = a[:, :2] * MAX_STEP
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(nrm > MAX_STEP, d / np.maximum(nrm, 1e-9) * MAX_STEP, d)
        self.ee = np.where(live[:, None], np.clip(self.ee + d, 0.0, 1.0), self.ee)
        grip = (a[:, 2] > 0.5) & live
        rising = grip & ~self.grip
        released = ~grip & self.grip & self.held & live
        self.grip = np.where(live, grip, self.grip)
        # held object follows the effector
        hm = self.held & live
        self.pos[hm, 0] = self.ee[hm]
        # release
        for i in np.nonzero(released)[0]:
            self.held[i] = False
            if np.abs(self.pos[i, 0] - self.goal[i]).max() < GOAL_R:
                self.picks_done[i] += 1
                if self.picks_done[i] >= self.n_picks[i]:
                    self.done[i] = True
                else:
                    self._respawn_target(i)
        # pick attempts
        can = grip & ~self.held & ~self.done
        for i in np.nonzero(can)[0]:
            dist = np.linalg.norm(self.pos[i] - self.ee[i], axis=1)
            dist = np.where(self.active[i], dist, np.inf)
            j = int(np.argmin(dist))
            if dist[j] < PICK_R:
                if j == 0:
                    self.held[i] = True
                    self.pos[i, 0] = self.ee[i]
                else:
                    self.done[i] = True
                    self.failed[i] = True
                    self.fail_kind[i] = "wrong_pick"
            elif rising[i]:
                self.failed_grasps[i] += 1
        # unexpected displacement of the target (drops it if held)
        self.t_pick = self.t_pick + (~self.done)
        for i in np.nonzero((self.t_pick == self.disp_at) & ~self.done)[0]:
            for _ in range(100):
                ang = self.rng.uniform(0, 2 * np.pi)
                mag = self.rng.uniform(0.25, 0.40)
                p = np.clip(self.pos[i, 0] + mag * np.array([np.cos(ang), np.sin(ang)], np.float32), 0.05, 0.95)
                if not under_occluder(p, self.occ[i], 0.02) and np.abs(p - self.goal[i]).max() > GOAL_R:
                    break
            self.pos[i, 0] = p
            self.held[i] = False
        # timeouts
        to = (self.t_pick >= T_PICK) & ~self.done
        self.done |= to
        self.failed |= to
        self.fail_kind[to] = "timeout"
        # blackouts
        self.black_left = np.maximum(self.black_left - 1, 0)
        start = (self.rng.random(self.n) < self.p_blackout) & (self.black_left == 0) & (self.t_pick >= 3)
        dur = self.rng.integers(self.dur_lo, self.dur_hi + 1)
        self.black_left = np.where(start, dur, self.black_left)
        self.t += 1
        return self.observe(), self.done.copy()

    # ------------------------------------------------------------------ observation
    def render(self):
        n = self.n
        img = np.broadcast_to(BG, (n, RES, RES, 3)).copy()
        gm = (np.abs(_X[None] - self.goal[:, 0, None, None]) < GOAL_R) & (np.abs(_Y[None] - self.goal[:, 1, None, None]) < GOAL_R)
        img[gm] = GOAL_COL
        for j in range(KMAX + 1):
            m = ((_X[None] - self.pos[:, j, 0, None, None]) ** 2 + (_Y[None] - self.pos[:, j, 1, None, None]) ** 2 < OBJ_R ** 2)
            m &= self.active[:, j, None, None]
            img[m] = COLORS[self.col[:, j]][np.nonzero(m)[0]]
        for o in range(NOCC):
            w = self.occ[:, o, 2]
            m = (np.abs(_X[None] - self.occ[:, o, 0, None, None]) < w[:, None, None] / 2) & \
                (np.abs(_Y[None] - self.occ[:, o, 1, None, None]) < self.occ[:, o, 3, None, None] / 2) & (w[:, None, None] > 0)
            img[m] = OCC_COL
        em = (_X[None] - self.ee[:, 0, None, None]) ** 2 + (_Y[None] - self.ee[:, 1, None, None]) ** 2 < EE_R ** 2
        img[em] = 255.0
        img += self.rng.normal(0, PIX_NOISE, img.shape).astype(np.float32)
        return np.clip(img, 0, 255)

    def observe(self):
        img = self.render()
        valid = self.black_left == 0
        img[~valid] = GRAY
        prop = np.concatenate([self.ee, self.held[:, None].astype(np.float32), self.grip[:, None].astype(np.float32)], 1)
        task = np.eye(N_COLORS, dtype=np.float32)[self.task]
        return {"img": img.astype(np.uint8), "valid": valid, "prop": prop.astype(np.float32), "task": task}

    def privileged(self):
        return {"ee": self.ee.copy(), "target": self.pos[:, 0].copy(), "goal": self.goal.copy(), "held": self.held.copy(),
                "distract": self.pos[:, 1:].copy(), "active": self.active[:, 1:].copy()}
