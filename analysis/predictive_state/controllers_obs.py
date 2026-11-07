"""EXP-0013 Phase 1.1: OBSERVATION-MATCHED non-learned baselines for OccPick2D-LA (numpy only, nothing trained).

Inputs: ONLY the observation dict {img uint8 (n,32,32,3), valid (n,), prop (n,4)=[x, y, held, grip], task one-hot (n,6)}, i.e. exactly
what the eventual learned agent receives. They never touch `env.walls`, `env.layout_ids`, `env.pos`, `env.goal`, `env.privileged()`
or `env_la.la_move`. (tests/test_controllers_obs.py runs them through a harness that only hands over the observation.)

  ObsA  "single-frame map": stateless. Parse the CURRENT frame (nearest-colour pixel classes) into wall mask / target / goal, build a
        32x32 occupancy grid, run a shortest-path value field, take a one-step descent. On a blacked-out frame it has no information and
        holds still (gripper kept as in proprio).
  ObsB  "map + object memory": the same parser and planner, plus a compact hand-coded memory: raw wall pixels OR-accumulated over all valid
        frames, last-seen target position, union bounding box of the goal pixels. Bridges blackouts and fills holes the end-effector
        disc leaves in the wall mask. No learning, no dynamics model of the world; end-effector position comes from proprioception.

Assumptions baked in by hand (a learned agent would have to acquire them): the colour code of the classes, the action->displacement
rule (displacement = MAX_STEP * action, magnitude-capped; collisions predicted from the PARSED occupancy only), the gripper rule of the
expert (close within CLOSE_R of the target estimate, release within RELEASE_R of the goal estimate). The only tuned parameter is the
wall dilation radius DIL (0 or 1 px), selected on the TRAIN layout range (run_phase11_benchmark.py select_dilation).
"""
from __future__ import annotations

import heapq

import numpy as np

from . import env as E
from . import env_la as L
from . import expert as X

R = E.RES
SPEED = X.SPEED
C_BG, C_GOAL, C_OCC, C_WALL, C_EE = 0, 1, 2, 3, 4          # class ids; target colours are 5 + colour index
CLASS_COLS = np.stack([E.BG, E.GOAL_COL, E.OCC_COL, L.WALL_COL, np.array([255.0, 255, 255], np.float32)] + list(E.COLORS)).astype(np.float32)
PX = ((np.arange(R) + 0.5) / R).astype(np.float32)
DIRS16 = np.stack([np.cos(np.linspace(0, 2 * np.pi, 16, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False))], -1).astype(np.float32)
NB8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]


def classify(img):
    """img (...,32,32,3) uint8 -> nearest-centroid class id (...,32,32)."""
    d = ((img[..., None, :].astype(np.float32) - CLASS_COLS) ** 2).sum(-1)
    return d.argmin(-1)


def dilate(m, r):
    if r <= 0:
        return m.copy()
    p = np.pad(m, r)
    out = np.zeros_like(m)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= p[r + dy:r + dy + R, r + dx:r + dx + R]
    return out


_PXX, _PXY = np.meshgrid(PX, PX)
_PXY2 = np.stack([_PXX.ravel(), _PXY.ravel()], -1)                # pixel centres, row-major (row = y)
_OFF = np.linspace(-0.06, 0.06, 17)
_OFFS = np.stack(np.meshgrid(_OFF, _OFF), -1).reshape(-1, 2)


def fit_disc(tm, ee_mask, r=E.OBJ_R):
    """Target centre by template fit. The end-effector disc is drawn over the target, so the visible target pixels are a crescent
    whose centroid is biased away from the end-effector (measured on TRAIN: the controller then oscillates around the target without
    ever getting within CLOSE_R of its estimate). Fit the known object radius instead: score = visible target pixels inside the disc
    minus visible non-target, non-end-effector pixels inside it; ties are averaged. The object radius is a hand-coded prior."""
    ys, xs = np.nonzero(tm)
    c0 = np.array([PX[xs].mean(), PX[ys].mean()])
    cand = c0[None] + _OFFS
    inside = np.linalg.norm(cand[:, None, :] - _PXY2[None], axis=-1) < r                  # (K, R*R)
    tmf, eef = tm.ravel(), ee_mask.ravel()
    score = (inside & tmf[None]).sum(1) - (inside & ~tmf[None] & ~eef[None]).sum(1)
    best = score == score.max()
    return cand[best].mean(0).astype(np.float32)


def parse(img, task_id):
    """One frame -> dict(wall bool (R,R)[row=y,col=x], tgt (2,)|None, goal_bbox (x0,x1,y0,y1)|None)."""
    cls = classify(img)
    wall_raw = cls == C_WALL
    ee = cls == C_EE
    wall = wall_raw | (ee & dilate(wall_raw, 1))               # the end-effector disc overdraws wall pixels: fill holes next to a wall
    tm = cls == 5 + task_id
    tgt = fit_disc(tm, ee) if tm.sum() >= 2 else None
    gm = cls == C_GOAL
    gb = None
    if gm.sum() >= 4:
        ys, xs = np.nonzero(gm)
        gb = (PX[xs.min()], PX[xs.max()], PX[ys.min()], PX[ys.max()])
    return {"wall": wall, "wall_raw": wall_raw, "tgt": tgt, "goal_bbox": gb}


def bbox_center(gb):
    return np.array([(gb[0] + gb[1]) / 2, (gb[2] + gb[3]) / 2], np.float32)


def geodesic(blocked, cell):
    """8-connected shortest-path distance (arena units) on the RxR grid from `cell`=(iy,ix); no corner cutting; inf where unreachable."""
    iy, ix = cell
    dist = np.full((R, R), np.inf, np.float32)
    dist[iy, ix] = 0.0
    pq = [(0.0, iy, ix)]
    s = 1.0 / R
    while pq:
        d0, y, x = heapq.heappop(pq)
        if d0 > dist[y, x]:
            continue
        for dy, dx, c in NB8:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < R and 0 <= xx < R) or blocked[yy, xx]:
                continue
            if dy and dx and (blocked[y + dy, x] or blocked[y, x + dx]):
                continue
            nd = d0 + c * s
            if nd < dist[yy, xx]:
                dist[yy, xx] = nd
                heapq.heappush(pq, (nd, yy, xx))
    return dist


def cell_of(p):
    return int(np.clip(p[1] * R, 0, R - 1)), int(np.clip(p[0] * R, 0, R - 1))


def nearest_free(blocked, cell):
    if not blocked[cell]:
        return cell
    best, bd = cell, 1e9
    for y in range(R):
        for x in range(R):
            if not blocked[y, x]:
                d = (y - cell[0]) ** 2 + (x - cell[1]) ** 2
                if d < bd:
                    best, bd = (y, x), d
    return best


def value(field, p):
    """windowed lookup: min over cells within 2 px of (field + Euclid distance to the cell centre); inf cells ignored."""
    iy, ix = cell_of(p)
    best, bc = np.inf, None
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            yy, xx = iy + dy, ix + dx
            if 0 <= yy < R and 0 <= xx < R and np.isfinite(field[yy, xx]):
                c = np.array([PX[xx], PX[yy]], np.float32)
                v = field[yy, xx] + float(np.linalg.norm(p - c))
                if v < best:
                    best, bc = v, c
    return best, bc


def line_clear(wall, a, b, step=0.015):
    k = max(int(np.linalg.norm(b - a) / step), 1)
    pts = a[None] + (b - a)[None] * (np.arange(1, k + 1)[:, None] / k)
    return not wall[np.clip((pts[:, 1] * R).astype(int), 0, R - 1), np.clip((pts[:, 0] * R).astype(int), 0, R - 1)].any()


class _ObsBase:
    uses_memory = False

    def __init__(self, dil=1):
        self.dil = dil

    def reset(self, n):
        self.n = n
        self.cache = {}
        self.mem_wall = [None] * n
        self.mem_tgt = [None] * n
        self.mem_gb = [None] * n
        self.effort = {"decisions": 0, "dijkstra": 0, "parses": 0}

    def _field(self, blocked, wp):
        key = (blocked.tobytes(), cell_of(wp))
        if key not in self.cache:
            self.cache[key] = geodesic(blocked, nearest_free(blocked, cell_of(wp)))
            self.effort["dijkstra"] += 1
        return self.cache[key]

    def _plan_move(self, ee, wp, wall_raw):
        """one-step descent on the parsed-map value; returns action xy in action units."""
        blocked = dilate(wall_raw, self.dil)
        field = self._field(blocked, wp)
        vec = wp - ee
        dist = float(np.linalg.norm(vec))
        if dist < SPEED * E.MAX_STEP and line_clear(wall_raw, ee, wp):
            return vec / E.MAX_STEP
        direct = vec / max(dist, 1e-9) * min(dist / E.MAX_STEP, SPEED)
        cands = [DIRS16[i] * SPEED for i in range(16)] + [direct]
        _, tc = value(field, ee)
        if tc is not None:
            tv = tc - ee
            nv = float(np.linalg.norm(tv))
            cands.append(tv / max(nv, 1e-9) * min(nv / E.MAX_STEP, SPEED))
        best, ba = np.inf, np.zeros(2, np.float32)
        for a in cands:
            p2 = np.clip(ee + a * E.MAX_STEP, 0, 1)
            if wall_raw[cell_of(p2)]:
                continue
            v, _ = value(field, p2)
            v += 1e-3 * float(np.linalg.norm(p2 - wp))
            if v < best:
                best, ba = v, a
        return ba

    def _decide(self, i, ee, held, grip, tgt, goal, wall_raw):
        out = np.zeros(3, np.float32)
        out[2] = float(grip)
        self.effort["decisions"] += 1
        wp = goal if held else tgt
        if wp is None or wall_raw is None:
            return out                                           # nothing known: hold still, keep gripper state
        near_t = tgt is not None and float(np.linalg.norm(tgt - ee)) < X.CLOSE_R
        near_g = goal is not None and float(np.linalg.norm(goal - ee)) < X.RELEASE_R
        out[2] = float((not near_g) if held else near_t)
        out[:2] = self._plan_move(ee, wp, wall_raw)
        nrm = float(np.linalg.norm(out[:2]))
        if nrm > SPEED:
            out[:2] *= SPEED / nrm
        return out

    def act(self, obs, alive):
        n = self.n
        out = np.zeros((n, 3), np.float32)
        for i in range(n):
            if not alive[i]:
                continue
            out[i] = self.act_one(i, obs["img"][i], bool(obs["valid"][i]), obs["prop"][i], int(obs["task"][i].argmax()))
        return out


class ObsA(_ObsBase):
    name = "obs_A_single_frame"
    uses_memory = False

    def act_one(self, i, img, valid, prop, task_id):
        ee, held, grip = prop[:2], prop[2] > 0.5, prop[3] > 0.5
        if not valid:
            self.effort["decisions"] += 1
            return np.array([0, 0, float(grip)], np.float32)
        self.effort["parses"] += 1
        pr = parse(img, task_id)
        goal = bbox_center(pr["goal_bbox"]) if pr["goal_bbox"] is not None else None
        tgt = ee if held else pr["tgt"]
        return self._decide(i, ee, held, grip, tgt, goal, pr["wall"])


class ObsB(_ObsBase):
    name = "obs_B_map_memory"
    uses_memory = True

    def act_one(self, i, img, valid, prop, task_id):
        ee, held, grip = prop[:2], prop[2] > 0.5, prop[3] > 0.5
        if valid:
            self.effort["parses"] += 1
            pr = parse(img, task_id)
            # memory stores the RAW wall class only (never the hole-filled mask): a filled effector ghost would thicken walls for ever
            self.mem_wall[i] = pr["wall_raw"] if self.mem_wall[i] is None else (self.mem_wall[i] | pr["wall_raw"])
            if pr["tgt"] is not None and not held:
                self.mem_tgt[i] = pr["tgt"]
            if pr["goal_bbox"] is not None:
                g = pr["goal_bbox"]
                m = self.mem_gb[i]
                self.mem_gb[i] = g if m is None else (min(m[0], g[0]), max(m[1], g[1]), min(m[2], g[2]), max(m[3], g[3]))
        goal = bbox_center(self.mem_gb[i]) if self.mem_gb[i] is not None else None
        tgt = ee if held else self.mem_tgt[i]
        plan_wall = self.mem_wall[i]
        if valid and plan_wall is not None:
            plan_wall = plan_wall | pr["wall"]                  # this frame's hole-filled mask is used for planning now, never remembered
        return self._decide(i, ee, held, grip, tgt, goal, plan_wall)


def run_episodes_obs(env, ctrl, T=None):
    """Closed loop with an observation-only controller. Returns the same dict shape as controllers_la.run_episodes."""
    T = T or (env.t_limit + 2)
    obs = env.reset()
    ctrl.reset(env.n)
    finish = np.full(env.n, -1)
    for t in range(T):
        if env.done.all():
            break
        a = ctrl.act(obs, ~env.done)
        obs, done = env.step(a)
        newly = done & (finish < 0)
        finish[newly] = env.t
    ok = env.picks_done >= env.n_picks
    return {"success": ok.astype(np.float32), "finish": finish, "fail_kind": env.fail_kind.copy(), "effort": dict(ctrl.effort)}
