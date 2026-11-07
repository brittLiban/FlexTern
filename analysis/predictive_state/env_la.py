"""EXP-0013 benchmark OccPick2D-LA (v1): the EXP-0012 pick-and-place task plus blocking walls and a decoy dead-end.

numpy-only and deterministic given seeds. `env.py` is NOT modified; `LAEnv` subclasses `env.BatchEnv`.

Geometry (per layout id, `la_layout`): a vertical divider wall at x ~ 0.40 splits the arena. The end-effector, target and
distractors start on the left, the goal zone is on the right, so every successful pick-and-place must cross the divider. The
divider has TWO gaps: an open gap, and a decoy gap that opens into a closed pocket (three more wall pieces). Walls are 0.08 thick
(>= 2 pixels at 32x32), drawn in a colour distinct from every other class, and painted over occluders, so they are always visible
(blackouts still blank the whole frame). The end-effector cannot enter a wall (per-axis blocking, i.e. it slides along it).

Dynamics (single source of truth, shared with the oracle planners): `la_move`. The step budget is `LA_T_PICK`.
Displacement events are not supported in v1 (a teleported target could land in or across a wall): p_disp must be 0.
"""
from __future__ import annotations

import heapq
import numpy as np

from . import env as E

NWALL = 6
WALL_T = 0.08
GAP_H = 0.08                      # gap half-height
POCKET_D = 0.20                   # pocket depth to the right of the decoy gap
WALL_COL = np.array([160.0, 110.0, 70.0], np.float32)
LA_T_PICK = 65                    # step budget per pick; LOCKED = ceil(1.5 x p95 of oracle finish steps on the TRAIN range), results/predictive_state_phase1/la_calibration.json

LA_CONDS = {
    "LA_NOM": E.Cond("LA_NOM", 0.02, (5, 15), 0.0, 2),
    "LA_OCC": E.Cond("LA_OCC", 0.05, (10, 30), 0.0, 3),
}


def la_layout(layout_id: int):
    """Deterministic layout -> (walls (NWALL,4) [cx, cy, w, h], goal (2,), info dict)."""
    rng = np.random.default_rng([int(layout_id), 4242])
    xw = rng.uniform(0.38, 0.42)
    lower = bool(rng.random() < 0.5)                         # open gap below (True) or above the decoy
    if lower:
        dec = rng.uniform(0.50, 0.62)
        opn = rng.uniform(0.10, dec - 0.27)
    else:
        dec = rng.uniform(0.38, 0.50)
        opn = rng.uniform(dec + 0.27, 0.90)
    gaps = sorted([(opn, "open"), (dec, "decoy")])
    pieces, lo = [], 0.0
    for g, _ in gaps:
        pieces.append((lo, g - GAP_H))
        lo = g + GAP_H
    pieces.append((lo, 1.0))
    walls = np.zeros((NWALL, 4), np.float32)
    k = 0
    for a, b in pieces:
        walls[k] = [xw, (a + b) / 2, WALL_T, b - a]
        k += 1
    x0, x1 = xw, xw + WALL_T / 2 + POCKET_D + WALL_T
    walls[k] = [(x0 + x1) / 2, dec + GAP_H + WALL_T / 2, x1 - x0, WALL_T]; k += 1          # pocket top
    walls[k] = [(x0 + x1) / 2, dec - GAP_H - WALL_T / 2, x1 - x0, WALL_T]; k += 1          # pocket bottom
    walls[k] = [x1 - WALL_T / 2, dec, WALL_T, 2 * GAP_H + 2 * WALL_T]                      # pocket back wall
    goal = np.array([rng.uniform(0.86, 0.90), rng.uniform(0.15, 0.85)], np.float32)
    info = {"xw": xw, "decoy": dec, "open": opn, "open_below": lower}
    return walls, goal, info


def _la_layout_full(layout_id):
    return la_layout(layout_id)


NWALL_ALL = NWALL


def walls_of(layout_id):
    return la_layout(layout_id)[0]


def inside(p, walls, margin=0.0):
    """p (..., 2); walls (..., NWALL_ALL, 4) broadcast-compatible with p[..., None, :]. -> bool (...)."""
    dx = np.abs(p[..., None, 0] - walls[..., :, 0])
    dy = np.abs(p[..., None, 1] - walls[..., :, 1])
    return ((dx < walls[..., :, 2] / 2 + margin) & (dy < walls[..., :, 3] / 2 + margin) & (walls[..., :, 2] > 0)).any(-1)


def la_move(ee, d, walls):
    """True end-effector dynamics. ee (...,2), d (...,2) displacement (already magnitude-capped), walls broadcastable.
    Axis-separated blocking: a blocked axis keeps its old coordinate (the effector slides along the wall)."""
    nx = np.clip(ee[..., 0] + d[..., 0], 0.0, 1.0)
    bx = inside(np.stack([nx, ee[..., 1]], -1), walls)
    x = np.where(bx, ee[..., 0], nx)
    ny = np.clip(ee[..., 1] + d[..., 1], 0.0, 1.0)
    by = inside(np.stack([x, ny], -1), walls)
    y = np.where(by, ee[..., 1], ny)
    return np.stack([x, y], -1).astype(np.float32)


def cap_step(a_xy):
    d = a_xy * E.MAX_STEP
    nrm = np.linalg.norm(d, axis=-1, keepdims=True)
    return np.where(nrm > E.MAX_STEP, d / np.maximum(nrm, 1e-9) * E.MAX_STEP, d)


class LAEnv(E.BatchEnv):
    def __init__(self, n, split, cond, seed, layout_ids=None, t_limit=None):
        conds = cond if isinstance(cond, list) else [cond] * n
        if any(c.p_disp != 0 for c in conds):
            raise ValueError("OccPick2D-LA v1 does not support displacement events (p_disp must be 0)")
        lo, hi = E.LAYOUT_RANGES[split]
        if layout_ids is None:
            layout_ids = np.random.default_rng([int(seed), 777]).integers(lo, hi, size=n)
        lays = [_la_layout_full(i) for i in layout_ids]
        self.walls = np.stack([l[0] for l in lays])
        self.goal_la = np.stack([l[1] for l in lays])
        self.la_info = [l[2] for l in lays]
        self.t_limit = LA_T_PICK if t_limit is None else t_limit
        super().__init__(n, split, cond, seed, layout_ids=layout_ids)

    def reset(self):
        self.goal = self.goal_la.copy()
        return super().reset()

    def _free_point(self, i, others, min_sep=0.12, avoid_occ=True):
        xw = self.la_info[i]["xw"]
        for _ in range(500):
            p = np.array([self.rng.uniform(0.08, xw - 0.10), self.rng.uniform(0.08, 0.92)], np.float32)
            if avoid_occ and E.under_occluder(p, self.occ[i], 0.02):
                continue
            if inside(p, self.walls[i], 0.04):
                continue
            if all(np.linalg.norm(p - o) > min_sep for o in others):
                return p
        return p

    def step(self, action):
        a = np.array(action, np.float32)
        live = ~self.done
        d = cap_step(a[:, :2])
        new = la_move(self.ee, d, self.walls)
        a[:, :2] = np.where(live[:, None], (new - self.ee) / E.MAX_STEP, 0.0)
        obs, _ = super().step(a)
        to = (self.t_pick >= self.t_limit) & ~self.done
        if to.any():
            self.done |= to
            self.failed |= to
            self.fail_kind[to] = "timeout"
        return obs, self.done.copy()

    def render(self):
        img = super().render()
        n = self.n
        for k in range(NWALL_ALL):
            w = self.walls[:, k, 2]
            m = (np.abs(E._X[None] - self.walls[:, k, 0, None, None]) < w[:, None, None] / 2) & \
                (np.abs(E._Y[None] - self.walls[:, k, 1, None, None]) < self.walls[:, k, 3, None, None] / 2) & (w[:, None, None] > 0)
            img[m] = np.clip(WALL_COL + self.rng.normal(0, E.PIX_NOISE, (int(m.sum()), 3)), 0, 255)
        em = (E._X[None] - self.ee[:, 0, None, None]) ** 2 + (E._Y[None] - self.ee[:, 1, None, None]) ** 2 < E.EE_R ** 2
        img[em] = np.clip(255.0 + self.rng.normal(0, E.PIX_NOISE, (int(em.sum()), 3)), 0, 255)
        return img

    def privileged(self):
        pv = super().privileged()
        pv["walls"] = self.walls.copy()
        return pv


# ---------------------------------------------------------------------------------------------------- geodesic field
G = 64
BIG = 10.0
WIN = 2                          # lookup window radius in cells
FIELD_MARGIN = 0.02              # field cells within this distance of a wall are treated as blocked (absorbs grid quantization)
_C = (np.arange(G) + 0.5) / G
_CX, _CY = np.meshgrid(_C, _C)               # [row=y, col=x]
_FIELDS: dict = {}


def geodesic_field(layout_id, point):
    """Shortest free-space distance (arena units) from every grid cell centre to `point` (8-connected Dijkstra, no corner cutting).
    Cells whose centre lies inside a wall are filled from their neighbours so positions hugging a wall get finite values."""
    ix = int(np.clip(point[0] * G, 0, G - 1)); iy = int(np.clip(point[1] * G, 0, G - 1))
    key = (int(layout_id), iy, ix)
    if key in _FIELDS:
        return _FIELDS[key]
    walls = walls_of(layout_id)
    blocked = inside(np.stack([_CX, _CY], -1), walls, FIELD_MARGIN)
    dist = np.full((G, G), np.inf)
    dist[iy, ix] = 0.0
    pq = [(0.0, iy, ix)]
    step = 1.0 / G
    nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]
    expanded = 0
    while pq:
        d0, y, x = heapq.heappop(pq)
        if d0 > dist[y, x]:
            continue
        expanded += 1
        for dy, dx, c in nb:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < G and 0 <= xx < G) or blocked[yy, xx]:
                continue
            if dy and dx and (blocked[y + dy, x] or blocked[y, x + dx]):
                continue
            nd = d0 + c * step
            if nd < dist[yy, xx]:
                dist[yy, xx] = nd
                heapq.heappush(pq, (nd, yy, xx))
    f = np.where(np.isfinite(dist), dist, BIG)
    _FIELDS[key] = (f.astype(np.float32), expanded, dist)
    return _FIELDS[key]


def lookup(field, p, want_cell=False):
    """Value of the field at a continuous position: min over free cells within WIN cells of (cell value + Euclidean distance to the
    cell centre). A position hugging a wall (inside the blocked margin) therefore reads the nearest free cells, never a value
    propagated through the wall. field (G,G) or (n,G,G) with p (n,...,2) leading dim n; p (...,2) for a single field."""
    cy = np.clip((p[..., 1] * G).astype(int), 0, G - 1)
    cx = np.clip((p[..., 0] * G).astype(int), 0, G - 1)
    best, bc = None, None
    for dy in range(-WIN, WIN + 1):
        for dx in range(-WIN, WIN + 1):
            yy, xx = np.clip(cy + dy, 0, G - 1), np.clip(cx + dx, 0, G - 1)
            if field.ndim == 3:
                idx = (np.arange(field.shape[0]).reshape((-1,) + (1,) * (yy.ndim - 1)), yy, xx)
                v = field[idx]
            else:
                v = field[yy, xx]
            ctr = np.stack([(xx + 0.5) / G, (yy + 0.5) / G], -1)
            v = v + np.linalg.norm(p - ctr, axis=-1)
            if best is None:
                best, bc = v, ctr
            else:
                m = v < best
                best = np.where(m, v, best)
                bc = np.where(m[..., None], ctr, bc)
    return (best, bc) if want_cell else best


def reachable(layout_id, a, b):
    """True if the grid shortest-path distance between two points is finite (free-space connectivity)."""
    _, _, raw = geodesic_field(layout_id, b)
    return bool(np.isfinite(raw[int(np.clip(a[1] * G, 0, G - 1)), int(np.clip(a[0] * G, 0, G - 1))]))
