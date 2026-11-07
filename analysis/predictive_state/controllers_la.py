"""EXP-0013 controllers for OccPick2D-LA. numpy-only. Diagnostic controls, NOT the proposed learned architecture.

INFORMATION AVAILABLE (see docs/experiments/EXP-0013-phase1.md for the table):
  greedy / reactive wall-follower : true end-effector, target, goal and held flag (privileged); walls ONLY through ray sensing
                                    of range <= RAY_R along a few directions (local sensing, no map).
  oracle shortest-path descent    : everything above plus the full wall map (via the geodesic field) -- a planner.
  oracle random-shooting MPC      : the same map and the true dynamics `la_move`, evaluated H steps ahead for N sampled plans.
The future learned controller gets only: 32x32 image (walls visible, blackouts), proprioception (x, y, held, grip), task colour.
All controllers use the expert's gripper rule (close within CLOSE_R of the target; hold until within RELEASE_R of the goal).
"""
from __future__ import annotations

import numpy as np

from . import env as E
from . import env_la as L
from . import expert as X

SPEED = X.SPEED
RAY_R = 0.25
DIRS16 = np.stack([np.cos(np.linspace(0, 2 * np.pi, 16, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False))], -1).astype(np.float32)


def _waypoint_and_grip(env):
    tgt, goal = env.pos[:, 0], env.goal
    wp = np.where(env.held[:, None], goal, tgt)
    near_t = np.linalg.norm(tgt - env.ee, axis=1) < X.CLOSE_R
    near_g = np.linalg.norm(goal - env.ee, axis=1) < X.RELEASE_R
    g = np.where(env.held, ~near_g, near_t).astype(np.float32)
    return wp, g


def _direct(env, wp):
    vec = wp - env.ee
    dist = np.linalg.norm(vec, axis=1, keepdims=True)
    step = np.minimum(dist, SPEED * E.MAX_STEP)
    return vec / np.maximum(dist, 1e-9) * step / E.MAX_STEP, dist[:, 0]


class Greedy:
    """C1: straight toward the current waypoint; the environment makes it slide along walls. No sensing, no memory."""
    name = "greedy"

    def reset(self, env):
        self.effort = {"sim_steps": 0, "decisions": 0, "expanded": 0}

    def act(self, env):
        wp, g = _waypoint_and_grip(env)
        d, _ = _direct(env, wp)
        return np.concatenate([d, g[:, None]], 1).astype(np.float32)


def ray_len(walls, p, u, maxlen, step=0.02):
    k = np.arange(1, int(maxlen / step) + 1)[:, None]
    pts = p[None] + k * step * u[None]
    hit = L.inside(np.clip(pts, 0, 1), walls) | (pts[:, 0] < 0) | (pts[:, 0] > 1) | (pts[:, 1] < 0) | (pts[:, 1] > 1)
    return maxlen if not hit.any() else float(k[np.argmax(hit), 0] * step)


class Reactive:
    """C2/C3: wall-following with local ray sensing. DIRECT toward the waypoint while a 0.10 ray is free; on a block, FOLLOW the
    wall along y (side with the longer free ray), return to DIRECT when the way ahead is free again. The tangent direction (y) is
    tailored to this benchmark's vertical divider -- deliberately generous to the baseline. retreat=True (C3) adds memory: after two
    side flips (boxed in, e.g. inside the decoy pocket) it backs out for RETREAT_N steps, takes the opposite side and ignores DIRECT
    for TABU steps so it does not re-enter the same gap."""
    RETREAT_N, TABU = 10, 14

    def __init__(self, retreat):
        self.retreat = retreat
        self.name = "reactive_memory" if retreat else "reactive"

    def reset(self, env):
        n = env.n
        self.mode = np.zeros(n, int); self.side = np.ones(n, int); self.follow_n = np.zeros(n, int); self.flips = np.zeros(n, int)
        self.ret_n = np.zeros(n, int); self.tabu = np.zeros(n, int); self.u0 = np.zeros((n, 2), np.float32)
        self.first_side = np.ones(n, int); self.prev_held = np.zeros(n, bool)
        self.effort = {"sim_steps": 0, "decisions": 0, "expanded": 0, "ray_casts": 0}

    def act(self, env):
        n = env.n
        wp, g = _waypoint_and_grip(env)
        out = np.zeros((n, 3), np.float32)
        out[:, 2] = g
        for i in range(n):
            if env.done[i]:
                continue
            if env.held[i] != self.prev_held[i]:
                self.mode[i], self.tabu[i], self.flips[i] = 0, 0, 0
                self.prev_held[i] = env.held[i]
            walls, p = env.walls[i], env.ee[i]
            vec = wp[i] - p
            dist = float(np.linalg.norm(vec))
            u = (vec / max(dist, 1e-9)).astype(np.float32)
            self.effort["decisions"] += 1
            if self.tabu[i] > 0:
                self.tabu[i] -= 1
            move = None
            if self.mode[i] == 2:
                move = -self.u0[i]
                self.ret_n[i] -= 1
                if self.ret_n[i] <= 0:
                    self.mode[i], self.side[i], self.tabu[i], self.flips[i], self.follow_n[i] = 1, -self.first_side[i], self.TABU, 0, 0
            else:
                self.effort["ray_casts"] += 1
                clear = ray_len(walls, p, u, min(0.10, max(dist, 0.02))) >= min(0.10, max(dist, 0.02)) - 1e-6
                if self.mode[i] == 0:
                    if clear:
                        move = u * min(SPEED, dist / E.MAX_STEP)
                    else:
                        cu = ray_len(walls, p, np.array([0, 1], np.float32), RAY_R)
                        cd = ray_len(walls, p, np.array([0, -1], np.float32), RAY_R)
                        self.effort["ray_casts"] += 2
                        self.mode[i], self.side[i], self.follow_n[i], self.flips[i] = 1, (1 if cu >= cd else -1), 0, 0
                        self.first_side[i], self.u0[i] = self.side[i], u
                if self.mode[i] == 1:
                    tang = np.array([0, self.side[i]], np.float32)
                    self.effort["ray_casts"] += 1
                    if ray_len(walls, p, tang, 0.06) < 0.05:
                        self.side[i] = -self.side[i]
                        self.flips[i] += 1
                        self.follow_n[i] = 0
                        tang = np.array([0, self.side[i]], np.float32)
                    if self.retreat and self.flips[i] >= 2:
                        self.mode[i], self.ret_n[i] = 2, self.RETREAT_N
                        move = -self.u0[i]
                    elif self.tabu[i] == 0 and self.follow_n[i] >= 3 and clear:
                        self.mode[i] = 0
                        move = u * min(SPEED, dist / E.MAX_STEP)
                    else:
                        move = tang * SPEED
                        self.follow_n[i] += 1
            nrm = float(np.linalg.norm(move))
            if nrm > SPEED:
                move = move / nrm * SPEED
            out[i, :2] = move
        return out


class Oracle:
    """Planning with TRUE dynamics (`env_la.la_move`) and the true wall map. Diagnostic control.
    kind='descent' : H=1, 16 headings + the direct step, cost = geodesic distance (shortest-path reference).
    kind='shoot'   : random shooting, N plans of H steps (heading random-walk, sigma 0.5 rad; plan 0 = direct step repeated),
                     cost = distance of the final position to the waypoint: 'euclid' or 'geodesic'.
    Final approach (waypoint within one step and a clear straight line) is a direct move, not planned."""

    def __init__(self, kind="descent", H=1, N=64, cost="geodesic", seed=0):
        self.kind, self.H, self.N, self.cost = kind, H, N, cost
        self.rng = np.random.default_rng(seed)
        self.name = "oracle_descent" if kind == "descent" else f"oracle_shoot_{cost}_H{H}"

    def reset(self, env):
        self.field = [None] * env.n
        self.key = [None] * env.n
        self.effort = {"sim_steps": 0, "decisions": 0, "expanded": 0}

    def _field(self, env, i, wp):
        key = (bool(env.held[i]), tuple(np.round(wp, 3)))
        if self.key[i] != key:
            f, ex, _ = L.geodesic_field(env.layout_ids[i], wp)
            self.field[i], self.key[i] = f, key
            self.effort["expanded"] += ex
        return self.field[i]

    def act(self, env):
        n = env.n
        wp, g = _waypoint_and_grip(env)
        direct, dist = _direct(env, wp)
        out = np.zeros((n, 3), np.float32)
        out[:, 2] = g
        live = ~env.done
        if not live.any():
            return out
        fields = np.stack([self._field(env, i, wp[i]) if live[i] else np.zeros((L.G, L.G), np.float32) for i in range(n)])
        walls = env.walls[:, None]                                    # (n,1,NW,4)
        if self.kind == "descent":
            # extra candidate: toward the free field cell that minimises value + distance (descent guarantee next to a wall)
            _, tgt_c = L.lookup(fields, env.ee, want_cell=True)
            toc = tgt_c - env.ee
            toc = toc / np.maximum(np.linalg.norm(toc, axis=1, keepdims=True), 1e-9) * np.minimum(np.linalg.norm(toc, axis=1, keepdims=True) / E.MAX_STEP, SPEED)
            seqs = np.concatenate([np.broadcast_to(DIRS16[None, :, None, :] * SPEED, (n, 16, 1, 2)), direct[:, None, None, :], toc[:, None, None, :]], 1)   # (n,18,1,2)
        else:
            N, H = self.N, self.H
            ang0 = self.rng.uniform(0, 2 * np.pi, (n, N, 1))
            ang = ang0 + np.cumsum(self.rng.normal(0, 0.5, (n, N, H)), -1) * (np.arange(H) > 0)
            seqs = np.stack([np.cos(ang), np.sin(ang)], -1).astype(np.float32) * SPEED
            seqs[:, 0] = direct[:, None, :]                          # plan 0: the greedy direct step repeated
        K, Hh = seqs.shape[1], seqs.shape[2]
        pos = np.broadcast_to(env.ee[:, None, :], (n, K, 2)).copy()
        for h in range(Hh):
            pos = L.la_move(pos, L.cap_step(seqs[:, :, h]), walls)
        self.effort["sim_steps"] += int(live.sum()) * K * Hh
        self.effort["decisions"] += int(live.sum())
        if self.cost == "euclid" and self.kind == "shoot":
            cost = np.linalg.norm(pos - wp[:, None, :], axis=-1)
        else:
            cost = L.lookup(fields, pos)
            cost = cost + 1e-3 * np.linalg.norm(pos - wp[:, None, :], axis=-1)      # tie-break inside one grid cell
        best = np.argmin(cost, axis=1)
        first = seqs[np.arange(n), best, 0]                       # action units (cap_step scales by MAX_STEP)
        clear_direct = np.array([ray_len(env.walls[i], env.ee[i], direct[i] / max(np.linalg.norm(direct[i]), 1e-9), max(dist[i], 0.02))
                                 >= max(dist[i], 0.02) - 1e-6 if (live[i] and dist[i] < SPEED * E.MAX_STEP) else False for i in range(n)])
        first = np.where(clear_direct[:, None], direct, first)
        out[:, :2] = first
        return out


def run_episodes(env, ctrl, T=None, collect_paths=False):
    """Closed loop with a privileged controller. Returns per-episode dict (success, finish step, failure kind, effort)."""
    T = T or (env.t_limit + 2)
    ctrl.reset(env)
    env.reset()
    finish = np.full(env.n, -1)
    for t in range(T):
        if env.done.all():
            break
        a = ctrl.act(env)
        _, done = env.step(a)
        newly = done & (finish < 0)
        finish[newly] = env.t
    ok = env.picks_done >= env.n_picks
    return {"success": ok.astype(np.float32), "finish": finish, "fail_kind": env.fail_kind.copy(), "effort": dict(ctrl.effort)}
