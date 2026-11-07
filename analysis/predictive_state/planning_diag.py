"""EXP-0013 Phase 1.1: planning-horizon diagnostic for OccPick2D-LA. PRIVILEGED controls (true dynamics `la_move`, true wall map).

Question: why does random-shooting with a geodesic terminal cost degrade from H=1 to H=5/10 (Phase 1: 1.00 -> 0.55 -> 0.02)?
Candidate causes, each isolated by one planner/option:
  sampling failure        rs_walk (Phase-1 parametrisation: heading random walk, sigma 0.5 rad) vs rs_const (constant-heading plans)
                          vs cem (iterative refit) vs beam (deterministic expansion of 16 headings + direct step)
  objective scaling       terminal cost only (agg='terminal') vs summed per-step cost (agg='sum')
  terminal value          cost = geodesic (exact cost-to-go: a terminal value) vs Euclid (no map-derived value)
  computational budget    every planner is run at a stated simulation budget B (sim steps per decision); N or W are derived from B
Receding horizon is common to all: the first action of the best plan is executed, then everything is replanned.
Final-approach rule (same as controllers_la.Oracle): waypoint within one step and straight line clear -> direct move.
"""
from __future__ import annotations

import numpy as np

from . import controllers_la as C
from . import env as E
from . import env_la as L

SPEED = C.SPEED
DIRS16 = C.DIRS16


class PlanDiag(C.Oracle):
    def __init__(self, kind, H, B, cost="geodesic", agg="terminal", seed=0, iters=4, elite_frac=0.125, sigma=0.5):
        super().__init__("shoot", H=H, N=1, cost=cost, seed=seed)
        self.pkind, self.B, self.agg, self.iters, self.elite_frac, self.sigma = kind, B, agg, iters, elite_frac, sigma
        self.name = f"{kind}|{cost}|{agg}|H{H}|B{B}"

    # ------------------------------------------------------------------------------------------ helpers
    def _cost_fn(self, fields, wp):
        if self.cost == "euclid":
            return lambda pos: np.linalg.norm(pos - wp[:, None, :], axis=-1)
        return lambda pos: L.lookup(fields, pos) + 1e-3 * np.linalg.norm(pos - wp[:, None, :], axis=-1)

    @staticmethod
    def _roll(pos, seq_h, walls):
        return L.la_move(pos, L.cap_step(seq_h), walls)

    def _eval(self, seqs, ee, walls, cost_fn):
        """seqs (n,K,H,2) action units -> (final-or-aggregated cost (n,K), final positions)."""
        n, K, H, _ = seqs.shape
        pos = np.broadcast_to(ee[:, None, :], (n, K, 2)).copy()
        acc = None
        for h in range(H):
            pos = self._roll(pos, seqs[:, :, h], walls)
            if self.agg == "sum":
                c = cost_fn(pos)
                acc = c if acc is None else acc + c
        self.effort["sim_steps"] += self._nl * K * H
        return (acc / H if self.agg == "sum" else cost_fn(pos)), pos

    # ------------------------------------------------------------------------------------------ planners
    def act(self, env):
        n = env.n
        wp, g = C._waypoint_and_grip(env)
        direct, dist = C._direct(env, wp)
        out = np.zeros((n, 3), np.float32)
        out[:, 2] = g
        live = ~env.done
        if not live.any():
            return out
        self._nl = int(live.sum())
        fields = None
        if self.cost == "geodesic":
            fields = np.stack([self._field(env, i, wp[i]) if live[i] else np.zeros((L.G, L.G), np.float32) for i in range(n)])
        cost_fn = self._cost_fn(fields, wp)
        walls = env.walls[:, None]
        ee, H = env.ee, self.H
        dir_ang = np.arctan2(direct[:, 1], direct[:, 0])
        if self.pkind in ("rs_walk", "rs_const"):
            N = max(self.B // H, 1)
            if self.pkind == "rs_walk":
                ang0 = self.rng.uniform(0, 2 * np.pi, (n, N, 1))
                ang = ang0 + np.cumsum(self.rng.normal(0, self.sigma, (n, N, H)), -1) * (np.arange(H) > 0)
            else:
                ang = (self.rng.uniform(0, 2 * np.pi, (n, 1, 1)) + np.linspace(0, 2 * np.pi, N, endpoint=False)[None, :, None]) * np.ones((1, 1, H))
            seqs = np.stack([np.cos(ang), np.sin(ang)], -1).astype(np.float32) * SPEED
            seqs[:, 0] = direct[:, None, :]
            c, _ = self._eval(seqs, ee, walls, cost_fn)
            first = seqs[np.arange(n), np.argmin(c, 1), 0]
        elif self.pkind == "cem":
            N = max(self.B // (H * self.iters), 2)
            ne = max(int(N * self.elite_frac), 2)
            mu = np.broadcast_to(dir_ang[:, None], (n, H)).copy()
            sd = np.full((n, H), np.pi / 2)
            best_c = np.full(n, np.inf)
            best_first = direct.copy()
            for _ in range(self.iters):
                ang = mu[:, None, :] + sd[:, None, :] * self.rng.normal(size=(n, N, H))
                seqs = np.stack([np.cos(ang), np.sin(ang)], -1).astype(np.float32) * SPEED
                seqs[:, 0] = np.stack([np.cos(mu), np.sin(mu)], -1).astype(np.float32) * SPEED
                c, _ = self._eval(seqs, ee, walls, cost_fn)
                order = np.argsort(c, 1)[:, :ne]
                el = np.take_along_axis(ang, order[:, :, None], 1)
                mu, sd = el.mean(1), np.maximum(el.std(1), 0.05)
                cb = np.take_along_axis(c, order[:, :1], 1)[:, 0]
                better = cb < best_c
                best_c = np.where(better, cb, best_c)
                best_first = np.where(better[:, None], seqs[np.arange(n), order[:, 0], 0], best_first)
            first = best_first
        elif self.pkind == "beam":
            W = max(self.B // (17 * H), 1)
            acts = np.concatenate([DIRS16 * SPEED, np.zeros((1, 2), np.float32)], 0)                    # 16 headings + (slot for direct)
            pos = np.broadcast_to(ee[:, None, :], (n, 1, 2)).copy()
            first_a = np.zeros((n, 1, 2), np.float32)
            for h in range(H):
                K = pos.shape[1]
                A = np.broadcast_to(acts[None, None], (n, K, 17, 2)).copy()
                A[:, :, 16] = direct[:, None, :]
                cand = L.la_move(np.broadcast_to(pos[:, :, None, :], (n, K, 17, 2)).reshape(n, K * 17, 2),
                                 L.cap_step(A.reshape(n, K * 17, 2)), walls)
                fa = A.reshape(n, K * 17, 2) if h == 0 else np.repeat(first_a, 17, axis=1)
                self.effort["sim_steps"] += self._nl * K * 17
                c = cost_fn(cand)
                keep = np.argsort(c, 1)[:, :W]
                pos = np.take_along_axis(cand, keep[:, :, None], 1)
                first_a = np.take_along_axis(fa, keep[:, :, None], 1)
            first = first_a[:, 0]
            c = None
        else:
            raise ValueError(self.pkind)
        self.effort["decisions"] += int(live.sum())
        clear = np.array([C.ray_len(env.walls[i], env.ee[i], direct[i] / max(np.linalg.norm(direct[i]), 1e-9), max(dist[i], 0.02))
                          >= max(dist[i], 0.02) - 1e-6 if (live[i] and dist[i] < SPEED * E.MAX_STEP) else False for i in range(n)])
        out[:, :2] = np.where(clear[:, None], direct, first)
        return out
