"""EXP-0013 Phase 1.1 closeout: PAIRED-REPLAY variant of OccPick2D-LA (protocol "la-paired-replay-1.1").

Why: in v1 (`env.BatchEnv`/`env_la.LAEnv`) ONE `np.random.Generator` serves spawning, blackout scheduling, pixel noise and (unsupported here)
displacement for the whole batch. `LAEnv.render` draws `normal(.., (em.sum(), 3))` with an end-effector pixel count that depends on where the
effector is, so two controllers that act differently consume different numbers of samples and every later blackout draw (and every other
episode in the batch) diverges. Same layout + same spawn therefore did NOT mean same blackout schedule.

What this module changes (RANDOMNESS ONLY; `env.py`/`env_la.py` are untouched and `step`/dynamics/`render` geometry are inherited verbatim):
  * spawn      : one stream per episode, keyed (seed, episode_id, SPAWN)               -> independent of batch size / order / actions
  * blackouts  : per episode, uniforms + durations for every step t precomputed from (seed, episode_id, BLACKOUT); `BatchEnv.step` reads them
                 through `_ExoRng` instead of a shared generator                        -> schedule is a pure function of (seed, episode_id, t)
  * pixel noise: one full-frame N(0, PIX_NOISE) field per (seed, episode_id, RENDER, t), added after the noise-free render and clipped
                 (v1 painted walls / effector with their own noise; per-pixel marginals are identical, only the realization differs)
  * layouts    : unchanged recipe by default (`default_rng([seed,777]).integers(lo,hi,n)`, so the 200 DEV episode layouts are the Phase 1.1 ones);
                 spawns are NOT bit-identical to v1 (they come from per-episode streams), which is unavoidable if they must not depend on batch order.
Not supported (asserted): displacement events (p_disp=0, as in LAEnv) and n_picks>1 (a respawn would make `t_pick`/`black_left` action dependent).
The reserved split "test" is refused.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import env as E
from . import env_la as L

PROTOCOL = "la-paired-replay-1.1"
SPAWN, BLACKOUT, RENDER = 1, 2, 3          # stream ids
TMAX = 128                                  # precomputed horizon (T_limit is 80)


def _gen(*key):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(k) for k in key])))


def blackout_draws(seed, episode_id, lo, hi):
    """(u (TMAX,), dur (TMAX,)) for one episode: depends only on (seed, episode_id, duration range)."""
    g = _gen(seed, episode_id, BLACKOUT)
    return g.random(TMAX), g.integers(lo, hi + 1, TMAX)


def noise_field(seed, episode_id, t):
    """Pixel-noise field (RES, RES, 3) float32 for one (episode, time step): depends only on (seed, episode_id, t)."""
    return _gen(seed, episode_id, RENDER, t).normal(0, E.PIX_NOISE, (E.RES, E.RES, 3)).astype(np.float32)


def reference_valid(u, dur, p_blackout, T):
    """Independent re-statement of the blackout recurrence in BatchEnv.step for an episode that is alive for steps 0..T-1
    (t_pick == k+1 at step k). Returns valid[0..T] (reset observation first). Used by the tests as an oracle."""
    left, out = 0, [True]
    for k in range(T):
        left = max(left - 1, 0)
        if u[k] < np.float32(p_blackout) and left == 0 and (k + 1) >= 3:
            left = int(dur[k])
        out.append(left == 0)
    return np.array(out)


class _ExoRng:
    """Stand-in for `BatchEnv.rng`: serves ONLY the draws `BatchEnv.step`/`render` make from precomputed, episode-keyed streams.
    Any other use raises: unknown methods (`uniform`, `permutation`, ...), a `random`/`integers` call whose shape or bounds differ from the
    blackout recurrence, and any `normal` call except the zero-mean PIX_NOISE render-noise placeholder made inside `render()` (the
    real noise is injected per (episode, t) afterwards). A Gaussian draw in the dynamics therefore fails visibly instead of returning zeros."""
    def __init__(self, env):
        self._env = env
        self.render_scope = False                   # set by PairedLAEnv.render around the noise-free super().render()

    def random(self, size):
        if size != self._env.n:
            raise RuntimeError(f"unsupported random(size={size!r}): only the per-episode blackout uniform (n={self._env.n}) is served")
        return self._env._u[:, self._env.t].copy()

    def integers(self, lo, hi):
        e = self._env
        if not (np.array_equal(lo, e.dur_lo) and np.array_equal(hi, e.dur_hi + 1)):
            raise RuntimeError("unsupported integers(): only the per-episode blackout duration draw is served")
        return self._env._dur[:, self._env.t].copy()

    def normal(self, loc, scale, size):
        if not (self.render_scope and loc == 0 and scale == E.PIX_NOISE):
            raise RuntimeError(f"unsupported normal(loc={loc!r}, scale={scale!r}): Gaussian draws are allowed only as zero-mean PIX_NOISE render noise inside render()")
        return np.zeros(size)                       # noise is injected per (episode, t) in `PairedLAEnv.render`

    def __getattr__(self, name):
        raise RuntimeError(f"shared-RNG draw '{name}' is not allowed in the paired-replay environment")


class PairedLAEnv(L.LAEnv):
    protocol = PROTOCOL

    def __init__(self, n, split, cond, seed, episode_ids=None, layout_ids=None, t_limit=None):
        if split == "test":
            raise ValueError("the reserved TEST layout range must not be used by exploratory/paired evaluations")
        self.seed = int(seed)
        if episode_ids is None:
            self.episode_ids = np.arange(n)         # protocol default: episode i of the batch IS episode id i (locked la-paired-replay-1.1)
        else:
            # Layout ids default to a function of the batch POSITION (`default_rng([seed,777]).integers(lo,hi,n)[i]`), so custom episode ids
            # would silently receive whatever layout their position implies. Make the pairing explicit instead of guessing.
            if layout_ids is None:
                raise ValueError("custom episode_ids require explicit layout_ids: the default layout of an episode depends on its batch position, not its id")
            self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
            if self.episode_ids.ndim != 1 or len(self.episode_ids) != n:
                raise ValueError(f"episode_ids must be 1-D with length n={n}")
            if len(np.unique(self.episode_ids)) != n:
                raise ValueError("duplicate episode_ids are not supported: they would share spawn, blackout and noise streams")
            if (self.episode_ids < 0).any():
                raise ValueError("episode_ids must be non-negative")
        if layout_ids is not None:                  # every path: explicit layouts must lie in the requested split (reserved TEST geometry is unreachable)
            lids = np.asarray(layout_ids)
            lo, hi = E.LAYOUT_RANGES[split]
            if lids.shape != (n,) or not np.issubdtype(lids.dtype, np.integer) or ((lids < lo) | (lids >= hi)).any():
                raise ValueError(f"layout_ids must be integers of shape ({n},) lying in the '{split}' range [{lo}, {hi})")
        conds = cond if isinstance(cond, list) else [cond] * n
        assert all(c.n_picks == 1 for c in conds), "paired replay requires n_picks == 1"
        draws = [blackout_draws(self.seed, e, c.dur[0], c.dur[1]) for e, c in zip(self.episode_ids, conds)]
        self._u = np.stack([d[0] for d in draws]).astype(np.float64)
        self._dur = np.stack([d[1] for d in draws])
        self._exo = _ExoRng(self)
        self.valid_log = []
        super().__init__(n, split, cond, seed, layout_ids=layout_ids, t_limit=t_limit)

    # BatchEnv.__init__ assigns self.rng = default_rng(seed); keep the guarded stand-in instead.
    @property
    def rng(self):
        return self._exo

    @rng.setter
    def rng(self, _):
        pass

    # ------------------------------------------------------------------ spawning (per-episode streams)
    def _free_point(self, i, others, min_sep=0.12, avoid_occ=True):
        g, xw = self._spawn_rng, self.la_info[i]["xw"]
        for _ in range(500):
            p = np.array([g.uniform(0.08, xw - 0.10), g.uniform(0.08, 0.92)], np.float32)
            if avoid_occ and E.under_occluder(p, self.occ[i], 0.02):
                continue
            if L.inside(p, self.walls[i], 0.04):
                continue
            if all(np.linalg.norm(p - o) > min_sep for o in others):
                return p
        return p

    def _schedule_disp(self, i):
        self.disp_at[i] = -1                       # p_disp == 0 is enforced by LAEnv

    def reset(self):
        """BatchEnv.reset with the spawn draws taken from per-episode streams (same logic otherwise)."""
        n = self.n
        self.goal = self.goal_la.copy()
        self.ee = np.zeros((n, 2), np.float32)
        self.pos = np.zeros((n, E.KMAX + 1, 2), np.float32)
        self.col = np.zeros((n, E.KMAX + 1), np.int64)
        self.active = np.zeros((n, E.KMAX + 1), bool)
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
        self.valid_log = []
        for i in range(n):
            self._spawn_rng = _gen(self.seed, self.episode_ids[i], SPAWN)
            cols = self._spawn_rng.permutation(E.N_COLORS)[: self.k[i] + 1]
            placed = []
            self.ee[i] = self._free_point(i, [], avoid_occ=False)
            placed.append(self.ee[i])
            for j in range(self.k[i] + 1):
                p = self._free_point(i, placed, avoid_occ=(j == 0))
                self.pos[i, j] = p
                self.col[i, j] = cols[j]
                self.active[i, j] = True
                placed.append(p)
            self.task[i] = self.col[i, 0]
            self._schedule_disp(i)
        return self.observe()

    # ------------------------------------------------------------------ observation
    def noise(self):
        return np.stack([noise_field(self.seed, e, self.t) for e in self.episode_ids])

    def render(self):
        self._exo.render_scope = True
        try:
            img = super().render()                  # noise-free (the guarded rng returns zeros for the PIX_NOISE placeholder only)
        finally:
            self._exo.render_scope = False
        self.last_noise = self.noise()
        return np.clip(img + self.last_noise, 0, 255)

    def observe(self):
        o = super().observe()
        self.valid_log.append(o["valid"].copy())
        return o

    # ------------------------------------------------------------------ provenance
    def spawn_digest(self):
        """Per-episode sha1 of the initial state (layout, goal, spawn, colours, task)."""
        assert self.t == 0, "spawn_digest is defined at reset"
        return [hashlib.sha1(b"".join(np.ascontiguousarray(a[i]).tobytes() for a in (self.walls, self.goal, self.ee, self.pos, self.col, self.active))
                             + np.int64(self.task[i]).tobytes()).hexdigest()[:16] for i in range(self.n)]

    def schedule_digest(self, T=82):
        """Per-episode sha1 of the closed-form blackout schedule over steps 0..T-1 (the scheduled `valid` sequence, action independent)."""
        out = []
        for i in range(self.n):
            v = reference_valid(self._u[i], self._dur[i], self.p_blackout[i], T)
            out.append(hashlib.sha1(v.astype(np.uint8).tobytes()).hexdigest()[:16])
        return out

    def scheduled_valid(self, T=82):
        return np.stack([reference_valid(self._u[i], self._dur[i], self.p_blackout[i], T) for i in range(self.n)])
