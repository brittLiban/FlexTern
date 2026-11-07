"""EXP-0013 Phase 1: one shared, instrumented inference step for the recurrent arms.

`evaluate.rollout` (batched quality evaluation) and `latency.run` (batch-1 timing) both drive `RecurrentRunner`, so trigger
decisions, forced observations and recurrent updates are the same code. The vision encoder only sees the rows that need a new
embedding (gather -> encode -> scatter into zeros); an all-skipped batch never calls it.

Observation status (blackout / trigger-skipped / finished) is tracked in `Counters` only. The model keeps its single `valid`
input (EXP-0012 interface), so a skipped frame and a blackout frame are still indistinguishable *to the network*.

Sub-batching is numerically safe for every arm: the encoder uses LayerNorm (no BatchNorm) and the fake-quant activation scales
are per sample / per token, so a row's embedding does not depend on which other rows are in the batch (up to float rounding
from batch-size-dependent kernels).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import torch

from . import models as M

KMAX_SKIP = 8          # same safety bound as evaluate.KMAX_SKIP (kept here to avoid a circular import)


def parse_trigger(trig):
    """'full' | 'fixed:K' | 'pix:TAU' | 'learned:TAU' -> (kind, param)"""
    if trig == "full":
        return "full", 0.0
    k, v = trig.split(":")
    if k not in ("fixed", "pix", "learned"):
        raise ValueError(trig)
    return k, float(v)


@dataclass
class Counters:
    """Executed-work counters. `enc_calls` counts PyTorch encoder forward calls, `enc_samples` the images actually processed."""
    steps: int = 0                 # runner.step calls (batch steps)
    rec_updates: int = 0           # recurrent-cell rows EXECUTED (== alive rows when select_alive, == n per step otherwise)
    policy_rows: int = 0           # policy-head rows EXECUTED
    trigger_rows: int = 0          # rows through the pixel-pool / trigger computation (always the full batch; not gated)
    active_steps: int = 0          # alive (unfinished) rows summed over steps (USEFUL work; executed work is rec_updates)
    finished_rows_processed: int = 0   # finished rows that still went through the cell/policy (0 when select_alive)
    enc_calls: int = 0
    enc_samples: int = 0
    requested: int = 0             # alive rows whose trigger asked for an observation and whose frame was valid (== enc_samples)
    blackout_rows: int = 0         # alive rows with an invalid (blacked-out) frame
    trigger_skipped_rows: int = 0  # alive rows with a valid frame that the trigger chose not to encode
    finished_rows: int = 0         # rows already done (never encoded, never counted as active)
    t_trigger: float = 0.0         # seconds (only if timing=True)
    t_encoder: float = 0.0
    t_update: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        d = asdict(self)
        d.pop("extra")
        d.update(self.extra)
        return d


def gated_encode(enc, img_u8, task, need, counters=None, z_dim=M.Z):
    """Embeddings (n, Z): `enc` applied to rows where `need` is True, exact zeros elsewhere. No encoder call if none are needed."""
    n = need.shape[0]
    if n == 1:                                   # batch-1 fast path (latency): no gather/scatter needed
        if not bool(need[0]):
            return torch.zeros(1, z_dim, device=img_u8.device)
        if counters is not None:
            counters.enc_calls += 1
            counters.enc_samples += 1
        return enc(M.to_x(img_u8), task)
    z = torch.zeros(n, z_dim, device=img_u8.device)
    idx = need.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return z
    if counters is not None:
        counters.enc_calls += 1
        counters.enc_samples += int(idx.numel())
    z[idx] = enc(M.to_x(img_u8[idx]), task[idx])
    return z


class RecurrentRunner:
    """Stateful per-step inference for a recurrent arm over n parallel episodes.

    step(img_u8, valid_env, prop, task, alive) -> (action (n,3), enc_mask (n,))
    All inputs are tensors on `dev`: img (n,32,32,3) uint8, valid_env/alive bool (n,), prop (n,4), task (n,N_COLORS).
    Causality: a_prev is the action executed before the current observation; the state is advanced every step (also when
    vision is skipped) from the zero embedding, the previous action and the current proprioception.
    """

    def __init__(self, net, trigger="full", n=1, dev="cpu", timing=False, select_alive=True, track=True):
        if net.cfg.kind != "rec":
            raise ValueError("RecurrentRunner is for recurrent arms only")
        self.net, self.dev, self.timing, self.select_alive, self.track = net, dev, timing, select_alive, track
        self.kind, self.par = parse_trigger(trigger)
        self.n = n
        self.h = net.init_state(n, dev)
        self.a_prev = torch.zeros(n, 3, device=dev)
        self.since = torch.zeros(n, device=dev, dtype=torch.long)
        self.acc = torch.zeros(n, device=dev)
        self.last_pool = None
        self.prev_task = None
        self.prev_held = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_grip = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t = 0
        self.counters = Counters()
        self.last_z = None

    def _tick(self):
        if not self.timing:
            return 0.0
        if self.dev != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def decide(self, img, valid_env, prop, task, alive):
        """Trigger decision only (no encoder, no state change except the fixed-rate accumulator). Returns enc_mask."""
        if self.kind == "full":
            return valid_env & alive, None
        pool = M.pool_feat(img)
        self.counters.trigger_rows += int(img.shape[0])
        forced = (self.since >= KMAX_SKIP) | (self.t == 0)
        if self.prev_task is not None:
            forced = forced | (task != self.prev_task).any(-1)
        held_now = prop[:, 2] > 0.5
        forced = forced | (self.prev_held & ~held_now & self.prev_grip)      # slip: object lost while gripper still closed
        if self.kind == "fixed":
            self.acc = self.acc + self.par             # fractional accumulator: encode when it reaches 1
            want = self.acc >= 1.0
            self.acc = torch.where(want & valid_env, self.acc - 1.0, self.acc).clamp(max=2.0)
        elif self.kind == "pix":
            want = ((pool - self.last_pool).pow(2).mean(-1).sqrt() > self.par) if self.last_pool is not None \
                else torch.ones_like(forced)
        else:
            u = self.net.trig(torch.cat([self.h, self.a_prev, pool], -1)).squeeze(-1)
            want = u > self.par
        self.prev_task = task
        return valid_env & (forced | want) & alive, pool

    @torch.no_grad()
    def step(self, img, valid_env, prop, task, alive):
        net, c = self.net, self.counters
        n_alive = int(alive.sum()) if self.n > 1 else int(bool(alive[0]))
        if n_alive == 0:                              # nothing to do: no trigger, encoder, cell or policy; state untouched
            c.steps += 1
            c.finished_rows += self.n
            self.t += 1
            self.last_z = torch.zeros(self.n, M.Z, device=self.dev)
            return torch.zeros(self.n, 3, device=self.dev), torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        t0 = self._tick()
        enc_mask, pool = self.decide(img, valid_env, prop, task, alive)
        t1 = self._tick()
        z = gated_encode(net.enc, img, task, enc_mask, c)
        t2 = self._tick()
        validf = enc_mask.float()
        if self.select_alive and n_alive < self.n:    # genuine selective execution: only alive rows enter the cell and policy
            idx = alive.nonzero(as_tuple=True)[0]
            h_sub = net.cell_step(self.h[idx], z[idx], validf[idx], prop[idx], task[idx], self.a_prev[idx])
            lg = net.policy(h_sub, prop[idx], task[idx], (z * validf[:, None])[idx])
            self.h = self.h.clone()
            self.h[idx] = h_sub
            act = torch.zeros(self.n, 3, device=self.dev)
            act[idx] = torch.cat([torch.tanh(lg[:, :2]), (lg[:, 2:3] > 0).float()], -1)
            rows = n_alive
            act_keep = self.a_prev.clone()
            act_keep[idx] = act[idx]
        else:                                         # full batch (also the legacy accounting when select_alive=False)
            self.h = net.cell_step(self.h, z, validf, prop, task, self.a_prev)
            logits = net.policy(self.h, prop, task, z * validf[:, None])
            act = torch.cat([torch.tanh(logits[:, :2]), (logits[:, 2:3] > 0).float()], -1)
            rows = self.n
            act_keep = act
        c.rec_updates += rows
        c.policy_rows += rows
        c.finished_rows_processed += rows - n_alive
        if self.kind != "full":
            self.since = torch.where(enc_mask, torch.zeros_like(self.since), self.since + 1)
            self.last_pool = pool if self.last_pool is None else torch.where(enc_mask[:, None], pool, self.last_pool)
        self.prev_held = prop[:, 2] > 0.5
        self.prev_grip = prop[:, 3] > 0.5
        self.a_prev = act_keep
        self.last_z = z
        t3 = self._tick()
        c.steps += 1
        if self.track:                                # one extra device->host sync per step; track=False isolates its cost
            n_req, n_black, n_skip = torch.stack(
                [enc_mask.sum(), (~valid_env & alive).sum(), (valid_env & alive & ~enc_mask).sum()]).tolist()
            c.requested += n_req
            c.blackout_rows += n_black
            c.trigger_skipped_rows += n_skip
        c.active_steps += n_alive
        c.finished_rows += self.n - n_alive
        if self.timing:
            c.t_trigger += t1 - t0
            c.t_encoder += t2 - t1
            c.t_update += t3 - t2
        self.t += 1
        return act, enc_mask
