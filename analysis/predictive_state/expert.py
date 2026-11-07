"""Privileged scripted expert for OccPick2D and demonstration collection (shared by every learned arm)."""
from __future__ import annotations

import numpy as np

from . import env as E

SPEED = 0.9           # fraction of MAX_STEP the expert commands
CLOSE_R = 0.03        # close gripper when this near the target
RELEASE_R = 0.04      # release when this near the goal centre


def expert_action(env: E.BatchEnv):
    """Uses true state. Returns (n,3) clean action."""
    tgt = env.pos[:, 0]
    goal = env.goal
    aim = np.where(env.held[:, None], goal, tgt)
    vec = aim - env.ee
    dist = np.linalg.norm(vec, axis=1, keepdims=True)
    step = np.minimum(dist, SPEED * E.MAX_STEP)
    d = vec / np.maximum(dist, 1e-9) * step / E.MAX_STEP
    near_t = np.linalg.norm(tgt - env.ee, axis=1) < CLOSE_R
    near_g = np.linalg.norm(goal - env.ee, axis=1) < RELEASE_R
    g = np.where(env.held, ~near_g, near_t).astype(np.float32)
    # keep the gripper closed once it is closed on the target
    return np.concatenate([d, g[:, None]], 1).astype(np.float32)


def collect(n_episodes, split, conds, seed, T, noise_levels=(0.0, 0.2, 0.4), batch=250):
    """Roll out the noisy expert. conds: list of Cond to sample per episode. Returns dict of arrays (E,T,...)."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("img", "valid", "prop", "task", "a_exec", "a_label", "mask", "target", "distract", "held", "layout")}
    for b0 in range(0, n_episodes, batch):
        nb = min(batch, n_episodes - b0)
        cs = [conds[int(rng.integers(len(conds)))] for _ in range(nb)]
        env = E.BatchEnv(nb, split, cs, seed=int(rng.integers(1 << 31)))
        sig = np.array([noise_levels[int(rng.integers(len(noise_levels)))] for _ in range(nb)], np.float32)
        obs = env.reset()
        buf = {k: [] for k in out}
        for t in range(T):
            pv = env.privileged()
            alive = ~env.done
            lab = expert_action(env)
            ex = lab.copy()
            ex[:, :2] = np.clip(ex[:, :2] + rng.normal(0, 1, (nb, 2)).astype(np.float32) * sig[:, None] * 0.5, -1, 1)
            for k, v in (("img", obs["img"]), ("valid", obs["valid"]), ("prop", obs["prop"]), ("task", obs["task"]),
                         ("a_exec", ex), ("a_label", lab), ("mask", alive), ("target", pv["target"]),
                         ("distract", pv["distract"]), ("held", pv["held"])):
                buf[k].append(v)
            buf["layout"].append(env.layout_ids)
            obs, _ = env.step(ex)
        for k in out:
            out[k].append(np.stack(buf[k], 1))
    return {k: np.concatenate(v, 0) for k, v in out.items()}


def rollout_expert(env: E.BatchEnv, T):
    """Noise-free privileged expert evaluated in closed loop; returns per-episode success summary."""
    env.reset()
    for _ in range(T):
        env.step(expert_action(env))
        if env.done.all():
            break
    return success_summary(env)


def success_summary(env: E.BatchEnv):
    ok = env.picks_done >= env.n_picks
    return {"success": ok.astype(np.float32), "picks_done": env.picks_done.copy(), "failed_grasps": env.failed_grasps.copy(),
            "fail_kind": env.fail_kind.copy()}
