"""Closed-loop evaluation with optional adaptive perception (encoder gating). Inference-time information only."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import env as E
from . import expert as X
from . import models as M

KMAX_SKIP = 8          # safety: never go longer than this without an encoder pass


def parse_trigger(trig):
    """'full' | 'fixed:K' | 'pix:TAU' | 'learned:TAU' -> (kind, param)"""
    if trig == "full":
        return "full", 0.0
    k, v = trig.split(":")
    return k, float(v)


@torch.no_grad()
def rollout(net, split, cond, seed, n, trigger="full", dev="cuda", T=None, collect=False, layout_ids=None):
    cfg = net.cfg
    kind, par = parse_trigger(trigger)
    if kind != "full" and cfg.kind != "rec":
        raise ValueError("adaptive perception only for recurrent arms")
    env = E.BatchEnv(n, split, cond, seed=seed, layout_ids=layout_ids)
    T = T or (E.T_PICK * cond.n_picks + 2)
    obs = env.reset()
    h = net.init_state(n, dev)
    a_prev = torch.zeros(n, 3, device=dev)
    since = torch.zeros(n, device=dev, dtype=torch.long)
    last_pool = None
    prev_task = None
    prev_held = torch.zeros(n, dtype=torch.bool, device=dev)
    prev_grip = torch.zeros(n, dtype=torch.bool, device=dev)
    enc_count = np.zeros(n)
    alive_steps = np.zeros(n)
    frames = []          # for S4
    toks = []            # for T16
    rec = {"h": [], "target": [], "held": [], "valid": [], "alive": [], "enc": []} if collect else None
    for t in range(T):
        alive = ~env.done
        if not alive.any():
            break
        img = torch.from_numpy(obs["img"]).to(dev)
        valid_env = torch.from_numpy(obs["valid"]).to(dev)
        prop = torch.from_numpy(obs["prop"]).to(dev)
        task = torch.from_numpy(obs["task"]).to(dev)
        x = M.to_x(img)
        enc_mask = valid_env.clone()
        if cfg.kind == "rec" and kind != "full":
            pool = M.pool_feat(img)
            forced = (since >= KMAX_SKIP) | (t == 0)
            if prev_task is not None:
                forced |= (task != prev_task).any(-1)
            held_now = prop[:, 2] > 0.5
            forced |= prev_held & ~held_now & prev_grip           # slip: object lost while gripper still closed
            if kind == "fixed":
                want = since >= int(par)
            elif kind == "pix":
                want = ((pool - last_pool).pow(2).mean(-1).sqrt() > par) if last_pool is not None else torch.ones_like(forced)
            else:
                u = net.trig(torch.cat([h, a_prev, pool], -1)).squeeze(-1)
                want = u > par
            enc_mask = valid_env & (forced | want)
            prev_task = task
        validf = enc_mask.float()
        if cfg.kind == "stack":
            frames.append((x.view(n, 3, 32, 32) * validf[:, None, None, None], validf))
            fs = [frames[-1 - k] if len(frames) > k else (torch.zeros_like(frames[-1][0]), torch.zeros(n, device=dev)) for k in range(4)]
            z = net.enc(torch.cat([f[0] for f in fs], 1), task)
            feat = torch.cat([z, torch.stack([f[1] for f in fs[1:]], -1), validf[:, None], prop, task, a_prev], -1)
            logits = net.pol(feat)
        else:
            z = net.enc(x, task)
            if cfg.kind == "react":
                logits = net.pol(torch.cat([z * validf[:, None], validf[:, None], prop, task, a_prev], -1))
            elif cfg.kind == "trans":
                toks.append(net.inp(torch.cat([z * validf[:, None], validf[:, None], prop, task, a_prev], -1)))
                win = torch.stack(toks[-16:], 1)
                L = win.shape[1]
                i = torch.arange(L, device=dev)
                allowed = i[None, :] <= i[:, None]
                hh = net.tf(win, mask=~allowed)[:, -1]
                logits = net.pol(torch.cat([hh, prop, task], -1))
            else:
                h = net.cell_step(h, z, validf, prop, task, a_prev)
                logits = net.policy(h, prop, task, z * validf[:, None])
        if cfg.kind == "rec" and kind != "full":
            since = torch.where(enc_mask, torch.zeros_like(since), since + 1)
            lp = M.pool_feat(img)
            last_pool = lp if last_pool is None else torch.where(enc_mask[:, None], lp, last_pool)
        act = torch.cat([torch.tanh(logits[:, :2]), (logits[:, 2:3] > 0).float()], -1)
        am = alive.copy()
        enc_count += (enc_mask.cpu().numpy() & am)
        alive_steps += am
        if collect:
            rec["h"].append(h.cpu().numpy() if cfg.kind == "rec" else np.zeros((n, 1), np.float32))
            pv = env.privileged()
            rec["target"].append(pv["target"]); rec["held"].append(pv["held"]); rec["valid"].append(obs["valid"].copy())
            rec["alive"].append(am.copy()); rec["enc"].append(enc_mask.cpu().numpy())
        prev_held = prop[:, 2] > 0.5
        prev_grip = prop[:, 3] > 0.5
        a_prev = act
        obs, _ = env.step(act.cpu().numpy())
    s = X.success_summary(env)
    res = {"success": s["success"], "picks_done": s["picks_done"], "failed_grasps": s["failed_grasps"],
           "fail_kind": s["fail_kind"], "enc_count": enc_count, "alive_steps": alive_steps,
           "enc_frac": float(enc_count.sum() / max(alive_steps.sum(), 1))}
    if collect:
        res["rec"] = {k: np.stack(v, 1) for k, v in rec.items()}
    return res


def calibrate(net, kind, target_frac, dev="cuda", seed=777, n=96, lo=None, hi=None, iters=9):
    """Find the trigger parameter whose encoder fraction on the DEV split (NOM+DIST mixture) is closest to target_frac.
    fixed:k is solved analytically (k = round(1/target))."""
    if kind == "fixed":
        k = max(1, int(round(1.0 / target_frac)) - 1)
        return float(k)
    conds = [E.TEST_CONDS["NOM"], E.TEST_CONDS["DIST"]]
    lo, hi = (0.0, 0.6) if kind == "pix" else (-0.5, 1.5)
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        fr = np.mean([rollout(net, "dev", c, seed, n // 2, f"{kind}:{mid}", dev)["enc_frac"] for c in conds])
        best = mid
        if fr > target_frac:      # too many encodes -> raise threshold
            lo = mid
        else:
            hi = mid
    return float(best)


def summarize(res):
    return {"success": float(res["success"].mean()), "picks_frac": float(res["picks_done"].mean()),
            "failed_grasps": float(res["failed_grasps"].mean()), "enc_frac": res["enc_frac"],
            "enc_per_ep": float(res["enc_count"].mean()), "steps_per_ep": float(res["alive_steps"].mean()),
            "fail_kinds": {k: int((res["fail_kind"] == k).sum()) for k in set(res["fail_kind"].tolist()) if k}}
