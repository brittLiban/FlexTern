"""State-usefulness and prediction analyses for the recurrent arms (EXP-0012).

probe:   ridge regression from the recurrent state h to the true target position on blackout frames (target not held).
         Fit on DEV LONG rollouts, steps <= 100; scored on TEST LONG rollouts, steps in (0,100] and > 160 (beyond the
         160-step training length). R^2 drop between windows measures state drift.
predict: open-loop multi-step latent prediction from teacher-forced state, k = 1..8, with a persistence baseline and a
         shuffled-action control (does the predictor use the action?). Cosine similarity to the EMA target embedding.
Usage: python -m analysis.predictive_state.analyze_state probe|predict ARMS SEEDS OUT.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from . import env as E
from . import evaluate as V
from . import expert as X
from . import models as M
from . import train as T


def _ridge(Xtr, ytr, lam=1.0):
    Xb = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ ytr)


def _r2(w, Xte, yte):
    Xb = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    p = Xb @ w
    return float(1 - ((p - yte) ** 2).sum() / ((yte - yte.mean(0)) ** 2).sum())


def _rows(rec, tmin, tmax, only_blackout=True):
    h, tg, held, valid, alive = (rec[k] for k in ("h", "target", "held", "valid", "alive"))
    T_ = h.shape[1]
    t = np.arange(T_)[None, :]
    m = alive & ~held & (t > tmin) & (t <= tmax)
    if only_blackout:
        m &= ~valid
    return h[m], tg[m], int(m.sum())


def probe(net, seed, n=60):
    cond = E.TEST_CONDS["LONG"]
    dv = V.rollout(net, "dev", cond, 5000 + seed, n, "full", collect=True)["rec"]
    ts = V.rollout(net, "test", cond, 9000 + seed, n, "full", collect=True)["rec"]
    Xtr, ytr, ntr = _rows(dv, 0, 100)
    w = _ridge(Xtr, ytr)
    out = {"n_fit": ntr}
    for name, (lo, hi) in {"early_le100": (0, 100), "late_gt160": (160, 10_000)}.items():
        Xte, yte, nte = _rows(ts, lo, hi)
        out[name] = {"r2": _r2(w, Xte, yte) if nte > 20 else None, "n": nte}
    alive = ts["alive"]
    h = ts["h"]
    nh = np.abs(h).mean(-1)
    t = np.arange(h.shape[1])[None, :]
    out["mean_abs_h_le100"] = float(nh[alive & (t <= 100)].mean())
    out["mean_abs_h_gt160"] = float(nh[alive & (t > 160)].mean()) if (alive & (t > 160)).any() else None
    out["max_abs_h"] = float(np.abs(h[alive]).max())
    out["nan"] = bool(np.isnan(h).any())
    return out


@torch.no_grad()
def predict(net, seed, n=250, t0=20, K=8, dev="cuda"):
    c = [E.TEST_CONDS["NOM"], E.TEST_CONDS["DIST"]]
    d = X.collect(n, "dev", c, seed=7000 + seed, T=t0 + K + 2, noise_levels=(0.0,))
    img = torch.from_numpy(d["img"]).to(dev)
    prop = torch.from_numpy(d["prop"]).to(dev)
    task = torch.from_numpy(d["task"]).to(dev)
    a = torch.from_numpy(d["a_exec"]).to(dev).float()
    ok = torch.from_numpy(d["mask"][:, : t0 + K + 1].all(1) & d["valid"][:, : t0 + K + 1].all(1)).to(dev)
    B = img.shape[0]
    Tn = t0 + K + 1
    tflat = task[:, :Tn].reshape(B * Tn, -1)
    zt = F.normalize(net.ema(M.to_x(img[:, :Tn]), tflat).view(B, Tn, M.Z), dim=-1)
    z = net.enc(M.to_x(img[:, :Tn]), tflat).view(B, Tn, M.Z)
    h = net.init_state(B, dev)
    a_prev = torch.zeros(B, 3, device=dev)
    for t in range(t0 + 1):                                      # teacher-forced with full observations up to t0
        h = net.cell_step(h, z[:, t], torch.ones(B, device=dev), prop[:, t], task[:, t], a_prev)
        a_prev = a[:, t]
    res = {"k": [], "cos_pred": [], "cos_shuffled_action": [], "cos_persistence": []}
    hp = h.clone()
    hs = h.clone()
    perm = torch.randperm(B, device=dev)
    for k in range(1, K + 1):
        t = t0 + k - 1
        zhat = F.normalize(net.pred(torch.cat([hp, a[:, t]], -1)), dim=-1)
        zsh = F.normalize(net.pred(torch.cat([hs, a[perm, t]], -1)), dim=-1)
        tgt = zt[:, t0 + k]
        res["k"].append(k)
        res["cos_pred"].append(float((zhat * tgt).sum(-1)[ok].mean()))
        res["cos_shuffled_action"].append(float((zsh * tgt).sum(-1)[ok].mean()))
        res["cos_persistence"].append(float((zt[:, t0] * tgt).sum(-1)[ok].mean()))
        zero = torch.zeros(B, device=dev)
        hp = net.cell_step(hp, torch.zeros_like(z[:, 0]), zero, prop[:, t + 1], task[:, t + 1], a[:, t])   # open loop: no observation
        hs = net.cell_step(hs, torch.zeros_like(z[:, 0]), zero, prop[:, t + 1], task[:, t + 1], a[perm, t])
    res["n"] = int(ok.sum())
    return res


def main():
    mode, arms, seeds, out = sys.argv[1], sys.argv[2].split(","), [int(s) for s in sys.argv[3].split(",")], sys.argv[4]
    ckdir = os.environ.get("PS_CKPT", "/home/liban/ps_ckpt")
    res = json.load(open(out)) if os.path.exists(out) else {}
    for arm in arms:
        for s in seeds:
            key = f"{arm}|{s}"
            path = f"{ckdir}/{arm}_s{s}.pt"
            if key in res or not os.path.exists(path):
                continue
            net, _ = T.load_net(path)
            res[key] = probe(net, s) if mode == "probe" else predict(net, s)
            json.dump(res, open(out, "w"), indent=1)
            print(key, json.dumps(res[key])[:300], flush=True)


if __name__ == "__main__":
    main()
