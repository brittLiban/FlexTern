"""Shared behaviour-cloning data generation and training loop (identical for every arm; see protocol_v1.json)."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from . import env as E
from . import expert as X
from . import models as M

DATA_DIR = os.environ.get("PS_DATA", os.path.expanduser("~/ps_data"))

# --- budget shared by all arms (locked in protocol_v1.json)
STEPS = int(os.environ.get("PS_STEPS", 6000))   # amendment 4: was 3000 (and batch 24)
BATCH = int(os.environ.get("PS_BATCH", 48))
LR = float(os.environ.get("PS_LR", 2e-3))
T_TRAIN = 160
N_TRAIN_EPISODES = 2000
DROP_P = 0.05
DROP_MAX = 8
JEPA_W = 1.0


def train_conds():
    cs = []
    for pb in (0.0, 0.02, 0.05):
        for dur in ((5, 15), (10, 30)):
            for pd in (0.0, 0.5, 1.0):
                for k in (1, 2, 3):
                    for npk in (1, 1, 2):
                        cs.append(E.Cond(f"tr", pb, dur, pd, k, npk))
    return cs


def gen_train(seed=0, n=N_TRAIN_EPISODES):
    path = os.path.join(DATA_DIR, f"train_s{seed}_n{n}.npz")
    if os.path.exists(path):
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    d = X.collect(n, "train", train_conds(), seed=1000 + seed, T=T_TRAIN)
    np.savez(path, **{k: v for k, v in d.items() if k not in ("layout",)}, layout=d["layout"])
    return path


def load_gpu(path, dev):
    paths = path if isinstance(path, (list, tuple)) else [path]
    g = {}
    for k in ("img", "valid", "prop", "task", "a_exec", "a_label", "mask"):
        parts = []
        for p in paths:
            d = np.load(p)
            parts.append(torch.from_numpy(d[k]).to(dev))
        g[k] = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
        del parts
    g["mask"] = g["mask"].bool()
    g["valid"] = g["valid"].bool()
    return g


def burst_drop(B, T, dev, gen=None):
    starts = torch.rand(B, T, device=dev, generator=gen) < DROP_P
    dur = torch.randint(1, DROP_MAX + 1, (B, T), device=dev, generator=gen)
    drop = torch.zeros(B, T, dtype=torch.bool, device=dev)
    for k in range(DROP_MAX):
        s = starts & (dur > k)
        drop |= torch.nn.functional.pad(s, (k, 0))[:, :T]
    drop[:, 0] = False                          # first frame always seen
    return drop


def sample_batch(g, B, dev, gen=None):
    n = g["mask"].shape[0]
    idx = torch.randint(0, n, (B,), device=dev, generator=gen)
    mask = g["mask"][idx]
    Tm = int(mask.sum(1).max().item())
    b = {k: g[k][idx, :Tm] for k in ("img", "prop", "task", "a_exec", "a_label")}
    b["mask"] = mask[:, :Tm]
    b["valid_env"] = g["valid"][idx, :Tm]
    b["valid_in"] = b["valid_env"] & ~burst_drop(B, Tm, dev, gen)
    b["a_exec"] = b["a_exec"].float()
    b["a_label"] = b["a_label"].float()
    return b


def train_arm(arm, seed, data_path, out_dir, steps=STEPS, dev="cuda", log_every=250):
    os.makedirs(out_dir, exist_ok=True)
    ck = os.path.join(out_dir, f"{arm}_s{seed}.pt")
    if os.path.exists(ck):
        return ck
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(10_000 + seed)
    net = M.build(arm).to(dev)
    g = load_gpu(data_path, dev)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    log = []
    for it in range(steps):
        b = sample_batch(g, BATCH, dev, gen)
        out = net.forward_seq(b, JEPA_W)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if net.ema is not None:
            net.update_ema()
        if it % log_every == 0 or it == steps - 1:
            rec = {"it": it, "t": round(time.time() - t0, 1)}
            rec.update({k: round(float(v), 5) for k, v in out.items() if k != "h"})
            log.append(rec)
            print(arm, seed, rec, flush=True)
    torch.save({"arm": arm, "seed": seed, "state": net.state_dict(), "steps": steps, "log": log,
                "train_seconds": time.time() - t0}, ck)
    return ck


def load_net(path, dev="cuda"):
    ck = torch.load(path, map_location=dev)
    net = M.build(ck["arm"]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck
