"""MEASURED batch-1 CPU per-control-step latency of the PyTorch reference model (EXP-0012).

Quantized arms have no packed kernel in this code base: their fake-quant graph is slower than FP32 and is NOT timed
(it would measure simulation overhead, not an implementation). Only the plain FP checkpoint's deployment path is timed:
pooled-pixel features + (optional) encoder + GRU cell + policy head (+ trigger head). The predictor head and EMA encoder
are training/analysis-only and excluded. Run on AC power with nothing else on the CPU (KI-021).
Usage: python -m analysis.predictive_state.latency CKPT.pt OUT.json
"""
from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np
import torch

from . import env as E
from . import evaluate as V
from . import models as M
from . import train as T


def pct(x):
    x = np.asarray(x) * 1e3
    return {"p50_ms": float(np.percentile(x, 50)), "p95_ms": float(np.percentile(x, 95)), "max_ms": float(x.max()), "mean_ms": float(x.mean()), "n": len(x)}


@torch.no_grad()
def run(net, trigger, threads, episodes=15, seed=31):
    torch.set_num_threads(threads)
    kind, par = V.parse_trigger(trigger)
    lat, enc_flags = [], []
    for ep in range(episodes):
        env = E.BatchEnv(1, "test", E.TEST_CONDS["DIST"], seed=seed + ep)
        obs = env.reset()
        h = net.init_state(1, "cpu")
        a_prev = torch.zeros(1, 3)
        since, last_pool, prev_task, prev_held, prev_grip = 0, None, None, False, False
        for t in range(E.T_PICK):
            if env.done[0]:
                break
            img = torch.from_numpy(obs["img"]); prop = torch.from_numpy(obs["prop"]); task = torch.from_numpy(obs["task"])
            valid = bool(obs["valid"][0])
            t0 = time.perf_counter()
            pool = M.pool_feat(img)
            do = valid
            if kind != "full":
                forced = since >= V.KMAX_SKIP or t == 0 or (prev_task is not None and bool((task != prev_task).any())) \
                    or (prev_held and prop[0, 2] < 0.5 and prev_grip)
                if kind == "pix":
                    want = last_pool is None or float((pool - last_pool).pow(2).mean().sqrt()) > par
                else:
                    want = float(net.trig(torch.cat([h, a_prev, pool], -1))) > par
                do = valid and (forced or want)
            z = net.enc(M.to_x(img), task) if do else torch.zeros(1, M.Z)
            vf = torch.tensor([1.0 if do else 0.0])
            h = net.cell_step(h, z, vf, prop, task, a_prev)
            lg = net.policy(h, prop, task, z * vf[:, None])
            act = torch.cat([torch.tanh(lg[:, :2]), (lg[:, 2:3] > 0).float()], -1)
            lat.append(time.perf_counter() - t0)
            enc_flags.append(do)
            since = 0 if do else since + 1
            last_pool = pool if (do or last_pool is None) else last_pool
            prev_task, prev_held, prev_grip, a_prev = task, bool(prop[0, 2] > 0.5), bool(prop[0, 3] > 0.5), act
            obs, _ = env.step(act.numpy())
    out = pct(lat)
    out["enc_frac"] = float(np.mean(enc_flags))
    out["enc_step"] = pct([l for l, f in zip(lat, enc_flags) if f]) if any(enc_flags) else None
    out["skip_step"] = pct([l for l, f in zip(lat, enc_flags) if not f]) if not all(enc_flags) else None
    return out


def main():
    ck, out = sys.argv[1], sys.argv[2]
    taus = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    net, _ = T.load_net(ck, "cpu")
    res = {"checkpoint": ck, "platform": platform.platform(), "torch": torch.__version__, "cpu_count": torch.get_num_threads(), "runs": {}}
    for th in (1, 4):
        for trig in ["full"] + list(taus.keys()):
            spec = "full" if trig == "full" else taus[trig]
            res["runs"][f"threads{th}|{trig}"] = run(net, spec, th)
            print(th, trig, res["runs"][f"threads{th}|{trig}"]["p50_ms"], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
