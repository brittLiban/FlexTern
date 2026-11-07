"""EXP-0013 Phase 1: before/after executed-encoder work, legacy (EXP-0012) vs gated rollout, on one checkpoint.
Counts come from a spy on net.enc.forward (actual executions), never inferred. Usage: python -m analysis.predictive_state.phase1_measure CKPT OUT.json [dev]"""
import json
import sys
import time

import numpy as np
import torch

from . import env as E
from . import evaluate as V
from . import train as T


def timed(fn, dev):
    if dev != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    if dev != "cpu":
        torch.cuda.synchronize()
    return r, time.perf_counter() - t0


def main():
    ck, out = sys.argv[1], sys.argv[2]
    dev = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    net, _ = T.load_net(ck, dev)
    spy = {"calls": 0, "samples": 0}
    orig = net.enc.forward

    def fwd(x, task):
        spy["calls"] += 1
        spy["samples"] += int(x.shape[0])
        return orig(x, task)
    net.enc.forward = fwd
    res = {"checkpoint": ck, "device": dev, "torch": torch.__version__, "episodes": 32, "cond": "NOM", "seed": 5, "runs": {}}
    cond = E.TEST_CONDS["NOM"]
    for trig in ("full", "fixed:0.25", "pix:0.03"):
        row = {}
        for name, fn in (("legacy", V.rollout_legacy), ("gated", V.rollout)):
            fn(net, "dev", cond, 5, 4, trig, dev, T=6)            # warm-up
            spy.update(calls=0, samples=0)
            r, dt = timed(lambda: fn(net, "dev", cond, 5, 32, trig, dev), dev)
            row[name] = {"executed_enc_calls": spy["calls"], "executed_enc_samples": spy["samples"],
                         "logical_enc_count": float(r["enc_count"].sum()), "alive_steps": float(r["alive_steps"].sum()),
                         "success": float(r["success"].mean()), "wall_s": dt, "per_episode_success": r["success"].tolist()}
        row["identical_per_episode_success"] = row["legacy"]["per_episode_success"] == row["gated"]["per_episode_success"]
        for k in ("legacy", "gated"):
            row[k].pop("per_episode_success")
        res["runs"][trig] = row
        print(trig, json.dumps({k: v for k, v in row.items()}), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
