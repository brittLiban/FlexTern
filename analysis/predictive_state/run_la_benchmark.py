"""EXP-0013 Phase 1: validate OccPick2D-LA with privileged controls (numpy only).
  calibrate : oracle shortest-path descent on the TRAIN layout range; writes the finish-step distribution used to LOCK the step budget
              (rule fixed in advance: LA_T_PICK = ceil(1.5 * p95 of oracle finish steps)). No reactive baseline is run here.
  evaluate  : all controllers on the DEV layout range (the TEST range is untouched), success vs step budget, effort.
Usage: python -m analysis.predictive_state.run_la_benchmark calibrate|evaluate OUT.json [n]"""
import json
import math
import sys
import time

import numpy as np

from . import controllers_la as C
from . import env_la as L

CAL_SEED, EVAL_SEED = 424242, 5001


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def calibrate(out, n=400):
    env = L.LAEnv(n, "train", L.LA_CONDS["LA_NOM"], CAL_SEED, t_limit=300)
    ctrl = C.Oracle("descent")
    r = C.run_episodes(env, ctrl, T=302)
    f = r["finish"][r["success"] > 0]
    p95 = float(np.percentile(f, 95))
    res = {"split": "train", "n": n, "seed": CAL_SEED, "oracle_success_at_300": float(r["success"].mean()), "finish_p50": float(np.median(f)),
           "finish_p95": p95, "finish_max": int(f.max()), "rule": "LA_T_PICK = ceil(1.5 * p95)", "locked_budget": math.ceil(1.5 * p95),
           "note": "base env.BatchEnv still enforces its own hard cap of 80 steps per pick"}
    json.dump(res, open(out, "w"), indent=1)
    print(res)


def controllers():
    cs = [C.Greedy(), C.Reactive(False), C.Reactive(True), C.Oracle("descent")]
    for cost in ("euclid", "geodesic"):
        for H in (1, 5, 10, 20):
            cs.append(C.Oracle("shoot", H=H, N=64, cost=cost, seed=H))
    return cs


def evaluate(out, n=200):
    res = {"split": "dev", "n": n, "seed": EVAL_SEED, "budget": L.LA_T_PICK, "controllers": {}}
    for ctrl in controllers():
        env = L.LAEnv(n, "dev", L.LA_CONDS["LA_NOM"], EVAL_SEED, t_limit=80)            # same layouts / spawns for every controller
        t0 = time.perf_counter()
        r = C.run_episodes(env, ctrl, T=82)
        wall = time.perf_counter() - t0
        ok = r["success"] > 0
        fin = r["finish"]
        row = {"success_at_budget": float((ok & (fin <= L.LA_T_PICK)).mean()), "ci95": wilson(int((ok & (fin <= L.LA_T_PICK)).sum()), n),
               "success_at_80": float(ok.mean()),
               "success_vs_budget": {str(b): float((ok & (fin <= b)).mean()) for b in (30, 40, 50, 60, 66, 80)},
               "median_finish_successful": float(np.median(fin[ok])) if ok.any() else None,
               "fail_kinds": {k: int((r["fail_kind"] == k).sum()) for k in set(r["fail_kind"].tolist()) if k},
               "effort": r["effort"], "wall_s": wall,
               "sim_steps_per_decision": r["effort"]["sim_steps"] / max(r["effort"]["decisions"], 1)}
        res["controllers"][ctrl.name] = row
        print(ctrl.name, {k: row[k] for k in ("success_at_budget", "success_at_80", "median_finish_successful", "sim_steps_per_decision")}, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else None
    {"calibrate": calibrate, "evaluate": evaluate}[mode](out, *( [n] if n else []))
