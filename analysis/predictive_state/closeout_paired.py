"""EXP-0013 Phase 1.1 closeout: corrected DIAGNOSTIC re-evaluation under the paired-replay protocol (env_paired.PairedLAEnv).

The frozen Phase 1.1 controllers are used unchanged (obs_A_single_frame, obs_B_map_memory with the committed dilation; privileged greedy,
reactive_memory, oracle_descent), same DEV seed 5001, same 200 sampled layout ids, same 65-step success metric / 80-step simulator cap.
This DEV set was already examined in Phase 1.1, so the result is a corrected re-evaluation, NOT an untouched holdout. No tuning.
Usage: python -m analysis.predictive_state.closeout_paired COND OUT.json      (COND in LA_CLEAR, LA_NOM, LA_OCC)
Writes raw per-episode rows only; every summary is derived from them by analyze_closeout.py.
"""
import json
import subprocess
import sys
import time

import numpy as np

from . import controllers_la as C
from . import controllers_obs as O
from . import env_la as L
from . import env_paired as P
from .closeout_repro_v1 import pack
from .run_phase11_benchmark import CONDS, EVAL_SEED

PHASE11_DIL = "results/predictive_state_phase1_1/dilation_selection.json"


def main(cn, out, n=200):
    dil = json.load(open(PHASE11_DIL))["selected_dilation"]
    ctrls = [("greedy", "priv", lambda: C.Greedy()), ("reactive_memory", "priv", lambda: C.Reactive(True)), ("oracle_descent", "priv", lambda: C.Oracle("descent")),
             ("obs_A_single_frame", "obs", lambda: O.ObsA(dil)), ("obs_B_map_memory", "obs", lambda: O.ObsB(dil))]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"protocol": P.PROTOCOL, "cond": cn, "cond_params": vars(CONDS[cn]) | {"dur": list(CONDS[cn].dur)}, "split": "dev", "seed": EVAL_SEED, "n": n, "dilation": dil,
           "budget": L.LA_T_PICK, "sim_cap": 80, "repo_sha": sha, "controllers": {}}
    for name, kind, mk in ctrls:
        env = P.PairedLAEnv(n, "dev", CONDS[cn], EVAL_SEED, t_limit=80)
        spawn, sched = env.spawn_digest(), env.schedule_digest(82)
        t0 = time.perf_counter()
        ctrl = mk()
        # run_episodes* call env.reset() themselves; the paired reset is deterministic so the digests taken above stay valid
        r = C.run_episodes(env, ctrl, T=82) if kind == "priv" else O.run_episodes_obs(env, ctrl)
        ok = r["success"] > 0
        steps = len(env.valid_log)
        realized = np.stack(env.valid_log, 1)
        scheduled = env.scheduled_valid(steps - 1)
        res["controllers"][name] = {
            "episode_id": env.episode_ids.tolist(), "layout_id": env.layout_ids.tolist(), "spawn_digest": spawn, "schedule_digest": sched,
            "success_at_budget": (ok & (r["finish"] <= L.LA_T_PICK)).astype(int).tolist(), "success_at_80": ok.astype(int).tolist(),
            "finish": r["finish"].tolist(), "fail_kind": r["fail_kind"].tolist(), "effort": r["effort"],
            "n_observations": steps, "realized_valid_equals_schedule": bool((realized == scheduled).all()), "valid_log": pack(env.valid_log),
            "wall_s": time.perf_counter() - t0}
        print(cn, name, float((ok & (r["finish"] <= L.LA_T_PICK)).mean()), float(ok.mean()), steps, res["controllers"][name]["realized_valid_equals_schedule"],
              round(time.perf_counter() - t0, 1), flush=True)
        json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
