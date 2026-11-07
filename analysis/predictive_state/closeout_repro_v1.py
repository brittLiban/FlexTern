"""EXP-0013 Phase 1.1 closeout, step 1: reproduce the HISTORICAL (v1, shared-RNG) paired run with the code unchanged and log, per controller,
the realized `valid` (blackout) sequence of every episode. A read-only subclass records `valid` in observe(); it does not touch any RNG, so
behaviour is identical to the original run. Purpose: (a) confirm the historical per-episode outcomes reproduce exactly, (b) measure how many
episodes saw different blackout schedules under different controllers (the pairing defect).
Usage: python -m analysis.predictive_state.closeout_repro_v1 COND OUT.json      (COND in LA_CLEAR, LA_NOM, LA_OCC)
"""
import json
import sys
import time

import numpy as np

from . import controllers_la as C
from . import controllers_obs as O
from . import env as E
from . import env_la as L
from .run_phase11_benchmark import CONDS, EVAL_SEED


class RecordingLAEnv(L.LAEnv):
    """v1 LAEnv + a passive log of the valid flag of every observation (no RNG use)."""
    def observe(self):
        o = super().observe()
        if not hasattr(self, "valid_log"):
            self.valid_log = []
        self.valid_log.append(o["valid"].copy())
        return o


def pack(log):
    """list of (n,) bool -> list of per-episode '0/1' strings (index = time step, 0 = reset observation)."""
    a = np.stack(log, 1).astype(int)
    return ["".join(map(str, r)) for r in a]


def main(cn, out, n=200):
    dil = json.load(open("results/predictive_state_phase1_1/dilation_selection.json"))["selected_dilation"]
    ctrls = [("greedy", "priv", lambda: C.Greedy()), ("reactive_memory", "priv", lambda: C.Reactive(True)), ("oracle_descent", "priv", lambda: C.Oracle("descent")),
             ("obs_A_single_frame", "obs", lambda: O.ObsA(dil)), ("obs_B_map_memory", "obs", lambda: O.ObsB(dil))]
    res = {"version": "v1-shared-rng-reproduction", "cond": cn, "seed": EVAL_SEED, "n": n, "dilation": dil, "controllers": {}}
    for name, kind, mk in ctrls:
        env = RecordingLAEnv(n, "dev", CONDS[cn], EVAL_SEED, t_limit=80)
        t0 = time.perf_counter()
        ctrl = mk()
        r = C.run_episodes(env, ctrl, T=82) if kind == "priv" else O.run_episodes_obs(env, ctrl)
        ok = r["success"] > 0
        res["controllers"][name] = {"layout_ids": env.layout_ids.tolist(), "success_at_budget": (ok & (r["finish"] <= L.LA_T_PICK)).astype(int).tolist(),
                                    "success_at_80": ok.astype(int).tolist(), "finish": r["finish"].tolist(), "fail_kind": r["fail_kind"].tolist(),
                                    "valid_log": pack(env.valid_log), "wall_s": time.perf_counter() - t0}
        print(cn, name, float((ok & (r["finish"] <= L.LA_T_PICK)).mean()), round(time.perf_counter() - t0, 1), flush=True)
        json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
