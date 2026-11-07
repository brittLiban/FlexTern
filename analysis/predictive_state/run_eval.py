"""Closed-loop evaluation driver. Usage:
python -m analysis.predictive_state.run_eval SPLIT ARMS SEEDS OUT.json [TRIGGER_SPEC]
TRIGGER_SPEC examples: full | learned@0.33,pix@0.33,fixed@0.33 (budget-calibrated on DEV, one tau per net).
Results are appended per (arm, seed, split, condition, trigger) so reruns skip finished entries."""
import json
import os
import sys

import numpy as np

from . import env as E
from . import evaluate as V
from . import train as T

N_EP = {"NOM": 300, "OCC": 300, "DIST": 300, "HARD": 300, "LONG": 100}
EVAL_SEED = {"dev": 5000, "test": 9000}


def main():
    split, arms, seeds, out = sys.argv[1], sys.argv[2].split(","), [int(s) for s in sys.argv[3].split(",")], sys.argv[4]
    trig_spec = sys.argv[5].split(",") if len(sys.argv) > 5 else ["full"]
    ckdir = os.environ.get("PS_CKPT", "/home/liban/ps_ckpt")
    conds = os.environ.get("PS_CONDS", "NOM,OCC,DIST,HARD,LONG").split(",")
    n_scale = float(os.environ.get("PS_NSCALE", 1.0))
    res = json.load(open(out)) if os.path.exists(out) else {}
    for arm in arms:
        for s in seeds:
            path = f"{ckdir}/{arm}_s{s}.pt"
            if not os.path.exists(path):
                print("missing", path)
                continue
            net, ck = T.load_net(path)
            for spec in trig_spec:
                if spec == "full":
                    trig = "full"
                else:
                    kind, bud = spec.split("@")
                    if net.cfg.kind != "rec" or not net.cfg.jepa:
                        continue
                    tau = V.calibrate(net, kind, float(bud))
                    trig = f"{kind}:{tau:.4f}"
                for cn in conds:
                    key = f"{arm}|{s}|{split}|{cn}|{spec}"
                    if key in res:
                        continue
                    n = max(20, int(N_EP[cn] * n_scale))
                    r = V.rollout(net, split, E.TEST_CONDS[cn], EVAL_SEED[split] + s, n, trig)
                    d = V.summarize(r)
                    d["trigger"] = trig
                    d["success_vec"] = r["success"].astype(int).tolist()
                    res[key] = d
                    json.dump(res, open(out, "w"))
                    print(key, round(d["success"], 3), "enc_frac", round(d["enc_frac"], 3), flush=True)


if __name__ == "__main__":
    main()
