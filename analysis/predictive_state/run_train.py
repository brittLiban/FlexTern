"""Train a list of arms x seeds (resumable: finished checkpoints are skipped). Usage: python -m analysis.predictive_state.run_train G,G0 0,1,2"""
import os
import sys

from . import train as T

if __name__ == "__main__":
    arms = sys.argv[1].split(",")
    seeds = [int(s) for s in sys.argv[2].split(",")]
    out = sys.argv[3] if len(sys.argv) > 3 else "/home/liban/ps_ckpt"
    path = [T.gen_train(0)] + [x for x in os.environ.get("PS_EXTRA_DATA", "").split(",") if x]
    for s in seeds:
        for a in arms:
            T.train_arm(a, s, path, out)
