"""EXP-0013 Phase 1: batch-1 CPU latency, EXP-0012 path (loaded from git history as _latency_exp0012.py) vs the shared-runner path.
Both skip at batch 1, so this is NOT a measurement of the skip fix; it checks the rebuilt path did not regress. Usage: CKPT OUT.json TAUS.json"""
import json, sys
from . import latency as NEW
from . import train as T
try:
    from . import _latency_exp0012 as OLD
except ImportError:
    OLD = None


def main():
    ck, out, taus = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
    net, _ = T.load_net(ck, "cpu")
    res = {}
    for th in (1, 4):
        for name, spec in [("full", "full")] + list(taus.items()):
            for tag, mod in (("old", OLD), ("new", NEW)):
                if mod is None:
                    continue
                r = mod.run(net, spec, th, episodes=10)
                res[f"threads{th}|{name}|{tag}"] = {k: r[k] for k in ("p50_ms", "p95_ms", "mean_ms", "n", "enc_frac")}
                if tag == "new":
                    res[f"threads{th}|{name}|new"]["component_mean_ms"] = r["component_mean_ms"]
                print(th, name, tag, {k: round(v, 3) for k, v in res[f"threads{th}|{name}|{tag}"].items() if k != "component_mean_ms"}, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
