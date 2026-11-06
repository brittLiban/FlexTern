"""Re-measure one timing cell several times (EXP-0011 anomaly investigation). Results are written to results/ternary_vs_int4/perf/<out>/ as new files;
nothing existing is overwritten. Usage: python -m analysis.ternary_vs_int4.repeat_cell MODEL RUNTIME THREADS P G KIND N [OUTDIR]"""
from __future__ import annotations

import json
import sys
import time

from . import config as C
from . import perf


def main(argv):
    model, runtime, threads, p, g, kind, n = argv[:7]
    out = perf.OUT / (argv[7] if len(argv) > 7 else "anomaly_repeat")
    out.mkdir(parents=True, exist_ok=True)
    for i in range(int(n)):
        rec = perf.run_one(model, runtime, int(threads), int(p), int(g), kind, C.REPS)
        f = out / f"{model}__{runtime}__t{threads}__p{p}_g{g}__{kind}__rep{i}_{int(time.time())}.json"
        f.write_text(json.dumps(rec, indent=1))
        print(i, "rc", rec["returncode"], "samples_s", [round(x / 1e9, 2) for x in rec.get("samples_ns", [])], "wall", round(rec["wall_s"]), flush=True)
        time.sleep(C.COOLDOWN_S)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
