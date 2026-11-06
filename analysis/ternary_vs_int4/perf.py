"""EXP-0011 Stage 1 latency/throughput/memory harness around llama-bench (identical tool in both runtimes).

Per (model, runtime, threads, prompt, gen) cell we run, as separate processes under /usr/bin/time -v (peak RSS):
  ttft   : llama-bench -pg P,1   -> time to process P tokens and emit the first token (excludes tokenization/sampling/server overhead)
  total  : llama-bench -pg P,G   -> time for P-token prefill + G generated tokens
Derived: decode_tps = (G-1) / (total - ttft); prefill_tps = P / ttft (approximate: ttft includes one decode step).
Results are written one JSON file per invocation (resumable, immutable). Order inside a round is shuffled with a fixed seed and cooled down
between invocations. This harness never claims energy.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from . import config as C
from . import envinfo

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "ternary_vs_int4" / "perf"


def parse_bench_json(text: str) -> dict:
    """llama-bench -o json prints a JSON array (log lines may precede it). Returns the first entry's timing fields."""
    start = re.search(r"^\[\s*$", text, flags=re.M)      # the JSON array opens on its own line; log lines may contain '[' elsewhere
    end = list(re.finditer(r"^\]\s*$", text, flags=re.M))[-1]
    arr = json.loads(text[start.start(): end.end()])
    e = arr[0]
    return {"samples_ns": e["samples_ns"], "avg_ns": e["avg_ns"], "n_prompt": e["n_prompt"], "n_gen": e["n_gen"], "n_depth": e.get("n_depth", 0),
            "model_size": e.get("model_size"), "model_n_params": e.get("model_n_params"), "build_commit": e.get("build_commit"),
            "cpu_info": e.get("cpu_info"), "backends": e.get("backends")}


def parse_time_v(text: str) -> dict:
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    wall = re.search(r"Elapsed \(wall clock\) time.*?:\s*([\d:.]+)", text)
    return {"max_rss_kb": int(rss.group(1)) if rss else None, "wall_str": wall.group(1) if wall else None}


def derive(ttft_ns: list[int], total_ns: list[int], p: int, g: int) -> dict:
    t1 = statistics.median(ttft_ns) / 1e9
    tt = statistics.median(total_ns) / 1e9
    dec = max(tt - t1, 1e-9)
    return {"ttft_s": t1, "total_s": tt, "decode_tps": (g - 1) / dec, "prefill_tps": p / t1}


def run_one(model_id: str, runtime: str, threads: int, p: int, g: int, kind: str, reps: int) -> dict:
    m = C.MODELS[model_id]
    n_gen = 1 if kind == "ttft" else g
    cmd = ["/usr/bin/time", "-v", str(C.RUNTIME_BIN[runtime] / "llama-bench"), "-m", str(C.MODELS_DIR / m["file"]), "-p", "0", "-n", "0",
           "-pg", f"{p},{n_gen}", "-t", str(threads), "-r", str(reps), "-ngl", "0", "-fa", C.FLASH_ATTN, "-o", "json"]
    snap = envinfo.snapshot(light=True)
    waited = 0
    while not snap["power_check"]["ok"] and waited < C.AC_WAIT_S:      # amendment 6: tolerate a short adapter blip, never time while it lasts
        time.sleep(10)
        waited += 10
        snap = envinfo.snapshot(light=True)
    if not snap["power_check"]["ok"]:
        raise SystemExit(f"refusing to time: power check failed {snap['power_check']} (needs AC and >= {C.MIN_CHARGE_PCT}% charge) after waiting {waited}s")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"model": model_id, "runtime": runtime, "threads": threads, "p": p, "g": g, "kind": kind, "reps": reps, "cmd": cmd, "returncode": r.returncode,
           "env_before": snap, "wall_s": time.time() - t0, "waited_for_ac_s": waited, "power_after": envinfo.power_check()}
    if r.returncode == 0:
        rec.update(parse_bench_json(r.stdout))
        rec.update(parse_time_v(r.stderr))
    else:
        rec["stderr_tail"] = r.stderr[-1500:]
    return rec


def path_for(stage: str, model_id: str, runtime: str, threads: int, p: int, g: int, kind: str) -> Path:
    return OUT / stage / f"{model_id}__{runtime}__t{threads}__p{p}_g{g}__{kind}.json"


def cells(stage: str):
    """Yield (model, runtime, threads, p_eff, g) work items for a stage: 'sweep' (thread selection) or 'main' (locked thread counts)."""
    perf_set = json.loads((REPO / "results" / "ternary_vs_int4" / "perf_models.json").read_text())["models"]
    if stage == "sweep":
        for p, g in C.THREAD_SWEEP_CELLS.values():
            for mid, m in C.MODELS.items():
                if mid not in perf_set:
                    continue
                for rt in m["runtimes"]:
                    for t in C.THREAD_SWEEP:
                        yield mid, rt, t, p, g
        return
    lock = json.loads((REPO / "results" / "ternary_vs_int4" / "thread_choice.json").read_text())
    for g in C.OUTPUTS:
        for pn in C.PROMPTS:
            p = C.p_eff(pn, g)
            for mid, m in C.MODELS.items():
                if mid not in perf_set:
                    continue
                for rt in m["runtimes"]:
                    yield mid, rt, lock[f"{mid}|{rt}|{C.regime(p)}"]["best_threads"], p, g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "main"])
    ap.add_argument("--budget-s", type=float, default=C.BUDGET_S, help="stop starting new invocations after this many seconds (foreground chunking)")
    ap.add_argument("--only-model", default=None)
    a = ap.parse_args(argv)
    work = list(cells(a.stage))
    if a.only_model:
        work = [w for w in work if w[0] == a.only_model]
    # shuffled order with a fixed seed => interleaved models, reproducible
    random.Random(C.ORDER_SEED).shuffle(work)
    t_start = time.time()
    done = skipped = 0
    for mid, rt, t, p, g in work:
        for kind in ("ttft", "total"):
            f = path_for(a.stage, mid, rt, t, p, g, kind)
            if f.exists():
                skipped += 1
                continue
            if time.time() - t_start > a.budget_s:
                print(f"budget reached: {done} run, {skipped} already done; re-invoke to continue", flush=True)
                return 0
            f.parent.mkdir(parents=True, exist_ok=True)
            rec = run_one(mid, rt, t, p, g, kind, C.SWEEP_REPS if a.stage == "sweep" else C.REPS)
            f.write_text(json.dumps(rec, indent=1))   # failed runs are kept too (negative results)
            done += 1
            print(f"{mid} {rt} t={t} P={p} G={g} {kind} rc={rec['returncode']} wall={rec['wall_s']:.0f}s rss={rec.get('max_rss_kb')}", flush=True)
            time.sleep(C.COOLDOWN_S)
    print(f"stage {a.stage} complete: {done} run, {skipped} already done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
