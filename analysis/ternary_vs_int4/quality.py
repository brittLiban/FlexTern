"""EXP-0011 Stage 1 quality harness: llama-perplexity built-in multiple-choice tasks (HellaSwag, Winogrande, ARC-Easy, ARC-Challenge).

The same tool and task subsets are used for every model (the subset is a deterministic function of the data file). Instruction-tuned models are
scored with plain log-likelihood prompts, so absolute numbers differ from the BitNet model card; only within-run comparisons are used.
Standard errors are binomial (n tasks), so differences below ~3 points are inside noise for n=400-500.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from . import config as C
from . import envinfo

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "ternary_vs_int4" / "quality"


def parse_score(task: str, text: str) -> dict:
    """Return {'acc': percent, 'n': tasks} parsed from llama-perplexity output."""
    if task == "hellaswag":
        # newer builds print "12<TAB>25.0%<TAB>[lo, hi]", bitnet.cpp 765741d prints "12<TAB>25.0" (no percent sign)
        rows = re.findall(r"^(\d+)\t([\d.]+)%?(?:\t|$)", text, flags=re.M)
        if not rows:
            raise ValueError("no hellaswag rows")
        n, acc = rows[-1]
        return {"acc": float(acc), "n": int(n)}
    if task == "winogrande":
        # this llama.cpp build prints running rows "<i>\t<acc%>\t<logprob> <logprob> <a> <b>" and no final line: the last row is the final score
        rows = re.findall(r"^(\d+)\t([\d.]+)\t\s*-?[\d.]+\s+-?[\d.]+", text, flags=re.M)
        if not rows:
            raise ValueError("no winogrande rows")
        n, acc = rows[-1]
        return {"acc": float(acc), "n": int(n)}
    m = re.search(r"Final result:\s*([\d.]+)\s*\+/-\s*([\d.]+)", text)
    n = re.search(r"(\d+) questions", text) or re.search(r"Evaluating (\d+) tasks", text)
    if not m:
        raise ValueError("no multiple-choice final result")
    return {"acc": float(m.group(1)), "n": int(n.group(1)) if n else C.QUALITY_TASKS[task]["n"]}


def stderr_points(acc: float, n: int) -> float:
    p = acc / 100.0
    return 100.0 * math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


def run_task(model_id: str, runtime: str, task: str) -> dict:
    spec, m = C.QUALITY_TASKS[task], C.MODELS[model_id]
    cmd = [str(C.RUNTIME_BIN[runtime] / "llama-perplexity"), "-m", str(C.MODELS_DIR / m["file"]), "-f", str(C.DATA_DIR / spec["data"]),
           spec["flag"], spec["n_flag"], str(spec["n"]), "-c", str(C.QUALITY_CTX), "-t", str(C.QUALITY_THREADS), "-ngl", "0"]
    cmd += list(m.get("extra_args", []))
    if "np" in spec:
        cmd += ["-np", str(spec["np"])]     # protocol_v1_amendment1: ARC has 5-option questions
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"model": model_id, "runtime": runtime, "task": task, "cmd": cmd, "returncode": r.returncode, "wall_s": time.time() - t0,
           "env_before": envinfo.snapshot(light=True)}
    out = r.stdout + "\n" + r.stderr
    if r.returncode == 0:
        try:
            rec.update(parse_score(task, out))
            rec["stderr_points"] = stderr_points(rec["acc"], rec["n"])
        except ValueError as e:
            rec["parse_error"] = str(e)
    rec["stdout_tail"] = r.stdout[-1500:]     # score rows go to stdout, perf lines to stderr (earlier runs kept only a combined 2500-char tail)
    rec["stderr_tail"] = r.stderr[-1500:]
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", help="default: all models, each on its primary runtime")
    ap.add_argument("--runtime", default=None)
    ap.add_argument("--task", action="append", help="default: all tasks")
    ap.add_argument("--budget-s", type=float, default=520)
    ap.add_argument("--tag", default="primary", help="'primary' (counts toward the decision) or e.g. 'crosscheck'")
    a = ap.parse_args(argv)
    models = a.model or list(C.MODELS)
    tasks = a.task or list(C.QUALITY_TASKS)
    t_start = time.time()
    for mid in models:
        rt = a.runtime or ("bitnet_old" if mid == C.TERNARY else "bitnet")   # primary scores: ternary on the validated bitnet.cpp; INT4 scores were produced earlier under bitnet.cpp main (amendment 3)
        for task in tasks:
            f = OUT / a.tag / f"{mid}__{rt}__{task}.json"
            if f.exists():
                continue
            if time.time() - t_start > a.budget_s:
                print("budget reached; re-invoke to continue", flush=True)
                return 0
            f.parent.mkdir(parents=True, exist_ok=True)
            rec = run_task(mid, rt, task)
            f.write_text(json.dumps(rec, indent=1))
            print(mid, rt, task, rec.get("acc"), rec.get("n"), f"rc={rec['returncode']} {rec['wall_s']:.0f}s", flush=True)
    print("quality run complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
