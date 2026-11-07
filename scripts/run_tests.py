"""Tiered Python test runner (EXP-0013 Phase 1.1 closeout test-budget policy).

  python scripts/run_tests.py smoke        numpy-only, seconds   : reserved-split guard, paired-replay env, LA env
  python scripts/run_tests.py predictive   predictive-state regression (needs torch): smoke + inference/executed-work + controllers + EXP-0012 env tests
  python scripts/run_tests.py benchmark    OccPick2D-LA benchmark tests: env, paired replay, observation-matched controllers, split guard
  python scripts/run_tests.py full         every tests/test_*.py module, run as separate chunks of <= BUDGET_S estimated seconds
  python scripts/run_tests.py full --chunk K   run only chunk K (1-based) of `full`, so each invocation stays under the 15-minute budget
  python scripts/run_tests.py list         tiers, modules and recorded durations

Estimated durations come from tests/test_budget.json (measured on the founder's laptop, Windows host Python 3.12, one module per process).
A chunk whose estimate exceeds BUDGET_S (900 s = the 15-minute rule) is refused unless --approved is passed (which also lifts the per-module timeout);
`full` splits modules into chunks that each fit the budget, and every chunk is an independently valid run (modules are independent processes). A single
module is never split; if one alone is estimated above the budget, it needs approval. The exit code is non-zero if any
module fails. Prints what actually ran, outcome, skipped count and measured seconds per module.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGET_S = 900
TIERS = {
    "smoke": ["test_split_guard", "test_env_paired", "test_transition_events", "test_env_la"],
    "benchmark": ["test_split_guard", "test_env_la", "test_env_paired", "test_transition_events", "test_controllers_obs"],
    "predictive": ["test_split_guard", "test_env_paired", "test_transition_events", "test_env_la", "test_controllers_obs", "test_predictive_state", "test_inference_skip", "test_inference_startup", "test_executed_work"],
}
BUDGET = json.loads((ROOT / "tests" / "test_budget.json").read_text())["modules_seconds"]
ALL = sorted(p.stem for p in (ROOT / "tests").glob("test_*.py"))


def est(mods):
    return sum(BUDGET.get(m, 120) for m in mods)


def chunks(mods):
    """Greedy chunking by recorded duration; a module whose own estimate exceeds the budget gets a chunk of its own and is refused without --approved."""
    out, cur = [], []
    for m in mods:
        if cur and est(cur + [m]) > BUDGET_S:
            out.append(cur); cur = []
        cur.append(m)
    return out + ([cur] if cur else [])


def run_module(m, approved=False):
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-m", "unittest", f"tests.{m}"], cwd=ROOT, capture_output=True, text=True, timeout=None if approved else BUDGET_S)
    except subprocess.TimeoutExpired:
        return {"module": m, "status": "TIMEOUT", "tests": None, "skipped": 0, "seconds": BUDGET_S}
    out = p.stderr + p.stdout
    ran = re.search(r"Ran (\d+) tests? in", out)
    skipped = re.search(r"skipped=(\d+)", out)
    status = "OK" if p.returncode == 0 else "FAIL"
    if status == "FAIL":
        print(out[-3000:])
    return {"module": m, "status": status, "tests": int(ran.group(1)) if ran else None, "skipped": int(skipped.group(1)) if skipped else 0, "seconds": round(time.perf_counter() - t0, 1)}


def only_missing(argv):
    return "--chunk" not in argv and "--approved" not in argv


def main():
    tier = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    approved = "--approved" in sys.argv
    if tier == "list":
        for k, v in TIERS.items():
            print(k, v, "est %ds" % est(v))
        print("full", ALL, "est %ds in %d chunks" % (est(ALL), len(chunks(ALL))))
        return 0
    mods = ALL if tier == "full" else TIERS[tier]
    if tier == "full" and only_missing(sys.argv):
        print("full without --chunk runs every chunk back to back (> 15 min expected); pass --chunk K, or --approved"); return 2
    if tier != "full" and est(mods) > BUDGET_S and not approved:
        print(f"refused: estimated {est(mods)} s exceeds the {BUDGET_S} s budget; pass --approved after founder approval"); return 2
    results = []
    only = int(sys.argv[sys.argv.index("--chunk") + 1]) if "--chunk" in sys.argv else None
    for i, ch in enumerate(chunks(mods) if tier == "full" else [mods]):
        if only is not None and i + 1 != only:
            continue
        if est(ch) > BUDGET_S and not approved:
            print(f"refused chunk {i + 1} {ch}: estimated {est(ch)} s exceeds the {BUDGET_S} s budget; get founder approval and pass --approved (a single module is never split)")
            return 2
        print(f"-- chunk {i + 1}: {ch} (estimated {est(ch)} s)", flush=True)
        for m in ch:
            r = run_module(m, approved); results.append(r); print(r, flush=True)
    print("TOTAL: modules", len(results), "tests", sum(r["tests"] or 0 for r in results), "skipped", sum(r["skipped"] for r in results),
          "seconds", round(sum(r["seconds"] for r in results), 1), "failed", [r["module"] for r in results if r["status"] != "OK"])
    return 0 if all(r["status"] == "OK" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
