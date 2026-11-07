"""EXP-0013 Phase 1.1 closeout: build results/predictive_state_phase1_1_closeout/reproducibility_manifest.json.
Records the commit, entry points and arguments, package versions, split/seed/episode provenance, frozen controller configuration, random-stream
policy, statistics version, sha256 of every raw result file, power-state snapshots and the observed test runs (parsed from the saved logs in
test_logs/). Usage: python -m analysis.predictive_state.closeout_manifest"""
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import analyze_closeout as A
from . import env_paired as P

R = Path("results/predictive_state_phase1_1_closeout")
FROZEN = Path("results/predictive_state_phase1_1")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def parse_logs():
    runs = []
    for f in sorted((R / "test_logs").glob("*.log")):
        txt = f.read_text(errors="replace")
        for m in re.finditer(r"\{'module': '([\w]+)', 'status': '(\w+)', 'tests': (\w+), 'skipped': (\d+), 'seconds': ([\d.]+)\}", txt):
            runs.append({"log": f.name, "module": m.group(1), "status": m.group(2), "tests": None if m.group(3) == "None" else int(m.group(3)),
                         "skipped": int(m.group(4)), "seconds": float(m.group(5))})
        for m in re.finditer(r"Ran (\d+) tests? in ([\d.]+)s\s+(OK|FAILED[^\n]*)", txt):
            runs.append({"log": f.name, "unittest_summary": f"Ran {m.group(1)} tests in {m.group(2)}s {m.group(3)}"})
    return runs


def main():
    raw = sorted(p for p in R.glob("*.json") if p.name != "reproducibility_manifest.json") + [R / "closeout_tables.md", R / "power_state.jsonl"]
    man = {
        "created_by": "analysis/predictive_state/closeout_manifest.py",
        "base_main_sha": "6cbd44b2f93721a4f63201ebbc796e62985e4f3a", "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "code_sha_at_manifest_time": git("rev-parse", "HEAD"), "working_tree_dirty_at_manifest_time": bool(git("status", "--porcelain")),
        "environment": {"host_python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
                        "wsl_timing": "torch 2.11.0+cu128, Linux-6.18 WSL2 Ubuntu (see timing_retime_train_trace.json)"},
        "splits": {"train": [0, 400], "dev": [1000, 1100], "test (reserved; unused by every closeout script)": [2000, 2200]},
        "evaluation": {"split": "dev", "seed": 5001, "n_episodes": 200, "episode_ids": "arange(200)", "layout_ids": "default_rng([5001,777]).integers(1000,1100,200) (87 unique)",
                       "conditions": {k: {"p_blackout": v.p_blackout, "dur": list(v.dur), "k": v.k} for k, v in __import__("analysis.predictive_state.run_phase11_benchmark", fromlist=["CONDS"]).CONDS.items()},
                       "step_budget_success": 65, "simulator_cap": 80, "dilation": json.load(open(FROZEN / "dilation_selection.json"))["selected_dilation"],
                       "controllers": list(A.CTRLS), "controller_changes_after_dev_observation": "none"},
        "random_stream_policy": {"protocol": P.PROTOCOL, "streams": {"spawn": "(seed, episode_id, 1)", "blackout": "(seed, episode_id, 2); uniforms+durations for t < %d" % P.TMAX,
                                                                      "render_noise": "(seed, episode_id, 3, t)"}, "shared_generator": "refused", "n_picks": "must be 1", "test_split": "refused"},
        "statistics": {"version": A.STATS_VERSION, "bootstrap_B": A.B, "bootstrap_seed": 0, "episode_ci": "Wilson 95%", "cluster_ci": "layout-clustered percentile bootstrap",
                       "zero_variance": "exact one-sided Clopper-Pearson over unique layouts", "mcnemar": "exact two-sided, '0 discordant pairs' reported without p"},
        "commands": [
            "python -m analysis.predictive_state.closeout_repro_v1 COND results/predictive_state_phase1_1_closeout/repro_v1_COND.json   (COND = LA_CLEAR, LA_NOM, LA_OCC; ~4 min each)",
            "python -m analysis.predictive_state.closeout_paired COND results/predictive_state_phase1_1_closeout/paired_replay_COND.json   (~5 min each)",
            "python -m analysis.predictive_state.analyze_closeout   -> closeout_summary.json, closeout_tables.md",
            "powershell -File scripts/capture_power_state.ps1 LABEL >> power_state.jsonl   (read-only)",
            "WSL (what was actually run): cp results/predictive_state/checkpoints/G_s0.pt /tmp/G_s0_repo.pt (same sha256); python -m analysis.predictive_state.phase11_timing /tmp/G_s0_repo.pt results/predictive_state_phase1_1_closeout/timing_retime_train_trace.json 12   (~6 min; trace split TRAIN by default; env TORCHDYNAMO_DISABLE=1)",
            "python scripts/run_tests.py smoke|benchmark|predictive|full --chunk K"],
        "timing": {"checkpoint": "results/predictive_state/checkpoints/G_s0.pt", "checkpoint_sha256": sha("results/predictive_state/checkpoints/G_s0.pt"),
                   "triggers": {"full": "full", "fixed": "fixed:0.25", "pix": "pix:0.0199", "learned": "learned:0.1445"}, "trace_split": "train",
                   "power_state": "host-reported AC (BatteryStatus 2, PowerOnline true) before and after; see power_state.jsonl",
                   "limits": "single checkpoint, single machine, WSL2 clocks/thermals uncontrolled, no energy measurement"},
        "not_independently_reproduced": ["historical timing on TEST traces (timing_paired*.json, op_counts.json): regenerable only with FLEXTERN_ALLOW_TEST_SPLIT=1; not re-run",
                                          "Phase 1.1 horizon grid (not re-run; re-tabulated only)"],
        "raw_files_sha256": {str(p): sha(p) for p in raw if p.exists()},
        "historical_files_unchanged": {"results/predictive_state_phase1_1/paired.json": sha(FROZEN / "paired.json"),
                                       "git_diff_vs_main_for_results_predictive_state_phase1_x": git("diff", "--stat", "6cbd44b", "--", "results/predictive_state_phase1", "results/predictive_state_phase1_1") or "empty"},
        "tests_observed_this_session": parse_logs(),
    }
    json.dump(man, open(R / "reproducibility_manifest.json", "w"), indent=1)
    print("manifest written;", len(man["raw_files_sha256"]), "raw files hashed;", len(man["tests_observed_this_session"]), "test log entries")


if __name__ == "__main__":
    main()
