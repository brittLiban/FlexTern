"""Write analysis/ternary_vs_int4/protocol_v1.json: the preregistered EXP-0011 Stage 1 protocol (constants, file hashes, runtime commits, code hashes).
Run once, commit, then measure. Later changes are amendments recorded in new files; this file is never edited."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from analysis.control_semantics import provenance

from . import bytes_model as bm
from . import config as C
from . import envinfo

HERE = Path(__file__).resolve().parent
CODE = ["config.py", "perf.py", "quality.py", "summarize.py", "envinfo.py", "bytes_model.py"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = HERE / "protocol_v1.json"
    if out.exists():
        print("protocol_v1.json already exists; refusing to overwrite (write an amendment instead)")
        return 1
    proto = {
        "experiment": "EXP-0011 Stage 1",
        "question": "At matched quality, does native ternary inference (BitNet b1.58 2B4T) substantially outperform strong optimized INT4 on this CPU?",
        "scope_limits": ["single x86 laptop CPU under WSL2; not an ARM/Jetson/edge target", "no energy measured (Stage 2 needs separate approval)",
                         "quality via log-likelihood multiple-choice subsets; absolute scores differ from the model card"],
        "models": {k: {**v, "sha256": sha(C.MODELS_DIR / v["file"]), "bytes": (C.MODELS_DIR / v["file"]).stat().st_size} for k, v in C.MODELS.items()},
        "datasets": {t["data"]: sha(C.DATA_DIR / t["data"]) for t in C.QUALITY_TASKS.values()},
        "quality": {"note": "tasks include MMLU (500) added after review; 4 tasks became 5", "tasks": {k: {kk: vv for kk, vv in v.items()} for k, v in C.QUALITY_TASKS.items()}, "ctx": C.QUALITY_CTX, "threads": C.QUALITY_THREADS,
                    "tolerance_points": C.QUALITY_TOLERANCE_POINTS, "tolerance_sensitivity": list(C.QUALITY_TOLERANCE_SENSITIVITY),
                    "macro": "unweighted mean of the four task accuracies", "primary_runtime_for_quality": "bitnet (one runtime for all models)"},
        "performance": {"prompts": list(C.PROMPTS), "outputs": list(C.OUTPUTS), "ctx_max": C.CTX_MAX, "prompt_rule": "p_eff = min(prompt, 4096 - gen)",
                        "reps": C.REPS, "sweep_reps": C.SWEEP_REPS, "cooldown_s": C.COOLDOWN_S,
                        "perf_set_rule": f"quality is measured first; timing runs only for BitNet and INT4 models with quality macro >= BitNet macro - {C.PERF_SET_TOLERANCE} "
                                         "(written to results/ternary_vs_int4/perf_models.json by `summarize select` before any timing)", "thread_sweep": list(C.THREAD_SWEEP), "thread_sweep_cells": {k: list(v) for k, v in C.THREAD_SWEEP_CELLS.items()},
                        "flash_attn": C.FLASH_ATTN, "min_charge_pct": C.MIN_CHARGE_PCT, "power_rule": "every timed invocation records AC/charge; the harness refuses to time on battery or below the minimum charge",
                        "budget_s": C.BUDGET_S,
                        "thread_rule": "per (model, runtime, regime short: prompt<=1024 / long: prompt>=2048): the thread count with the lowest median end-to-end latency at that regime's sweep cell, fixed before the main run",
                        "order_seed": C.ORDER_SEED, "tool": "llama-bench -pg P,1 (TTFT) and -pg P,G (total) under /usr/bin/time -v, -ngl 0"},
        "decision_rule": {
            "comparator": "per metric per cell, the fastest INT4 (model, runtime) in the matched pool = INT4 models with quality macro >= BitNet macro - tolerance (conservative for ternary)",
            "empty_pool": "if the pool is empty the outcome is NO_MATCHED_COMPARATOR: no ratios are computed (no fallback to weaker models); a quality-vs-speed table of all timed models is reported instead",
            "fixed_comparator_also_reported": C.PRIMARY_COMPARATOR,
            "headline": "geometric mean over the 8 grid cells of the end-to-end latency ratio (INT4 time / ternary time)",
            "labels": {"SPEED_PASS": f">= {C.SPEED_PASS}", "SPEED_MIDDLE": f"[{C.SPEED_MIDDLE}, {C.SPEED_PASS})", "SPEED_LOW": f"< {C.SPEED_MIDDLE}"},
            "flags": ["MIXED_DECODE_WIN_PREFILL_LOSS (decode geomean >= 1.3 and TTFT geomean < 1.0)",
                      f"AUDIT_DECODE_EXCEEDS_BANDWIDTH_IMPLIED_RATIO (decode geomean > geomean of per-cell (INT4 bytes streamed per token incl. KV) / (ternary bytes) x {C.BYTES_CEILING_MARGIN}; bytes from GGUF tensor types, see bytes_model.py)",
                      "WIN_NOT_UNIFORM_ACROSS_CELLS (worst cell < 1.3 while geomean >= 1.3)",
                      "COMPARATOR_ABOVE_TOLERANCE_TERNARY_DISADVANTAGED (a chosen comparator's macro quality exceeds BitNet's by more than the tolerance: the comparison is not matched)"],
            "interpretation": "Speed/memory only. Not an energy result, not evidence about FlexTern hardware. The 2.0/1.3 energy thresholds in the founder directive apply to Stage 2.",
        },
        "preregistered_prediction": {
            "basis": "static GGUF tensor accounting before any timing (bytes_model.py)",
            "statement": "BitNet 2B4T streams more bytes per decoded token than Qwen2.5-1.5B Q4_K_M because its tied 128k-vocabulary embedding/head is stored in F16 "
                         "(about 0.66 GB of its 1.18 GB) and its KV cache per token is larger (5 KV heads x 30 layers vs 2 x 28). So a bandwidth-bound decode should NOT be faster "
                         "for BitNet than for that comparator; a decode speed-up above the bandwidth-implied ratio would indicate a compute-side effect of the I2_S kernels. "
                         "HYPOTHESIZED, to be checked by the measurements.",
        },
        "pre_lock_disclosures": [
            "protocol was revised after an independent pre-lock review and before any measurement: Qwen2.5-3B re-added, MMLU added, empty-pool fallback removed, "
            "bytes-ceiling audit replaced by tensor-level accounting, per-regime thread sweep, explicit flash-attn, power refusal rule, shorter foreground budget",
            "before the lock I ran load-only smoke checks and two 1-repetition llama-bench smoke runs (BitNet tg8 about 14.5 tok/s; a 64-token-prompt/16-token run showing "
            "BitNet about 14.9 vs Qwen2.5-1.5B Q4_K_M about 17.8-18.7 tok/s decode). They were used only to validate the harness and are not data; the preregistered "
            "prediction below was derived from static GGUF accounting, but it was written after those smoke numbers were visible, so it is not blind",
        ],
        "bytes_profiles": {m: bm.profile(m) for m in C.MODELS},
        "missing_by_design": ["TL2 kernel for the ternary model was not built (ternary side uses I2_S only: it may be under-optimized)",
                              "no OpenVINO run (BitNet is not supported there; INT4-only comparison would not test the question on this AMD CPU)",
                              "no energy, no CPU frequency/temperature"],
        "runtimes": envinfo.runtime_versions(),
        "environment": envinfo.snapshot(),
        "code_sha256": {f: sha(HERE / f) for f in CODE},
        "repo_state": provenance.repo_state(),
    }
    out.write_text(json.dumps(proto, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
