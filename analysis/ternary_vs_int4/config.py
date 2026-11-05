"""Preregistered configuration of EXP-0011 Stage 1. Everything here is hashed into protocol_v1.json by lock.py before any quality or
performance measurement. Changing a value after the lock is a documented amendment, never a silent edit."""
from __future__ import annotations

from pathlib import Path

EXP_ROOT = Path.home() / "exp0011"            # WSL ext4 working area (runtimes, models, data); not in git
MODELS_DIR = EXP_ROOT / "models"
DATA_DIR = EXP_ROOT / "data"
RUNTIME_BIN = {"bitnet": EXP_ROOT / "BitNet" / "build" / "bin",              # bitnet.cpp main 0b341e5: UNUSABLE for BitNet 2B4T here (PPL 97-119, see amendment 3); kept for provenance
               "bitnet_old": EXP_ROOT / "BitNet_old" / "build" / "bin",      # bitnet.cpp 765741d (2025-05-21), contemporary with the model release: reproduces the reference perplexity
               "upstream": EXP_ROOT / "llama.cpp" / "build" / "bin"}

# Candidate selection (documented before any quality or performance test): one ternary model, five INT4 GGUF models spanning quality.
# `card_avg` is Microsoft's own 16-task average from the BitNet 2B4T model card (vendor-reported, NOT our measurement); it is listed only to
# show why each candidate was chosen. Quality for the decision is measured here (QUALITY_TASKS).
MODELS = {
    "BITNET_2B4T_I2S": dict(file="bitnet-b1.58-2B-4T-i2_s.gguf", fmt="ternary I2_S (1.58-bit)", runtimes=["bitnet_old"], card_avg=54.19,
                            why="the ternary model under test"),
    "QWEN25_1P5B_Q4KM": dict(file="qwen2.5-1.5b-instruct-q4_k_m.gguf", fmt="INT4 Q4_K_M", runtimes=["upstream", "bitnet_old"], card_avg=55.23,
                             why="highest card average of the comparison set: the closest quality match to BitNet"),
    "QWEN25_1P5B_Q40": dict(file="qwen2.5-1.5b-instruct-q4_0.gguf", fmt="INT4 Q4_0", runtimes=["upstream"], card_avg=55.23,
                            why="same model in the simplest 4-bit block format (no k-quant overhead)"),
    "QWEN25_3B_Q4KM": dict(file="qwen2.5-3b-instruct-q4_k_m.gguf", fmt="INT4 Q4_K_M", runtimes=["upstream"], card_avg=None,
                           why="stronger, larger comparator (not in the BitNet card); re-added after review because the 1.5B-class models alone could leave "
                               "no INT4 model at or above BitNet's quality. First attempt from the Qwen repo was too slow, the bartowski mirror is used"),
    "LLAMA32_1B_Q4KM": dict(file="llama-3.2-1b-instruct-q4_k_m.gguf", fmt="INT4 Q4_K_M", runtimes=["upstream"], card_avg=44.90,
                            why="smaller, lower-quality comparator, same tokenizer family as BitNet"),
    "SMOLLM2_1P7B_Q4KM": dict(file="smollm2-1.7b-instruct-q4_k_m.gguf", fmt="INT4 Q4_K_M", runtimes=["upstream"], card_avg=48.70,
                              why="mid-quality comparator with a different architecture"),
}
TERNARY = "BITNET_2B4T_I2S"
PRIMARY_COMPARATOR = "QWEN25_1P5B_Q4KM"   # fixed comparator also reported separately (the per-cell oracle comparator is the headline)
INT4 = [m for m in MODELS if m != TERNARY]

# --- quality (llama-perplexity built-in tasks; the task subset is a deterministic function of the data file, identical for every model)
QUALITY_CTX = 2048
QUALITY_THREADS = 8
QUALITY_TASKS = {
    "hellaswag": dict(flag="--hellaswag", n_flag="--hellaswag-tasks", n=400, data="hellaswag_val_full.txt"),
    "winogrande": dict(flag="--winogrande", n_flag="--winogrande-tasks", n=500, data="winogrande-debiased-eval.csv"),
    "arc_easy": dict(flag="--multiple-choice", n_flag="--multiple-choice-tasks", n=500, data="arc-easy-validation.bin", np=5),   # np=5: some ARC questions have 5 options (amendment 1)
    "arc_challenge": dict(flag="--multiple-choice", n_flag="--multiple-choice-tasks", n=299, data="arc-challenge-validation.bin", np=5),
    "mmlu": dict(flag="--multiple-choice", n_flag="--multiple-choice-tasks", n=500, data="mmlu-test.bin"),   # knowledge task closest to a Q&A workload
}
QUALITY_TOLERANCE_POINTS = 3.0     # a model is "quality-matched" to BitNet if its macro-average accuracy is >= BitNet's minus this
QUALITY_TOLERANCE_SENSITIVITY = (0.0, 5.0)

# --- performance grid. Context rule: prompt + output <= 4096 (BitNet 2B4T's trained maximum), so the nominal 4096 prompt becomes 4096 - G.
PROMPTS = (512, 1024, 2048, 4096)
OUTPUTS = (128, 256)
CTX_MAX = 4096
REPS = 3
SWEEP_REPS = 2                      # thread sweep only; the main grid uses REPS
COOLDOWN_S = 10
# Performance is measured only for the ternary model and INT4 models whose quality macro >= BitNet's minus the widest sensitivity tolerance
# (anything weaker can never be the quality-matched comparator, so timing it would not change the decision). Decided from quality results,
# written to results/ternary_vs_int4/perf_models.json by `summarize select` before any timing.
PERF_SET_TOLERANCE = 5.0
# protocol_v1_amendment2 (after quality, before any timing): because BitNet is below EVERY INT4 model in quality, the matched-pool rule selects all INT4 models;
# timing all of them (about 4+ hours) adds little. Timing is restricted to the INT4 models nearest to BitNet in size and quality plus the primary comparator.
PERF_SET_OVERRIDE = None   # amendment 4: the original rule (pool at the widest tolerance) applies again
THREAD_SWEEP = (4, 8, 12, 16)
# thread count per (model, runtime, regime) = argmin total latency at that regime's sweep cell; fixed before the main run.
# short regime (prompt <= 1024) is decode-dominated, long regime (prompt >= 2048) is prefill-heavy
THREAD_SWEEP_CELLS = {"short": (512, 128), "long": (2048, 128)}


def regime(p: int) -> str:
    return "short" if p <= 1024 else "long"

FIXED_THREADS = 8                   # also reported, for fairness to a fixed physical-core count
ORDER_SEED = 20261105
FLASH_ATTN = "1"                    # pinned explicitly for both runtimes (their "auto" resolution may differ)
MIN_CHARGE_PCT = 30                 # refuse to time while discharging; require at least this battery charge on AC
BUDGET_S = 240                      # start no new invocation after this many seconds in one foreground call (tool timeout is 600 s)


def p_eff(prompt: int, gen: int) -> int:
    return min(prompt, CTX_MAX - gen)


# --- Stage 1 decision rule (speed/memory only, NOT energy: Stage 2 owns every energy claim). Ratios are INT4 time / ternary time,
# i.e. >1 means ternary is faster. Comparator = the fastest INT4 (model, runtime, threads) among those quality-matched to BitNet.
SPEED_PASS = 2.0
SPEED_MIDDLE = 1.3
BYTES_CEILING_MARGIN = 1.2           # decode is largely memory-bound: a measured decode speed-up above (INT4 file bytes / ternary file bytes) x this is flagged for audit
TTFT_PARITY = 1.0
