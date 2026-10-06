# Handoff

Date: 2026-10-06
Agent: Claude (Claude Code, model Fable 5.1), implementation role
Current commit: the EXP-0004 commit ("research: evaluate INT4 head and KV cache on Falcon-E 1B"), followed by a docs-only fixup commit recording this hash

## Objective

FT-CURRENT, EXP-0004: can Falcon-E 1B (ADR-0004 revision) keep acceptable quality with an INT4 output head, a ~4-bit KV cache, and both. Complete.

## What I Completed

Feasibility audit; an auditable INT4 RTN g128 head quantizer with explicit nibble packing; the transformers KIVI-style `QuantizedCache` (hqq backend) exercised through real prefill, residual flushes and per-step retrieval, in two grouping variants; lm-eval on six tasks for baseline and INT4 head plus a KV-invariance control; teacher-forced incremental perplexity through the real cache at 2k and 4k context with per-position buckets; greedy generation at three prompt lengths; KV memory sweep 128–16k; latency; candidate-format report; EXP-0003 re-evaluation with measured bits; 11 unit tests (all 11 executed with the model loaded under `FLEXTERN_SLOW_TESTS=1`; 3 are gated behind that flag by default); EXP-0004 record; ADR-0006; plan and completion reviews.

## What I Changed

- New: `analysis/model_precision/` (9 modules), `tests/test_model_precision.py`, `scripts/run_model_precision.ps1`, `results/model_precision/` (JSON, CSV, two Markdown reports), ADR-0006, EXP-0004, this handoff, two reviews.
- Updated: `PROJECT_STATE.md`, `CURRENT_WORK.md`, `THESIS.md` (H6), `ROADMAP.md` (model-side track), `OPEN_QUESTIONS.md` (Q-019, Q-021 evidence), `KNOWN_ISSUES.md` (KI-005, KI-006, KI-007), `README.md`, `analysis/README.md`.
- Untouched: RTL, `results/roofline/`, `results/workload_profile/`, `BENCHMARKS.md` (hardware scoreboard; software quality results stay in the EXP record).
- Outside the repository: `C:\Users\liban\.wslconfig` (WSL2 memory 6 GB, swap 6 GB) and a Python venv at `~/ftvenv` in WSL Ubuntu with torch 2.11.0+cu128, transformers 5.18.0, hqq, lm_eval, datasets. The Hugging Face cache on the Windows side holds the Falcon-E checkpoint and the WikiText/lm-eval datasets.

## Tests / Validation Run

```
# Windows host (Python 3.12)
python -m unittest tests.test_reference tests.test_analysis tests.test_analysis_cross_model tests.test_roofline
# WSL Ubuntu, ~/ftvenv
python -m unittest tests.test_model_precision -v                        # 8 OK, 3 skipped
FLEXTERN_SLOW_TESTS=1 python -m unittest tests.test_model_precision -v  # 11 OK, 71 s, GPU
python analysis/model_precision/audit.py
python analysis/model_precision/run_experiment.py --phases lmeval|incremental|generation|memory|latency ...
python analysis/model_precision/report.py && python analysis/model_precision/roofline_reconnect.py
```

Exact phase invocations and their arguments are in `results/model_precision/manifest_*.json`. The runner `scripts/run_model_precision.ps1 -Wsl` chains them; under the 6 GB WSL cap, run phases one at a time (KI-007).

## Actual Results

All MEASURED, one checkpoint, pilot scale; see EXP-0004 R1–R8. Headline: INT4 head +0.67% WikiText perplexity, tasks within one stderr, head 134 MB → 35 MB logical; 4-bit KV +0.17% (per-token groups) to +1.07% (default strided groups) teacher-forced token perplexity at 2k–4k, cache 3.5× smaller; combined additive; resident memory shrinks but attention reads do not on this runtime; prefill 4–8× slower with the quantized cache; greedy long-prompt loop occurs at baseline too, earlier under quantization.

## Remaining Problems

- KV quality is pilot scale (≈3.9k scored tokens per context, no stderr), ≤ 4k context, greedy only.
- No calibration-based head quantizer for comparison; one recipe per component.
- The `QuantizedCache` dequantizes the whole history each step; there is no path in this repo that reads 4-bit K/V directly, so the EXP-0003 "logical" rows remain hypothetical.
- The quanto backend is unusable without a CUDA toolkit (KI-006); results use hqq.

## Important Context

- Loglikelihood tasks in lm-eval never touch the KV cache (demonstrated: B == baseline exactly). Never cite them as KV evidence.
- transformers' default `axis_key=axis_value=0` groups across tokens and drifts on every residual flush; `axis=1` (per-token along head_dim) is idempotent. `common.KV_INT4` keeps the default for fidelity to upstream; `KV_INT4_AXIS1` is the recommended variant.
- `cache_unpacked_ternary_weights` unpacks the ternary weights once to BF16 (identical logits, 2× decode speed, 3.17 GB resident). Use `--cache-unpacked` for anything incremental; stock `BitLinear` otherwise.
- `TORCHDYNAMO_DISABLE=1`, `HF_HOME` on `/mnt/c`, `TMPDIR=$HOME/tmp` (WSL `/tmp` is a 1.5 GB tmpfs) are required in WSL.
- `results/model_precision/lm_eval_limit20.json` is the 20-sample pipeline pilot; ignore it for numbers.

## Questions for Founder / ChatGPT

1. Is +0.5–1% perplexity with no task-accuracy change an acceptable loss for a FlexTern system format? No threshold was predeclared.
2. Should the candidate format (INT4 g128 head, 4-bit g64 per-token KV with BF16 residual window) be recorded as an ADR-level assumption for hardware work, or wait for a second checkpoint (BitNet 2B4T, 128k-vocabulary tied head) first?
3. Q-022: both representations only pay in traffic if attention and the head consume compressed operands directly. Is FlexTern willing to own that engine, or does it stay a GPU-kernel problem?

## Recommended Immediate Next Consideration

The bandwidth-matched streaming ternary dot-product engine study that the EXP-0003 handoff proposed: RTL plus generic synthesis of a 256–1,024-lane engine that consumes packed 2-bit bytes at line rate with 20–22-bit accumulation over K = 2048–13312, measured against the lanes-per-bandwidth break-even. EXP-0004 weakened the model-side objection to the ternary-first system on one checkpoint at pilot scale (the head and KV can be stored at ~4 bits with small, measured cost), so the premise every ceiling rests on, that the ternary stage can run at the bandwidth bound, is now the largest untested assumption in the thesis. Carry the EXP-0004 candidate format as the fixed interface assumption for that study (INT4 head bytes, 4.5-bit KV bytes), and do not widen it to attention or head engines until Q-022 is decided. The alternative, repeating EXP-0004 on BitNet 2B4T, is cheaper but answers a narrower question.
