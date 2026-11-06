# Review — EXP-0011 Stage 1: protocol (pre-lock) and results (pre-timing)

Reviewer: advisor agent, consulted twice by the implementing Claude agent: before the protocol lock (design) and after quality results and the first commit (results and claims). It saw the implementer's transcript, so it was not blind to the implementer's reasoning, and it did not re-run any experiment. A review is advice, not an accepted change.
Date: 2026-11-06 · Scope: `analysis/ternary_vs_int4/`, `tests/test_ternary_vs_int4.py`, `results/ternary_vs_int4/`, `docs/experiments/EXP-0011-*.md`, KI-020/KI-021.

## Objections and dispositions

### Pre-lock review (design)
1. **Matched-pool risk: the basket and the 1.5B-class models could leave no INT4 model at BitNet's quality; the empty-pool fallback would compare against weaker models.** Adopted: Qwen2.5-3B re-added (bartowski mirror), MMLU added, the fallback removed (NO_MATCHED_COMPARATOR). Outcome: with valid scores the tolerance-0 pool is still empty by 0.27 points (a tie within noise) and the tolerance-3 pool is {Qwen2.5-3B, SmolLM2}.
2. **The file-size 'bytes ceiling' was wrong.** Adopted: tensor-level accounting from GGUF types (weights streamed + KV at depth). It found the F16 tied head (0.66 of 1.18 GB). Also found: gguf-py mis-sizes bitnet.cpp's I2_S type (corrected in `bytes_model.py`).
3. **Fixes before running:** AC/charge recorded per invocation and refusal to time on battery (adopted, and it fired); explicit flash-attn (adopted, both runtimes); verify the `-pg` JSON on a real run (done); foreground budget 240 s (adopted); second thread-sweep regime (adopted); fixed-comparator column (adopted); TL2 (not built; recorded as a limitation).

### Results review (after the quality runs)
4. **The remaining +16% perplexity gap may be the missing pre-tokenizer.** Confirmed: with `tokenizer.ggml.pre=llama-bpe` the release-era runtime gives 16.80 vs 16.88 for the HF checkpoint. Amendment 5; BitNet quality re-run; the earlier scores retained as negative results. This moved BitNet's macro from 55.5 to 56.7 and the tolerance-3 pool from {Qwen2.5-1.5B, 3B, SmolLM2} to {3B, SmolLM2}.
5. **BitNet and INT4 were scored by different `llama-perplexity` versions.** Cross-check done: Qwen2.5-1.5B Q4_K_M on the May-2025 build gives macro 53.58 vs 53.61 on the main build (per-task offsets up to 1.4). INT4 scores kept.
6. **Unsupported claims in the first commit.** Corrected: KI-021 no longer attributes the 537 s stall to the battery transition (it was on AC); the 'repository's own instructions' phrase was removed (main was built by hand and `setup_env.py` was not run; a time-boxed run of the documented flow was not done); the +25% validity gate is labelled post-hoc; the HellaSwag reparse is disclosed.
7. **MMLU does not discriminate** (near chance for every model). Acknowledged; locked macro kept; a no-MMLU macro is reported post-hoc as sensitivity (BitNet 62.5, Qwen2.5-3B 61.0, SmolLM2 60.3, Qwen2.5-1.5B 58.0).
8. **Framing for timing.** Written into the record before any timing result: the ternary side uses a May-2025 I2_S kernel and no TL2, INT4 uses current upstream; a low ratio does not imply the format cannot win.

## Unresolved questions (not decided by this review)
- Why bitnet.cpp main fails numerically on this model as built here (could be a build option, the submodule branch, or the kernel change); a documented-flow run was not made.
- Whether the old runtime's I2_S kernel is representative of what a modern ternary kernel or hardware would deliver.
- The 537 s stall: laptop sleep, WSL2 behaviour or the old runtime on INT4. To be re-measured first.
- Quality is a zero-shot log-likelihood proxy at 300-500 tasks per benchmark; it says nothing about application accuracy (instruction following, citations, hallucination).
- No energy has been measured; every efficiency statement remains speed/bytes only.

## What the reviewer did not do
It did not re-execute any run, inspect the model weights, or verify the vendors' speed claims.
