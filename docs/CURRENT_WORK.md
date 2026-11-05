# Current Work

There is exactly one active engineering objective at a time. This file is the single active objective. The founder and ChatGPT choose the next objective after review; do not automatically start a backlog item.

---

**Objective ID:** EXP-0011 — Ternary vs optimized INT4 at matched quality (Stage 1: quality, latency, throughput, memory on existing hardware)

**Title:** At matched model quality, can native ternary inference (BitNet b1.58 2B4T) substantially outperform strong, optimized INT4 inference on commercially relevant edge workloads?

**Status:** Stage 1 **in progress** (protocol locked before measurement: `analysis/ternary_vs_int4/protocol_v1.json`). **Stage 2 (physical energy) is NOT approved**: it needs separate founder approval before any equipment purchase or hardware-dependent measurement; the plan is `experiments/EXP-0011-stage2-energy-plan.md`. The previous objective (EXP-0010C) is complete with its preregistered outcome CHANGE_WORKLOAD; its records are unchanged.

**Owner / agent:** Claude (implementation); founder + ChatGPT (research direction/review).

## Scope (from the founder directive of 2026-11-05)

1. Stage 1: quality (HellaSwag, Winogrande, ARC-E, ARC-C, MMLU subsets), prefill, time to first token, decode tokens/s, total latency, peak/resident memory at 512/1024/2048/4096-token contexts and 128/256-token outputs, BitNet 2B4T (bitnet.cpp) against several INT4 GGUF models (bitnet.cpp and upstream llama.cpp), on the existing laptop under WSL2. No energy claims.
2. Preregistered protocol, evidence ledger, immutable result manifests, negative results retained, adversarial review, commit.
3. Parallel commercial research and outreach preparation only (`reviews/MEMO-2026-11-05-commercial-validation-package.md`): no interviews, demand or commitments exist; outreach is founder-led.
4. Stop at the next evidence-based decision gate. No EXP-0012, no hardware purchase, no architectural redesign without authorization.

## Primary records
`experiments/EXP-0011-ternary-vs-int4-stage1.md`, `experiments/EXP-0011-stage2-energy-plan.md`, `reviews/MEMO-2026-11-05-commercial-validation-package.md`, `analysis/ternary_vs_int4/` (harness), `tests/test_ternary_vs_int4.py`. Strategy context: `reviews/MEMO-2026-11-04-workload-selection-decision.md`, `reviews/MEMO-2026-11-04-project-homerun-opportunity-search.md`. The second strategic report named in the directive is not in the repository.
