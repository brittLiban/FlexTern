# FlexTern Project State

**Last verified:** 2026-10-06 (EXP-0004 model-precision run on Falcon-E 1B; 11 model-precision tests PASS in WSL with the model loaded; 82 earlier host tests PASS; EXP-0001/0002/0002B/0003 artifacts unchanged)

**Current phase:** Phase 2 — workload characterization and system bottleneck analysis, now joined by the model-side track (ADR-0006). Complete; awaiting founder/ChatGPT review.

**Current objective:** FT-CURRENT (EXP-0004, INT4 head and ~4-bit KV feasibility on Falcon-E 1B) is **complete**. No next objective has been set. See `CURRENT_WORK.md`.

**Current architecture status:** Unchanged since Phase 1. Y-Core v0.1 and Y-Dot16 v0.1 implemented and verified. No hardware built since. System scope fixed by ADR-0005: ternary-first, not ternary-only. A candidate numerical format for the non-ternary components exists (`../results/model_precision/candidate_format.md`) and is **not** adopted.

**What is working:**
- Phase 1 RTL, testbenches, synthesis flow (EXP-0001).
- Spec-driven workload profiler over three pinned checkpoints (EXP-0002/0002B).
- Analytical bottleneck model consuming those profiles, 26 tests, one-command runner (EXP-0003).
- GPU model-precision harness under WSL2 (INT4 head quantizer, quantized-cache evaluation, lm-eval wrapper, resumable phases, report generation), 11 tests (EXP-0004). Needs WSL2 because Smart App Control blocks PyTorch on the host (KI-005).

**What has been proven (MEASURED):**
- EXP-0001: ternary primitive needs no multiplier; 5–12× fewer generic cells than INT8 multiply.
- EXP-0002/0002B: across two families, ternary {-1,0,+1} 2-bit storage, per-tensor post-accumulation scale, INT8 per-token activations, ~40% unstructured zeros, two dot lengths per model all divisible by 256, BF16 non-ternary head.
- EXP-0004 (one checkpoint, pilot scale): Falcon-E 1B tolerates an INT4 RTN g128 output head (+0.67% WikiText-2 perplexity, task accuracies within one stderr, head 134 MB → 35 MB logical) and a 4-bit KIVI-style KV cache (+0.17–1.07% teacher-forced perplexity at 2k–4k, per-token grouping best, cache 3.5× smaller), additively. On a PyTorch runtime the quantized cache is dequantized before every attention call: resident memory shrinks, attention traffic does not, prefill is 4–8× slower.

**What has been modeled (ANALYTICAL, EXP-0003, re-evaluated in EXP-0004):**
- Ideal ternary-only end-to-end ceilings: BitNet ≤1.8× at any context (BF16 head dominates); Falcon-E ≥3× below ~2.6k/7k context and ≥2× below ~10.7k/18k, falling to 1.4–1.6× at 32k (KV cache). Batching lowers the ceiling. With the measured INT4 head and 4.5-bit KV, Falcon-E 1B's ceiling at 32k would rise from 1.42× to 2.52× only if hardware reads the compressed forms directly; what this runtime attains is 1.47×.

**What has NOT been proven:**
- Any physical result: FPGA utilization, frequency, ASIC area, power, bandwidth, tokens/sec.
- That a streaming ternary engine can consume packed weights at line rate (every EXP-0003 ceiling assumes it).
- The real cost of non-ternary compute (Q-022); head/KV quantization quality beyond one checkpoint, 4k context, and greedy decoding (Q-019); generalization beyond two families (Q-018); INT8 activation persistence (Q-021).
- Any benefit specific to FlexTern hardware from the EXP-0004 representations: both are category A techniques any GPU can use today.

**Current measured results:** `BENCHMARKS.md` (hardware, EXP-0001); `../results/workload_profile/cross_model/comparison.md` (workloads); `../results/roofline/summary.md` (analytical ceilings); `../results/model_precision/quality_comparison.csv` and `memory_comparison.csv` (model-side quality and memory, EXP-0004).

**Current blockers:** None technical. Awaiting direction.

**Current major uncertainty:** Whether the ternary block can actually be built to stream 2-bit weights at line rate with 20–22-bit accumulation at K = 2048–13312, which is the premise of every ceiling in EXP-0003; and whether FlexTern will own an attention/head path that consumes 4-bit operands directly, without which the model-side savings of EXP-0004 are memory-only.

**Immediate next decision point:** Founder and ChatGPT review EXP-0004 (acceptable-loss threshold; whether the candidate format becomes a hardware assumption now or after a second checkpoint) and choose the next objective. The handoff `handoffs/2026-10-06-exp-0004-model-precision.md` recommends one.
