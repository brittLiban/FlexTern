# FlexTern Project State

**Last verified:** 2026-10-06 (EXP-0003 model run; 26 roofline tests PASS; EXP-0002/0002B profiles and Phase 1 flow unchanged)

**Current phase:** Phase 2 — workload characterization and system bottleneck analysis. Complete; awaiting founder/ChatGPT review.

**Current objective:** FT-CURRENT (EXP-0003, analytical compute-vs-memory bottleneck model) is **complete**. No next objective has been set. See `CURRENT_WORK.md`.

**Current architecture status:** Unchanged since Phase 1. Y-Core v0.1 and Y-Dot16 v0.1 implemented and verified. No hardware built in Phase 2. System scope fixed by ADR-0005: ternary-first, not ternary-only.

**What is working:**
- Phase 1 RTL, testbenches, synthesis flow (EXP-0001).
- Spec-driven workload profiler over three pinned checkpoints (EXP-0002/0002B).
- Analytical bottleneck model consuming those profiles, with 26 tests and a one-command runner (EXP-0003).

**What has been proven (MEASURED):**
- EXP-0001: ternary primitive needs no multiplier; 5–12× fewer generic cells than INT8 multiply.
- EXP-0002/0002B: across two families, ternary {-1,0,+1} 2-bit storage, per-tensor post-accumulation scale, INT8 per-token activations, ~40% unstructured zeros, two dot lengths per model all divisible by 256, BF16 non-ternary head.

**What has been modeled (ANALYTICAL, EXP-0003):**
- Ideal ternary-only end-to-end ceilings: BitNet ≤1.8× at any context (BF16 head dominates); Falcon-E ≥3× below ~2.6k/7k context and ≥2× below ~10.7k/18k, falling to 1.4–1.6× at 32k (KV cache). Batching lowers the ceiling. Edge-class bandwidth (100–250 GB/s) is saturated by 400–1,000 Y-Core lanes at 1 GHz, so ternary compute is not the decode limiter there.

**What has NOT been proven:**
- Any physical result: FPGA utilization, frequency, ASIC area, power, bandwidth, tokens/sec.
- That a streaming ternary engine can consume packed weights at line rate (every EXP-0003 ceiling assumes it).
- The real cost of non-ternary compute (Q-022), any head/KV quantization quality (Q-019, hypotheticals), generalization beyond two families (Q-018), INT8 persistence (Q-021).

**Current measured results:** `BENCHMARKS.md` (hardware, EXP-0001); `../results/workload_profile/cross_model/comparison.md` (workloads); `../results/roofline/summary.md` (analytical ceilings).

**Current blockers:** None technical. Awaiting direction.

**Current major uncertainty:** Whether the ternary block can actually be built to stream 2-bit weights at line rate with 20–22-bit accumulation at K = 2048–13312, which is the premise of every ceiling in EXP-0003; and, for product scope, which non-ternary component (head for 128k-vocabulary models, KV for long context) FlexTern chooses to address.

**Immediate next decision point:** Founder and ChatGPT review EXP-0003's feasibility regions and choose the next objective. The handoff `handoffs/2026-10-06-exp-0003-roofline.md` recommends one.
