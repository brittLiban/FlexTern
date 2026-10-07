# FlexTern Project State

**Last verified:** 2026-10-07 (EXP-0006 memory-residency analysis; 17 new tests PASS; 98 earlier non-RTL tests PASS; EXP-0001..0005 artifacts and RTL unchanged)

**Current phase:** Phase 2 — workload characterization and system bottleneck analysis — plus the model-side track (ADR-0006) and the co-design gates: Gate 1 (EXP-0005, dataflow) and Gate 2 (EXP-0006, memory residency). Complete; awaiting founder/ChatGPT review.

**Current objective:** FT-CURRENT (EXP-0006, model-resident memory feasibility) is **complete**. No next objective has been set. See `CURRENT_WORK.md`.

**Current architecture status:** Y-Core v0.1 and Y-Dot16 v0.1 (Phase 1) plus experimental packed consumers from EXP-0005 (`y_stream_engine`, `y_memory_bank`, `y_near_memory_engine`), verified at RTL and generic-synthesis level; **not frozen, no ADR**. The memory organization for the bank-local form is exploratory (EXP-0006): tens of banks of several MB on one SRAM-heavy die for 1B–3B models is inside the demonstrated public envelope; no memory technology is chosen (Q-023). System scope fixed by ADR-0005: ternary-first, not ternary-only. The EXP-0004 candidate non-ternary format is not adopted.

**What is working:**
- Phase 1 RTL, testbenches, synthesis flow (EXP-0001).
- Spec-driven workload profiler over three pinned checkpoints (EXP-0002/0002B).
- Analytical bottleneck model (EXP-0003); GPU model-precision harness (EXP-0004); packed-ternary engines, testbenches, synthesis sweep and traffic models (EXP-0005).
- Source-backed memory-residency analysis package with provenance tests (EXP-0006).

**What has been proven (MEASURED):**
- EXP-0001: ternary primitive needs no multiplier; 5–12× fewer generic cells than INT8 multiply.
- EXP-0002/0002B: across two families, ternary 2-bit storage, per-tensor post-accumulation scale, INT8 per-token activations, ~40% unstructured zeros, two dot lengths per model all divisible by 256, BF16 non-ternary head.
- EXP-0004 (one checkpoint, pilot scale): Falcon-E 1B tolerates an INT4 output head (+0.67% perplexity) and a 4-bit KV cache (+0.17–1.07%), additively.
- EXP-0005: the canonical 2-bit packing is consumed directly at initiation interval 1 with 333 flip-flops of state at 256 terms/cycle; 25,466 generic cells, no multiplier, versus 69,526 for an INT4 structural baseline; a bank-local engine gives bit-identical results reading every weight byte locally and none globally.

**What has been modeled (ANALYTICAL / INFERRED):**
- EXP-0003: ideal ternary-only ceilings ≤1.8× for BitNet at any context, ≥3× for Falcon-E below ~2.6k/7k context, 1.4–1.6× at 32k.
- EXP-0005: bank-local placement removes the ternary weight stream from the global path (Falcon-E 1B: 399 MB → 2.8 MB per token) if 396–730 MB stay resident; useful bank counts ≤ 128 (collector bound).
- EXP-0006 (public evidence, INFERRED): Falcon-E 1B needs 440 MB physical at 2 bits (10% margin) = 404–648 mm² at the demonstrated whole-SRAM-heavy-die density tier (0.68–1.09 MB/mm², N7), inside the Graphcore 900 MB / 823 mm² envelope; 3B is at the top of that envelope; INT4 moves 3B outside one reticle at every whole-die anchor; ≥7B needs multi-die SRAM or DRAM-class capacity. Bandwidth never forces residency at 1B–3B (one LPDDR5X channel streams ~120 tok/s batch 1); the resident form's case is energy (SRAM 0.16–1.6 pJ/bit at a 45 nm anchor vs DRAM 3.9–15) and latency. After residency, head + KV are 88–99.7% of global bytes.

**What has NOT been proven:**
- Any physical result: FPGA utilization, frequency, ASIC area, power, bandwidth, tokens/sec, energy per token, cost. No FlexTern silicon or design exists.
- Modern-node SRAM read energy (the number the residency energy argument rests on); Y-Core bank area; latency benefit.
- The real cost of the non-ternary consumer (Q-022), now the dominant external traffic after residency; quality beyond one checkpoint (Q-019); generalization beyond two families (Q-018); INT8 activation persistence (Q-021).

**Current measured results:** `BENCHMARKS.md` (generic synthesis, EXP-0001 and EXP-0005); `../results/workload_profile/cross_model/comparison.md`; `../results/roofline/summary.md`; `../results/model_precision/quality_comparison.csv`; `../results/hardware_gate/summary.md`; `../results/memory_residency/summary.md` (analytical/inferred, not measured).

**Current blockers:** None technical. Awaiting direction. No FPGA for months (by founder statement).

**Current major uncertainty:** Whether an energy/latency advantage justifies 400–650 mm² of SRAM-heavy silicon for 1B–3B models when external LPDDR5X streaming already serves ~100 tok/s, and what model size FlexTern targets (on-die residency ends near 3B ternary parameters per reticle).

**Immediate next decision point:** Founder and ChatGPT review EXP-0006's two gate verdicts, decide the target model size and whether the resident form is pursued (Q-023, coupled to Q-015), and choose the next objective. The handoff `handoffs/2026-10-07-exp-0006-memory-residency.md` recommends one.
