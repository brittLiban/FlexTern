# FlexTern Project State

**Last verified:** 2026-10-07 (EXP-0005 RTL simulation and generic synthesis; 26 hardware-gate tests PASS; 82 earlier host tests PASS; 11 model-precision tests PASS in WSL; EXP-0001..0004 artifacts unchanged)

**Current phase:** Phase 2 — workload characterization and system bottleneck analysis — plus the model-side track (ADR-0006), now joined by the first hardware dataflow experiment (EXP-0005, Gate 1 of the co-design program). Complete; awaiting founder/ChatGPT review.

**Current objective:** FT-CURRENT (EXP-0005, packed ternary consumer: streaming vs banked near-memory) is **complete**. No next objective has been set. See `CURRENT_WORK.md`.

**Current architecture status:** Y-Core v0.1 and Y-Dot16 v0.1 (Phase 1) plus experimental packed consumers from EXP-0005: `y_stream_engine` (row-quad native consumption of the checkpoint packing, II = 1), `y_memory_bank` and `y_near_memory_engine` (bank-local, bit-identical). Implemented and verified at RTL and generic-synthesis level; **not frozen, no ADR**. System scope fixed by ADR-0005: ternary-first, not ternary-only. A candidate numerical format for the non-ternary components exists (`../results/model_precision/candidate_format.md`) and is not adopted.

**What is working:**
- Phase 1 RTL, testbenches, synthesis flow (EXP-0001).
- Spec-driven workload profiler over three pinned checkpoints (EXP-0002/0002B).
- Analytical bottleneck model consuming those profiles (EXP-0003).
- GPU model-precision harness under WSL2 (EXP-0004).
- Packed-ternary engines, testbenches that act as the memory system, Python reference and vector generator, 14-point generic synthesis sweep, workload/traffic/capacity/reconnection models, one-command runner (EXP-0005).

**What has been proven (MEASURED):**
- EXP-0001: ternary primitive needs no multiplier; 5–12× fewer generic cells than INT8 multiply.
- EXP-0002/0002B: across two families, ternary {-1,0,+1} 2-bit storage, per-tensor post-accumulation scale, INT8 per-token activations, ~40% unstructured zeros, two dot lengths per model all divisible by 256, BF16 non-ternary head.
- EXP-0004 (one checkpoint, pilot scale): Falcon-E 1B tolerates an INT4 RTN g128 output head (+0.67% WikiText-2 perplexity) and a 4-bit KIVI-style KV cache (+0.17–1.07% teacher-forced perplexity at 2k–4k), additively; on a PyTorch runtime the quantized cache is dequantized before every attention call.
- EXP-0005: the canonical 2-bit packing is consumed directly (decode = 2-bit decrement, four output rows per byte) at initiation interval 1 on all five real K values, extremes, reserved codes, tails, backpressure and real checkpoint slices, with 333 flip-flops of state at 256 terms/cycle; 25,466 generic cells (96–115 cells/term from 64 to 1024 terms/cycle), no multiplier, versus 69,526 for an INT8 × INT4 structural baseline; a bank-local engine gives bit-identical results reading every weight byte locally and none globally.

**What has been modeled (ANALYTICAL):**
- EXP-0003: ideal ternary-only ceilings ≤1.8× for BitNet at any context, ≥3× for Falcon-E below ~2.6k/7k context, 1.4–1.6× at 32k; 400–1,000 Y-Core lanes at 1 GHz saturate 100–250 GB/s.
- EXP-0005: bank-local placement removes the ternary weight stream from the global path (Falcon-E 1B: 399 MB → 2.8 MB per token) if 396–730 MB stay resident; activation delivery costs NB × K per matvec with 13 KB buffers per bank, or the full weight-byte count without them; the global limiter then becomes the head and KV cache. Required clocks for 100/250 GB/s: 1.56/3.9 GHz at 64 B/cycle, 391/977 MHz at 256 B/cycle, 98/244 MHz for 64 banks × 16 B — TARGETS, UNMEASURED.

**What has NOT been proven:**
- Any physical result: FPGA utilization, frequency, ASIC area, power, bandwidth, tokens/sec. No FPGA target exists.
- That hundreds of megabytes of near-compute memory are economical in any technology, or that bank reads beat fabric reads in energy.
- The real cost of the non-ternary consumer (Q-022), now the global limiter after near-memory; quality beyond one checkpoint (Q-019); generalization beyond two families (Q-018); INT8 activation persistence (Q-021).

**Current measured results:** `BENCHMARKS.md` (generic synthesis, EXP-0001 and EXP-0005); `../results/workload_profile/cross_model/comparison.md`; `../results/roofline/summary.md`; `../results/model_precision/quality_comparison.csv`; `../results/hardware_gate/summary.md`.

**Current blockers:** None technical. Awaiting direction. No FPGA for months (by founder statement).

**Current major uncertainty:** Whether the near-memory direction's residency requirement (396–730 MB near compute) is realizable at a cost that beats a streaming engine fed from external memory, and how the head and KV cache are consumed once the ternary stream leaves the global path.

**Immediate next decision point:** Founder and ChatGPT review EXP-0005's two gate verdicts and choose between the near-memory direction (memory-technology question next) and the streaming direction (pipelined engine, required clock), and whether FlexTern owns the non-ternary consumer. The handoff `handoffs/2026-10-07-exp-0005-hardware-gate.md` recommends one objective.
