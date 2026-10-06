# Handoff

Date: 2026-10-07
Agent: Claude (Claude Code, model Fable 5.1), implementation role
Current commit: the EXP-0005 commit ("research: compare packed streaming and near-memory ternary dataflow"), followed by a docs-only fixup commit recording this hash

## Objective

FT-CURRENT, EXP-0005: packed ternary hardware consumer, streaming baseline vs banked near-memory dataflow (Gate 1 of the co-design program). Complete.

## What I Completed

Python golden reference reusing the profiler's pack/unpack; RTL for `ternary_unpack4`, `sum_tree`, `y_stream_engine` (BYTES × ROWQ), `y_memory_bank`, `y_near_memory_engine` (NB banks), `int4_stream_baseline`; two testbenches that act as the memory system; 26 tests (packing, reference, RTL at every canonical K, extremes, reserved code, tails, backpressure, II = 1, consecutive ops, bit-identity of both engines, checkpoint slices from all three models, INT4 baseline, analytical-model invariants); a 14-point generic synthesis sweep; workload mapping, four-level traffic model, capacity sweeps, required clocks, EXP-0003/0004 reconnection; EXP-0005 record with separate Gate 1A / 1B verdicts; ARCHITECTURE §3; canonical-doc updates; completion review.

## What I Changed

- New: `rtl/{ternary_unpack4,sum_tree,y_stream_engine,y_memory_bank,y_near_memory_engine,int4_stream_baseline}.sv`, `tests/{y_stream_engine_tb,y_near_memory_engine_tb}.sv`, `tests/test_hardware_gate.py`, `analysis/hardware_gate/` (reference, vectors, workload_mapping, traffic_model, synth_points, run_gate), `synth/exp0005/*.ys` (14 generated scripts), `scripts/run_exp0005.ps1`, `results/hardware_gate/` (CSVs, summary.md, synth logs, environment.json, checkpoint_slices.json), EXP-0005, this handoff, the completion review.
- Updated: `ARCHITECTURE.md` (§3 packed consumers, renumbered), `PROJECT_STATE.md`, `CURRENT_WORK.md`, `THESIS.md` (H2, H3, EXP-0005 paragraph), `ROADMAP.md`, `OPEN_QUESTIONS.md` (Q-005, Q-009, Q-010, Q-011, Q-022 evidence; none closed), `KNOWN_ISSUES.md` (KI-008), `BENCHMARKS.md` (EXP-0005 rows), `README.md`, `analysis/README.md`.
- Untouched: Phase-1 RTL and results, `results/roofline/`, `results/model_precision/`, `results/workload_profile/`. No ADR (nothing frozen).

## Tests / Validation Run

```
python -m unittest tests.test_hardware_gate -v            # 26 tests; run per class in the foreground (see KI-007)
python -m unittest tests.test_reference tests.test_analysis tests.test_analysis_cross_model tests.test_roofline   # 82 OK
(WSL) python -m unittest tests.test_model_precision       # 8 OK, 3 skipped (slow tests gated)
python analysis\hardware_gate\synth_points.py --tool wsl  # 14 points, Yosys 0.52 native, 0.3–80 s each
python analysis\hardware_gate\run_gate.py                 # tables + summary
```

`scripts\run_exp0005.ps1 -SynthTool wsl` chains them. `FLEXTERN_FULL_RTL=1` runs every case on every engine (the default subset takes ~25 minutes of Icarus time; the streaming class alone is 18 minutes because the 256-term engine simulates slowly).

## Actual Results

See EXP-0005 R1–R7 and `results/hardware_gate/summary.md`. Headline (MEASURED): the packing is consumed directly at II = 1 with 333 flip-flops of state at 256 terms/cycle; 25,466 generic cells (96–115 cells/term from 64 to 1024 terms/cycle), no multiplier, versus 69,526 for the INT4 structural baseline; bank-local engine bit-identical, 147 → 123 cells/term at 64 → 256 terms per bank. ANALYTICAL: near-memory cuts Falcon-E 1B's global bytes from 399 MB to 2.8 MB per token for the ternary part if 396 MB stay resident; activation delivery NB × K per matvec; without bank buffers the broadcast equals the weight bytes; the global limiter becomes head + KV. Gate 1A PASS (structural/cycle); Gate 1B PASS (dataflow/global-traffic), physical INCONCLUSIVE.

## Remaining Problems

- No physical target: Fmax, utilization, bandwidth, energy, area unmeasured.
- The single-cycle 64–256-input adder trees (longest path 42–51 gates) would need pipelining; not done.
- YoWASP Yosys 0.69 cross-check of the 256-term point did not finish in 10 minutes; the sweep is Yosys 0.52 only (KI-008). Tool sensitivity of absolute counts is 1.1–1.8× per EXP-0001.
- Near-memory top synthesized at NB = 4 only; larger NB scaled from one bank plus the measured collector, whose one-group-per-cycle output is the cycle limiter beyond 64 banks (collector-bound tables in `workload_totals.csv`).
- The harness killed background runs under host memory pressure (KI-007); everything was redone in foreground chunks.

## Important Context

- Packing: `packed[r, c]` = four OUTPUT rows {r, r+N/4, r+2N/4, r+3N/4} at input c; code − 1 is the Y-Core code; reserved 11 → 0. The testbenches un-stride results with `row = i·N/4 + rowq`. Do not "fix" this to adjacent rows.
- The row-quad K-parallel form is the cheap consumer for one contiguous stream; output-parallel (many rows × 1 byte) costs 3× the cells and 41× the flip-flops at equal throughput and needs strided streams. Banks are where output parallelism lives.
- Memories are ports on purpose; keep them that way or synthesis counts become meaningless (Yosys would map them to flip-flops).
- Near-memory overhead per operation is NB + 3 cycles (fixed-priority collector drains NB simultaneous results), and the collector's one-group-per-cycle output bounds every tensor at ≥ N/4 cycles: beyond 64 banks × 16 B the K = 2048 tensors are collector-bound (Falcon-E 1B utilization 0.57 at 256 banks, 0.16 at 1,024). Any study of hundreds of banks must start with the collector and the broadcast.
- Lockstep activation broadcast without per-bank buffers costs exactly the weight bytes; the buffered design (13 KB per bank) is the one modeled as the default.
- Yosys scripts set parameters with `chparam -set` + `rename -top` because `hierarchy -chparam` asserts on recursive `sum_tree` in Yosys 0.69.

## Questions for Founder / ChatGPT

1. Gate 1B's benefit presumes 396–730 MB of weight residency near compute. Is FlexTern prepared to make a memory-technology question (SRAM/eDRAM/stacked/PIM) its next hardware question, or does it stay with a streaming engine fed by HBM/LPDDR?
2. After near-memory, the global limiter is the head and KV cache (Q-022). Does FlexTern own that consumer or not?
3. Should the canonical synthesis tool move to native Yosys for anything larger than Phase 1 (KI-008)?

## Recommended Immediate Next Consideration

Pipeline and re-synthesize the streaming engine's adder trees so that the longest combinational path is bounded (e.g. ≤ 12 gates) at 256–1024 terms/cycle, and record the cost in cells and latency against the single-cycle version, with the same testbenches proving II = 1 is preserved. This is the one structural question that stands between today's "II = 1 in RTL" and a defensible required-clock statement when an FPGA arrives; it needs no hardware, no new dataflow decision, and it does not pre-empt the memory-technology or head/KV questions, which are founder decisions. The alternative, a hierarchical collector and broadcast study for NB ≥ 256, matters only if the founder chooses the near-memory direction.
