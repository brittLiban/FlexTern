# Review — EXP-0005 packed ternary consumer: streaming vs near-memory (completion)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent)
Date: 2026-10-07
Scope: EXP-0005 and its artifacts (`rtl/`, `tests/`, `analysis/hardware_gate/`, `synth/exp0005/`, `results/hardware_gate/`, the EXP-0005 record, ARCHITECTURE §3, the handoff, and the THESIS / OPEN_QUESTIONS / PROJECT_STATE / BENCHMARKS wording) as they stood immediately before the EXP-0005 commit (eef813f). Per the founder's instruction that the reviewer receive a different prompt from the implementation, the following sixteen points were put to the reviewer verbatim: attack (1) whether the checkpoint packing orientation was interpreted correctly, (2) whether the streaming baseline receives an unfair handicap, (3) whether the near-memory design assumes resident weights for free, (4) global versus local traffic accounting, (5) activation broadcast accounting, (6) partial-sum/output traffic, (7) memory capacity assumptions, (8) generic synthesis treatment of memory arrays, (9) whether II=1 is confused with absolute bandwidth, (10) whether generic logic depth is confused with Fmax, (11) accumulator width/correctness, (12) whether near-memory's benefit would apply equally to INT4, (13) whether the claimed ternary-specific benefit survives a fair INT4 reference, (14) whether dataflow improvement is confused with energy or end-to-end speedup, (15) whether the Gate 1A and Gate 1B verdicts follow from the evidence, (16) whether the recommended next objective follows.

## Overall Assessment

The packing remap, accumulator widths per K, flip-flop scaling across engine widths, the NB + 3 collector overhead, the traffic sums and the 581.1 MB EXP-0003 reconnection all recompute from the artifacts. Three defects were found and fixed before commit:

1. **Near-memory cycle counts ignored the collector bound the record itself documents.** The top emits one four-output group per cycle, so a tensor can never finish in fewer cycles than it has row-quads; `nearmem_cycles` returned only `rows_per_bank · K / b`. Every configuration with NB · b > K was overstated on the K = 2048 tensors (71% of Falcon-E weights): Falcon-E 1B at 1,024 banks × 16 B publishes 44,544 cycles per token when the row-quad groups alone need 150,528; 4,096 banks was off by roughly 18×. Fixed to `max(rows_per_bank · K / b, rowq)`, utilization recomputed from it, a collector-bound caveat added to the near-memory rows of `bandwidth_requirements.csv`, tables regenerated and the record's R4 and Interpretation re-worded. Gate 1B's byte verdict does not move; the large-NB cycle claims do.
2. **BENCHMARKS.md marked six rows "Reproducible? Yes" when only the 256-term streaming point had been run twice.** The whole 14-point sweep was re-run into a scratch directory and every point's gate-level breakdown diffed before the rows kept that mark.
3. **The flip-flop budget sentence did not reconcile.** Rewritten as the enumeration 88 (accumulators) + 4·TW (chunk sums) + 88 (output) + 64 (counters/config) + 16 (`y_rowq`) + 16 (`s1_rowq`) + 7 (flags) − log2(BYTES) always-zero low bits of `col` stripped by Yosys = 327 / 333 / 336 / 339 at B = 16 / 64 / 128 / 256, which matches the measured counts exactly.

Also added: a zero-handling sentence (zeros contribute 0 through Y-Core's select; no bandwidth saving from zeros is claimed); a limitation on the memory-interface asymmetry (bank weight memories synchronous, activation buffers combinational on both engines); a BitNet utilization correction (0.99, not 1.0, at 64 banks).

## What Looks Strong

- The dataflow was derived from the storage layout rather than assumed: byte (r, c) → rows {r, r+N/4, r+2N/4, r+3N/4}, consecutive bytes along K; both forms (row-quad K-parallel and output-parallel) were built and synthesized so the choice is measured, not argued.
- Verification is cycle-counted: II = 1 is an equality assertion on every streaming run, back-pressure runs must be byte-identical, both engines are diffed against each other and the exact-integer reference, and five real checkpoint slices from all three models pass.
- Memories are ports, so the only inferred memory is the bank FIFO (512 flip-flops) and no synthesis number hides an N × K array.
- The traffic model keeps the four levels apart and reports the lockstep-broadcast result (delivered activation bytes = weight bytes) rather than burying it.

## Concerns (verdicts on the sixteen points)

1. **Packing orientation.** Correct. Mutation test on adjacent-row ordering fails as required; checkpoint slices bit-identical.
2. **Streaming handicap.** None. If anything the streaming testbench is advantaged (zero-latency source versus the bank's synchronous read plus FIFO). Both engines share the same external activation-buffer assumption.
3. **Resident weights for free.** Both architectures load once (Level A). The real asymmetry is now stated: streaming can use one monolithic, possibly external, memory; near-memory needs the same 396–730 MB partitioned into NB compute-adjacent banks with local ports. Capacity is counted; bank cost and technology are INCONCLUSIVE and labeled so.
4. **Global versus local.** Separated; local reads equal streaming's global weight bytes by construction, and the record says they never disappear.
5. **Broadcast.** Buffered (NB × K) and lockstep (= weight bytes) both modeled; three link bounds as counts only, no wire energy. The lockstep result is the strongest negative finding in the experiment and is reported as such.
6. **Partial sums / outputs.** Zero partial sums by row ownership, verified in RTL; outputs counted as 32-bit words. The collector serialization is where output traffic leaks into cycles (defect 1, fixed).
7. **Capacity.** From measured sizes including tail waste; per-layer and whole-model residency distinguished; no technology chosen.
8. **Memory arrays in synthesis.** Ports; only the FIFO is inferred and it is counted as flip-flops. Correct handling for a generic flow.
9. **II = 1 versus bandwidth.** Cycle-counted assertion; required clocks labeled UNMEASURED everywhere; no GB/s is claimed.
10. **Depth versus Fmax.** `ltp -noff` reported in gates; the record states that single-cycle 64–256-input trees would be pipelined. Not confused.
11. **Accumulator widths.** 20/20/21/22/22 bits (format bound, −128 reachable) verified by test and by simulated extremes at every K; INT4 baseline 25 bits.
12. **INT4 gets the same near-memory gain.** Yes, and the model says so: Level B global bytes are identical for 2-bit and 4-bit weights.
13. **Ternary-specific benefit versus a fair INT4 reference.** Half the local bytes and capacity; 2.7× fewer cells at equal terms per cycle against a `*`-based structural baseline, explicitly not a GPU.
14. **Dataflow versus energy / speedup.** Every table carries the disclaimer; no speedup or energy number exists in the record.
15. **Gate verdicts.** 1A follows. 1B follows at the byte level, with defect 1 fixed so that the cycle tables no longer overstate large-NB throughput.
16. **Next objective.** Pipelining the adder trees follows from the measured 42–51-gate paths, is the one structural gap before any required-clock statement, needs no hardware, and pre-empts no founder decision. Defensible.

## Assumptions Being Made

- Batch 1, stream-once weights, INT8 activations arriving once per distinct input vector, 32-bit outputs, untimed preload.
- Generic cells are structure counts from Yosys 0.52; the canonical 0.69 cross-check did not complete (KI-008), and EXP-0001 measured 1.1–1.8× sensitivity between versions.
- The near-memory top was synthesized at NB = 4; larger NB scale analytically from one bank plus the measured collector.

## Potential Failure Modes

- Reading "global bytes fall 140×" as a system speedup or an energy result.
- Quoting a required clock as an achieved clock.
- Forgetting that near-memory's byte advantage is format-neutral and claiming it as ternary-specific.
- Scaling the near-memory engine to hundreds of banks without the collector and broadcast redesign the cycle tables now show is needed.

## Recommended Changes

All applied before commit: the collector-bound cycle law and regenerated tables; the full-sweep reproducibility re-run behind the BENCHMARKS rows; the flip-flop budget reconciliation; the zero-handling and memory-interface sentences; the BitNet utilization figure; the one-line commit trailer.
