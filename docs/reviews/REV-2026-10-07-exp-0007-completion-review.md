# Review — EXP-0007 datacenter thesis falsification (completion)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent)
Date: 2026-10-07
Scope: EXP-0007 and its artifacts (`analysis/datacenter_gate/`, `tests/test_datacenter_gate.py`, `results/datacenter_gate/`, the EXP-0007 record, the handoff, and the THESIS / OPEN_QUESTIONS / PROJECT_STATE / ARCHITECTURE wording) as they stood immediately before the EXP-0007 commit (925510b). Per the founder's instruction that the reviewer receive a different prompt from the implementation, the following twenty-five points were put to the reviewer verbatim: attack (1) conventional baseline strength, (2) batching assumptions, (3) continuous-serving realism, (4) weight-reuse math, (5) prefill/decode separation, (6) KV accounting, (7) compressed-KV fairness, (8) head accounting, (9) tile-capacity assumptions, (10) pipeline fill/drain, (11) activation traffic, (12) tensor-parallel communication, (13) interconnect assumptions, (14) HBM peak versus sustained bandwidth, (15) product-spec misuse, (16) FP4/INT4 fairness, (17) model-resident INT4 fairness, (18) 70B scaling assumptions, (19) 400B scaling assumptions, (20) energy extrapolation, (21) economic proxies, (22) whether Gate 3A follows, (23) whether Gate 3B follows, (24) whether Gate 3C follows, (25) whether the recommended next objective attacks the largest remaining uncertainty.

## Overall Assessment

The model's structure is sound and its arithmetic reproduces (parameter counts, KV per sequence, energies, microbatch logic, topology counts, interconnect bytes, the MLPerf anchor). Several sentences in the record contradicted the model's own tables, two of them inside the Gate 3C verdict; all were corrected before commit:

1. **A latency advantage was claimed that the tables refute.** The resident system is slower than the baseline in every cell at the TARGET compute density (1B 0.39 vs 0.17 ms per token; 70B 16.7 vs 7.0 ms), and a capacity-minimal baseline understates the conventional node's interactive latency by up to 8× (tensor parallelism). The claim was removed from the Interpretation and the verdict; the surviving regime is movement-energy only.
2. **The surviving batch range was overstated** ("batch ≤ 4–8") against the crossover table, which shows ≥ 2× conservative only at batch 1–2 for 7B–400B and to batch 8 for single-tile 1B at 2k. Corrected to per-scale numbers in the record, handoff, CURRENT_WORK, PROJECT_STATE, THESIS and OPEN_QUESTIONS.
3. **Baseline strength on latency** stated: the capacity-minimal GPU count is pessimistic for System A's small-batch latency; a note was added to the summary and record. It strengthens the negative result.
4. Numeric prose errors fixed: 70B KV per sequence (336 MB at 2k, 1.34 GB at 8k, 5.37 GB at 32k, 21.5 GB at 128k; the first draft halved them); 400B HBM saving (1 of 9 GPUs at batch 128/32k, not 4 of 9); prefill figures (8k × 16 prompts in 4.3 s on A, resident tiles 17–550× slower, with the two drivers named); System A weight bytes at batch 16 (2.41 GB weights, 2.78 GB with head and KV).
5. **Resident operating point.** The frontier between pipelined and batch-sequential modes is now stated (C ties A on weight energy only when the microbatch is ≥ ~0.18 of the batch, i.e. ≤ ~5 microbatches in flight on 40 stages); the microbatch now uses a ceiling so every sequence is served; the tile-capacity sensitivity at 2 GB is reported (microbatch 7 at batch 64, still ≈ 1× A).
6. Presentation: C/D head bytes moved from the HBM column to the local column in the founder table; the summary leads with the conservative ratio and explains that the large upper ends come from the 8 KB-array SRAM figure that EXP-0006's review excluded for multi-MB tiles; the KV-stack over-provisioning in the economic proxies is stated.

## What Looks Strong

- The baseline is current-generation and fair: FP4 charged at its real 4.5 effective bits, native FP8/NVFP4 KV, tensor parallelism with reuse B, continuous batching, 70% HBM utilization; its bound lands within 16% of a measured MLPerf result without tuning.
- KV capacity is gated first for every system, which is what makes long-context cells honest for both sides.
- The reuse arithmetic (tensor-parallel node: whole batch; resident pipeline: microbatch) is tested and is the decisive mechanism; the model finds the negative result the prompt asked it to look for.
- EXP-0003 batch-1 bytes, EXP-0005 local reads and EXP-0006 capacities are reproduced by tests, not restated.

## Concerns (verdicts on the twenty-five points)

1. **Baseline strength.** Adequate on bytes and energy; under-strong on latency (fixed by statement, not by code: the capacity-minimal GPU count stays, the pessimism is disclosed).
2. **Batching.** Ideal reuse B for the tensor-parallel node; monotone and infinite-batch tests pass.
3. **Continuous serving.** Utilization factors are assumptions; applied where they bite; at 70B they favor the resident system and still do not help it. Honest.
4. **Weight-reuse math.** Correct and decisive; frontier sentence added.
5. **Prefill/decode.** Separate; prose corrected.
6. **KV accounting.** Per sequence, context-linear, capacity-gated; prose corrected.
7. **Compressed KV.** NVFP4 KV granted to the baseline; resident KV forced to DRAM at any useful context.
8. **Head.** Consistent; < 2% of bytes at batch ≥ 16; batch-1 and tied-head cases flagged.
9. **Tile capacity.** 256 MB–2 GB from EXP-0006; KV correctly excluded from tile SRAM.
10. **Pipeline fill/drain.** Bubble = 1 − in-flight/stages, latency = stages × microbatch time; tested; batch-sequential mode shown.
11. **Activation traffic.** 8 kB per hop, < 0.1% of bytes; conserved by test.
12. **Tensor-parallel communication.** Ring all-reduce counted for the node and for split layers in C/D.
13. **Interconnect.** Sourced tier ranges; never the limiter at pipeline granularity.
14. **HBM peak vs sustained.** 70% applied; peaks labeled.
15. **Product specs.** Used as ceilings only; MLPerf as sanity only.
16. **FP4/INT4 fairness.** 4.5 effective bits; ternary 2.0.
17. **INT4 resident.** Same algorithm, same term rate (favors D); halves to ternary's tiles only.
18. **70B.** Llama-3.1 shapes sourced; capacity-only; 17.1 vs 17.5 GB distinction made.
19. **400B.** Capacity-only; splits counted; HBM-saving prose corrected.
20. **Energy.** Movement-only, labeled, conservative ratios used in verdicts; display cleaned.
21. **Economics.** Proxies, no prices; KV-stack over-provisioning disclosed.
22. **Gate 3A follows** with the corrected batch bounds.
23. **Gate 3B follows**; INCONCLUSIVE leaning FAIL is right given the TARGET density.
24. **Gate 3C follows** after fixes 1–3; before them it overstated the surviving regime.
25. **Next objective.** Q-023 (modern SRAM read energy) is the load-bearing number for every surviving regime and the batching result does not depend on it; defensible. Runner-up: a compute-energy figure for ternary versus FP4 arithmetic, the one term the movement-only verdict omits and the term System B's case rests on.

## Assumptions Being Made

- Compute energy excluded; resident compute density is a TARGET; serving utilization and link tiers are assumptions with ranges; hypothetical shapes only; no scheduler simulated.

## Potential Failure Modes

- Reading any throughput as achieved performance or any mJ/token as a chip figure.
- Quoting the upper ends of the energy ranges, which rest on an inapplicable SRAM figure.
- Treating the batch-1–2 interactive regime as commercially established rather than as the one regime the arithmetic leaves open.

## Recommended Changes

All applied before commit: the six items above, the ceiling microbatch, the two-line commit trailer.
