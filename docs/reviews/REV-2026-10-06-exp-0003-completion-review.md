# Review — EXP-0003 bottleneck model (completion)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent)
Date: 2026-10-06
Scope: EXP-0003 and its artifacts (`analysis/roofline_models.py`, `analysis/flextern_roofline.py`, `tests/test_roofline.py`, `results/roofline/`, ADR-0005, the Q-019 reframe, the handoff) as they stood immediately before the EXP-0003 commit. Per the founder's instruction that the reviewer receive a different prompt from the implementation, the following twelve questions were put to the reviewer verbatim: attack (1) the Amdahl / ternary-only ceiling math, (2) the KV-cache traffic equations, (3) batch amortization, (4) overlap assumptions, (5) attention operation counts, (6) head treatment, (7) context crossover calculations, (8) whether short-context/edge conclusions actually follow, (9) whether hypothetical KV INT8/INT4 sensitivity is overstated, (10) whether the experiment accidentally assumes impossible residency, (11) whether 2×/3× theoretical ceilings are confused with achievable speedups, (12) whether the recommended next objective follows from evidence.

## Overall Assessment

The model is small, auditable, and internally consistent; no number in EXP-0003 disagrees with the generated `summary.md` (checked: S0–S2 rows, the opportunity curve, crossovers, 2×/3× regions, batch-equals-KV values, accumulator bits). One content gap was found and closed before commit: the record claimed INT4 as the serious low-precision baseline but showed no comparison; a `system_storage_comparison.csv` and section were added. Two wordings were corrected (a crossover quoted beyond a model's supported context; an "infx" cell).

## What Looks Strong

- The memory-bound identity (ideal ceiling = total bytes / non-ternary bytes) organizes the experiment and is tested to 1e-9.
- Inputs are the three canonical profiles, loaded with identity and revision asserts; KV and terms/byte invariants reproduce EXP-0002B from equations.
- Twenty required validations are covered by 26 tests, including break-even agreement with sweep transitions and residency monotonicity.

## Concerns (verdicts on the twelve questions)

1. **Amdahl math.** Correct and pinned: compute-only elimination equals 1.00× wherever the ternary stage is memory-bound; ideal ≥ partials; ceiling → 1 at large context. Caveat correctly stated: ceilings above the byte ratio at high bandwidth reflect slow ternary compute on the abstract engine, not opportunity.
2. **KV equations.** Reproduce 314,572,800 (BitNet @4k) and 805,306,368 (Falcon 1B @32k); write term included; GQA via nKV. Sound.
3. **Batch.** Weights ÷B, KV and per-token compute unchanged; division commutes with the per-stage max, so per-token times are exact. The ceiling falling with batch is a correct consequence.
4. **Overlap.** Two per-stage bounds, no cross-stage prefetch, documented. Within-stage overlap of KV streaming with attention compute is the contestable assumption; it is the optimistic bound by design and the conservative bound is reported beside it.
5. **Attention counts.** QKᵀ + AV reconcile with EXP-0002's 629,145,600 at 4k; softmax at 4 ops/score is coarse and matters only in the compute-bound regime.
6. **Head.** Correct for tied and untied; embedding lookup ignored as one row.
7. **Crossovers.** Algebra confirmed by sweep transitions; "always" and "beyond maximum context" cases handled.
8. **Edge conclusions.** Follow from the tables and are correctly made vocabulary-dependent. With the INT4 framing added, the edge wedge cannot be read as multi-fold against serious low-precision competitors in bytes.
9. **KV INT8/INT4.** Not overstated; the S6 result that attention compute at the assumed rate erases the gain is the right caution.
10. **Residency.** No hit rates; "ideal" policies labeled; the very large ratios after head residency are ceilings and read as such.
11. **Ceilings vs achievable.** Distinguished throughout; "ceiling" wording must be kept in the report.
12. **Next objective.** Follows: 400–1,000 lanes at 1 GHz saturate edge bandwidth, and every ceiling presumes the ternary stage runs at the bandwidth bound. Caveat added to the handoff: generic synthesis gives cells and depth, not clock.

## Assumptions Being Made

- The non-ternary compute rate (1e12 op-eq/s default) has no evidence and governs every ≥1 TB/s and compressed-KV conclusion (Q-022).
- Activations on-chip; stream-once weights; coarse op counts for "other".

## Potential Failure Modes

- Quoting any ceiling as a speedup.
- Reading the realized 2-bit storage advantage (under 2× in bytes against an INT4-weight system with identical head and KV) as a multi-fold system claim.

## Benchmark / Scientific Concerns

- None blocking. INT4/INT8/BF16 compute and energy are explicitly not modeled.

## Architecture Questions

- Whether a streaming engine (unpack + lanes + 20–22-bit accumulation at K up to 13312) can be built at line rate: the premise of all ceilings.

## Alternative Approaches Worth Considering

- KV-compression research as the next step, if the founder decides long-context serving is the first market. Deferred by the implementing agent pending Q-019.

## Questions for Founder / ChatGPT

- Q-019 (head vs KV as the gating component for the chosen market), Q-022 (how attention and the head execute).

## Recommended Actions

1. Commit as `research: model FlexTern system bottleneck boundaries`, then a docs-only hash fixup; push under this task's authorization after staged-file inspection.
2. Keep "ceiling" language in all reporting.

## Severity

Informational. No defects in the delivered model; one omitted comparison and two wordings fixed during review.
