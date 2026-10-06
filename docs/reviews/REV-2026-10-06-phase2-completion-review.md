# Review — Phase 2 workload characterization (completion)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent)
Date: 2026-10-06
Scope: EXP-0002 and its artifacts in `results/workload_profile/`, `analysis/`, `tests/test_analysis.py`, the updated canonical docs, and the Phase 2 handoff, as they stood immediately before the Phase 2 commit (dd27adb).

## Overall Assessment

The objective is complete and the deliverables are consistent with the generated data. No number quoted in EXP-0002 was found to disagree with `workload_summary.md`: ratios, per-layer ranges, dead-row sums, KV-cache bytes, and accumulator bit widths were checked. Four small items were raised and resolved before commit (below).

## What Looks Strong

- Input is verified end to end: pinned revision, file size, SHA-256 against the Hugging Face etag, and a byte-range reconciliation proving every byte belongs to exactly one classified tensor.
- Counting is exact, not sampled, over all 2.08 B weights, with in-tool asserts that totals reconcile with the config-derived expectation.
- The official ternarization and activation quantization were reproduced from pinned reference code, and the ternarization was cross-checked on real BF16 latents.
- The dead-channel analysis turned a suspicious g = 16 statistic into a concrete, explained structure instead of leaving it as "block sparsity".
- Labels (MEASURED / INFERRED / HYPOTHETICAL) are applied consistently, and the traffic model states what invalidates it.

## Concerns

- Every finding rests on one checkpoint (Q-018). The ranking of next experiments should be re-examined if a second native ternary model becomes available.
- The traffic model assumes stream-once, batch-1 decode: the most pessimistic weight model. It establishes plausibility that bytes dominate, not that they do on any device.
- The recommended next objective (accumulation at real K) deliberately parks the largest byte stream (the BF16 head) behind a scope question (Q-019). That is a defensible sequencing choice, but the founder should make it consciously.

## Assumptions Being Made

- That the packed repo, not the BF16 repo, is the deployed artifact. The model card says so; the 98–99.4% code agreement confirms they differ slightly.
- That static MAC counts are a usable proxy for "where the work is" absent runtime data.

## Potential Failure Modes

- A reader quoting the 56% head-traffic figure as a measurement. It is INFERRED under stated assumptions.
- A future agent designing block-sparse hardware from the 42% zero rate without reading the independence result.

## Benchmark / Scientific Concerns

- None blocking. Items resolved during this review: the committed cross-check JSON now comes from the committed script (network returned); Phase 1 simulations were re-run after comment-only RTL edits and pass; a per-layer dead-channel range in EXP-0002 was corrected (layers 5–29: 0.3–4.4%); the o_proj dead-column decomposition was added.

## Architecture Questions

- The ranking contradicts the Phase 1 handoff, which proposed accumulator studies at N = 16/64/256. The real K values (2560, 6912) dictate 256-wide tiles composing to those lengths with 20/21-bit outputs, so the Phase 1 proposal was right in direction and wrong in parameters. Handoffs sit at the bottom of the source-of-truth hierarchy; the data wins.
- The checkpoint packs four output rows per byte at one K position. This is a dataflow fact a Y-Array design must either adopt or repack.

## Alternative Approaches Worth Considering

- A memory-feed experiment could be well-posed for the ternary layers alone (521 MB/token at 2 bits) even before Q-019 is decided, if the founder wants the memory question first.

## Questions for Founder / ChatGPT

- Q-019 (scope of the BF16 head) and Q-018 (second model), as listed in the handoff.

## Recommended Actions

1. Commit as `research: characterize real ternary transformer workload`, then a docs-only fixup recording the hash.
2. Do not push under this task's authorization.
3. Carry Q-019 to the next direction discussion before choosing between EXP-0003 (accumulation) and a memory-feed experiment.

## Severity

Informational. No defects found; the concerns are about interpretation limits already stated in EXP-0002.
