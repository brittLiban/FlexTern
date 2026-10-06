# Review — EXP-0003 roofline / bottleneck model plan (pre-implementation)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent)
Date: 2026-10-06
Scope: The plan for the analytical compute-vs-memory bottleneck model before any code was written. Inputs: the three canonical `model_summary.json` files (ADR-0004), `cross_model/comparison.md`, EXP-0002/EXP-0002B, and the founder's EXP-0003 prompt including the Q-019 scope reframe.

## Overall Assessment

Inputs are all machine-readable and the stage decomposition is standard. The experiment will be legible only if it is organized around one analytical identity rather than a table of rate-dependent numbers.

## What Looks Strong

- Per-token normalization with batch handled per stream (weights ÷ B, KV not) is the right accounting.
- Two overlap bounds, per stage, summed; no cross-stage prefetch assumed.
- Crossovers derived algebraically and checked against sweep transitions.

## Concerns

- **Organizing identity.** In the memory-bound regime every stage time is bytes/BW, so the ideal ternary-only ceiling equals `total_bytes / non_ternary_bytes`, independent of compute rate, clock, and overlap choice. Hence ceiling > 2× iff ternary bytes > KV + head + norm bytes, and > 3× iff ternary bytes > 2× non-ternary bytes. Derive it, test it (model output at low BW equals the byte ratio), and use it to explain the opportunity curve; then show how the compute-bound regime departs from it.
- **Batch direction.** Batching shrinks weight streams but not KV, so the ceiling should *fall* with batch at fixed context. Verify the model shows it.
- **The non-ternary compute rate has no evidence behind it.** Make it an explicit abstract parameter, sweep it, and show which conclusions move. In the memory-bound regime it should not matter.
- **Reproduce, don't copy.** KV at 4096 (BitNet) must equal 314,572,800 bytes and Falcon 1B at 32,768 must equal 805,306,368 from the equation; terms/byte must be exactly 4.0 (2-bit) and 5.0 (5-trit) at batch 1, 4B at batch B.
- **Residency without hit rates.** Policies: none; head-if-fits; one-layer working buffer (reduces nothing for stream-once decode; say so); greedy whole-tensor fill; ideal any-byte; ideal including KV.
- **Zero handling.** Three cases, packed bytes identical in all three.
- **Plots.** matplotlib is absent; tables are the deliverable. State it.

## Assumptions Being Made

- Activations stay on-chip (reported as counts only).
- Softmax counted as ~4 op-equivalents per score.
- "Other" work has ops but negligible external bytes.

## Potential Failure Modes

- Reading theoretical ceilings as achievable speedups.
- Letting the bottleneck map explode across every parameter; vary one at a time in `sensitivity.csv`.

## Benchmark / Scientific Concerns

- Context sweeps beyond a model's maximum must be flagged hypothetical (BitNet above 4096).
- No tokens/sec figure may be presented as a measurement.

## Architecture Questions

- Whether BitNet's head caps its ceiling at any context while Falcon's does not; if so the "edge wedge" conclusion is vocabulary-dependent.

## Alternative Approaches Worth Considering

- A single global `max(Σcompute, Σmemory)` bound. Only if labeled "requires cross-stage prefetch"; not part of the two requested bounds.

## Questions for Founder / ChatGPT

- None blocking. The scope reframe of Q-019 is a durable decision and deserves ADR-0005.

## Recommended Actions

1. Write ADR-0005 (ternary-first, not ternary-only; head/attention/KV are system-level scope) and reframe Q-019.
2. Open CURRENT_WORK with Started 2026-10-06.
3. Pin Amdahl definitions (compute-only: ternary stage → its memory time; BW-only: → its compute time; ideal: → 0) so tests 16 and 17 follow.
4. Put the twelve attack points verbatim into the completion review as questions put to the reviewer.
5. Persist this review.

## Severity

Informational.
