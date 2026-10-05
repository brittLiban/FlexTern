# EXP-0001 — Y-Core v0.1 vs conventional multiply-based datapath

Status: Not started
Date: —
Commit: —

This record is a stub created with the objective definition. It will be filled in as FT-CURRENT (`../CURRENT_WORK.md`) proceeds. No measurements exist.

## Research Question

On a fixed FPGA target under identical methodology, how does a single-element ternary datapath (Y-Core: activation × weight ∈ {-1, 0, +1}) compare to a conventional multiply-based datapath in logic resources, achievable clock frequency, and, if measurable, power?

## Hypothesis

H1 from `../THESIS.md`: the Y-Core is materially smaller and/or lower-power than the multiply-based baseline at the same activation width. **HYPOTHESIZED**, untested.

## Why It Matters

This is the first and most basic test of the FlexTern thesis. If H1 fails at the primitive level, the project must re-examine whether its advantage lies elsewhere (memory, encoding, sparsity) or does not exist.

## Setup

Not yet defined. Blocked on Q-001 through Q-004 in `../OPEN_QUESTIONS.md`.

## Hardware / Software Environment

Not yet defined.

## Baseline

Not yet defined. See Q-004. Must be decided before any synthesis run.

## Method

Not yet defined.

## Metrics

Planned: LUT/FF/DSP utilization, maximum frequency after place-and-route, estimated power if the toolchain supports it. Final list to be fixed before the first run and not changed afterward without annotation.

## Results

None. Not yet run.

## Interpretation

None.

## Surprises

None.

## Limitations

Known in advance:
- Single-element comparison; says nothing about composed structures.
- FPGA-specific; not an ASIC result.
- Baseline fairness depends on Q-004.

## What This Proves

Nothing yet.

## What This Does NOT Prove

Nothing yet. When complete, this section must explicitly state that the result does not establish system-level, transformer-scale, or GPU-relative advantages.

## Artifacts / Logs

None.

## Follow-up Questions

None yet.
