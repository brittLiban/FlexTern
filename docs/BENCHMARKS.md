# FlexTern Benchmarks

This is the canonical scoreboard. It contains **only validated measurements**. Every row must reference an experiment record in `experiments/` and a commit hash. Numbers that are estimated, projected, or targeted do not belong in the main table.

## Validated measurements

| Experiment | Commit | Hardware | Workload | Baseline | Metric | Result | Reproducible? | Notes |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | No validated measurements yet. |

## Projections and targets (not measurements)

This section is deliberately separate. Anything here is **TARGET** or **SPECULATIVE** and must be labeled as such. Nothing in this section may be quoted as a result.

None recorded.

## Rules

- A row is added only after the experiment record is complete and the result has been reproduced from a clean checkout.
- FPGA measurements and ASIC projections are never placed in the same table.
- If methodology changes, old rows are kept and annotated, not replaced.
- Reproducible? means: a second agent or run, from the stated commit, obtained the same number within stated tolerance.
