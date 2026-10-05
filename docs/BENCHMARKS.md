# FlexTern Benchmarks

This is the canonical scoreboard. It contains **only validated measurements**. Every row references an experiment record in `experiments/` and a commit. Numbers that are estimated, projected, or targeted do not belong in the main table.

## Validated measurements

All rows: generic Yosys synthesis, `synth -flatten`, default ABC, no target device. Metric is total generic logic cells (`$scopeinfo` excluded). Full breakdowns and logs in `../results/`.

| Experiment | Commit | Hardware | Workload | Baseline | Metric | Result | Reproducible? | Notes |
|---|---|---|---|---|---|---|---|---|
| EXP-0001 | 823902e | None (generic Yosys 0.69, YoWASP) | Y-Core, 1 × (int8 × ternary) | INT8 × INT8 multiplier | generic cells | 33 vs 407 (12.3×) | Yes, `scripts/run_all.ps1` | No `$mul` inferred in Y-Core. |
| EXP-0001 | 823902e | None (generic Yosys 0.69, YoWASP) | Y-Dot16, 16-element ternary dot | INT8 16-element dot | generic cells | 1,330 vs 8,646 (6.5×) | Yes | Ratio shrinks with scale; accumulation is shared cost. |
| EXP-0001 | 823902e | None (generic Yosys 0.52, WSL) | Y-Core | INT8 multiplier | generic cells | 60 vs 423 (7.1×) | Yes, `results/crosscheck_yosys052/` | Cross-version check. Small-design counts are tool-sensitive. |
| EXP-0001 | 823902e | None (generic Yosys 0.52, WSL) | Y-Dot16 | INT8 dot16 | generic cells | 1,694 vs 8,886 (5.2×) | Yes | Cross-version check. |
| EXP-0001 | 823902e | Icarus Verilog 12 | Y-Core functional | Python reference / exact math | pass/fail | PASS, 768 + 256 cases | Yes | Exhaustive valid domain. |
| EXP-0001 | 823902e | Icarus Verilog 12 | Y-Dot16 functional | TB math + Python reference | pass/fail | PASS, 26 + 10,000 + 2,000 | Yes | Includes independent Python-generated vectors. |

Results in rows 1-6 were generated from the tree committed as 823902e.

## Projections and targets (not measurements)

This section is deliberately separate. Anything here is **TARGET** or **SPECULATIVE** and must be labeled as such. Nothing in this section may be quoted as a result.

None recorded.

## Rules

- A row is added only after the experiment record is complete and the result has been reproduced from a clean checkout.
- FPGA measurements and ASIC projections are never placed in the same table.
- If methodology changes, old rows are kept and annotated, not replaced.
- Reproducible? means: a second agent or run, from the stated commit, obtained the same number within stated tolerance.
