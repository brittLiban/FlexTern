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

### EXP-0005 — packed ternary consumers (generic synthesis, Yosys 0.52 native under WSL2, `synth -flatten`, default ABC; see KI-008 for why not the YoWASP build)

| Experiment | Commit | Hardware | Workload | Baseline | Metric | Result | Reproducible? | Notes |
|---|---|---|---|---|---|---|---|---|
| EXP-0005 | eef813f | None (generic Yosys 0.52, WSL) | `ternary_unpack4`, one packed byte → 4 Y-Core codes | — | generic cells | 15 (path 4 gates) | Yes, `scripts/run_exp0005.ps1 -SynthTool wsl` | 2-bit decrement per weight. |
| EXP-0005 | eef813f | None (generic Yosys 0.52, WSL) | `y_stream_engine` 64 B/cycle = 256 terms/cycle, 22-bit acc | INT8 × INT4 `int4_stream_baseline` at 256 terms/cycle | generic cells | **25,466** vs 69,526 (2.7×) | Yes; whole sweep run twice, identical | 333 FFs, 0 `$mul`, longest path 42 gates (no FF); baseline 231 FFs, 256 `$mul`, 53 gates. |
| EXP-0005 | eef813f | None (generic Yosys 0.52, WSL) | `y_stream_engine` 16/128/256 B/cycle | — | generic cells | 7,371 / 49,769 / 98,472 | Yes | 115 / 97 / 96 cells per term. |
| EXP-0005 | eef813f | None (generic Yosys 0.52, WSL) | `y_memory_bank` 16/32/64 B/cycle | — | generic cells | 9,450 / 16,728 / 31,387 | Yes | includes 4-entry FIFO and address generator; memories are ports. |
| EXP-0005 | eef813f | None (generic Yosys 0.52, WSL) | `y_near_memory_engine` NB=4 × 16 B (256 terms/cycle) | — | generic cells | 38,983 | Yes | 3,749 FFs; collector included. |
| EXP-0005 | eef813f | Icarus Verilog 12 | streaming and near-memory engines functional | Python reference (exact ints) | pass/fail | PASS, 26 tests incl. 5 checkpoint slices | Yes | II = 1 cycle-counted; both engines byte-identical. |

Every EXP-0005 synthesis point was run twice (sweep and a scratch re-run) with byte-identical STAGE 2 cell breakdowns. EXP-0005 rows are structure counts from a different Yosys version than rows 1-6 and must not be ratioed against them (EXP-0001 measured 1.1–1.8× version sensitivity). Required clocks for 100/250 GB/s in `../results/hardware_gate/bandwidth_requirements.csv` are TARGETS, not measurements, and are deliberately kept out of this table.

## Projections and targets (not measurements)

This section is deliberately separate. Anything here is **TARGET** or **SPECULATIVE** and must be labeled as such. Nothing in this section may be quoted as a result.

None recorded.

## Rules

- A row is added only after the experiment record is complete and the result has been reproduced from a clean checkout.
- FPGA measurements and ASIC projections are never placed in the same table.
- If methodology changes, old rows are kept and annotated, not replaced.
- Reproducible? means: a second agent or run, from the stated commit, obtained the same number within stated tolerance.
