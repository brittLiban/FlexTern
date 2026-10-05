# Handoff

Date: 2026-10-05
Agent: Claude (Claude Code, model Fable 5.1), implementation role
Current commit: the Phase 1 commit (see `git log`; this file is part of it)

## Objective

FT-CURRENT, Phase 1: Y-Core v0.1 + Y-Dot16 + first generic synthesis comparison against INT8 multiply baselines. Complete.

## What I Completed

Everything in the Phase 1 definition of done. Toolchain installed, all RTL written and verified, all four modules synthesized with two Yosys versions, results and documentation written, one-command reproduction working.

## What I Changed

- New: `reference/ternary_op.py`, `tests/test_reference.py`
- New: `rtl/y_core.sv`, `rtl/y_dot16.sv`, `rtl/int8_mul_baseline.sv`, `rtl/int8_dot16_baseline.sv`
- New: `tests/y_core_tb.sv`, `tests/y_dot16_tb.sv`, `tests/int8_baseline_tb.sv`
- New: `scripts/gen_vectors.py`, `scripts/run_all.ps1`
- New: `synth/*.ys` (four scripts)
- New: `results/*_synth.txt`, `results/synth_environment.txt`, `results/benchmark.md`, `results/crosscheck_yosys052/`
- New: `docs/decisions/ADR-0002-y-core-v0-1-interface.md`
- Updated: `README.md`, `.gitignore`, `docs/CURRENT_WORK.md`, `docs/PROJECT_STATE.md`, `docs/ARCHITECTURE.md`, `docs/BENCHMARKS.md`, `docs/KNOWN_ISSUES.md`, `docs/OPEN_QUESTIONS.md`, `docs/THESIS.md`, `docs/ROADMAP.md`, `docs/experiments/EXP-0001-y-core.md`

## Tests / Validation Run

```
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

All steps PASS. Also run by hand: each testbench individually, a mutation test on Y-Core (broken negate → 256 mismatches, exit 1), and the four synthesis scripts under WSL Yosys 0.52.

## Actual Results

See `docs/experiments/EXP-0001-y-core.md` and `results/benchmark.md`. Headline, Yosys 0.69 generic cells: Y-Core 33 vs INT8 mul 407; Y-Dot16 1,330 vs INT8 dot16 8,646. No `$mul` in any FlexTern module.

## Remaining Problems

- KI-001: WASM Yosys spends ~13 min on the INT8 dot16 baseline. Cosmetic.
- KI-002: Smart App Control blocks native Windows Yosys. Worked around.
- KI-003: small-design cell counts swing ~2× between Yosys versions.
- Housekeeping outside the repo: `C:\oss-cad-suite` (extracted, blocked, several GB) and the 600 MB archive in the session scratchpad can be deleted. I did not delete them.

## Important Context

- The RTL deliberately uses flat packed ports and a `genvar`/`integer` style so both Icarus `-g2012` and Yosys `read_verilog -sv` accept it. Unpacked array ports would break Yosys.
- Sign extension is explicit everywhere. Do not "simplify" `$signed({{4{p[8]}}, p})` to `p`; Verilog drops signedness in mixed expressions.
- `run_all.ps1` requires both exit code 0 and the testbench's success token. `vvp` returns 0 after `$finish` even on failure, so the token is the real check.
- `$scopeinfo` cells are deleted before every `stat`. If you compare against logs from before this change, subtract 16 from the flattened totals.
- `tests/vectors/` is regenerated with a fixed seed and is git-ignored.
- The Y-Dot16 accumulator is a naive sequential chain. That was a deliberate v0.1 choice, not an oversight, and it is now the largest cost in the design.

## Questions for Founder / ChatGPT

1. Is the INT8 baseline the right comparison going forward, or should the next baseline be INT4 or a packed-ternary-on-INT8 datapath to separate "reduced function" from "architectural" savings?
2. Should the next measurement be physical (a real FPGA target for Fmax/LUT/DSP) or structural (larger N, accumulator variants)? These answer different questions.
3. Does the accumulation finding change the thesis emphasis from "no multiplier" toward "cheap wide accumulation of ±x terms"?

## Recommended Immediate Next Consideration

The element-level advantage is real but gets diluted by accumulation at 16 elements. Before scaling to Y-Array, decide whether to characterize the accumulator (carry-save, tree, sparsity-skip, width) at N = 16/64/256 so the project knows where the logic actually goes before committing to an array dataflow.
