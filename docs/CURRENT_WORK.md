# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** Phase 1 — Y-Core v0.1 + Y-Dot16 + first generic synthesis comparison against an INT8 multiply-based datapath.

**Status:** Complete. Awaiting founder/ChatGPT review of results. No next objective has been set.

**Owner / agent:** Claude (implementation)

**Started:** 2026-10-05
**Completed:** 2026-10-05

## Why we are doing this

The entire FlexTern thesis (`THESIS.md`, hypothesis H1) rests on the assumption that a ternary datapath is materially cheaper than a multiply-based one. Before building anything larger, we need to know whether the primitive advantage is real at the RTL/logic level, how large it is in generic cell counts, and whether the primitive composes correctly into a dot product.

## Question we are trying to answer

Can activation × ternary-weight be implemented correctly without a multiplier, and how does the resulting generic logic compare to a conventional signed INT8 × INT8 multiplier, at single-element and 16-element dot-product scale?

## Scope

- Python golden reference (`reference/ternary_op.py`) with unit tests.
- Y-Core v0.1 RTL (`rtl/y_core.sv`): combinational, 8-bit signed activation, 2-bit ternary weight, 9-bit signed result. Exhaustive testbench over all 768 valid inputs plus the reserved encoding.
- Y-Dot16 RTL (`rtl/y_dot16.sv`): 16 Y-Cores plus adder tree, 13-bit signed output. Deterministic, 10,000 randomized, and Python-generated cross-check vectors.
- INT8 baselines (`rtl/int8_mul_baseline.sv`, `rtl/int8_dot16_baseline.sv`) with their own correctness testbench.
- Yosys generic synthesis of all four modules with logs in `results/`.
- `scripts/run_all.ps1` reproducing the whole flow on Windows.
- `results/benchmark.md` with measured values only.

Scope note: the original FT-CURRENT definition excluded Y-Dot16. Founder direction on 2026-10-05 added it. Interface parameters are fixed by ADR-0002.

## Explicitly out of scope

- Y-Array or anything beyond a 16-element dot product.
- Memory interfaces, weight unpacking beyond the Y-Core's own decode.
- Any neural-network workload or BitNet model.
- FPGA bitstream, physical place-and-route, timing closure, power measurement.
- ASIC projection of any kind.
- GPU comparison.
- Compiler, runtime, CUDA compatibility.
- Pipelining or clocked designs.

## Definition of done

All items met; see `experiments/EXP-0001-y-core.md` for evidence.

- [x] Python reference exists and its tests pass.
- [x] `y_core.sv` exists; all 768 valid cases verified; -128 × -1 = +128 verified by name.
- [x] `y_dot16.sv` exists; deterministic, randomized, and Python cross-check tests pass.
- [x] INT8 baselines exist and pass their own exhaustive/randomized tests.
- [x] All four modules synthesize in Yosys; logs saved in `results/`.
- [x] `results/benchmark.md` contains only tool-obtained numbers.
- [x] `scripts/run_all.ps1` runs the full flow and stops on failure.
- [x] Documentation updated per `AGENTS.md` §9.

## Required validation

Met. Functional correctness by exhaustive and randomized simulation; independence by Python-vs-RTL cross-check; synthesis by Yosys `check -assert` passing on every module.

## Relevant files

`reference/ternary_op.py`, `tests/test_reference.py`, `rtl/*.sv`, `tests/*_tb.sv`, `scripts/gen_vectors.py`, `scripts/run_all.ps1`, `synth/*.ys`, `results/*.txt`, `results/benchmark.md`.

## Relevant ADRs

- ADR-0001 — Operating model.
- ADR-0002 — Y-Core v0.1 interface, encoding, and measurement baseline.

## Relevant experiments

- EXP-0001 — Y-Core v0.1 and Y-Dot16 vs INT8 baseline (complete).

## Known uncertainties

- Generic Yosys cell counts are a structure measurement. They do not predict FPGA LUT counts, ASIC area, frequency, or power on any real technology.
- The INT8 baseline computes a strictly more general function with a wider output. Part of the measured difference is reduced function, not architectural insight.
- Single-element and 16-element results say nothing about system-level behavior once memory, control, and interconnect are included.

## Questions requiring founder decision

None block this objective. For what to do next, see the handoff `handoffs/2026-10-05-phase1-y-core.md` and `OPEN_QUESTIONS.md`.
