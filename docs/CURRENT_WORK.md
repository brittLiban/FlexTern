# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** Build and validate Y-Core v0.1 and establish a controlled synthesis comparison against a conventional multiply-based datapath.

**Status:** Defined, not started. Blocked on founder decisions listed at the bottom.

**Owner / agent:** Claude (implementation), once unblocked.

**Started:** —

## Why we are doing this

The entire FlexTern thesis (`THESIS.md`, hypothesis H1) rests on the assumption that a ternary datapath is materially cheaper than a multiply-based one. This has never been measured in this project. Before building anything larger, we need to know whether the primitive advantage is real, how large it is, and what it depends on.

## Question we are trying to answer

On a fixed target and under identical methodology, how does a Y-Core (activation × ternary weight) compare to a conventional multiply-based datapath in logic resources, achievable clock, and, where measurable, power?

## Scope

- Y-Core v0.1: a single-element ternary operation (+1 → +act, 0 → skip/zero, -1 → -act) in RTL, with a testbench proving functional correctness against a reference model.
- A conventional baseline datapath implementing the equivalent operation with a multiplier, written with comparable care, same activation width, same target.
- Synthesis of both on the same target with the same tool settings and constraints.
- Collection of resource utilization, timing, and (if the flow supports it) power estimates.
- Experiment record `EXP-0001-y-core.md` with method, raw results, interpretation, and limitations.

## Explicitly out of scope

- Y-Dot, Y-Array, or any accumulation structure.
- Memory interfaces, weight packing/unpacking logic beyond what the Y-Core itself needs.
- Any neural-network workload.
- Any ASIC projection.
- Any GPU comparison.
- Compiler, runtime, or software of any kind beyond test scaffolding.
- Choosing the long-term FPGA platform. The target for this experiment is a measurement vehicle, not a commitment.

## Definition of done

- Y-Core RTL and baseline RTL exist in `rtl/` with testbenches in `tests/` that pass.
- Both designs synthesize on the chosen target with scripts in `synth/` that reproduce the result from a clean checkout.
- Raw tool reports are stored in `results/` and referenced from the experiment record.
- `EXP-0001-y-core.md` is complete, including limitations and what the result does *not* prove.
- `BENCHMARKS.md` carries the validated numbers with commit hash.
- `PROJECT_STATE.md` is updated.
- Results are reported to the founder. **The next objective is not started.**

## Required validation

- Functional: exhaustive or randomized testbench showing Y-Core output equals reference for all three weight values across the activation range, including sign edge cases.
- Fairness: the baseline must not be deliberately naive. Same activation width, same pipelining discipline, same constraints. If the baseline uses a DSP block and the Y-Core does not, both the raw numbers and a DSP-equivalent accounting must be reported.
- Reproducibility: a second run from a clean checkout yields the same resource and timing numbers.

## Relevant files

None yet. Expected: `rtl/`, `tests/`, `synth/`, `results/`.

## Relevant ADRs

- ADR-0001 — Operating model (how this objective was chosen and how its results will be handled).

## Relevant experiments

- EXP-0001 — Y-Core v0.1 vs conventional datapath (stub; not started).

## Known uncertainties

- A single-element comparison may show a large relative advantage that shrinks or vanishes once accumulation, control, and memory are included. This experiment cannot answer that, and the record must say so.
- FPGA LUT/DSP accounting does not translate directly to ASIC area. Results are FPGA-specific.
- The "skip" path for weight = 0 may or may not have a measurable benefit at single-element scale.

## Questions requiring founder decision

These block the start of implementation. Each is also tracked in `OPEN_QUESTIONS.md`.

1. **Target FPGA and toolchain** for this measurement (Q-001). Any available board/tool is acceptable as a vehicle, but the choice should be made explicitly and recorded, and it does not commit the project long-term.
2. **Activation bit width** for v0.1 (Q-002). Recommendation if undecided: pick one width (e.g. 8-bit signed) and treat width sweeps as a possible follow-up, not part of v0.1.
3. **Ternary weight encoding** at the Y-Core boundary (Q-003). The Y-Core needs some 2-bit (or other) representation at its input. This choice affects the comparison only slightly but should be explicit.
4. **Definition of the conventional baseline** (Q-004). Specifically: signed activation × signed N-bit weight (what N?), with or without a DSP block, with or without accumulation. This materially affects fairness and must be decided before synthesis.
