# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** EXP-0002B — Validate the EXP-0002 workload findings against an independent native ternary family (TII Falcon-E 1B/3B) before any Youssouf Architecture decision hardens around BitNet-specific numbers.

**Status:** Complete. Awaiting founder/ChatGPT review. No next implementation objective has been set.

**Owner / agent:** Claude (implementation)

**Started:** 2026-10-06
**Completed:** 2026-10-06

## Why we are doing this

Every workload-driven conclusion in EXP-0002 (dot lengths, zero rate, dead channels, scale placement, INT8 activations, head dominance, 256-divisibility, accumulator widths) came from one Microsoft checkpoint. Hardware decisions must not overfit to it.

## Question we are trying to answer

Which hardware-relevant workload properties are model-specific, and which look stable enough across independent ternary families to inform the first architecture?

## Scope

- Generalize the Phase 2 profiler to a per-model spec table; no model constants in the analysis code.
- Fully profile Falcon-E-1B-Base and Falcon-E-3B-Base (packed `main` branches, pinned); cross-check ternarization and scale convention on the 1B `prequantized` latents.
- Regression-check that the Microsoft EXP-0002 numeric outputs are unchanged after the refactor.
- Produce a cross-model comparison and classify each property as LIKELY GENERALIZING / MODEL-SPECIFIC / UNCERTAIN.
- Document implications for Y-Core/Y-Architecture and for the planned roofline experiment's input space.

## Explicitly out of scope

Any RTL change, Y-Array, accumulator variants, roofline model, zero skipping, 256-wide hardware, compiler, ASIC, FPGA, accelerator comparisons, tokens/sec claims.

## Definition of done

All met; evidence in `experiments/EXP-0002B-cross-model-validation.md`.

- [x] Canonical docs read first; Falcon-E provenance recorded (ADR-0004).
- [x] Falcon-E 1B and 3B fully characterized; 1B cross-checked on latents (100% with BF16 emulation).
- [x] Architecture comparison, real K values, divisibility table.
- [x] Activation quantization, scale semantics, ternary encoding verified from reference code and data.
- [x] Weight distribution, sparsity structure, output head, accumulator widths for every observed K.
- [x] Microsoft EXP-0002 results reproduce: 10,929 numeric values identical.
- [x] Invariants classified; Y-Core and roofline implications documented.
- [x] Advisor plan and completion reviews persisted in `reviews/`.
- [x] No hardware implemented; one next objective recommended.

## Required validation

Met: 37 unit tests (`tests/test_analysis.py`, `tests/test_analysis_cross_model.py`), in-tool reconciliation asserts on all three checkpoints, programmatic regression diff on BitNet outputs.

## Relevant files

`analysis/`, `tests/test_analysis*.py`, `scripts/run_workload_profile.ps1`, `results/workload_profile/` (BitNet canonical, `falcon-e-1b/`, `falcon-e-3b/`, `cross_model/`).

## Relevant ADRs

ADR-0002 (assumptions under test), ADR-0003 (superseded), ADR-0004 (reference workload set).

## Relevant experiments

EXP-0001, EXP-0002, EXP-0002B.

## Known uncertainties

Two families are evidence, not proof (Q-018). Both inference activation quantizers share transformers' code; independent evidence is TII's training kernel. INT8 persistence is a tracked risk (Q-021).

## Questions requiring founder decision

None block this objective. See the EXP-0002B handoff.
