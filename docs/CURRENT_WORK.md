# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** EXP-0003 — Analytical compute-vs-memory bottleneck model: how much end-to-end benefit ternary specialization can deliver across the measured BitNet/Falcon-E envelope before non-ternary work (KV cache, output head, attention, other) caps it.

**Status:** Complete. Awaiting founder/ChatGPT review. No next implementation objective has been set.

**Owner / agent:** Claude (implementation)

**Started:** 2026-10-06
**Completed:** 2026-10-06

## Why we are doing this

FlexTern's thesis treats multi-fold end-to-end gains as important. EXP-0002/EXP-0002B showed that non-ternary streams can exceed the ternary stream. Before building any datapath or memory-feed hardware, the project needed to know where ternary specialization has system-level headroom and where it does not.

## Question we are trying to answer

Across three reference workloads and sweeps of context, batch, bandwidth, ternary throughput, packing, activation precision, head/KV precision, and residency: which component limits decode, and what is the Amdahl ceiling on end-to-end speedup from ternary specialization alone?

## Scope

Analytical model only (`analysis/roofline_models.py`, `analysis/flextern_roofline.py`), consuming the three canonical `model_summary.json` files; component-level stage model with optimistic and conservative overlap bounds; analytical crossovers verified against numerical sweeps; ternary-only ceilings; 2×/3× feasibility regions; residency, packing, activation and KV precision sensitivities; tile utilization; a concise prefill treatment; tests; EXP-0003 record; ADR-0005; Q-019 reframe.

## Explicitly out of scope

Any RTL, Y-Array, accumulator trees, KV quantization, attention hardware, compiler, runtime, FPGA synthesis, ASIC estimates, tile selection, clock claims, measured tokens/sec, model training or quantization, competitor benchmarks.

## Definition of done

All met; evidence in `experiments/EXP-0003-compute-memory-roofline.md` and `../results/roofline/`.

- [x] Canonical docs read; three workload summaries consumed programmatically with identity asserts.
- [x] Ternary compute/traffic, head, attention, KV, and other modeled separately.
- [x] Context, batch, bandwidth, ternary-compute, clock, non-ternary-rate sweeps complete.
- [x] 2-bit and 5-trit packing; INT8 and hypothetical 4-bit activations; BF16 and hypothetical INT8/INT4 head and KV.
- [x] Residency policies over 0–1 GB; optimistic and conservative overlap bounds.
- [x] Break-even and crossover equations derived and confirmed by sweep transitions.
- [x] Ternary-only ceilings, context crossovers, 2×/3× regions; edge hypothesis evaluated.
- [x] 26 tests pass covering the twenty required validations.
- [x] Advisor plan and completion reviews persisted; ADR-0005; Q-019 reframed; Q-022 added.
- [x] One next objective recommended; nothing started.

## Required validation

Met: `tests/test_roofline.py` (26 tests), in-driver asserts (identity, revisions, KV reproduction, terms/byte).

## Relevant files

`analysis/roofline_models.py`, `analysis/flextern_roofline.py`, `tests/test_roofline.py`, `scripts/run_roofline.ps1`, `results/roofline/`.

## Relevant ADRs

ADR-0002, ADR-0004, ADR-0005.

## Relevant experiments

EXP-0001, EXP-0002, EXP-0002B, EXP-0003.

## Known uncertainties

All hardware rates are abstract. The non-ternary compute rate has no evidence (Q-022). Ceilings are upper bounds on an impossible ideal, not achievable speedups.

## Questions requiring founder decision

None block this objective. See the EXP-0003 handoff.
