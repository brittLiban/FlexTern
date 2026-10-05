# FlexTern

FlexTern is an exploratory hardware/software research project investigating whether AI compute designed around **ternary and ultra-low-precision models** can deliver materially better inference economics than conventional accelerator architectures.

The long-term aspiration is a full-stack platform (compiler, runtime, accelerator architecture, silicon). The near-term work is far smaller: a single ternary primitive, the **Y-Core**, and a controlled comparison against a conventional multiply-based datapath.

## Current stage

**Phase 1 — ternary primitive.** No hardware has been implemented or measured yet. The first objective is defined in [`docs/CURRENT_WORK.md`](docs/CURRENT_WORK.md). See [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) for what is and is not proven.

## Repository layout

```
AGENTS.md        How humans and agents work in this repo. Read first.
docs/            Canonical project documentation (see docs/INDEX.md)
  decisions/     Architecture Decision Records (ADR-XXXX)
  experiments/   Experiment records (EXP-XXXX), including negative results
  reviews/       Independent reviewer critiques
  handoffs/      Agent-to-agent handoff notes
reference/       External references, papers, datasheets, notes
rtl/             Hardware description sources
tests/           Simulation and verification
synth/           Synthesis / implementation scripts and constraints
results/         Raw measurement outputs referenced by experiments
scripts/         Helper scripts
```

## How to run current validation

Nothing to run yet. No RTL, tests, or synthesis flows exist. This section will be updated when the first validated flow lands.

## Documentation

Start at [`docs/INDEX.md`](docs/INDEX.md). Agents start at [`AGENTS.md`](AGENTS.md).

## Research disclaimer

FlexTern is research. Nothing here is a product. No performance, efficiency, or cost advantage has been demonstrated. Performance and architecture claims in the documentation are labeled as known, measured, inferred, hypothesized, target, or speculative. Any unlabeled performance statement should be treated as unverified.
