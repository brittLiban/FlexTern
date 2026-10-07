# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** EXP-0007 — Datacenter thesis falsification: does ternary model-resident inference survive batching, HBM, KV and rack-scale reality? (Gate 3 of the co-design program.)

**Status:** Complete. Awaiting founder/ChatGPT review. No next objective has been started.

**Owner / agent:** Claude (implementation and research)

**Started:** 2026-10-07
**Completed:** 2026-10-07

## Why we are doing this

Every FlexTern result before this was batch 1 against a BF16 reference. The datacenter form of the thesis had never faced a conventional accelerator with FP4 weights, HBM, native low-bit KV, continuous batching and current interconnect.

## Question we are trying to answer

Gate 3A: does the resident ternary advantage survive weight amortization across batch? Gate 3B: can 7B–400B residency be distributed across tiles without interconnect, activation, pipeline or capacity costs erasing it? Gate 3C: which regimes, if any, keep structural headroom against a strong baseline?

## Outcome (one paragraph; details in EXP-0007)

**Gate 3A: FAIL beyond small batch.** A tensor-parallel FP4 HBM node reads each weight once per step for the whole batch; a resident pipeline reads each stage's weights once per microbatch (B/stages), so at 70B (40 stages) the resident movement-energy advantage is 5× at batch 1, 2.7× at batch 2, 1.3× at batch 4, and below 1× from batch 8 to 256; single-tile 1B keeps ≥ 2× only to batch 8 at 2k (2 at 8k). **Gate 3B: INCONCLUSIVE leaning FAIL.** Distribution is feasible (42–255 tiles, < 0.1% interconnect bytes, 2-bit halves tiles versus INT4) but KV must live in per-tile DRAM and, at the TARGET compute density, the resident 70B/400B systems deliver 2–4× lower aggregate throughput than one HBM node. **Gate 3C: PASS only for small-batch (1–2 at 7B–400B, ≤ 8 for single-tile 1B), short-context (≤ 8k) interactive serving with no latency advantage, and for ternary streaming on conventional HBM hardware (2–3× fewer weight bytes at small batch); FAIL as a general datacenter architecture.** Prefill on resident tiles at the TARGET density is 17–550× slower than the baseline. Movement-only energy; no tokens/MW claimed. Q-023 (modern SRAM read energy) remains unresolved and load-bearing for the edge/resident path.

## Scope (as executed)

Four systems (A FP4 HBM node sourced to B200/MI355X class with native FP8/NVFP4 KV; B ternary streaming; C ternary resident; D INT4 resident, same topology); 1B/3B canonical and 7B/70B/400B hypothetical shapes; batch 1–256; context 128–131k; prefill and decode; three KV and three head precisions, 32k/128k vocab; tile capacities 256 MB–2 GB; two pipeline modes; serving utilizations; KV-capacity gates; crossover thresholds; MLPerf sanity anchor (baseline within 16% of measured); 16 tests; provenance tests.

## Explicitly out of scope

New RTL, adder-tree pipelining, FPGA, ASIC process, training, native low-bit attention or head engines, model-format freeze, rack/PCB/interposer/protocol design, tokens/MW claims, competitor-superiority claims, EXP-0008.

## Required validation

`tests/test_datacenter_gate.py` (16 tests) and the existing non-RTL suites (115 tests) pass; RTL untouched.

## Relevant files

`analysis/datacenter_gate/`, `results/datacenter_gate/`, `tests/test_datacenter_gate.py`.

## Relevant ADRs

ADR-0002, ADR-0004, ADR-0005, ADR-0006. No new ADR.

## Relevant experiments

EXP-0003 (batch-1 bytes reproduced), EXP-0004 (candidate KV/head), EXP-0005 (local reads reproduced), EXP-0006 (capacities reproduced), EXP-0007 (this).

## Known uncertainties

Compute energy unsourced; resident compute density is a TARGET; serving utilization and link tiers are assumptions; modern SRAM read energy unresolved (Q-023).

## Questions requiring founder decision

Whether to pursue any datacenter form (only the batch-1–2 interactive regime and ternary-as-a-format on conventional hardware survive at 7B+) or concentrate on the edge/resident 1B–3B form; whether Q-023 is now the commercialization-critical measurement. See the handoff `handoffs/2026-10-07-exp-0007-datacenter-gate.md`.
