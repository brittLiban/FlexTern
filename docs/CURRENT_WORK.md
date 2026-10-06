# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** EXP-0005 — Packed ternary hardware consumer: streaming baseline vs banked near-memory dataflow (Gate 1 of the model–hardware co-design program).

**Status:** Complete. Awaiting founder/ChatGPT review. No next objective has been started.

**Owner / agent:** Claude (implementation)

**Started:** 2026-10-07
**Completed:** 2026-10-07

## Why we are doing this

EXP-0004 showed that a compressed representation only helps if the consumer operates on it without expanding it first; the stock runtime unpacks 2-bit ternary weights to BF16 every forward. Every EXP-0003 ceiling assumes a ternary engine that consumes packed 2-bit bytes at the bandwidth bound. Neither the datapath nor the dataflow had been built. The founder has no FPGA and does not expect one for months, so this was intentionally an RTL + simulation + generic synthesis + analytical data-movement experiment.

## Question we are trying to answer

Gate 1A: can a hardware datapath consume the real checkpoint packing directly, at initiation interval 1, without a persistent full-weight expansion? Gate 1B: does keeping packed weights local to memory banks and computing there materially reduce global weight movement once activation broadcast, output and partial-sum traffic and capacity are counted?

## Outcome (one paragraph; details in EXP-0005)

**Gate 1A: PASS at the structural/cycle level.** The checkpoint packing (four output rows per byte at one K position, code = q+1) is consumed directly by a row-quad engine; decode is a 2-bit decrement; II = 1 at every canonical K with 333 flip-flops of state at 256 terms/cycle; 25,466 generic cells, no multiplier (INT4 structural baseline: 69,526). Absolute Fmax and bandwidth UNMEASURED (no FPGA). **Gate 1B: PASS at the dataflow/global-traffic level.** A bank-local engine with bit-identical results removes the ternary weight stream from the global path (Falcon-E 1B: 399 MB → 2.8 MB per token) at the cost of whole-model residency (396–730 MB), NB × K activation delivery per matvec with 13 KB buffers per bank (or a broadcast equal to the weight bytes without them), and a collector; the global limiter becomes the head and KV cache. The gain in global bytes is format-neutral (INT4 gets it too); ternary's specific contribution is half the local bytes and capacity and 2.7× fewer cells per term. Physical energy, cost and memory technology INCONCLUSIVE.

## Scope (as executed)

See EXP-0005 Setup and Method. 14 synthesis points (Yosys 0.52 native in WSL; KI-008), 26 tests, five checkpoint slices from all three models, four-level traffic model, capacity sweeps, required clocks for 100/250/500 GB/s labeled UNMEASURED, EXP-0003/0004 reconnection.

## Explicitly out of scope

FPGA board selection or toolchains, physical timing, power claims, analog/true compute-in-memory, INT4 head or 4-bit attention engines, freezing the model format, ISA, compiler, ASIC node, memory technology choice, competitor claims, EXP-0006.

## Required validation

`tests/test_hardware_gate.py` (26 tests, all pass in foreground runs), the two RTL testbenches, existing suites (82 host tests, 11 model-precision tests) still pass.

## Relevant files

`rtl/`, `tests/`, `analysis/hardware_gate/`, `synth/exp0005/`, `scripts/run_exp0005.ps1`, `results/hardware_gate/`.

## Relevant ADRs

ADR-0002 (encoding, baseline, generic synthesis), ADR-0004 (checkpoints), ADR-0005 (system scope), ADR-0006 (model-side track). No new ADR: nothing is frozen.

## Relevant experiments

EXP-0001 (primitive), EXP-0002/0002B (packing, K values, zeros), EXP-0003 (break-even), EXP-0004 (compression is not enough), EXP-0005 (this).

## Known uncertainties

No physical target; single-cycle adder trees unpipelined; near-memory top synthesized at NB = 4 only; Yosys 0.69 cross-check incomplete; memory technology for residency unchosen.

## Questions requiring founder decision

Whether to pursue the near-memory direction (and therefore a memory-technology question) or a streaming engine fed by external memory; whether FlexTern owns the head/KV consumer (Q-022); whether the canonical synthesis tool moves to native Yosys. See the handoff `handoffs/2026-10-07-exp-0005-hardware-gate.md`.
