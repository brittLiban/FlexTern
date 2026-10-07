# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** EXP-0006 — Model-resident memory feasibility for the Youssouf Architecture (Gate 2 of the co-design program).

**Status:** Complete. Awaiting founder/ChatGPT review. No next objective has been started.

**Owner / agent:** Claude (implementation and research)

**Started:** 2026-10-07
**Completed:** 2026-10-07

## Why we are doing this

EXP-0005 showed that bank-local consumption removes the ternary weight stream from the global path only if the whole packed model (396–730 MB for the reference checkpoints) stays resident close to compute. The project had no evidence that such residency can be built at acceptable capacity, bandwidth, area, energy, packaging and cost.

## Question we are trying to answer

Gate 2A: is there a technically credible, publicly evidenced memory organization that holds the packed ternary weights near compute with enough local bandwidth for the EXP-0005 dataflow? Gate 2B: does 2-bit ternary materially improve that feasibility versus INT4 in the same architecture?

## Outcome (one paragraph; details in EXP-0006)

**Gate 2A: PASS at the published-technology envelope for Falcon-E 1B, marginal PASS for Falcon-E 3B and BitNet 2B4T (BitNet within 7 mm² of the reticle at low density; its tied 657 MB head stays external); residency is of the ternary weights only, head and KV remain external** (one monolithic SRAM-heavy die of the Graphcore/Dojo class: 404–648 mm² for 440 MB at 0.68–1.09 MB/mm², inside the demonstrated 900 MB / 823 mm²; local bandwidth exceeds need by orders of magnitude; activation delivery ≤ 12% of weight bytes up to 128 banks); **FAIL for on-die residency at ≥ 7B; INCONCLUSIVE for DRAM-class near-memory** (capacity trivial, compute-in-stack evidence is one research prototype, weight-read energy class unchanged). **Gate 2B: PASS structurally at 1B–3B** (2-bit keeps every canonical model inside one reticle where INT4 moves 3B outside one reticle at every whole-die anchor (cache-only-die anchor: only with ≤ 5% margin and no compute) and 1B to a marginal fit; local bandwidth, bank capacity and local-read energy halve), **FAIL as a technology-class change at ≥ 7B**. Bandwidth never forces residency at these sizes (one LPDDR5X channel streams ~120 tok/s of Falcon-E 1B); the resident form is an energy/latency choice resting on an unmeasured modern-node SRAM read energy. After residency the head and KV cache are 88–99.7% of global bytes.

## Scope (as executed)

22 sources with provenance and 8 explicit assumptions; capacity at 2/4/8/16 bits with head and metadata separate and 5–30% margins; hypothetical 7B–70B scaling; residency policies (full, layer, tiled, hot set); 16–256 banks with the EXP-0005 collector law; bandwidth at 10–250 tok/s with 50–80% utilization; area ranges by density tier and tier crossings against the 858 mm² reticle; energy ranges per movement; seven package organizations; economics proxies; INT4 fairness; sensitivity thresholds; EXP-0003/0004/0005 reconnection at 128/2k/8k/32k context.

## Explicitly out of scope

Buying hardware, FPGA flows, analog compute-in-memory as the subject, head/KV engines, new model architectures, training, freezing a memory technology (no ADR), product pricing models, EXP-0007.

## Required validation

`tests/test_memory_residency.py` (17 tests) and the existing non-RTL suites (98 tests) pass; RTL untouched.

## Relevant files

`analysis/memory_residency/`, `results/memory_residency/`, `tests/test_memory_residency.py`.

## Relevant ADRs

ADR-0002, ADR-0004, ADR-0005, ADR-0006. No new ADR.

## Relevant experiments

EXP-0003 (byte accounting), EXP-0004 (candidate head/KV bits), EXP-0005 (dataflow, traffic, collector bound), EXP-0006 (this).

## Known uncertainties

Modern-node SRAM read energy unmeasured; Y-Core bank area unmeasured; eDRAM availability and multi-die SRAM stacking assumed from absence of evidence; no cost or latency model.

## Questions requiring founder decision

Whether an energy/latency advantage justifies 400–650 mm² of SRAM-heavy silicon when LPDDR5X streaming already serves 100 tok/s at 1B; target model size (couples to Q-015); whether Q-022 is next. See the handoff `handoffs/2026-10-07-exp-0006-memory-residency.md`.
