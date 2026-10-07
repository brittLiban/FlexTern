# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** EXP-0008 — Modern SRAM energy and latency validation for the edge / model-resident FlexTern architecture (resolves or narrows Q-023).

**Status:** Complete (awaiting founder/ChatGPT review). No next objective has been set.

**Owner / agent:** Claude (implementation and research)

**Started:** 2026-10-07
**Completed:** 2026-10-07

## Why we are doing this

Every surviving FlexTern advantage after EXP-0007 is a memory-movement energy claim whose SRAM side rested on a 45 nm tutorial figure (EXP-0006 planning 1.6 pJ/bit). Leakage of hundreds of megabytes of always-on SRAM was never modeled. The edge/resident 1B–3B form cannot be judged on that basis.

## Question we are trying to answer

Gate 4A: do modern multi-bank SRAM reads still beat LPDDR5X streaming on dynamic energy for batch-1 edge decode? Gate 4B: after leakage, does residency keep a total memory-power advantage at realistic duty cycles? Gate 4C: does residency buy latency once streaming is prefetched? Gate 4D: is the 1B–3B resident form still promising enough for continued hardware work?

## Scope

Primary-source SRAM macro energy/latency at 5–7 nm and Intel 18A; LPDDR5X datasheet IDD arithmetic (device read, self-refresh, power-down, idle) plus PHY bracket; wire energy; a 28 nm measured leakage anchor with a modern-node range; bank decomposition and read-width sweep; token-rate and duty-cycle sweeps with retention factors; break-even thresholds; latency with prefetch; cold start; INT4 fairness; head/KV external cases; EXP-0006/0007 reconnection; provenance tests.

## Explicitly out of scope

New RTL, attention/KV hardware, MRAM/RRAM, rack design, foundry or node selection, model changes, hardware purchases, chip TDP or J/token claims, EXP-0009.

## Definition of done

Done: see `experiments/EXP-0008-modern-sram-energy-latency.md` (Gate 4A PASS at order of magnitude; 4B INCONCLUSIVE; 4C FAIL as a latency claim; 4D INCONCLUSIVE for hardware commitment; Q-023 narrowed) and `handoffs/2026-10-07-exp-0008-sram-validation.md`.

## Required validation

`tests/test_sram_validation.py` (15 OK); all non-RTL suites pass in one process after the `datacenter_gate/sources.py` → `dc_sources.py` rename.

## Relevant files

`analysis/sram_validation/`, `results/sram_validation/`, `tests/test_sram_validation.py`.

## Relevant ADRs

ADR-0002, ADR-0004, ADR-0005, ADR-0006.

## Relevant experiments

EXP-0004 (candidate head/KV), EXP-0005 (bank organization), EXP-0006 (residency capacities, old energy ranges), EXP-0007 (small-batch regime).

## Known uncertainties

No modern-node measured leakage per MB was found (assumption A4 spans the Gate 4B verdict); retention factors are hypotheses; the E02 read width, bank-composition overhead, wire distances, LPDDR typical-device factor and PHY bracket are assumptions.

## Questions requiring founder decision

1. What duty cycle and token rate does the target edge product run at? 2. Does the EXP-0004 candidate head/KV become the edge baseline? 3. Is a predictive-PDK modeled leakage figure acceptable as T2 evidence if no published modern-node standby figure exists?
