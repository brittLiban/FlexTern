# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** EXP-0009 — SRAM retention, leakage and wake feasibility for layer-banked model-resident inference (continues Q-023).

**Status:** Complete (awaiting founder/ChatGPT review). No next objective has been set.

**Owner / agent:** Claude (implementation and research)

**Started:** 2026-10-07
**Completed:** 2026-10-07

## Why we are doing this

EXP-0008 left the edge/resident form hanging on leakage: 440–811 MB of always-on SRAM lost to LPDDR5X below ~23 tok/s or ~85% duty at its planning figure, modern-node leakage was unmeasured, and the retention factors were hypotheses. A transformer runs its layers sequentially, so most banks might sit in a low-leakage retentive state while one or two layers work.

## Question we are trying to answer

Gate 5A: is modern SRAM retention leakage materially below active standby? Gate 5B: can retained domains wake quickly and cheaply enough during sequential execution? Gate 5C: with leakage, retention, wake energy and head/KV counted, does the resident 1B–3B edge form keep a material memory-power advantage across realistic duty cycles? Gate 5D: is large SRAM residency still the preferred edge memory architecture?

## Scope

Tiered sources on SRAM power modes (3 nm / 7 nm T1 standby, 22 nm FD-SOI retention and wake, old-node temperature data, 7 nm PDK ratios, compiler mode definitions); a Tier-2 ASAP7 cell SPICE corroboration; per-layer schedules from the profiles; leakage by mode versus voltage and temperature; wake latency and energy by domain; P0/P1/P2/P4/P8/PS/PT policies and a shutdown-and-reload comparison; token-rate, duty-cycle, session, temperature, voltage-domain and wake sweeps; INT4 fairness; head/KV; eleven break-even thresholds; EXP-0008 reconciliation; tests.

## Explicitly out of scope

New RTL or a retention controller, physical power gating, an SRAM compiler, foundry or node selection, low-bit attention, training, the 70B rack, freezing the EXP-0004 candidate, battery life, chip TDP, BOM, EXP-0010.

## Definition of done

Done: see `experiments/EXP-0009-sram-retention-leakage-wake.md` (Gate 5A INCONCLUSIVE; 5B PASS on latency, INCONCLUSIVE on energy; 5C INCONCLUSIVE, split by temperature; 5D INCONCLUSIVE; Q-023 narrowed) and `handoffs/2026-10-07-exp-0009-sram-retention.md`.

## Required validation

`tests/test_sram_retention.py` (21 OK); all non-RTL suites pass in one process.

## Relevant files

`analysis/sram_retention/` (incl. `spice/`), `results/sram_retention/`, `tests/test_sram_retention.py`.

## Relevant ADRs

ADR-0002, ADR-0004, ADR-0005, ADR-0006.

## Relevant experiments

EXP-0004 (candidate head/KV), EXP-0005 (bank organization), EXP-0006 (capacities), EXP-0007 (datacenter, not reopened), EXP-0008 (dynamic energy, old leakage planning).

## Known uncertainties

The hot result rests on a Tier-2 temperature coefficient (ASAP7 ×4.9 at 85 °C) applied to Tier-1 25 °C figures; the deep-retention ratios (0.076 cold / 0.23 hot) are modeled; wake energy of a multi-MB domain is an assumption; the 7 nm standby figure was quoted from a summary page; hold margin has no variation analysis.

## Questions requiring founder decision

1. What junction temperature and duty cycle does the target edge product run at? 2. Can FlexTern obtain a foundry/compiler SRAM datasheet under NDA with sleep-mode leakage versus temperature and wake cycles? 3. Should the edge form plan on shutdown-and-reload from a nonvolatile store for idle gaps rather than retention?
