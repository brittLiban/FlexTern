# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** EXP-0009A — Correctness audit and energy-model repair of EXP-0009 (SRAM wake energy B7, shutdown-and-reload economics B14).

**Status:** Complete (awaiting founder/ChatGPT review). No next objective has been set.

**Owner / agent:** Claude (implementation and research)

**Started:** 2026-10-08
**Completed:** 2026-10-08

## Why we are doing this

An external review of the repository found that EXP-0009's wake-energy assumption stated a formula giving 1.2 nJ/MB but reported 1.2 µJ/MB, and that its shutdown-and-reload comparison never said where the weights persist while the SRAM is off. Both feed Gate 5B, 5C and 5D. Before another architectural decision, the project needed to know whether these defects change the EXP-0009 conclusions.

## Question we are trying to answer

Is the B7 discrepancy a unit error, an omitted physical term, or a wrong model, and what is the defensible wake energy of a multi-MB retention domain? Where do the weights live during shutdown, and what do that store's standby, reload energy, bandwidth and latency do to the retention-versus-reload comparison? Do Gates 5A–5D change?

## Scope

Audit of `analysis/sram_retention/` and `results/sram_retention/`; a component wake-energy model in SI units with an in-rush consistency check; DRAM-backed and flash-backed reload accounting on consistent boundaries, with a symmetric option for the streaming baseline; a V_ret supply-implementation sensitivity; regeneration of every affected artifact with the original run preserved; original-versus-corrected comparison; 15 new tests; all non-RTL suites; documentation.

## Explicitly out of scope

EXP-0010, any change to the Youssouf Architecture, a retention controller, FPGA/ASIC/foundry/compiler selection, a nonvolatile subsystem design, compiler/runtime work, reopening EXP-0007, new backlog.

## Definition of done

Done: see `experiments/EXP-0009A-energy-accounting-correction.md` (B7 explained and corrected: a 1000× unit error plus omitted bitline, periphery and decap terms, now 2.8 nJ/MB planning in a 1.6–7.7 range; B14 corrected: DRAM- and flash-backed variants priced; Gate 5B energy half INCONCLUSIVE → bounded PASS; 5A, 5C, 5D unchanged in kind with the hot continuous case now hinging on the retention-rail supply scheme, B18) and `handoffs/2026-10-08-exp-0009a-energy-accounting-correction.md`.

## Required validation

`tests/test_sram_retention_correction.py` (15 OK); `tests/test_sram_retention.py` (21 OK); all non-RTL suites pass in one process (182 OK).

## Relevant files

`analysis/sram_retention/{retention_evidence,wake,policy,compare,report}.py`, `results/sram_retention/` (corrected) and `results/sram_retention/superseded_899976e/` (original), `tests/test_sram_retention_correction.py`.

## Relevant ADRs

ADR-0002, ADR-0004, ADR-0005, ADR-0006. No new ADR.

## Relevant experiments

EXP-0008 (cold start, energies, LPDDR modes), EXP-0009 (the corrected record).

## Known uncertainties

No measured modern-node macro wake charge exists (range 1.6–7.7 nJ/MB, all immaterial); flash reload energy is device-side datasheet maximum (host controller/PHY unmodeled); the retention-rail supply scheme (separate rail versus linear derivation) is unknown and decides the 85 °C continuous verdict; the hot temperature coefficient remains Tier-2.

## Questions requiring founder decision

1. Does the target compiler's retention mode deliver the array rail at V_ret from a separate supply (B4 planning) or derive it linearly from V_dd (B18: half the saving, 85 °C continuous 0.87×)? 2. The EXP-0009 questions stand: product junction temperature and duty cycle; NDA access to compiler sleep-mode data (now including the rail supply scheme and the wake charge).
