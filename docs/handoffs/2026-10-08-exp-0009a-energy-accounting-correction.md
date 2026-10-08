# Handoff

Date: 2026-10-08
Agent: Claude (Claude Code, model Fable 5.1), implementation and research role
Current commit: recorded by a docs-only fixup commit after the EXP-0009A commit

## Objective

FT-CURRENT, EXP-0009A: correctness audit and energy-model repair of EXP-0009 (B7 wake energy, B14 shutdown-and-reload). Complete.

## What I Completed

Confirmed three defects in EXP-0009 (899976e): (1) B7 applied 1.2 µJ/MB where its own formula gives 1.2 nJ/MB, with the PLANNING and LOW levels derived from the wrong number; (2) B14 priced shutdown-and-reload with no standby cost for the weights' persistent copy, LPDDR read energy and flash-class bandwidth; (3) P2+S sessions charged one whole-model wake on top of the per-token layer wakes. Rebuilt the wake energy from rail capacitances in SI units (cell rail, bitlines, periphery, decap and grid; E = C V_active dV per switched rail; 1.6 / 2.8 / 7.7 nJ/MB) with an in-rush check against B6; priced PX-DRAM (B16) and PX-FLASH (B17) with sourced standby, reload energy, bandwidth and latency, plus a symmetric DRAM-off option for the streaming baseline; added the V_ret supply-scheme sensitivity (B18) and the resident-side DRAM-background sensitivity (B19); regenerated every artifact with the 899976e run preserved; wrote 15 tests; updated the record, the canonical docs and KI-010.

## What I Changed

- Modified: `analysis/sram_retention/retention_evidence.py` (R11 numbers from Table I; new R19 SK hynix UFS 3.1 datasheet, R20 Micron UFS page, R21 arXiv 2508.06978; B7, B14 rewritten; B16–B19 added; new constants, superseded ones kept as reproduction inputs), `wake.py` (component model, in-rush, superseded reproduction), `policy.py` (wake-model switch, V_ret supply option, `reload()`, backing-aware `session()`, break-even solvers, transition-count fix), `compare.py` (thresholds 11a/11b/11s/12a–12d, `reload_rows`, `correction_rows`, sensitivities, walkthrough PX rows), `report.py` (three new CSVs, correction section in the summary).
- New: `results/sram_retention/{wake_components,reload_economics,correction_comparison}.csv`, `results/sram_retention/superseded_899976e/` (25 original files), `tests/test_sram_retention_correction.py`, `docs/experiments/EXP-0009A-energy-accounting-correction.md`, this handoff, `docs/reviews/REV-2026-10-08-exp-0009a-correction-review.md`.
- Updated: EXP-0009 (correction banner only; tables untouched), `PROJECT_STATE.md`, `CURRENT_WORK.md`, `THESIS.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `OPEN_QUESTIONS.md` (Q-023), `KNOWN_ISSUES.md` (KI-010 closed), `README.md`, `analysis/README.md`. `BENCHMARKS.md` unchanged. No ADR.
- Untouched: RTL, EXP-0001..0008 code and results, the SPICE decks.

## Tests / Validation Run

```
python analysis\sram_retention\report.py                                  # 2.5 s, 28 files + superseded_899976e/
python -m unittest tests.test_sram_retention_correction -v                # 15 OK
python -m unittest tests.test_sram_retention -v                           # 21 OK
python -m unittest tests.test_reference tests.test_analysis tests.test_analysis_cross_model tests.test_roofline tests.test_memory_residency tests.test_datacenter_gate tests.test_sram_validation tests.test_sram_retention tests.test_sram_retention_correction tests.test_hardware_gate.TestPackingInterpretation tests.test_hardware_gate.TestWorkloadAndTrafficModels   # 182 OK in one process
```

Python 3.12.10 on the host. EXP-0005 RTL classes not re-run (no RTL changed). The SPICE sweep was not re-run (unaffected).

## Actual Results

Wake energy 1200 → 2.84 nJ/MB (planning); 0.53 mJ → 1.3 µJ per token (38% → 0.09% of the read energy). P2 at 25 tok/s: 70.4 → 57.2 mW, 4.90× → 6.03× (25 °C); 257.7 → 244.5 mW, 1.34× → 1.41× (85 °C). Crossovers ~1.4 / ~17 tok/s and ~3.5% / ~61% duty, essentially unchanged. Shutdown-vs-retention break-even 1.2 s / 0.08 s → 2.8 s / 0.18 s (DRAM-backed, 14 mJ, 33 ms) or 29 s / 1.8 s (flash-backed, ~300 mJ, 0.2 s); DRAM-backed shutdown never beats the streaming baseline at idle; flash-backed beats it hot only beyond ~5 s (P2 sessions). B18: a linearly derived retention rail doubles the retention ratio and flips the 85 °C continuous case to 0.87×. Gate 5B energy half: INCONCLUSIVE → bounded PASS (INFERRED, unmeasured). 5A, 5C, 5D unchanged in kind.

## Remaining Problems

- No measured modern-node wake charge (range spans 5×, all immaterial); B6's LOW latency level is not in-rush-consistent for 256 KB macros (documented).
- Flash reload energy is a device-side datasheet maximum at 25 °C; the host UFS controller and M-PHY are unmodeled (HIGH level only).
- The retention-rail supply scheme (B18) is unknown and verdict-relevant at 85 °C.
- The hot result still rests on the Tier-2 ASAP7 temperature coefficient (EXP-0009 limitation 1, unchanged).
- EXP-0009's R11 table printed PS-at-25%-duty totals (1.41× / 1.58× for Falcon-E 1B) that do not match its own `system_reconnect.csv` (1.37× / 1.53×, identical before and after the correction); the CSV governs. Noted in EXP-0009A, not edited in the 899976e tables.

## Important Context

- The corrections move numbers in opposite directions: B7 favours residency (removes a 1000× inflated penalty), B14 disfavours shutdown-and-reload (adds the backing store's cost). Neither flips a gate on its own; B18 would.
- `Scenario(..., wake_model="exp0009_original")` and `backing="unbacked_superseded"` reproduce the 899976e numbers for traceability (tested against the snapshot).
- Everything is analytical; nothing here is a silicon measurement.

## Questions for Founder / ChatGPT

1. How is the retention rail supplied in the target compiler's retention mode (separate rail at V_ret versus linear derivation)? This now decides the 85 °C continuous verdict.
2. The EXP-0009 questions stand: product junction temperature and duty cycle; NDA access to compiler sleep-mode data, now including the rail supply scheme and the sleep-to-active charge.

## Recommended Immediate Next Consideration

No new agent experiment. Take the B18 question and the thermal/duty envelope to the founder; with answers, re-run `analysis/sram_retention/report.py` with the Tier-1 inputs in place of B1, B2, B4, B18.
