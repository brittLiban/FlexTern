# Review — EXP-0009A energy-accounting correction of EXP-0009 (completion)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent with a review brief that differed from the implementation prompt: reproduce the component arithmetic and the datasheet derivations, check transition counting and the PX-DRAM/baseline boundary in both views, and test whether any revised gate verdict overstates certainty)
Date: 2026-10-08
Scope: EXP-0009A as committed at 8ab0a24 (`analysis/sram_retention/`, `tests/test_sram_retention_correction.py`, `results/sram_retention/` and `superseded_899976e/`, the EXP-0009A record, the EXP-0009 banner, the canonical-doc edits, the handoff). The four corrections below were applied in the docs fixup commit that follows 8ab0a24.

## Overall Assessment

The correction is sound and complete for the task's definition of done. Every figure in the record reproduces from the committed code and the three defects are what the record says they are: a unit slip (1.2 nJ/MB applied as 1.2 µJ/MB, with the PLANNING and LOW levels derived from the wrong number), an unpriced backing store with mixed energy and bandwidth sources, and a session-level double count of the whole-model wake under a rolling window. The audit did not stop at fixing the exponent: it rebuilt the wake energy from four rail components in SI units, checked the implied in-rush against B6, found that B6's LOW latency level is not in-rush-consistent for 256 KB macros, and anchored the class with the MAPG Table I numbers it actually read. The reload comparison is now stated on explicit boundaries (weights-only versus system view) with the streaming baseline given the same DRAM-off option, which is the only way the long-gap rows mean anything.

## What was verified

- B7 arithmetic at every level: cell rail 4 nF × 0.75 V × 0.40 V = 1.20 nJ, bitline 0.72, periphery 0.45, decap 0.47, total 2.84 nJ/MB; DEEP_SLEEP exit 0.25 / 0.54 / 4.05 and LIGHT_SLEEP 0 / 0 / 1.80 follow from the rail sets; in-rush 4.9 mA per macro at planning, 54 mA at LOW.
- Headroom and thresholds: 11,180 / 2.84 = 3,900× at 25 tok/s, 1,118 / 2.84 = 390× at 250 tok/s; threshold 7 at 9.5 µs equals 1.27 µJ / 134 mW; the 5 and 100 tok/s cells in R2 match `token_rate_sweep.csv` in both the corrected and the superseded runs.
- R5 matches `reload_economics.csv` cell for cell (13.7 / 0.91 s at LOW, 53.4 / 3.35 s at HIGH).
- UFS derivations against the extracted datasheet text: hibernate 60 µA × 2.5 V + 360 µA × 1.2 V = 0.58 mW, sleep 0.43 mW, 2-lane active 1.485 W at 2,000 MB/s = 92.8 pJ/bit; R21's 102.4 pJ/b is within 10%.
- Transition counting: PS(P0) charges one whole-model wake, PS(window) zero, PX one reload; nothing overlaps.
- "A DRAM-backed shutdown never beats the streaming baseline at idle" holds in both views: same die, same retaining mode, plus the SRAM residual.
- The R11 transcription discrepancy in EXP-0009 (1.41× / 1.58× in the record against 1.37× / 1.53× in its own CSV, unchanged by this correction) is real and correctly left unedited with the CSV declared governing.
- The test map covers every item in the task's §5 list; 182 non-RTL tests pass in one process; test 13 pins the unaffected artifacts byte-for-byte to the snapshot.

## Corrections required (applied in the fixup commit)

1. `correction_comparison.csv` printed the original threshold 11 as 1.174 s because the comparison recomputed it with the PS session wake subtracted, while EXP-0009 published 1.223 s from E_reload / (P_ret − P_residual). The original-model row now uses the 899976e formula and test 12 pins it to the snapshot within 2 ms.
2. The EXP-0009 banner said every wake figure was "1000× too high" while EXP-0009A R1 says ~420× against the component total. The banner now says three orders of magnitude, 1000× against the stated formula and ~400× against the corrected total.
3. B19 as built is asymmetric the other way: the streaming side's background is evaluated on weight bytes only while the resident side's new column gets background on head+KV bytes. Conservative against residency and not verdict-relevant; a sentence in B19 now says so.
4. The record's environment line lacked the pypdf version used to read the three PDFs; added.

## Concerns that remain (no action required now)

- The B7 range (1.6–7.7 nJ/MB) is an estimate cross-checked against one 32 nm core-level model; it is immaterial at every point, which is why Gate 5B's energy half can be stated as a bound, but it remains unmeasured and the record says so.
- Flash reload energy is a device-side datasheet maximum; the host UFS controller and M-PHY are covered only by the HIGH level. A real product figure could sit above HIGH.
- B18 is the right single open question: it is the only remaining assumption that flips a hot verdict on its own (1.41× → 0.87× at 85 °C continuous). The implementation correctly surfaced it as a founder question rather than choosing.
- The duty-cycle sweep's DRAM idle convention (cheaper of power-down and self-refresh) is inherited from EXP-0009 and listed as limitation 6; it favours the baseline hot by at most 4.4 mW × (1 − d).

## Verdict on the gates

5A INCONCLUSIVE, unchanged: no correction touches leakage. 5B: PASS on latency unchanged; the energy half moves from INCONCLUSIVE to a bound (INFERRED, unmeasured), defensible because even the HIGH end of the range is 0.25% of the read energy. 5C INCONCLUSIVE split by temperature, unchanged in kind; the long-idle statement is correctly replaced by the backing-store-conditional one and the hot continuous case is now explicitly conditional on B18. 5D INCONCLUSIVE, unchanged; wake energy leaves the open list.

## Recommendation

Commit as is, record the hash, push, no new objective. Take B18 and the thermal/duty envelope to the founder.
