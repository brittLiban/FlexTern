# Review — EXP-0009 SRAM retention, leakage and wake feasibility (completion)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent with a review brief that differed from the implementation prompt: attack the twenty-six points below, recompute, require corrections before commit)
Date: 2026-10-07
Scope: EXP-0009 and its artifacts (`analysis/sram_retention/`, `tests/test_sram_retention.py`, `results/sram_retention/`, the EXP-0009 record, the handoff, and the canonical-doc wording) as they stood immediately before the EXP-0009 commit. Everything below was applied before the commit; the "as submitted" state is described so the correction is traceable.

## Overall Assessment

The arithmetic reproduces (R01 → 0.21 mW/MB, R02 → 0.52, the ASAP7 ×4.88 temperature factor, the 0.076 / 0.231 retention ratios, the 22 µJ layer wake and 0.528 mJ/token, the P2 totals at both temperatures); the 21 tests are real. Four things had to go into the model or the record before the gate verdicts could be written, two of which change what Gate 5C says: the obvious idle alternative (shut down and reload) was not priced, so the record would have said "residency loses at low duty" where a trivially available policy wins; the hot result rests on a Tier-2 temperature coefficient and had to be stated as the first limitation rather than the fifth; the LPDDR baseline's datasheet maxima are hot-corner values charged at every temperature; and the two headline negatives (a retained 440 MB array leaks more than a self-refreshing DRAM at 85 °C; the only retention mode with a modern T1 figure saves 10% and moves nothing) had to be made explicit with their numbers.

## Corrections required and applied

1. **Shutdown-and-reload comparison row (new B14, policy PX).** Reload is 13.2 mJ for Falcon-E 1B (EXP-0008 cold start) and 0.1–0.4 s of first-token latency; break-even idle gap against retention is 1.2 s at 25 °C and 0.08 s at 85 °C (threshold 11). Added to `session_sweep.csv`, the walkthrough and `break_even.csv`, labeled as a data-loss mode and kept out of the retention tables. It changes the 5C regime statement for long idle gaps at both temperatures.
2. **Limitation ordering and the pessimistic cold ratio.** The 85 °C verdict is 0.33 mW/MB (T1, 25 °C) × 4.88 (ASAP7, T2); there is no T1 modern coefficient. The 25 °C retention ratio (0.076) depends on the predictive cell's gate-versus-subthreshold mix; a sensitivity row applies the hot ratio (0.23) cold (P2 4.9× → 3.8×). Both are now limitation 1 and in `mode_sensitivity.csv`.
3. **LPDDR temperature asymmetry disclosed (new B15).** IDD2N / IDD2P are hot-corner datasheet maxima charged at every temperature; effect ≤ 27 mW of 345 mW at 25 °C, in residency's favour. Planning left as EXP-0008 defined it; a sensitivity row halves them (4.90× → 4.70×).
4. **Explicit negatives.** (a) At 85 °C a retained 440 MB array leaks 164 mW against 9.8 mW of DRAM self-refresh; no retention policy rescues low-duty hot operation; only P2 above ~17 tok/s wins, at 1.34× (1.2–1.3× with head/KV). (b) Light sleep (10%, T1) and periphery-off deep sleep (T3) leave EXP-0008's verdict unchanged; every improvement comes from the V_ret array-voltage mode (T2). Both are in R3, R10 and the Gate 5A/5C statements.
5. Threshold 1 printed a retention ratio of 2.13 at 25 °C; now "any ratio ≤ 1 wins".
6. `gate_5c_cell_verdict` in `primary_comparison.csv` renamed `heuristic_flag_not_a_verdict`; verdicts live in the record.
7. Paced P2 stated as planning (no engine clock claimed) and burst as the sensitivity that assumes the 8 TB/s TARGET engine (258 → 212 mW at 85 °C).
8. EXP-0008's flat 0.60 mW/MB located at ~55 °C on the new curve; its 23 tok/s and 85% duty crossovers restated as warm-case P0 results now bracketed by 11.5 / 60 tok/s and 47% / never.
9. INT4 stated plainly: 2.25× absolute power, ratios within 4%, verdict unchanged.
10. R02's 16.9 µW figure keeps its "quoted from a summary page, not confirmed" tag in the record because it sets the HIGH anchor and half the planning geometric mean.

## What Looks Strong

- Power modes are kept apart and a data-loss mode is never counted as retention (test 17).
- Two Tier-1 modern macro anchors for active standby, with the 3 nm figure read from the paper's own abstract.
- The Tier-2 SPICE is used for ratios only, is reproducible from committed decks, and is corroborated on the retention-mode temperature slope by two measured old-node sources.
- The layer schedule is programmatic from the profiles; capacity conservation, wake counting, transition frequency and the break-even equations are tested.
- The thresholds are stated in units the founder can act on, and the one that decides the gate (temperature) is explicit.

## Concerns (verdicts on the twenty-six points)

1. **Source quality.** Two T1 modern active-standby figures; T1 retention only at 22 nm FD-SOI and older planar nodes; deep retention at modern nodes is T2/T3. Stated.
2. **Modern-node relevance.** The modes documented at 3–7 nm (light sleep, periphery-off) do not change the verdict; the one that does is modeled. Stated as the central caveat.
3. **Terminology.** Active / idle-standby / light sleep / deep sleep / retention / shutdown separated per source (`retention_evidence.csv`, `modes_reported`).
4. **Data actually retained.** Shutdown rows carry "data lost"; PX prices the reload; test 17 guards it.
5. **Measured vs modeled.** Every table row carries its label; the summary header says which tier each input is.
6. **Voltage scaling.** ASAP7 power ratio at 0.25 / 0.35 / 0.45 V with nominal hold SNM 128 / 178 / 228 mV; no variation, no verified modern DRV (R18 excluded). Adequate for a range, not for a design point.
7. **Temperature scaling.** The decisive T2 input; corroborated only for retention modes by old-node data; limitation 1.
8. **Macro-size mismatch.** Carried from EXP-0008 (bank overhead A12); retention adds the per-layer rail assumption B8.
9. **Voltage-domain granularity.** Macro-level switches are supported by every source; layer domains are an assumption; the sweep shows four-layer domains already lose hot.
10. **Layer scheduling.** Sequential layer order from the profiles; head external; no attention-time model, which only affects wake slack where the margin is 1000×.
11. **Wake latency.** Hidden under every swept assumption at ≤ 250 tok/s paced; the high case stalls burst execution only. PASS is defensible.
12. **Wake energy.** Assumption-only; 38% of read energy at planning; the threshold table shows 9× headroom at 25 tok/s and parity at 250 tok/s. INCONCLUSIVE is right.
13. **Transition frequency.** Every-layer-every-token is the planning worst case; PT and PS are cheaper; tests 9 and 11.
14. **Wake behind compute.** Paced timing gives milliseconds of slack; burst timing gives 2 µs, which the planning wake meets and the high case does not. Both shown.
15. **Session duration.** Irrelevant to the ordering once one wake or reload per session is counted (R6).
16. **Duty-cycle math.** Reproduces EXP-0008 under its convention (test 12); the planning convention now charges the DRAM its cheapest idle mode.
17. **Timeout policy.** Break-even time 3.9 ms (25 °C) / 1.0 ms (85 °C); tested (14).
18. **LPDDR fairness.** Unchanged from EXP-0008 by design; the temperature asymmetry is disclosed (B15) with a sensitivity row.
19. **Head/KV accounting.** Identical on both sides; the candidate remains a candidate (R11).
20. **INT4 fairness.** Identical algorithm (test 15); ratios within 4%.
21. **Gate 5A** INCONCLUSIVE follows: modeled yes, measured no.
22. **Gate 5B** split verdict follows from 11 and 12.
23. **Gate 5C** INCONCLUSIVE with named regimes follows; after correction 1 the long-idle regime is served by reload.
24. **Gate 5D** INCONCLUSIVE follows: the uncertainty moved, it was not resolved.
25. **Moved vs resolved.** Moved: from "leakage per MB" to "operating temperature plus foundry retention mode plus wake energy". Resolved: dynamic energy (EXP-0008) and wake latency. The record says so.
26. **Next objective.** The verdict-spanning unknown (measured modern retention leakage vs temperature) is not in the public literature; it needs vendor compiler data under NDA and a product thermal/duty envelope, both founder actions. Recommending that rather than inventing an agent-runnable substitute is correct.

## Assumptions Being Made

B1–B15 as listed in the record; the two that span the verdict are B2 (temperature coefficient) and B4 (V_ret retention ratio).

## Potential Failure Modes

- Reading the 25 °C rows as the product case; a generating edge SoC may sit anywhere between 40 and 85 °C.
- Reading RETENTION_VRET as an available foundry mode; it is modeled.
- Reading the PX rows as retention; they lose the weights.
- Quoting any mW figure as a chip figure; all are memory-subsystem only.

## Recommendation

Commit with the corrections above; the Tier-2 SPICE must be described as corroboration in the commit message; nothing under `build/` may be staged. Narrow Q-023 to the measured modern retention leakage versus temperature, the multi-MB wake energy and the product thermal/duty envelope. One next objective: obtain the vendor/compiler retention data and the product envelope (founder action) and re-run this package with Tier-1 inputs.
