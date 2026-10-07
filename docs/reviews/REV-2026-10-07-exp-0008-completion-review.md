# Review — EXP-0008 modern SRAM energy and latency validation (completion)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent with a review brief that differed from the implementation prompt: attack the twenty-four points below, recompute, require corrections before commit)
Date: 2026-10-07
Scope: EXP-0008 and its artifacts (`analysis/sram_validation/`, `tests/test_sram_validation.py`, `results/sram_validation/`, the EXP-0008 record, the handoff, and the OPEN_QUESTIONS / PROJECT_STATE / THESIS wording) as they stood immediately before the EXP-0008 commit (b48237f). Everything below was applied before the commit; the "as submitted" numbers are kept so the correction is traceable.

## Overall Assessment

The arithmetic reproduces (the 3.59 pJ/bit IDD4R derivation, the 440.4 MB provisioning, the capacities, the duty-cycle rows, the break-even equations). The structural defect in the submitted version was that the model was tilted toward residency in four places at once, which the "try to kill it" standard does not allow: the LPDDR side was built from datasheet maxima while the SRAM side was built from a best-case low-voltage measurement; a 256 Kb macro figure stood in for a 6 MB bank with no composition overhead; the leakage planning figure was a 28 nm planar part at 0.5 V, which is below what a modern HD FinFET macro leaks at nominal voltage; and the LPDDR channel was charged full idle-standby power for the whole active period although a controller enters power-down between bursts. None of these changes the direction of Gate 4A. Together they move the planning dynamic ratio from 14.7× to 9.1×, the continuous break-even from 8.6 to 23 tok/s, and the duty-cycle break-even at 25 tok/s from 33% to 85% without a retention mode. Gate 4B was always going to be INCONCLUSIVE; after the corrections it is INCONCLUSIVE with the verdict-spanning uncertainty stated as one number (modern leakage per MB).

## Corrections required and applied

1. **LPDDR baseline symmetry (A6).** E06's IDD4R2H is a datasheet maximum under a stringent pattern (datasheet note 6), so 3.59 pJ/bit is a ceiling. Levels rebuilt the same way on both sides: LOW typical device 2.5 + PHY 1.0 = 3.5; PLANNING typical device 2.5 + PHY 1.5 = 4.0 (was 5.0); HIGH maximum device 3.6 + PHY 2.5 = 6.1 (was 7.0). The typical-device factor 0.7× is an ASSUMPTION and is labeled.
2. **Bank-composition overhead (new A12).** E02 is a 256 Kb macro and E01 is a per-bit figure; the resident banks are 3–25 MB built from 12–100 macros. A bank-level decode / macro-select / output-mux term of 0.05 / 0.10 / 0.15 pJ/bit was added to the resident total (ASSUMPTION, labeled as the price of composing multi-MB banks from sub-MB macros).
3. **Leakage planning figure (A4).** Moved from 0.30 mW/MB (E05, 28 nm at 0.5 V) to 0.60 mW/MB (E11's ~100 pA/cell FinFET class at ~0.75 V: 100 pA × 8e6 cells/MB × 0.75 V). E05 is kept as a reduced-voltage planar context point only. LOW 0.05 now explicitly means reduced retention-capable voltage; HIGH 1.0 is the hot corner.
4. **Retention factors labeled HYPOTHESIZED.** The 0.2× and 0.1× factors have no modern-node evidence (E05's 1000× is 28 nm). The CSV labels, summary headers and the record now say so, and the Gate 4B statement names the retention mode as unverified.
5. **LPDDR active-period idle (new A13).** The channel-busy fraction W·R / (17.07 GB/s × 70%) is charged IDD2N and the remainder IDD2P in both the token-rate and duty-cycle sweeps (was: IDD2N for the whole active period, which inflated LPDDR's active power by ~25 mW on ~400 mW for 1B at 25 tok/s).
6. **Gate 4C stated in both halves.** Access latency is irrelevant once a 1.5 KB prefetch is in flight (FAIL on the latency claim); bandwidth per x16 channel is the real limiter above ~30 tok/s (1B) / ~16 tok/s (3B) / ~23 tok/s (BitNet), and that is a channel-count question, not a latency one.
7. **Precision of the 4A claim.** Two T1 points (E01 at 0.4 V, E02 per access with an inferred width) and one T2 projection support an order-of-magnitude statement ("roughly 10×, range 5.6–18×"), not a three-figure ratio.
8. **Q-023 disposition:** narrow, do not close. Dynamic read energy at modern nodes is now sourced (T1); leakage per MB and retention behaviour at modern nodes are not.

## What Looks Strong

- Both sides of the dynamic comparison now rest on T1 primary figures (TSMC N5 JSSC 2020 per-bit read energy, a 7 nm VLSI 2020 macro per-access energy, the Micron LPDDR5X datasheet IDD table) with every derivation reproduced by a test.
- Leakage is modeled at all, which EXP-0006 did not do, and it is reported separately from dynamic energy with its own boundary.
- The break-even thresholds are stated in the units a founder can act on: tokens per second, duty cycle, mW/MB, pJ/bit.
- The head/KV external case is carried through, so the edge advantage is not quoted on weights alone.
- EXP-0006's 1.6 pJ/bit planning figure and 5–15 pJ/bit LPDDR range are reproduced and then replaced, with the old numbers kept in `exp0006_reconnect.csv`.

## Concerns (verdicts on the twenty-four points)

1. **Source quality.** E01/E02/E06 are T1; E03 contributes frequency and density only; E04 is a 2011 projection (T2) used for wire energy and as a cross-check; E05 is T1 but 28 nm; E07/E08/E09/E11/E12 are T3 secondary. Adequate for dynamic energy; inadequate for leakage. Stated in the record.
2. **Modern-node.** The SRAM dynamic figures are 5 nm and 7 nm (T1); the leakage figures are not modern-node measurements. Correct labels after fix 3.
3. **Measured vs modeled.** E01, E02, E05, E06 are silicon measurements or datasheet specifications; everything composed from them (voltage scaling, width normalization, bank overhead, wire distance) is modeled and labeled A1–A13.
4. **Macro-size mismatch.** The decisive defect as submitted; fixed by A12. Residual risk: 0.10 pJ/bit is an assumption, not a measurement; the local-wire threshold (3.8 pJ/bit) shows the verdict tolerates a much larger overhead.
5. **Read-width normalization.** E02's width is not in the abstract; 64–256 bits assumed (A3), giving 23–91 fJ/bit, and the planning figure takes the pessimistic end. The width sweep shows the per-bit total is flat from 128 to 2048 bits because the per-access fixed cost is amortized; width is not where the result lives.
6. **Macro vs cache energy.** E10 (Horowitz) cache-access figures are now T4 historical and used only for the EXP-0006 reconnection; the model uses macro figures plus explicit bank and wire terms instead of a cache figure.
7. **Hidden peripheral energy.** Macro periphery is inside E01/E02 by their stated boundary; bank periphery is A12; delivery wires and registers are A5; decode is amortized per access (A9). Nothing is claimed to be free except what the sources include.
8. **Logical bank vs subarray.** The 6.19 MB logical bank (EXP-0005/0006) decomposes into 4–25 macros, one switching per access (A9); the others leak only. Tested for capacity conservation.
9. **Leakage assumptions.** Fix 3 and fix 4. The modern leakage per MB spans 0.05–1.0 mW/MB and that range spans the Gate 4B verdict; this is the single unmeasured number.
10. **Temperature.** LPDDR self-refresh is given at 25 °C and 85 °C from E06; SRAM leakage temperature dependence is folded into the LOW/HIGH range without a measured coefficient. Disclosed.
11. **Local wire.** 0.115–0.15 pJ/bit/mm (E04, T2) at 0.5 / 2 / 8 mm; planning uses the 2 mm tile distance. Threshold: the advantage survives up to ~3.8 pJ/bit of wire (~25 mm), so the floorplan is not where the result lives.
12. **LPDDR boundary fairness.** Fix 1. The device figure is T1; the PHY bracket is T3; the typical-device factor is an assumption. Both sides now use typical for LOW/PLANNING and maximum for HIGH.
13. **LPDDR idle/background.** Self-refresh (0.77–9.8 mW), power-down (5.4 mW) and idle standby (32.1 mW) per x16 channel from E06; fix 5 charges them by busy fraction. SoC PHY idle power is not modeled (A7), which favours streaming slightly.
14. **Duty-cycle math.** Average power = d × active + (1 − d) × idle on both sides, with the LPDDR idle at power-down; the resident idle at leakage × retention factor. Tested against hand arithmetic. The retention rows are HYPOTHESIZED.
15. **Latency pipeline.** Bytes in flight to hide 90 ns at 17.07 GB/s = 1,536 B, under the 4 KB prefetch assumption; the engine consumes sequentially. No scheduler or bank conflict modeled.
16. **First-word vs steady state.** Separated: first word 90 ns vs 0.82 ns; steady state is a bandwidth statement (one-channel floor 33 / 61 / 44 ms per token). Fix 6.
17. **INT4 fairness.** Same organization and term rate for 4.5-bit weights; capacity 2.25× and leakage 2.25×; INT4 3B provisioning (1.82 GB) is beyond one reticle at EXP-0006 densities, which the record states rather than scoring.
18. **EXP-0006 conclusion.** Its dynamic ratio (3.1–9.4×, planning 1.6 pJ/bit) is superseded by 5.6–18.3× (planning 9.1×); its energy-class framing stands; its silence on leakage is the material change.
19. **Gate 4A** PASS at order-of-magnitude on T1 evidence; the three-figure ratio is not claimed.
20. **Gate 4B** INCONCLUSIVE; the verdict flips on the unmeasured leakage number and on an unverified retention mode.
21. **Gate 4C** FAIL as a latency claim; bandwidth per channel is the real limiter. Both halves stated.
22. **Gate 4D** INCONCLUSIVE for hardware commitment, sufficient for continued analysis; contingent on the EXP-0004 candidate head/KV being adopted (a founder decision).
23. **Edge-product claims.** None made: no TDP, no J/token, no battery life. Memory-subsystem energy and power only, with the boundary on every row.
24. **Next objective.** Resolving the leakage term (A4) with modern-node evidence is the one measurement that changes the 4B and 4D verdicts; the dynamic result does not depend on it. Defensible.

## Assumptions Being Made

- SRAM macro energy scales as V²; the E02 width is 64–256 bits; the bank overhead is 0.05–0.15 pJ/bit; wire distances 0.5–8 mm; LPDDR typical device = 0.7 × maximum; PHY 1.0–2.5 pJ/bit; SoC PHY idle = 0; retention factors 0.2/0.1; prefetch 4 KB; cold-start bandwidth 1–4 GB/s.

## Potential Failure Modes

- Reading the duty-cycle rows with retention 0.1 as evidence; they are hypotheses.
- Quoting the dynamic ratio as a chip-level figure; it is memory-subsystem only.
- Reading Gate 4A as a product verdict; Gate 4B is where the product lives.

## Recommendation

Commit with the corrections above. Narrow Q-023 to the leakage question. One next objective: obtain a modern-node measured leakage and retention figure for HD SRAM macros (published standby power per Mb at 5–7 nm, or a predictive-PDK SPICE characterization as a modeled corroboration), because that number spans the only verdict that matters for the edge form.
