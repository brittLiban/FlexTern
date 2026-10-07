# Handoff

Date: 2026-10-07
Agent: Claude (Claude Code, model Fable 5.1), implementation and research role
Current commit: b48237f (EXP-0008), followed by a docs-only fixup commit recording this hash

## Objective

FT-CURRENT, EXP-0008: modern SRAM energy and latency validation for the edge / model-resident form (Gate 4). Complete.

## What I Completed

Twelve tiered sources (three T1 SRAM macro papers at 5 nm / 7 nm / 18A, the Micron LPDDR5X datasheet IDD tables, Keckler 2011 wire and DRAM figures, a 28 nm leakage anchor, secondary PHY/leakage/timing references), thirteen tagged assumptions, and a reproducible analytical package (`analysis/sram_validation/`) that derives per-bit energies by boundary, decomposes banks into macros, sweeps read width and wire distance, models leakage with token-rate and duty-cycle sweeps and retention factors, solves break-even thresholds, separates first-word latency from bandwidth, models cold start, scores the three canonical models at 2-bit and 4.5-bit with BF16 and candidate head/KV external, and reconnects EXP-0004/0005/0006/0007; 15 tests; the EXP-0008 record with Gate 4A/4B/4C/4D verdicts; docs; completion review with eight corrections applied before commit.

## What I Changed

- New: `analysis/sram_validation/{evidence,model,report}.py`, `tests/test_sram_validation.py`, `results/sram_validation/` (21 files), EXP-0008, this handoff, the completion review.
- Renamed: `analysis/datacenter_gate/sources.py` → `dc_sources.py` (module-name collision with `memory_residency/sources.py` when suites share a process); imports updated in `systems.py`, `workloads.py`, `report.py`, `tests/test_datacenter_gate.py`; `results/datacenter_gate/` regenerated with identical content apart from the commit stamp.
- Updated: `PROJECT_STATE.md`, `CURRENT_WORK.md` (complete), `THESIS.md`, `ARCHITECTURE.md` (§4 note and change log), `ROADMAP.md`, `OPEN_QUESTIONS.md` (Q-005, Q-015, Q-019, Q-023 narrowed), `README.md`, `analysis/README.md`. `BENCHMARKS.md` and `KNOWN_ISSUES.md` unchanged. No ADR.
- Untouched: RTL, all earlier results except the datacenter commit stamp.

## Tests / Validation Run

```
python -m unittest tests.test_sram_validation -v                          # 15 OK
python -m unittest tests.test_reference tests.test_analysis tests.test_analysis_cross_model tests.test_roofline tests.test_memory_residency tests.test_datacenter_gate tests.test_sram_validation tests.test_hardware_gate.TestPackingInterpretation tests.test_hardware_gate.TestWorkloadAndTrafficModels   # OK in one process
python analysis\sram_validation\report.py                                 # seconds
python analysis\datacenter_gate\report.py                                 # seconds
```

EXP-0005 RTL test classes were not re-run (no RTL or testbench changed).

## Actual Results

See EXP-0008 R1–R11 and `results/sram_validation/summary.md`. Headline: resident SRAM reads cost 0.33–0.62 pJ/bit (macro + bank overhead + wires) against 3.5–6.1 pJ/bit for LPDDR5X through the SoC path, roughly 10× (5.6–18×); EXP-0006's 1.6 pJ/bit planning figure was ~3.6× too high and its 5–15 pJ/bit LPDDR range too high as well. Leakage, never counted before, is the verdict: 264–486 mW standing power at the planning 0.6 mW/MB for 440–811 MB, so residency wins total memory power only above ~23 tok/s continuous or ~85% duty at 25 tok/s (32–40% with a HYPOTHESIZED 0.1× retention mode). Access latency is hidden by a 1.5 KB prefetch; per-channel bandwidth (~30 / 16 / 23 tok/s per x16 channel) is the real streaming limit. With head and KV external the total advantage is 1.4–3.3× at BF16 and 2.2–5.8× with the EXP-0004 candidate. Gate 4A PASS (order of magnitude), 4B INCONCLUSIVE (spans the unmeasured leakage range), 4C FAIL as a latency claim, 4D INCONCLUSIVE for hardware commitment. Q-023 narrowed to leakage/retention, not closed.

## Remaining Problems

- No modern-node measured SRAM leakage per MB or retention-mode factor was found; A4 and the retention factors are the load-bearing assumptions.
- E02's read width, the bank-composition overhead, the wire distances, the LPDDR typical-device factor and the PHY bracket are assumptions with ranges.
- Memory-subsystem energy only; compute and the non-ternary engines are outside the boundary.

## Important Context

- The resident pJ/bit is mostly delivery (wire + registers), not the macro read; the floorplan assumption matters more than the macro paper.
- The planning leakage (0.60 mW/MB) sits just under the 25 tok/s threshold (0.65) and far above the 5 tok/s threshold (0.14); the gate flips inside the 0.05–1.0 mW/MB range.
- Both systems were built symmetrically after review: typical figures for LOW/PLANNING and maxima for HIGH on both sides; the pre-correction numbers (14.7×, 8.6 tok/s, 33% duty) are in the review for traceability.
- The datacenter-package rename touches EXP-0007 code but not its results.

## Questions for Founder / ChatGPT

1. What duty cycle and token rate does the target edge product run at? The break-even tables turn into a verdict once that is fixed.
2. Does the EXP-0004 candidate head/KV become the baseline for the edge form?
3. Is a modeled (predictive-PDK SPICE) leakage figure acceptable as T2 evidence if no published modern-node macro standby figure can be found?

## Recommended Immediate Next Consideration

Resolve the leakage term (assumption A4) with modern-node evidence: a targeted search for published 5–7 nm HD SRAM macro standby power per Mb at nominal and retention voltage across temperature (ISSCC/VLSI/JSSC macro papers that report standby current), corroborated by a predictive-PDK SPICE characterization of a 6T HD cell and its periphery as a modeled T2 figure; then re-run `analysis/sram_validation/report.py` with the measured range in place of A4. That one number spans the Gate 4B and 4D verdicts; the dynamic result (Gate 4A) does not depend on it.
