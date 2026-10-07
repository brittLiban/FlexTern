# Handoff

Date: 2026-10-07
Agent: Claude (Claude Code, model Fable 5.1), implementation and research role
Current commit: 899976e (EXP-0009), followed by a docs-only fixup commit recording this hash

## Objective

FT-CURRENT, EXP-0009: SRAM retention, leakage and wake feasibility for layer-banked model-resident inference (Gate 5). Complete.

## What I Completed

Eighteen tiered sources on SRAM power modes (two Tier-1 modern-node standby figures at 3 nm and 7 nm, a Tier-1 22 nm FD-SOI retention and wake measurement, Tier-1 old-node temperature data, Tier-2 7 nm PDK retention ratios, Tier-3 compiler mode definitions), a Tier-2 ASAP7 6T-cell SPICE corroboration (ngspice + OpenVAF-compiled BSIM-CMG; voltage 0.25–0.80 V, 25–105 °C, hold SNM), fifteen tagged assumptions, and a reproducible package (`analysis/sram_retention/`) that builds per-layer schedules from the profiles, models active / light-sleep / deep-sleep / V_ret-retention leakage versus temperature, wake latency and energy by domain, the P0 / P1 / P2 / P4 / P8 / PS / PT policies and a shutdown-and-reload comparison (PX), token-rate, duty-cycle, session, temperature, voltage-domain and wake sweeps, INT4 fairness, head/KV reconnection, eleven break-even thresholds, a Falcon-E 1B walkthrough and the EXP-0008 P0 reconciliation; 21 tests; the EXP-0009 record with Gate 5A–5D verdicts; docs; completion review with ten corrections applied before commit.

## What I Changed

- New: `analysis/sram_retention/{retention_evidence,retention,wake,layer_schedule,policy,compare,report}.py`, `analysis/sram_retention/spice/` (decks, README, raw cell tables), `tests/test_sram_retention.py`, `results/sram_retention/` (25 files), EXP-0009, this handoff, the completion review.
- Updated: `PROJECT_STATE.md`, `CURRENT_WORK.md` (complete), `THESIS.md`, `ARCHITECTURE.md` (§4 note and change log), `ROADMAP.md`, `OPEN_QUESTIONS.md` (Q-005, Q-015, Q-023 narrowed), `KNOWN_ISSUES.md` (KI-009 ngspice/BSIM-CMG tooling note), `README.md`, `analysis/README.md`. `BENCHMARKS.md` unchanged. No ADR.
- Untouched: RTL, EXP-0001..0008 code and results.
- Not committed (gitignored `build/`): ASAP7 model cards, OpenVAF, BSIM-CMG Verilog-A, the compiled `.osdi`, downloaded PDFs.

## Tests / Validation Run

```
python -m unittest tests.test_sram_retention -v                           # 21 OK
python -m unittest tests.test_reference tests.test_analysis tests.test_analysis_cross_model tests.test_roofline tests.test_memory_residency tests.test_datacenter_gate tests.test_sram_validation tests.test_sram_retention tests.test_hardware_gate.TestPackingInterpretation tests.test_hardware_gate.TestWorkloadAndTrafficModels   # OK in one process
python analysis\sram_retention\report.py                                  # seconds
```

EXP-0005 RTL test classes were not re-run (no RTL changed). The SPICE sweep is reproducible from `analysis/sram_retention/spice/README.md` (WSL).

## Actual Results

See EXP-0009 R1–R12 and `results/sram_retention/summary.md`. Headline: modern SRAM standby leakage is 0.21–0.52 mW/MB at 25 °C (T1) and ~1.6 mW/MB at 85 °C (T2 scaling); EXP-0008's 0.60 was a ~55 °C point. A V_ret retention mode (T2) cuts leakage 13× cold but only 4× hot. With a two-layer active window, a resident Falcon-E 1B beats LPDDR5X at 25 °C from ~1.4 tok/s continuous and ~4–6% duty (4.9× weights-only, 2.3–3.5× with head/KV), but at 85 °C only above ~17 tok/s (1.34×) and never below ~65% duty; a retained 440 MB array leaks more than a self-refreshing DRAM, so long idle gaps are served by shutdown-and-reload. Wake latency is hidden everywhere; wake energy is 38% of the read energy at planning with 9× headroom at 25 tok/s. Gate 5A INCONCLUSIVE, 5B PASS on latency / INCONCLUSIVE on energy, 5C INCONCLUSIVE split by temperature, 5D INCONCLUSIVE. Q-023 narrowed, not closed.

## Remaining Problems

- The hot verdict rests on a Tier-2 temperature coefficient and the deep-retention ratios on a predictive cell; no modern-node measured retention leakage per MB exists in public literature (searched: ISSCC/VLSI/JSSC abstracts, compiler documentation, vendor pages).
- Wake energy of a multi-MB domain is an assumption with a 60× range.
- R02's standby figure was quoted from a summary page the session could not open.
- Hold SNM is nominal, no variation; the real DRV is statistical.

## Important Context

- Temperature, not leakage-per-MB, is now the decisive unknown: the same hardware wins 4.9× at 25 °C and 1.34× at 85 °C, and loses at every low duty cycle when hot.
- The modes modern papers document (light sleep, periphery-off) move nothing; the result depends on an array-voltage retention mode that compilers advertise (T3) and a predictive cell supports (T2) but no modern silicon paper reports.
- Retention is the wrong idle mode beyond ~1 s cool / ~0.1 s hot; the obvious alternative is to lose the weights and reload (13 mJ, 0.1–0.4 s).
- Paced P2 is planning; burst execution and race-to-idle are better if the engine runs at the 8 TB/s TARGET.
- The datacenter conclusion (EXP-0007) was not reopened.

## Questions for Founder / ChatGPT

1. What junction temperature and duty cycle does the target edge product run at? The verdict flips between 48 °C (25% duty) and 94 °C (continuous 25 tok/s).
2. Can FlexTern obtain a foundry/compiler SRAM datasheet (TSMC N5/N3 HD compiler via Arm or Synopsys, under NDA) with light-sleep / deep-sleep / shutdown leakage versus temperature and wake cycles?
3. Should the edge form plan on a nonvolatile weight store with shutdown-and-reload for idle, rather than retention?

## Recommended Immediate Next Consideration

Replace the Tier-2 inputs that span the verdict with Tier-1 data: obtain a modern (5–3 nm) HD SRAM compiler datasheet under NDA (standby, light-sleep, deep-sleep and shutdown leakage versus temperature and voltage, wake cycles, switch overhead) and fix the product's thermal and duty envelope, then re-run `analysis/sram_retention/report.py` with those inputs in place of B1, B2, B4, B6 and B7. This is a founder action (NDA access and a product decision), not an agent experiment; the public literature does not contain the number, and no further modeling changes which way the gate falls.
