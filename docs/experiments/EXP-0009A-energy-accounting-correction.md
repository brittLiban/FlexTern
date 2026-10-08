# EXP-0009A — Correctness audit and energy-model repair of EXP-0009 (B7 wake energy, B14 shutdown-and-reload)

Status: Complete (awaiting founder/ChatGPT review). Correction addendum to EXP-0009; supersedes EXP-0009's wake-energy and shutdown-and-reload figures and the energy half of its Gate 5B verdict. EXP-0009's tables are left as generated at 899976e and are marked by a banner.
Date: 2026-10-08
Commit: recorded by a docs-only fixup commit after the EXP-0009A commit; corrected results generated from the EXP-0009A tree.

Primary data: `results/sram_retention/` (corrected run of every EXP-0009 artifact plus `wake_components.csv`, `reload_economics.csv`, `correction_comparison.csv`); `results/sram_retention/superseded_899976e/` (the 25 artifacts of the 899976e run, unmodified). Model: `analysis/sram_retention/{retention_evidence,wake,policy,compare,report}.py`. Tests: `tests/test_sram_retention_correction.py` (15), `tests/test_sram_retention.py` (21, unchanged and still passing).

## Research Question

Does the dimensional inconsistency in EXP-0009's wake-energy assumption (a formula giving 1.2 nJ/MB, a model applying 1.2 µJ/MB) and the incomplete shutdown-and-reload comparison (no persistent backing memory) materially change EXP-0009's Gate 5A–5D conclusions? This is a correctness task, not an attempt to improve or defend the SRAM-resident edge form.

## Hypothesis

None to confirm. Two outcomes were held open: the 1000× discrepancy is a unit slip around a formula that is otherwise complete, or the formula omits physical terms large enough to justify the reported magnitude. For the reload comparison, the expectation was that charging the backing store would lengthen the break-even gap and could reverse the "shutdown wins long hot gaps" statement.

## Why It Matters

Gate 5B's energy half, the long-idle regime of Gate 5C and the "retention versus reload" planning question in Gate 5D all rest on these two assumptions. A corrected negative result would be worth more than a favourable unsupported one.

## Setup

**Audit scope.** EXP-0009's `retention_evidence.py` (B1–B15), `wake.py`, `policy.py`, `compare.py`, `report.py`, the 21 tests and all 25 artifacts, re-derived by hand before any code was changed.

**New sources (T1–T3).** R11 (MAPG, DATE 2012) re-read from Table I: a 32 nm core of 2.891 mm² logic has C_core 7.53 nF, Q 7.53 nC at 1.0 V, "wake-up energy" 3.30 nJ, C_decap = 20% of C_core (T2, old node; anchors the nJ-per-mm² class). R19 SK hynix UFS 3.1 datasheet HN8Tx5DxHKX07x Rev 1.1 (T1, device side): active 2-lane ICC 330 mA at 2.5 V + ICCQ 550 mA at 1.2 V = 1.485 W at 2,000 MB/s sequential read (92.8 pJ/bit), 1-lane 1.365 W at 1,000 MB/s (171 pJ/bit); Sleep 0.43 mW, Hibernate 0.58 mW (256 GB); PowerDown both rails off. R20 Micron UFS 3.1 product page (T3): active 960 mW max, sleep 0.2 mW. R21 Kyung, Yun, Ahn, arXiv 2508.06978 (T3, secondary derivation; its cited primary was not read): flash read 102.4 pJ/b including I/O, DRAM 4.2–9.2 pJ/b. R03's 41-cycle break-even was checked as an energy anchor and rejected: at NPU-D's 1.75 GHz and B1's 0.33 mW/MB it implies ~6 pJ/MB for its source-biased SLEEP exit, 200× below any rail-charge estimate; R03 anchors latency and ratios only. Two figures seen only in search summaries (a 1.5 µs break-even for an 8 KB ULVR macro; a 72 pJ wake for a 16 kB 28 nm macro) were excluded under the R18 precedent.

**Corrected B7 (wake energy per decimal MB, 8e6 cells; ASSUMPTION, physics-based, R11 method).** Supply energy of a hard-switched rail recharged from V_active: E = C × V_active × ΔV per rail (KNOWN: the supply delivers Q = C ΔV at V_active; half is stored, half dissipated in the switch; this is the expression EXP-0009 stated). Components, LOW / PLANNING / HIGH:

| rail | capacitance per MB | swing | energy nJ/MB (L / P / H) |
|---|---|---|---|
| (a) cell array rail (PU sources, high-node loads) | 0.3 / 0.5 / 1.0 fF per cell = 2.4 / 4 / 8 nF | V_ret → V_active, 0.35 → 0.75 V | 0.72 / 1.20 / 2.40 |
| (b) bitlines, two per column | 0.10 / 0.15 / 0.20 fF per cell per line = 1.6 / 2.4 / 3.2 nF | ΔV (follow the array rail) / full V_active at HIGH (discharged) | 0.48 / 0.72 / 1.80 |
| (c) periphery virtual rail | 0.4 / 0.8 / 1.6 nF (R11 ~2.6 nF/mm² of logic × ~0.3 mm² per MB at EXP-0006 densities) | full V_active | 0.23 / 0.45 / 0.90 |
| (d) decoupling and grid | 10% / 20% / 50% of the switched capacitance (R11: 20%) | as above | 0.14 / 0.47 / 2.55 |
| RETENTION_VRET exit (a+b+c+d) | | | **1.57 / 2.84 / 7.65** |
| DEEP_SLEEP_NOMINAL exit (b+c+d; array rail unchanged, bitlines follow it: b = 0 unless discharged) | | | 0.25 / 0.54 / 4.05 |
| LIGHT_SLEEP exit (bitline precharge only) | | | 0 / 0 / 1.80 |

Sequencer 10 pJ per macro unchanged; sleep entry 0 (stored charge is dissipated, not drawn). V_active = 0.75 V (EXP-0008 A4); the ASAP7 ratio deck's nominal is 0.70 V and is used for ratios only. In-rush implied by B7 at B6's wake times: planning 3.8 nC/MB → 0.97 nC per 256 KB macro / 200 ns = 4.9 mA per macro, 39 mA per group of 8; HIGH 2.6 mA per macro at 1 µs; the superseded 1.2 µJ/MB would have needed 4 µF/MB and ~2 A per macro. B6's LOW level (10 ns per macro, 64 in parallel, from R03's 4 KB segments) implies 54 mA per macro and 3.4 A per group and is therefore a latency bound, not an in-rush-consistent design point (recorded in B7).

**Corrected B14 (shutdown-and-reload, data lost), split into B16 and B17.** PX-DRAM (B16): the persistent copy sits in the LPDDR5X die; reload = weights × 8 × (4.0 pJ/bit LPDDR path + 0.50 pJ/bit SRAM write path, i.e. EXP-0008's resident read path with the macro write at 54.1/32.8 × read) = 14.3 mJ for Falcon-E 1B in 33 ms (one x16 channel at 70%); backing standby = the DRAM's data-retaining idle mode at the idle temperature (self-refresh 0.77–9.8 mW for gaps ≥ 1 s, power-down 5.4 mW otherwise), and in the weights-only view also power-down during the session because nothing else uses the die; in the system view (head/KV external) that background is already counted on both sides. PX-FLASH (B17): UFS-class NAND; read energy 46 / 93 / 171 pJ/bit (R19 2-lane max, 1-lane; LOW = half the 2-lane maximum, the EXP-0008 A6 typical-versus-max convention; R21 corroborates) + 0.50 pJ/bit SRAM write = 147 / 296 / 543 mJ for Falcon-E 1B; bandwidth 4 / 2 / 1 GB/s (0.10 / 0.20 / 0.40 s first-token latency); standby 0 / 0.43 / 0.58 mW (PowerDown with host re-init / Sleep / Hibernate). Host-side UFS controller and M-PHY are in no datasheet read and are covered only by HIGH. Symmetric baseline option: the streaming system may also power its DRAM off for long gaps and reload from flash (flash read + 4.0 pJ/bit DRAM write); priced as a column. PX-UNBACKED reproduces the superseded EXP-0009 row (reload 13.2 mJ at LPDDR energy, no standby, 1–4 GB/s) for traceability only.

**Other changes.** B18 (sensitivity): EXP-0009's RETENTION_VRET ratio uses array power I(V_ret)·V_ret, which is the supply draw only if the retention rail is delivered at V_ret (separate rail or efficient converter; R01 is dual-rail). If V_ret is derived from V_dd by a linear element (source-bias diode as in R04's modes, or an LDO) the ratio is the current ratio, 2.0× worse at 0.35 V. B19 (sensitivity): the resident side's own head/KV DRAM background in the system view. Transition counting: a P2+S session no longer charges a whole-model wake on top of the window's per-token layer wakes (the superseded model did). Every other input (B1–B6, B8–B13, B15), the leakage model, the schedules, the LPDDR baseline and the dynamic energies are unchanged.

## Hardware / Software Environment

Python 3.12.10 on the host; `pypdf` in the WSL venv to read the three PDFs (not committed). No hardware; the Tier-2 SPICE was not re-run (unaffected).

## Baseline

LPDDR5X streaming as EXP-0008/0009 defined it (B10). The corrected PX rows are judged against the same baseline and, where stated, against the baseline with its DRAM off and a flash reload.

## Method

Verify head f1c65de; re-derive B7 and B14 by hand; cross-check against R11 and R03; obtain and read the UFS datasheet and the flash-energy paper; snapshot `results/sram_retention/` to `superseded_899976e/`; rebuild `wake.py` from components with a reproduction switch; add reload accounting, backing-aware sessions and break-even solvers to `policy.py`; add thresholds 11a/11b/11s and 12a–12d, the reload table and the correction comparison to `compare.py`; regenerate; write 15 tests mapped to the audit's validation list; run all non-RTL suites in one process; independent review (`docs/reviews/REV-2026-10-08-exp-0009a-correction-review.md`); update the record and canonical docs.

## Results

### R1. Confirmed defects (each separately)

| # | Defect | Evidence | Effect |
|---|---|---|---|
| D1 | B7 unit error: formula 4 nF/MB × 0.75 V × 0.40 V = 1.2 nJ/MB, model constant `WAKE_UJ_PER_MB_ARRAY_SWING = 1.2` (µJ); PLANNING 0.15 and LOW 0.02 µJ/MB defined as fractions of it | `retention_evidence.py` 899976e lines 164–165; `policy.py` `wake_uj_mb`; implied 4 µF/MB and ~2 A in-rush per macro | every wake-energy figure in EXP-0009 is ~420× too high against the component total (1000× against the cell-rail term alone) |
| D2 | B14 backing memory unpriced: PX idle charged only the SRAM residual (0.002×); reload at LPDDR energy (13.2 mJ) but timed at 1–4 GB/s (flash class); no reload write path beyond the macro | `policy.py` 899976e `e_reload`, `p_shutdown`, `first_token_stall_us` | PX rows and threshold 11 understate the idle alternative; the hot 300 s "PX wins 1.63×" was an artefact of a free backing store |
| D3 | Transition double count: `session(in_session="window")` added one whole-model wake per session to a policy whose per-token layer wakes already cover the first token | `policy.py` 899976e `session()` | 0.53 mJ per session in the superseded model (now 0; immaterial after D1 but a counting error) |

Suspected limitations, not defects: B18 (retention-rail supply scheme) and B19 (resident-side DRAM background) are boundary choices EXP-0009 made implicitly; they are now explicit sensitivities. EXP-0009's R11 table printed PS-at-25%-duty totals for Falcon-E 1B of 1.41× / 1.58× while its own `system_reconnect.csv` (identical before and after this correction for that column) holds 1.37× / 1.53×; a transcription discrepancy in the record, the CSV governs.

### R2. Original versus corrected, Falcon-E 1B, planning (INFERRED; `correction_comparison.csv`)

| Quantity | °C | EXP-0009 (899976e) | Corrected (EXP-0009A) | Cause |
|---|---:|---:|---:|---|
| Wake energy per MB, V_ret exit | | 1,200 nJ | 2.84 nJ (1.57–7.65) | D1 |
| Layer wake (18.4 MB) / whole model (440 MB) | | 22 µJ / 528 µJ | 0.053 µJ / 1.27 µJ | D1 |
| P2 transition energy per token; share of the 1.40 mJ read | | 528 µJ; 38% | 1.27 µJ; 0.09% | D1 |
| P2 wake power at 25 tok/s | | 13.2 mW | 0.032 mW | D1 |
| P2 resident total at 25 tok/s; LPDDR/resident | 25 | 70.4 mW; 4.90× | 57.2 mW; 6.03× | D1 |
| P2 resident total at 25 tok/s; LPDDR/resident | 85 | 257.7 mW; 1.34× | 244.5 mW; 1.41× | D1 |
| P2 at 5 / 100 tok/s | 25 | 2.30× / 6.06× | 2.51× / 8.03× | D1 |
| P2 at 5 / 100 tok/s | 85 | 0.33× / 3.23× | 0.34× / 3.72× | D1 |
| PT0 (race-to-idle) at 25 tok/s | 25 / 85 | 59.3 mW 5.81× / 212.9 mW 1.62× | 46.2 mW 7.47× / 199.7 mW 1.73× | D1 |
| P0 at 25 tok/s | 25 / 85 | 180.3 mW 1.91× / 744.2 mW 0.46× | unchanged | no wake term |
| Threshold 4 headroom (wake per MB at which P2 stops beating P0, 25 tok/s, over the model) | 25 / 85 | 9× / 38× (parity at 250 tok/s) | 3,900× / 16,000× (390× at 250 tok/s) | D1 |
| Threshold 7, idle time after which whole-model retention pays | 25 / 85 | 3.9 ms / 0.97 ms | 9.5 µs / 2.3 µs | D1 |
| Threshold 8, P2 crossover | 25 / 85 | 1.45 / 17.5 tok/s | 1.38 / 16.8 tok/s | D1 |
| Threshold 8, PT0 crossover | 25 / 85 | 0.49 / 13.7 tok/s | 0.46 / 13.1 tok/s | D1 |
| Threshold 9, P2+S duty crossover at 25 tok/s | 25 / 85 | 3.6% / 65% | 3.5% / 61% | D1 |
| Threshold 9, P0 and PS duty crossovers | | 47% / 5.9%; never / never | unchanged | no wake term |
| Threshold 10, temperature at which P2 continuous loses | | 93.6 °C | 94.7 °C | D1 |
| Reload energy | | 13.2 mJ | 14.3 mJ (DRAM) / 296 mJ (flash; 147–543) | D2 |
| First-token latency after shutdown | | 0.1–0.4 s | 33 ms (DRAM) / 0.10–0.40 s (flash) | D2 |
| Backing standby while idle (gap ≥ 1 s) | 25 / 85 | 0 | DRAM 0.77 / 9.8 mW; flash 0.43 mW (0–0.58) | D2 |
| Threshold 11, gap above which shutdown beats retention (64-token P0 session) | 25 | 1.2 s | 2.8 s (DRAM) / 28.7 s (flash) | D2 |
| Threshold 11 | 85 | 0.08 s | 0.18 s (DRAM) / 1.83 s (flash) | D2 |
| Threshold 12 (new), gaps over which shutdown beats the LPDDR baseline, P0 session | 25 | n/a | DRAM: up to ~1,300 s; flash: always | D2 |
| Threshold 12, P0 session | 85 | n/a (PX 1.63× at 300 s) | DRAM: never (best 0.87×); flash: beyond ~166 s | D2 |
| Threshold 12, P2 session | 85 | n/a | DRAM: up to ~158 s; flash: beyond ~5.3 s | D2 |
| P2+S session wake charged per session | | 0.53 mJ | 0 | D3 |

### R3. Walkthrough at 25 tok/s, corrected (planning; every mW; INFERRED)

| °C | policy | dynamic | active leak | retention leak | wake | resident total | LPDDR total | LPDDR / resident (EXP-0009) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 25 | P0 all active | 34.9 | 145.3 | 0 | 0 | 180.3 | 344.6 | 1.91× (1.91×) |
| 25 | P1 one layer | 34.9 | 6.1 | 10.6 | 0.03 | 51.6 (+43 µs stall/token) | 344.6 | 6.68× (5.32×) |
| 25 | P2 current + next | 34.9 | 12.1 | 10.1 | 0.03 | 57.2 | 344.6 | 6.03× (4.90×) |
| 25 | P4 window | 34.9 | 24.2 | 9.2 | 0.03 | 68.4 | 344.6 | 5.04× (4.23×) |
| 25 | P2 burst (8 TB/s) | 34.9 | 0.0 | 11.0 | 0.03 | 46.0 | 344.6 | 7.49× (5.82×) |
| 25 | PT0 race-to-idle | 34.9 | 0.2 | 11.0 | 0.03 | 46.2 (+43 µs) | 344.6 | 7.47× (5.81×) |
| 85 | P0 all active | 34.9 | 709.3 | 0 | 0 | 744.2 | 344.6 | 0.46× (0.46×) |
| 85 | P1 one layer | 34.9 | 29.6 | 157.3 | 0.03 | 221.8 | 344.6 | 1.55× (1.47×) |
| 85 | P2 current + next | 34.9 | 59.1 | 150.4 | 0.03 | 244.5 | 344.6 | 1.41× (1.34×) |
| 85 | P4 window | 34.9 | 118.2 | 136.7 | 0.03 | 289.9 | 344.6 | 1.19× (1.14×) |
| 85 | P2 burst (8 TB/s) | 34.9 | 0.1 | 164.1 | 0.03 | 199.1 | 344.6 | 1.73× (1.62×) |
| 85 | PT0 race-to-idle | 34.9 | 0.9 | 163.9 | 0.03 | 199.7 | 344.6 | 1.73× (1.62×) |

Race-to-idle (PT0) and burst P2 now beat paced P2 at every rate and temperature on energy because the whole-model wake costs 1.3 µJ; P2 paced stays the planning policy because it claims no engine clock (B9), PT0 is strictly better if the 8 TB/s TARGET engine exists. The hot result is leakage-decided: at 85 °C the retained array (150–164 mW) is 60% of the P2 total; wake is 0.01%.

### R4. Sessions and the idle alternatives, corrected (Falcon-E 1B, 64 tokens at 25 tok/s; resident average mW and LPDDR/resident; INFERRED; `session_sweep.csv`)

P0 in session:

| °C | gap | PS retain | PX-DRAM (B16) | PX-FLASH (B17) | superseded PX | LPDDR baseline | baseline with DRAM off + flash reload |
|---:|---:|---|---|---|---|---:|---:|
| 25 | 1 s | 132.7 (1.87×) | 137.8 (1.80×) | 213.1 (1.16×) | 133.4 (1.86×) | 248.0 | 334.3 |
| 25 | 10 s | 45.5 (1.56×) | 39.8 (1.78×) | 60.9 (1.16×) | 38.0 (1.86×) | 70.9 | 95.1 |
| 25 | 60 s | 18.0 (0.83×) | 8.8 (1.68×) | 12.8 (1.16×) | 7.9 (1.89×) | 14.8 | 19.4 |
| 25 | 300 s | 12.5 (0.29×) | 2.7 (1.38×) | 3.2 (1.14×) | 1.9 (1.98×) | 3.7 | 4.4 |
| 85 | 1 s | 581 (0.43×) | 546 (0.46×) | 619 (0.40×) | 539 (0.46×) | 250.6 | 334.3 |
| 85 | 10 s | 282 (0.28×) | 163 (0.48×) | 177 (0.44×) | 154 (0.51×) | 78.0 | 95.1 |
| 85 | 60 s | 188 (0.13×) | 41.7 (0.56×) | 37.0 (0.64×) | 32.0 (0.73×) | 23.5 | 19.4 |
| 85 | 300 s | 169 (0.07×) | 17.5 (0.72×) | 9.1 (1.39×) | 7.7 (1.63×) | 12.6 | 4.4 |

P2 in session:

| °C | gap | PS(P2) | P2+PX-DRAM | P2+PX-FLASH | LPDDR baseline |
|---:|---:|---|---|---|---:|
| 25 | 1 s | 44.2 (5.61×) | 49.3 (5.03×) | 124.6 (1.99×) | 248.0 |
| 25 | 10 s | 20.5 (3.46×) | 14.7 (4.81×) | 35.8 (1.98×) | 70.9 |
| 25 | 60 s | 12.9 (1.15×) | 3.8 (3.90×) | 7.8 (1.91×) | 14.8 |
| 25 | 300 s | 11.4 (0.32×) | 1.6 (2.26×) | 2.2 (1.69×) | 3.7 |
| 85 | 1 s | 222 (1.13×) | 187 (1.34×) | 260 (0.97×) | 250.6 |
| 85 | 10 s | 180 (0.43×) | 61.0 (1.28×) | 74.9 (1.04×) | 78.0 |
| 85 | 60 s | 167 (0.14×) | 21.2 (1.11×) | 16.5 (1.42×) | 23.5 |
| 85 | 300 s | 165 (0.08×) | 13.3 (0.95×) | 4.9 (2.59×) | 12.6 |

Reading: a DRAM-backed shutdown is cheap to reload (14 mJ, 33 ms) but its idle floor is the same self-refreshing die the baseline idles on, so in the weights-only view it can win only through the session and converges to a loss at long gaps (0.72× hot at 300 s, where EXP-0009 printed 1.63×). A flash-backed shutdown has a near-zero idle floor but a 300 mJ reload (21 tokens of LPDDR streaming energy), so it needs gaps of ~29 s (cool) / ~1.8 s (hot) to beat retention and, hot, ~5 s (P2 sessions) to ~170 s (P0 sessions) to beat the baseline. The symmetric baseline (DRAM off, flash reload) shows the streaming system cannot escape the same reload cost. EXP-0009's "idle gaps beyond ~1 s cool / ~0.1 s hot are served by shutdown-and-reload" is therefore replaced by: beyond ~3 s / ~0.2 s if the copy lives in DRAM, beyond ~29 s / ~1.8 s if it lives in flash, and only the flash case ever beats the streaming baseline when hot.

### R5. Reload economics by level (Falcon-E 1B; `reload_economics.csv`)

| backing | level | reload mJ | reload s | standby while idle | break-even gap vs retention, 25 / 85 °C |
|---|---|---:|---:|---|---|
| DRAM | planning | 14.3 | 0.033 | 0.77 / 9.8 mW self-refresh (5.4 mW power-down < 1 s) | 2.8 s / 0.18 s |
| flash | low | 147 | 0.099 | 0 (powered down, host re-init unmodeled) | 14 s / 0.9 s |
| flash | planning | 296 | 0.198 | 0.43 mW (UFS sleep) | 29 s / 1.8 s |
| flash | high | 543 | 0.396 | 0.58 mW (hibernate) | 53 s / 3.4 s |

### R6. Sensitivities touched by the correction (Falcon-E 1B, 25 tok/s; `mode_sensitivity.csv`, `system_reconnect.csv`, `int4_comparison.csv`, `voltage_domain_sweep.csv`)

| case | 25 °C P2 / PS 25% duty | 85 °C P2 / PS 25% duty |
|---|---|---|
| planning (corrected) | 6.03× / 1.63× | 1.41× / 0.29× |
| superseded wake energy (1.2 µJ/MB) | 4.90× / 1.63× | 1.34× / 0.29× |
| HIGH wake energy (7.65 nJ/MB) | 6.02× / 1.63× | 1.41× / 0.29× |
| **B18: V_ret derived linearly (current ratio 0.152 / 0.463)** | 5.12× / 1.41× | **0.87× / 0.21×** |
| LIGHT_SLEEP (T1 mode) / DEEP_SLEEP_NOMINAL | 2.06× / 2.46× | 0.51× / 0.63× |
| V_ret 0.25 V / 0.45 V | 6.63× / 5.11× | 1.80× / 1.12× |

Head/KV external at 2k, P2 at 25 tok/s, total ratio BF16 / candidate (EXP-0009 in parentheses): Falcon-E 1B 25 °C 2.40× / 3.98× (2.26 / 3.50), 85 °C 1.26× / 1.35× (1.21 / 1.29); 3B 2.97× / 4.60× (2.72 / 3.94), 1.32× / 1.40× (1.26 / 1.33); BitNet 1.52× / 2.54× (1.49 / 2.37), 1.15× / 1.29× (1.12 / 1.25). With the resident side's own head/KV DRAM background charged (B19): 1B 25 °C 2.23× / 3.67×, 85 °C 1.21× / 1.32×. PS at 25% duty totals are unchanged (no wake term). INT4 fairness: P2 25 °C 6.03× ternary vs 5.79× INT4, 85 °C 1.41× vs 1.36×; transition 1.3 vs 2.9 µJ/token; ratios within 4%, verdict unchanged. Voltage domains at 85 °C: 1 / 2 / 4 / 8 layers / whole model 1.41× / 1.19× / 0.91× / 0.61× / 0.46× (EXP-0009: 1.34 / 1.14 / 0.87 / 0.60 / 0.46); the granularity conclusion stands and is leakage-driven. Temperature sweep, P2 at 25 tok/s: 25 / 50 / 75 / 85 / 105 °C = 6.03× / 4.01× / 1.99× / 1.41× / 0.68×. Crossovers for 3B and BitNet: P2 1.36 / 1.31 tok/s cool, 16.0 / 16.1 tok/s hot; P2+S duty 3.7% / 3.5% cool, 61% / 60% hot.

### R7. Reproduction (MEASURED by test)

`Scenario(wake_model="exp0009_original")` and `backing="unbacked_superseded"` reproduce the 899976e walkthrough P2 totals, ratios, PX rows and threshold 11 against the snapshot (test 12); `leakage_temperature.csv`, `retention_voltage.csv`, `layer_schedule.csv`, `exp0008_reconnect.csv` and every P0 walkthrough cell are byte-identical before and after (test 13); EXP-0008's inputs are untouched (tests 13, 22).

## Interpretation

- **B7 was a unit slip around an incomplete formula, not a hidden physical term.** No capacitance on the array rail, bitlines, periphery or decap reaches a thousandth of what 1.2 µJ/MB implies; R11's core-level charge and the in-rush arithmetic both put a multi-MB domain in the nJ-per-MB class. The corrected range (1.6–7.7 nJ/MB) spans 5× and is immaterial at every point: wake energy is 0.09% of the read energy at planning and 0.25% at HIGH, independent of token rate.
- **The correction helps residency slightly and does not change which regime it wins in.** The cool P2 figure rises from 4.9× to 6.0×, the hot one from 1.34× to 1.41×; crossovers move by less than 5% of their value. Leakage decides the hot case, as before.
- **The idle alternative is only as cheap as its backing store.** EXP-0009's reload row was a DRAM-energy, flash-bandwidth, free-standby hybrid. Priced consistently, shutdown wins long gaps against retention on both backings but against the streaming baseline only from flash and only beyond seconds (P2 sessions) to minutes (P0 sessions) when hot. The planning statement "long gaps are served by reload" survives only with "from flash, beyond ~30 s cool / ~2 s hot, at a 0.2 s first-token penalty" attached.
- **A new verdict-relevant assumption surfaced (B18).** Whether the retention rail is supplied at V_ret or derived linearly is a 2× effect on the whole retention saving; it flips the 85 °C continuous case (1.41× → 0.87×). EXP-0009 chose the favourable reading implicitly.
- **Against EXP-0008 and EXP-0007:** nothing in EXP-0008 (dynamic energies, leakage anchors, cold-start energy as an input) changes; EXP-0008's cold-start row is now understood as the DRAM-backed reload energy without a write path. EXP-0007 is not reopened.

## Surprises

1. The error was exactly three orders of magnitude and survived a 26-point review because every downstream figure (38% of read energy, 9× headroom, parity at 250 tok/s) looked plausible for a physical cost that was never measured.
2. With the correct wake energy, race-to-idle beats every windowed policy; the window matters for latency hiding, not energy.
3. A DRAM-backed shutdown cannot beat the streaming baseline during idle by construction; the hot "shutdown wins" rows were the free backing store talking.
4. The flash reload of a 1B model costs ~21 tokens of streaming energy; at 25 tok/s a session shorter than a second does not pay for it.

## Limitations

1. No measured modern-node macro wake charge exists; the component model is an estimate with explicit ranges, cross-checked only against a 32 nm core-level model (R11).
2. Flash read energy is a device-side datasheet maximum at 25 °C over 100 ms; the host UFS controller and M-PHY energy, the resume time from PowerDown and any file-system overhead are unmodeled (HIGH level is the only allowance).
3. The DRAM standby during PX-DRAM sessions in the weights-only view is a boundary choice; the system view removes it. Both are reported.
4. B18 is a sensitivity, not a measurement; which scheme a given compiler uses is unknown.
5. EXP-0009's limitations 1–8 stand (Tier-2 temperature coefficient, modeled retention ratios, nominal hold SNM, datasheet IDD maxima, memory-subsystem boundary, assumed rails and sequencer, idle temperature).
6. The duty-cycle sweep charges the DRAM the cheaper of power-down and self-refresh at the idle temperature (EXP-0009 B10); at 85 °C that is power-down without refresh, which cannot retain data over long idle and favours the baseline by ≤ 4.4 mW × (1 − d). Left as defined; the session table uses self-refresh.

## What This Proves

- INFERRED (physics, R11 class anchor, dimensional test): the wake energy of a multi-MB SRAM retention domain is nJ per MB, 1.6–7.7 nJ/MB here, and is immaterial (0.09–0.25% of the read energy) to every EXP-0009 comparison.
- INFERRED (T1 datasheet modes and energies): a shutdown-and-reload idle policy beats retention beyond ~3 s / ~0.2 s (DRAM-backed) or ~29 s / ~1.8 s (flash-backed) at 25 / 85 °C, and beats the streaming baseline hot only from flash and only beyond ~5 s (P2) to ~170 s (P0) gaps.
- MEASURED (by test): the corrected model reproduces the 899976e figures under its superseded inputs and leaves every unaffected artifact byte-identical.

## What This Does NOT Prove

- Any silicon figure. Nothing here is a measurement of a FlexTern design or of a foundry macro.
- That the hot continuous case wins: it depends on B18 and on EXP-0009's Tier-2 temperature coefficient.
- Anything about a nonvolatile subsystem design; flash is priced from a datasheet, not designed.

## Gate verdicts (revised; EXP-0009 verdict in parentheses)

**Gate 5A — Modern retention leakage: INCONCLUSIVE (unchanged).** No correction touches the leakage anchors or ratios. New: the modeled saving assumes a separately supplied retention rail; a linear derivation halves it (B18).

**Gate 5B — Wake feasibility: PASS on latency (unchanged); energy PASS as a bound, INFERRED and unmeasured (was INCONCLUSIVE).** The corrected transition energy is 1.3 µJ/token at planning (0.09% of the read energy) and 3.4 µJ at HIGH (0.25%), at every rate to 250 tok/s, with 390× headroom where EXP-0009 reported parity. No plausible capacitance reaches the threshold; no modern measurement exists.

**Gate 5C — Retention-enabled edge power: INCONCLUSIVE, regime-split by temperature (unchanged in kind).** Cool: PASS for continuous generation above ~1.4 tok/s and duty above ~3.5–6% at 25 tok/s (6.0× weights-only, 2.4–4.0× with head/KV). Hot: FAIL below ~17 tok/s continuous and at every duty below ~61%; above 17 tok/s 1.41× (1.26–1.35× with head/KV), and 0.87× under B18. Long idle gaps: served by a flash-backed reload beyond ~29 s cool / ~1.8 s hot against retention; a DRAM-backed reload never beats the streaming baseline at idle (replaces EXP-0009's ~1 s / ~0.1 s statement).

**Gate 5D — Model-resident hardware direction: INCONCLUSIVE (unchanged).** The open inputs are the product temperature, the existence of a foundry array-voltage retention mode and, newly, how its rail is supplied; wake energy leaves the list. No ADR, no SRAM selection, no edge commitment.

**Q-023:** narrowed a third time. Resolved: wake energy (bounded immaterial). Open: measured retention leakage versus temperature; the retention-rail supply scheme; the product thermal and duty envelope.

## Artifacts / Logs

`results/sram_retention/` (corrected; `summary.md` carries a correction notice and the original-versus-corrected table), `results/sram_retention/superseded_899976e/`, `analysis/sram_retention/`, `tests/test_sram_retention_correction.py`, review `docs/reviews/REV-2026-10-08-exp-0009a-correction-review.md`, handoff `docs/handoffs/2026-10-08-exp-0009a-energy-accounting-correction.md`, KI-010. Reproduce: `python analysis/sram_retention/report.py`; `python -m unittest tests.test_sram_retention_correction tests.test_sram_retention -v`.

## Follow-up Questions

- Does the target compiler's retention mode supply the array rail at V_ret from a separate rail, or derive it from V_dd by a linear element (B18)? This now decides the 85 °C continuous verdict.
- The EXP-0009 questions stand: measured modern-node retention leakage versus temperature and wake charge under NDA; the product's junction temperature and duty cycle.
- Is a 0.2 s first-token penalty after idle acceptable for the target product? If not, the flash-backed idle policy is unavailable and the hot low-duty case has no winning policy.
