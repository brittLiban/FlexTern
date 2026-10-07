# EXP-0009 — SRAM retention, leakage and wake feasibility for layer-banked model-resident inference (Gate 5)

Status: Complete (awaiting founder/ChatGPT review)
Date: 2026-10-07
Commit: 899976e ("research: evaluate SRAM retention for model residency (EXP-0009)"); results generated from that tree. The hash is recorded by a docs-only fixup commit.

Primary data: `results/sram_retention/` (`summary.md`, `sources.csv`, `assumptions.csv`, `retention_evidence.csv`, `asap7_cell_leakage.csv`, `hold_snm.csv`, `leakage_temperature.csv`, `retention_voltage.csv`, `wake_latency_energy.csv`, `wake_slack_sweep.csv`, `voltage_domain_sweep.csv`, `layer_schedule.csv`, `retention_policy_sweep.csv`, `token_rate_sweep.csv`, `duty_cycle_sweep.csv`, `session_sweep.csv`, `temperature_sweep.csv`, `int4_comparison.csv`, `mode_sensitivity.csv`, `break_even.csv`, `system_reconnect.csv`, `exp0008_reconnect.csv`, `primary_comparison.csv`, `falcon_e_1b_walkthrough.csv`, `environment.json`). Model: `analysis/sram_retention/{retention_evidence,retention,wake,layer_schedule,policy,compare,report}.py` plus the Tier-2 SPICE decks in `analysis/sram_retention/spice/`. Tests: `tests/test_sram_retention.py` (21 tests).

## Research Question

Can most of a resident ternary model's SRAM spend most of its time in a low-leakage data-retentive state while only the current or imminently needed layer banks stay active, without wake latency, wake energy, sequencing or granularity costs large enough to erase the EXP-0008 dynamic advantage (Gate 5A retention leakage, 5B wake feasibility, 5C retention-enabled edge power, 5D hardware direction)? Continues Q-023.

## Hypothesis

Sequential layer execution lets a rolling window of active layers carry the token while the rest of the model sits in retention; if retention leakage is several times below active standby, wake hides behind the previous layer and transition energy is small against the saved leakage, residency stays attractive at low duty cycles. The experiment was built to falsify this.

## Why It Matters

EXP-0008 found that the 440–811 MB of always-on SRAM a resident 1B–3B model needs leaks enough to lose to LPDDR5X below ~23 tok/s continuous or ~85% duty at its planning leakage, with modern-node leakage unmeasured and the retention factors hypothesized. The edge form stands or falls on this.

## Setup

**Sources (R01–R18; T1 measured silicon or datasheet, T2 primary simulation/PDK, T3 secondary, T4 historical).** R01 TSMC 3 nm HD 6T 563 kbit dual-rail SRAM, JSSC 2025: 27.5 µW/Mbit standby at 25 °C in its one-cycle-latency low-leakage mode (bitline prechargers off during NOP, 10% below normal standby, one-cycle exit), 3.89 pJ/access (T1). R02 Socionext 7 nm ultra-high-density 256 Kb macro, VLSI 2020 (EXP-0008 E02): 16.9 µW standby at 25 °C (T1, figure quoted from the paper's summary page; the abstract does not restate it). R03 ReGate, arXiv 2025, 7 nm PDK NPU study: SLEEP retains data at 25% of active static power, OFF loses data at 0.2%, wake 4 / 10 cycles, break-even 41 / 82 cycles, 4 KB segments (T2). R04 Synopsys memory-compiler power modes: light sleep up to 50% standby saving (array source-biased, fast wake), deep sleep up to 70% (periphery off, array retained), shutdown (lost) (T3). R05 22 nm FD-SOI RISC-V MCU, arXiv 2023: 128 KiB SRAM retention 3.2 µW at 25 °C and 142 µW at 125 °C, 75× below active with periphery off and reverse body bias, power-down ~99% but lost, wake 200 ns with in-rush limiting, 2.1% area per 4 KiB macro (T1, planar FD-SOI, not FinFET). R06 Renesas 40 nm, ISQED 2014: deep standby 1.86 µW/Mb at 25 °C and 643 µW/Mb at 170 °C (T1 for mechanism and temperature, T4 for node). R07 55 nm compiler: retention 4×, sleep 50× lost (T2, simulated). R08 Toshiba 65 nm ISSCC 2014: 27 fA/bit retention, 7 ns access (T4). R09 Renesas 65 nm SOTB: 13.7 nW/Mbit with reverse body bias (T4). R10 Intel 22 nm Xeon L3 slice shutoff; 65 nm Tulsa sleep transistors > 2× (T1 mechanism / T4). R11 MAPG, DATE 2012: wake energy Q = C·ΔV and in-rush-limited wake, 5–14 ns core wake at 32 nm (T2 method). R12 Qin et al. 2004: DRV < 300 mV at 130 nm, 10 mV margin over 25–125 °C (T4). R13 ASAP7 predictive 7 nm PDK 6T cell simulated here (T2). R14 EXP-0008 E05 28 nm (T1, not modern). R15 FinFET ~100 pA/cell statements (T3). R16 Samsung 3 nm GAA 256 Mb, JSSC 2022, and R17 TSMC 7 nm 256 Mb, ISSCC 2017: silicon exists, no leakage in the accessible abstracts (T1, unused). R18 an unverified "279 mV DRV at 5 nm" claim found only in a search summary (excluded).

**Power modes kept distinct.** ACTIVE / IDLE-STANDBY (nominal rails, periphery on) / LIGHT SLEEP (bitline prechargers off, one-cycle exit; R01) / DEEP SLEEP, array at nominal (periphery off; R04) / DATA RETENTION at V_ret (periphery off, array supply lowered; R05, R06, R13) / BODY-BIAS retention (FD-SOI/SOTB only; R05, R09) / SHUTDOWN (data lost; R03, R04, R10). Shutdown is never used as retention; it is priced separately as a reload policy (PX, B14).

**Tier-2 SPICE (R13).** ngspice 45.2 (Debian, WSL) with BSIM-CMG 110 compiled from Verilog-A by OpenVAF 23.5.0 (the Debian ngspice has no built-in level-72 model), ASAP7 r1p7 model cards converted to OSDI. A 6T cell of single-fin `sram`-Vt devices in hold (WL = 0, BL = BLB = VDD) was swept over 0.25–0.80 V and 25 / 50 / 75 / 85 / 105 °C at TT (SS/FF at two points) for the current drawn from VDD, BL and BLB; the hold VTC was swept for a nominal (no-variation) hold static noise margin. Array only, no periphery. Used for ratios, never for absolutes. Decks and the raw tables are committed; the PDK, OpenVAF and the Verilog-A sources are not.

**Assumptions (B1–B15, every constant tagged in `retention_evidence.py`).** B1 active standby at 25 °C, macro incl. periphery: LOW 0.21 (R01) / PLANNING 0.33 (geometric mean) / HIGH 0.52 mW/MB (R02); EXP-0008's flat 0.60 is kept only as a reproduction input. B2 temperature scaling from the ASAP7 cell at nominal voltage: ×1.60 (50 °C), ×3.42 (75 °C), ×4.88 (85 °C), ×10.2 (105 °C) versus 25 °C; at 0.35 V ×3.1 / 9.6 / 14.9 / 34 (MODELED, T2; the old-node measured retention modes R05 1.46×/10 °C and R06 1.50×/10 °C agree with the low-voltage curve; there is no T1 modern-node coefficient). B3 periphery share of macro standby 0.30 (R01's prechargers alone are 10%). B4 mode ratios to active standby at the same temperature: LIGHT_SLEEP 0.90 (R01, T1); DEEP_SLEEP_NOMINAL 1 − B3 = 0.70; RETENTION_VRET (periphery off, array at V_ret) = 0.70 × ASAP7 array power ratio, which is temperature dependent: 0.076 at 25 °C and 0.231 at 85 °C for 0.35 V (0.037 / 0.150 at 0.25 V; 0.153 / 0.331 at 0.45 V); BODY_BIAS 0.013 (R05) as an FD-SOI bound only; R03's 0.25 sits inside the range. B5 V_ret planning 0.35 V (hold state converges and the nominal hold SNM is 178 mV at 0.35 V across 25–105 °C; 128 mV at 0.25 V; no variation analysis; no verified modern DRV). B6 wake latency: per 256 KB macro LOW 10 ns (R03) / PLANNING 200 ns (R05) / HIGH 1 µs, macros waking in groups of 64 / 8 / 1 (in-rush limit), floor 2 ns (R01). B7 wake energy: 4 nF/MB of array rail (KNOWN physics, order of magnitude) × 0.75 V × 0.4 V swing = 1.2 µJ/MB for exiting V_ret retention (HIGH-class, applied to RETENTION_VRET); periphery-only restore 0.15 µJ/MB (DEEP_SLEEP_NOMINAL); 0.02 µJ/MB R03-class; 10 pJ sequencer per macro; sleep entry free. B8 granularity: one switch set per 256 KB macro is supported by every compiler source; layer domains are groups of macros on a shared retention rail; swept 1 / 2 / 4 / 8 layers and whole model. B9 engine timing: burst layer time = layer bytes / 8 TB/s (EXP-0006 TARGET; 1 TB/s sensitivity); paced layer time = token period / L. No physical clock. B10 LPDDR5X unchanged from EXP-0008 (4.0 pJ/bit; IDD2N for the busy fraction, IDD2P for the rest; idle = cheaper of power-down 5.4 mW and self-refresh 0.77–9.8 mW at the idle temperature). B11 dynamic energies carried (0.4405 / 4.0 pJ/bit). B12 usage sweeps are not forecasts. B13 idle periods optionally at 40 °C when generating at 85 °C. B14 PX shutdown-and-reload: array off (0.002× residual), reload 13.2 mJ (EXP-0008 cold start) and 0.1–0.4 s per session. B15 disclosure: datasheet IDD maxima are hot-corner values charged at every temperature while the SRAM side is 25 °C-measured at 25 °C (favours residency at 25 °C by up to ~27 mW of 345 mW; sensitivity row halves them).

**Workloads and schedule.** Falcon-E 1B (24 layers × 66.06 M weights = 16.5 MB per layer at 2 bits, 440.4 MB provisioned), Falcon-E 3B (32 × 91.2 M = 22.8 MB, 810.9 MB), BitNet 2B4T (30 × 69.5 M = 17.4 MB, 578.9 MB), from `weight_distribution_by_layer.csv` (sums equal the ternary totals); executed layer 0 → L−1 then the external head. INT4 at 4.5 bits with the identical algorithm.

**Policies.** P0 all active; P1 / P2 / P4 / P8 rolling windows (every layer wakes once per token; paced is planning, burst at 8 TB/s is the sensitivity); PS session-active with whole-model retention between sessions; PT timeout 0 / 1 / 10 / 100 / 1000 ms; PX shutdown-and-reload (comparison only). Sweeps: 1–250 tok/s; duty 1–100% at 25 tok/s; sessions of 16–1024 tokens with 0–300 s gaps; 25 / 50 / 75 / 85 (/105) °C.

## Hardware / Software Environment

Python 3.12 on the host; ngspice and OpenVAF in WSL for the Tier-2 cell simulation (seconds per point). No hardware.

## Baseline

LPDDR5X streaming exactly as EXP-0008 defined it (B10). Where the DRAM idles it is charged its cheapest datasheet mode at the idle temperature.

## Method

Verify head 3df1e6d; collect and tier sources; separate power modes; build the leakage model (B1–B5) with the SPICE ratios; the wake model (B6–B7); the layer schedule; the policies; the LPDDR comparison with head/KV external; the break-even thresholds; INT4; 21 tests; independent review (`docs/reviews/REV-2026-10-07-exp-0009-completion-review.md`) whose corrections (PX reload policy, limitation ordering, LPDDR temperature disclosure, explicit negatives, threshold wording, flag rename) were applied before commit.

## Results

### R1. Leakage by mode and temperature (planning level, mW per decimal MB; INFERRED, T1 anchors × T2 scaling)

| °C | T-factor at V_nom | active standby | light sleep (T1 mode) | deep sleep, array nominal | retention at 0.35 V | retention ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 1.00 | 0.330 | 0.297 | 0.231 | 0.025 | 0.076 |
| 50 | 1.60 | 0.528 | 0.476 | 0.370 | 0.079 | 0.149 |
| 75 | 3.42 | 1.129 | 1.016 | 0.790 | 0.240 | 0.212 |
| 85 | 4.88 | 1.611 | 1.450 | 1.127 | 0.373 | 0.231 |
| 105 | 10.2 | 3.356 | 3.020 | 2.349 | 0.857 | 0.256 |

Two things decide everything below. First, EXP-0008's single planning number (0.60 mW/MB) sits at ~55 °C on this curve; the T1 anchors are 0.21–0.52 at 25 °C and the hot corner is 1.6 at 85 °C. Second, retention saves 13× at 25 °C but only 4.3× at 85 °C: in the ASAP7 cell the room-temperature leakage is gate-leakage dominated and falls steeply with voltage, while the hot leakage is subthreshold dominated and nearly voltage independent (cell current 25.9 → 5.6 pA from 0.70 to 0.35 V at 25 °C; 127 → 84 pA at 85 °C). Whether real HD FinFET cells share this mix at 25 °C is not measured; the pessimistic case (hot ratio 0.23 applied cold) is in `mode_sensitivity.csv`.

### R2. Wake latency and energy (ASSUMPTION ranges B6/B7; Falcon-E 1B)

| Domain | MB | macros | wake latency low / planning / high | wake energy low / planning / high | per token if every layer wakes (planning mode) |
|---|---:|---:|---|---|---:|
| one 256 KB macro | 0.26 | 1 | 10 ns / 200 ns / 1 µs | 5 nJ / 38 nJ / 0.31 µJ | |
| one layer | 18.4 | 72 | 20 ns / 1.8 µs / 72 µs | 0.37 / 2.75 / 22 µJ | 0.53 mJ (exiting V_ret: 22 µJ × 24) |
| whole model | 440 | 1,721 | 0.27 µs / 43 µs / 1.7 ms | 8.8 / 66 / 528 µJ | |

Layer time is 1.67 ms at 25 tok/s paced (2.1 µs burst at 8 TB/s). A P2 window hides the planning layer wake under every swept condition; the high-case 72 µs layer wake is hidden paced but stalls burst execution (`wake_slack_sweep.csv`); P1 (no lead) stalls 43 µs per token at planning, 0.1% of a 40 ms token. Exiting V_ret retention costs 0.53 mJ per token when every layer transitions every token, 38% of the 1.40 mJ dynamic read energy; the per-MB wake energy at which this stops paying is 11 µJ/MB at 25 tok/s (9× the planning figure) and 1.1 µJ/MB at 250 tok/s (parity). No measured modern-node macro wake energy exists.

### R3. Falcon-E 1B walkthrough, 25 tok/s, batch 1 (planning; every mW; INFERRED)

| °C | policy | dynamic | active leak | retention leak | wake | resident total | LPDDR dyn + bg = total | LPDDR / resident |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 25 | P0 all active | 34.9 | 145.3 | 0 | 0 | 180.3 | 317.1 + 27.5 = 344.6 | 1.91× |
| 25 | P1 one layer | 34.9 | 6.1 | 10.6 | 13.2 | 64.8 (+43 µs stall/token) | 344.6 | 5.32× |
| 25 | P2 current + next | 34.9 | 12.1 | 10.1 | 13.2 | 70.4 | 344.6 | 4.90× |
| 25 | P4 window | 34.9 | 24.2 | 9.2 | 13.2 | 81.6 | 344.6 | 4.23× |
| 25 | P2 burst (8 TB/s) | 34.9 | 0.0 | 11.0 | 13.2 | 59.2 | 344.6 | 5.82× |
| 25 | PT0 race-to-idle | 34.9 | 0.2 | 11.0 | 13.2 | 59.3 (+43 µs) | 344.6 | 5.81× |
| 85 | P0 all active | 34.9 | 709.3 | 0 | 0 | 744.2 | 344.6 | 0.46× |
| 85 | P1 one layer | 34.9 | 29.6 | 157.3 | 13.2 | 234.9 | 344.6 | 1.47× |
| 85 | P2 current + next | 34.9 | 59.1 | 150.4 | 13.2 | 257.7 | 344.6 | 1.34× |
| 85 | P4 window | 34.9 | 118.2 | 136.7 | 13.2 | 303.1 | 344.6 | 1.14× |
| 85 | P2 burst (8 TB/s) | 34.9 | 0.1 | 164.1 | 13.2 | 212.3 | 344.6 | 1.62× |
| 85 | PT0 race-to-idle | 34.9 | 0.9 | 163.9 | 13.2 | 212.9 | 344.6 | 1.62× |

At 5 tok/s: 25 °C P2 31.9 vs 73.2 mW (2.30×); 85 °C P2 219 vs 73 mW (0.33×). At 100 tok/s: 25 °C P2 215 vs 1,300 mW (6.06×); 85 °C 402 vs 1,300 (3.23×). At 85 °C the retained 440 MB alone leaks 164 mW, more than a self-refreshing LPDDR5X die (9.8 mW) and almost half the whole streaming budget at 25 tok/s; no policy below P2 at ~17 tok/s rescues it.

### R4. Token-rate crossovers (planning; INFERRED)

| Model | °C | P0 wins above | P2 wins above | PT0 wins above | P2 ratio at 5 / 25 / 100 tok/s |
|---|---:|---:|---:|---:|---|
| Falcon-E 1B | 25 | 11.5 tok/s | 1.4 | 0.5 | 2.30× / 4.90× / 6.06× |
| Falcon-E 1B | 85 | 60.0 | 17.5 | 13.7 | 0.33× / 1.34× / 3.23× |
| Falcon-E 3B | 25 | 11.7 | 1.4 | 0.7 | 2.44× / 4.95× / 6.07× |
| Falcon-E 3B | 85 | — | 16.8 | — | 0.34× / 1.36× / 3.29× |
| BitNet 2B4T | 25 | — | 1.4 | — | 2.43× / 5.01× / 6.08× |
| BitNet 2B4T | 85 | — | 16.9 | — | 0.34× / 1.37× / 3.29× |

EXP-0008's 23 tok/s P0 crossover was a ~55 °C result; it is now bracketed by 11.5 tok/s (25 °C) and 60 tok/s (85 °C) for P0, and moved to 1.4–17.5 tok/s by the P2 window.

### R5. Duty-cycle crossovers at 25 tok/s (planning; INFERRED; retention rows T2)

| Model | °C | P0 (no retention) wins above | PS (retain when idle) wins above | P2+S wins above | PS ratio at 25% duty (idle at same T / idle at 40 °C) |
|---|---:|---|---|---|---|
| Falcon-E 1B | 25 | 47% | 5.9% | 3.6% | 1.63× / — |
| Falcon-E 1B | 85 | never | never | 65% | 0.29× / 0.43× |
| Falcon-E 3B | 25 | 48% | 6.5% | 3.0% | 1.57× / — |
| Falcon-E 3B | 85 | never | never | ~65% | 0.28× / 0.42× |
| BitNet 2B4T | 25 | ~47% | 6.1% | ~3.5% | 1.61× / — |
| BitNet 2B4T | 85 | never | never | ~65% | 0.29× / 0.43× |

EXP-0008's 85% duty crossover (no retention mode, 0.60 mW/MB) becomes 47% at 25 °C and "never" at 85 °C for P0, and 6% (PS) or 4% (P2+S) at 25 °C with V_ret retention. Cooling the idle periods to 40 °C does not rescue the hot case (0.43×), because the 440 MB still leaks 22 mW retained at 40 °C against 0.77 mW of DRAM self-refresh.

### R6. Sessions and the shutdown alternative (Falcon-E 1B, 64 tokens at 25 tok/s; INFERRED)

| °C | idle gap | PS (retain idle) res / LPDDR | ratio | PX (shut down + reload 13 mJ) res / LPDDR | ratio | P2+PX res | ratio |
|---:|---:|---|---:|---|---:|---:|---:|
| 25 | 1 s | 133 / 248 mW | 1.87× | 133 / 248 | 1.86× | | |
| 25 | 10 s | 46 / 71 | 1.55× | 38 / 71 | 1.86× | | |
| 25 | 60 s | 18 / 15 | 0.83× | 7.9 / 15 | 1.89× | | |
| 25 | 300 s | 12.5 / 3.7 | 0.29× | 1.9 / 3.7 | 1.98× | | |
| 85 | 1 s | 581 / 251 | 0.43× | 539 / 251 | 0.46× | 189 | 1.32× |
| 85 | 10 s | 282 / 78 | 0.28× | 154 / 78 | 0.51× | 55 | 1.43× |
| 85 | 60 s | 188 / 24 | 0.13× | 32 / 24 | 0.73× | 12 | 1.94× |
| 85 | 300 s | 169 / 13 | 0.07× | 7.7 / 13 | 1.63× | 3.6 | 3.48× |

Retention is the wrong idle mode for long gaps: the retained array leaks 11 mW (25 °C) / 164 mW (85 °C) against 0.77 / 9.8 mW of DRAM self-refresh, so shutdown-and-reload (data lost, 13 mJ and 0.1–0.4 s first-token latency) beats retention beyond 1.2 s (25 °C) / 0.08 s (85 °C) of idle (threshold 11). The best hot policy is P2 inside the session and PX between sessions; the best cool policy is P2 in-session with retention for short gaps and PX for long ones. Session length does not change the ordering (one model wake of 0.07 mJ or one reload of 13 mJ per session amortizes over any session longer than a few tokens: 16-token sessions with 10 s gaps give P2+PX 3.7× at 25 °C).

### R7. Temperature (Falcon-E 1B, 25 tok/s, planning; INFERRED)

| °C | active mW/MB | retention mW/MB | P0 resident | P2 resident | LPDDR | P0 ratio | P2 ratio | PS 25% duty ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.330 | 0.025 | 180 | 70 | 345 | 1.91× | 4.90× | 1.63× |
| 50 | 0.528 | 0.079 | 268 | 99 | 345 | 1.29× | 3.47× | 0.95× |
| 75 | 1.129 | 0.240 | 532 | 186 | 345 | 0.65× | 1.85× | 0.43× |
| 85 | 1.611 | 0.373 | 744 | 258 | 345 | 0.46× | 1.34× | 0.29× |
| 105 | 3.356 | 0.857 | 1,513 | 517 | 345 | 0.23× | 0.67× | 0.14× |

Verdict-flipping temperatures (threshold 10): P2 continuous 25 tok/s loses above 94 °C; P0 continuous above 59 °C; PS at 25% duty above 48 °C.

### R8. Voltage-domain granularity (Falcon-E 1B, 85 °C, 25 tok/s; INFERRED)

| domain | MB | active window | wake latency | wake energy | resident mW | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 layer | 18 | 2 layers | 1.8 µs | 22 µJ | 258 | 1.34× |
| 2 layers | 37 | 4 | 3.6 µs | 44 µJ | 303 | 1.14× |
| 4 layers | 73 | 8 | 7.2 µs | 88 µJ | 394 | 0.87× |
| 8 layers | 147 | 16 | 14 µs | 176 µJ | 576 | 0.60× |
| whole model | 440 | 24 | 43 µs | 528 µJ | 757 | 0.46× |

Retention needs layer-granular (or finer) domains: at four-layer domains the hot case already loses. Macro-granular switches are supported by every source; a per-layer shared retention rail is an assumption (B8).

### R9. INT4 fairness (identical policy, mode, voltage, temperature, wake model; INFERRED)

Falcon-E 1B at 25 °C, 25 tok/s: P2 ternary 70 mW (4.90×) vs INT4 158 mW (4.71×); PS 25% duty 53 mW (1.63×) vs 120 mW (1.56×); layer wake 1.8 vs 4.2 µs; transition 0.53 vs 1.19 mJ/token. At 85 °C: P2 258 mW (1.34×) vs 580 mW (1.29×). Ternary changes absolute memory power by 2.25× and the number of macros per domain, not the verdict against LPDDR (ratios within 4%), because both the resident and the streaming side scale with bytes.

### R10. Mode and level sensitivity (Falcon-E 1B, 25 tok/s; `mode_sensitivity.csv`)

| mode (25 °C → 85 °C) | retention ratio | P2 ratio | PS 25% duty ratio |
|---|---|---|---|
| LIGHT_SLEEP (the only mode with a T1 modern figure) | 0.90 | 2.04× → 0.51× | 0.63× → 0.14× |
| DEEP_SLEEP_NOMINAL (periphery off) | 0.70 | 2.43× → 0.63× | 0.74× → 0.16× |
| RETENTION_VRET 0.45 V | 0.153 → 0.331 | 4.27× → 1.07× | 1.46× → 0.25× |
| RETENTION_VRET 0.35 V (planning) | 0.076 → 0.231 | 4.90× → 1.34× | 1.63× → 0.29× |
| RETENTION_VRET 0.25 V | 0.037 → 0.150 | 5.29× → 1.68× | 1.84× → 0.34× |
| BODY_BIAS bound (FD-SOI only) | 0.013 | 6.84× → 3.31× | 1.94× → 0.47× |
| pessimistic: hot ratio 0.23 applied at 25 °C | 0.23 | 3.79× | 1.24× |
| IDD2N/IDD2P halved at 25 °C (B15) | 0.076 | 4.70× | 1.56× |
| leakage level LOW / HIGH at 25 °C (V_ret 0.35) | | 6.98× / 3.45× | 2.51× / 1.10× |
| leakage level LOW / HIGH at 85 °C | | 2.31× / 0.78× | 0.49× / 0.17× |

Every improvement over EXP-0008 comes from lowering the array voltage (T2); the modes with modern T1/T3 evidence (light sleep, periphery-off deep sleep) do not move EXP-0008's verdict.

### R11. Head and KV external in both systems (planning, 25 tok/s; INFERRED)

| Model | °C | context | case | weights-only P2 | total P2 | total PS at 25% duty |
|---|---:|---:|---|---:|---:|---:|
| Falcon-E 1B | 25 | 2k | BF16 / candidate | 4.90× | 2.26× / 3.50× | 1.41× / 1.58× |
| Falcon-E 1B | 85 | 2k | BF16 / candidate | 1.34× | 1.21× / 1.29× | 0.37× / 0.31× |
| Falcon-E 3B | 25 | 2k | BF16 / candidate | 4.95× | 2.72× / 3.94× | 1.43× / 1.55× |
| Falcon-E 3B | 85 | 2k | BF16 / candidate | 1.36× | 1.26× / 1.33× | 0.33× / 0.29× |
| BitNet 2B4T | 25 | 2k | BF16 / candidate | 5.01× | 1.49× / 2.37× | 1.20× / 1.41× |
| BitNet 2B4T | 85 | 2k | BF16 / candidate | 1.37× | 1.12× / 1.25× | 0.49× / 0.35× |

### R12. EXP-0008 reconciliation (MEASURED reproduction by test)

With retention disabled and the EXP-0008 flat 0.60 mW/MB input, P0 reproduces EXP-0008's 1.397 / 12.68 mJ per token, 264 mW static, 22.9 tok/s break-even and its duty-cycle rows (`exp0008_reconnect.csv`, tests 2 and 12).

## Interpretation

- **Retention leakage (INFERRED, T2-dependent):** modern T1 evidence covers only a 10% light-sleep saving; the 4–13× savings that change the verdict require lowering the array voltage, which the predictive 7 nm cell supports (178 mV nominal hold margin at 0.35 V) but no modern silicon measurement confirms. The saving is strongly temperature dependent: 13× cold, 4× hot.
- **Wake (INFERRED, assumption-dependent):** latency is never the problem at edge token rates (1.8 µs per layer against 1.67 ms of layer time at 25 tok/s; even a 1.7 ms whole-model wake is 4% of a token). Energy is a real but bounded cost: 0.53 mJ/token (38% of the read energy) at planning, with 9× headroom at 25 tok/s and parity at 250 tok/s.
- **Policy:** P2 (current + next layer, paced) is the planning policy; burst execution and race-to-idle are better still if the engine runs at the TARGET bandwidth; P1 stalls negligibly; P4/P8 and coarser domains cost leakage without buying anything; for idle gaps beyond ~1 s (cool) or ~0.1 s (hot) shutdown-and-reload beats retention; session length is irrelevant.
- **Temperature is the verdict:** at 25 °C a retention-enabled resident Falcon-E 1B beats LPDDR5X above 1.4 tok/s continuous and above 4–6% duty, 2.3–3.5× in total with external head/KV; at 85 °C it wins only above ~17 tok/s continuous (1.34×, 1.2–1.3× with head/KV) and loses at every duty cycle below ~65%, retention or not. EXP-0008's single planning point was a ~55 °C result.
- **Against the prior experiments:** Gate 4B's uncertainty has not been resolved; it has been moved from "leakage per MB" to "operating temperature and whether a V_ret retention mode exists at the foundry". Gate 4D is unchanged. EXP-0007's datacenter conclusion is untouched (not reopened). EXP-0004's candidate remains the lever for the edge total (3.5× vs 2.3× at 25 °C). EXP-0005's bank organization is what the macro-granular switches sit under.

## Surprises

1. Retention helps least where it is needed most: at 85 °C the array leakage is subthreshold-dominated and nearly voltage independent, so lowering V_ret saves 4× instead of 13×.
2. A retained 440 MB array (164 mW at 85 °C, 11 mW at 25 °C) leaks more than a self-refreshing LPDDR5X die (9.8 / 0.77 mW); for long idle gaps the right answer is to lose the data and reload, not to retain it.
3. EXP-0008's "hot nominal" planning leakage was a lukewarm point (~55 °C); the hot corner is 2.7× worse and the room-temperature point 1.8× better.
4. Wake latency is irrelevant at edge rates; wake energy matters (38% of the read energy per token at planning when every layer transitions every token).
5. The modes modern papers actually document (light sleep, periphery-off deep sleep) do not change EXP-0008's verdict at all.

## Limitations

1. The whole hot result rests on a Tier-2 temperature coefficient (ASAP7 ×4.88 at 85 °C) applied to Tier-1 25 °C macro figures; no modern-node measured coefficient for nominal-voltage FinFET standby was found. The 25 °C retention ratio (0.076) depends on the predictive cell's gate-versus-subthreshold leakage mix; if real HD cells are subthreshold-dominated cold, the P2 result at 25 °C is 3.8× rather than 4.9×.
2. No modern-node measured retention leakage per MB, wake latency for a multi-MB domain, or wake energy exists; B4, B6 and B7 are modeled.
3. R02's 16.9 µW standby figure (half of the planning anchor) was quoted from a summary page the session could not open.
4. Hold SNM is nominal TT with no variation; the real data-retention voltage is a statistical quantity and may be above 0.35 V.
5. Datasheet IDD maxima are hot-corner values charged at every temperature (B15; favours residency at 25 °C by ≤ 8%).
6. Memory-subsystem power only; compute, control, activation movement and the non-ternary engines are outside the boundary; no chip TDP, J/token, battery life, area or cost.
7. The layer-granular retention rail, the sequencer and the in-rush staggering are assumptions; no RTL or controller was built.
8. Idle temperature (B13) is an assumption; thermal behaviour of a real device was not modeled.

## What This Proves

- INFERRED (T1 anchors, T2 scaling): the leakage of a 440–811 MB resident array spans 0.21–0.52 mW/MB at 25 °C and ~1.6 mW/MB at 85 °C; EXP-0008's single figure was a mid-temperature point.
- INFERRED: with a V_ret retention mode and a two-layer active window, residency beats LPDDR5X streaming at 25 °C from ~1.4 tok/s continuous and ~4–6% duty, and at 85 °C only above ~17 tok/s continuous.
- INFERRED: wake latency is hideable at every swept edge rate; wake energy is bounded at ≤ 38% of the read energy at planning.
- INFERRED: shutdown-and-reload beats retention for idle gaps beyond ~1 s (cool) / ~0.1 s (hot).

## What This Does NOT Prove

- That any foundry's modern HD SRAM offers an array-voltage retention mode with these ratios, or what its wake costs are.
- Anything about the real operating temperature or duty cycle of a FlexTern edge product.
- Any chip-level figure.

## Gate verdicts

**Gate 5A — Modern retention leakage: INCONCLUSIVE.** The only modern-node (3 nm) measured low-leakage mode saves 10% with a one-cycle exit; the deep-retention ratios that matter (0.076–0.23) come from a predictive 7 nm cell (T2) and compiler marketing ceilings (T3); the one measured deep retention (75×, R05) is FD-SOI with body bias. Modeled evidence says retention is materially below active standby, measured modern evidence does not yet.

**Gate 5B — Wake feasibility: PASS on latency, INCONCLUSIVE on energy.** A per-layer wake of 1.8 µs (planning) to 72 µs (high) hides behind 1.67 ms of layer time at 25 tok/s under a two-layer window; even a 1.7 ms whole-model wake is 4% of a token. Transition energy at planning is 0.53 mJ/token (38% of the read energy) with 9× headroom at 25 tok/s and none at 250 tok/s, and no measured modern figure behind it.

**Gate 5C — Retention-enabled edge power: INCONCLUSIVE, regime-split by temperature.** Cool (≤ 50 °C): PASS for continuous generation above ~1.5 tok/s and for duty above ~4–6% at 25 tok/s (2.3–3.5× total with head/KV external at 2k). Hot (85 °C): FAIL below ~17 tok/s continuous and at every duty below ~65%; above 17 tok/s continuous the advantage is 1.34× (1.2–1.3× with head/KV). Long idle gaps are served by reload, not retention, at both temperatures.

**Gate 5D — Model-resident hardware direction: INCONCLUSIVE.** The decision has moved from "how much does modern SRAM leak" to "what temperature does the product run at, and does the foundry offer an array-voltage retention mode"; neither is resolved here. SRAM residency remains the best edge candidate at room temperature and an unproven one for hot operation. No ADR, no SRAM selection, no edge commitment.

**Q-023:** narrowed again, not closed. Resolved: dynamic energy (EXP-0008); wake latency (hideable). Open: measured modern-node retention leakage and its temperature coefficient; wake energy of a multi-MB domain; the product's thermal and duty envelope.

## Artifacts / Logs

`results/sram_retention/`; `analysis/sram_retention/` (incl. `spice/`); `tests/test_sram_retention.py`; review `docs/reviews/REV-2026-10-07-exp-0009-completion-review.md`. Third-party PDK models, OpenVAF and the Verilog-A sources live under the gitignored `build/` and are not committed.

## Follow-up Questions

- What is the measured standby and retention leakage versus temperature of a modern (5–3 nm) HD SRAM compiler instance, and does it offer an array-voltage retention mode (foundry/compiler data under NDA)?
- What junction temperature and duty cycle does the target edge product actually see?
- What does a multi-MB retention domain cost to wake (energy, in-rush, sequencer), measured?
- Should the edge form plan on shutdown-and-reload with a nonvolatile weight store rather than retention for idle gaps beyond a second?
