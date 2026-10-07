# EXP-0008 — Modern SRAM energy and latency validation for the edge / model-resident architecture (Gate 4)

Status: Complete (awaiting founder/ChatGPT review)
Date: 2026-10-07
Commit: b48237f ("research: validate modern SRAM energy and latency (EXP-0008)"); results generated from that tree. The hash is recorded by a docs-only fixup commit.

Primary data: `results/sram_validation/` (`summary.md`, `sources.csv`, `assumptions.csv`, `modern_sram_macros.csv`, `macro_energy_scaling.csv`, `bank_decomposition.csv`, `macro_latency.csv`, `leakage_ranges.csv`, `local_wire_ranges.csv`, `lpddr_energy.csv`, `dynamic_energy.csv`, `token_rate_sweep.csv`, `duty_cycle_sweep.csv`, `break_even.csv`, `latency_comparison.csv`, `cold_start.csv`, `int4_comparison.csv`, `system_reconnect.csv`, `exp0006_reconnect.csv`, `primary_comparison.csv`, `environment.json`). Model: `analysis/sram_validation/{evidence,model,report}.py`. Tests: `tests/test_sram_validation.py` (15 tests).

## Research Question

Do modern (5–7 nm) multi-bank SRAM reads still beat LPDDR5X streaming on dynamic energy for batch-1 edge decode of a 1B–3B ternary model (Gate 4A); does residency keep a total memory-power advantage once the leakage of hundreds of megabytes of always-on SRAM is counted at realistic token rates and duty cycles (Gate 4B); does residency buy latency once streaming is prefetched (Gate 4C); and is the 1B–3B resident edge form still promising enough for continued hardware work (Gate 4D)? This narrows Q-023.

## Hypothesis

EXP-0006's planning figure for resident SRAM reads (1.6 pJ/bit, the top of a 45 nm Horowitz range) overstated modern SRAM energy; a modern macro figure keeps a multi-fold dynamic advantage over LPDDR5X; leakage, which EXP-0006 did not model, decides whether the advantage survives at edge duty cycles. The experiment was built to find the conditions under which residency loses.

## Why It Matters

After EXP-0007 every surviving FlexTern advantage (edge/resident 1B–3B, small-batch interactive serving) is a memory-movement-energy claim whose SRAM side rested on a 45 nm tutorial figure, and leakage had never been counted. The edge form cannot be judged on that basis.

## Setup

**Sources (E01–E12, tiers T1 primary silicon measurement or vendor datasheet, T2 published projection, T3 secondary, T4 historical).** E01 TSMC N5 135 Mb SRAM, JSSC 2020: 32.8 fJ/bit read, 54.1 fJ/bit write at 0.4 V (T1). E02 7 nm ultra-high-density 256 Kb macro, VLSI 2020: 5.8 pJ per read access, 0.82 ns access (T1; width not stated, 64–256 bits assumed, A3). E03 Intel 18A SRAM, ISSCC 2025: 5.6 GHz, 38.1 Mb/mm² (T1, frequency and density only). E04 Keckler et al., IEEE Micro 2011: SRAM 14 pJ per 64-bit read at 40 nm, 1.8–2.4 pJ projected at 10 nm; wire 150 → 115 fJ/bit/mm; DRAM interface 20–30 → 2 pJ/bit, access 8–15 → 2.5 pJ/bit (T2/T4). E05 28 nm HKMG embedded SRAM: 296 nW/kB idle at 0.5 V, 0.23 nW/kB in data-retention sleep at 0.46 V (T1, not modern node). E06 Micron LPDDR5X Y52Q datasheet IDD tables (T1). E07 secondary PHY-level 3–5 pJ/bit (T3). E08 Micron 110 mW/GB marketing figure (T3, superseded). E09 LPDDR6 ~0.8× claims (T3). E10 Horowitz ISSCC 2014 (T4). E11 FinFET ~100 pA/cell standby statements (T3). E12 LPDDR5X typical timings (T3).

**Assumptions (A1–A13; every constant in `evidence.py` carries a tag).** A1 SRAM macro read energy LOW 33 / PLANNING 90 / HIGH 150 fJ/bit (E01 at 0.4 V; E02 at 64 bits; E01 scaled by V² to ~0.75 V = 115 fJ/bit sits between). A2 E ∝ V². A3 E02 width. A4 leakage LOW 0.05 / PLANNING 0.60 / HIGH 1.0 mW per provisioned MB (E11's 100 pA/cell × 8e6 cells/MB × 0.75 V = 0.6; E05's 0.296 mW/MB at 0.5 V kept as a reduced-voltage planar context point); retention-mode factors 1 / 0.2 / 0.1 are HYPOTHESIZED. A5 delivery: wire 0.115–0.15 pJ/bit/mm (E04) × 0.5 / 2 / 8 mm plus 0.02 pJ/bit registers; planning uses the 2 mm tile distance. A6 LPDDR5X per bit: device from E06 IDD4R arithmetic (16 × 1.80 + 382 × 1.05 + 0.2 × 0.90 + 121 × 0.50) mW / 136.5 Gb/s = 3.59 pJ/bit, a datasheet maximum; typical device 0.7× = 2.5 (ASSUMPTION); PHY/controller 1.0–2.5 (E07, E04); levels LOW 3.5 / PLANNING 4.0 / HIGH 6.1, built the same way as the SRAM side (typical for LOW/PLANNING, maximum for HIGH). A7 LPDDR background per x16 channel: self-refresh 0.77 mW (25 °C) to 9.8 mW (85 °C), power-down 5.4 mW, idle standby 32.1 mW (E06); SoC PHY idle not modeled. A8 latencies: SRAM access 0.82 ns, cycle 0.18–0.5 ns; LPDDR device first access ~38 ns, SoC-loaded ~90 ns; prefetch 4 KB. A9 a 6.19 MB logical bank (EXP-0005/0006) built from 256 KB–2 MB macros, one macro switching per access, per-access fixed cost 0.5–1.5 pJ amortized over 128–2048-bit reads. A10 cold start at 1–4 GB/s with SRAM write = 1.65 × read (E01). A11 10% capacity margin (carried from EXP-0006). A12 bank-composition overhead (bank decode, macro select, output mux) 0.05 / 0.10 / 0.15 pJ/bit. A13 during the active period the LPDDR channel is busy for W·R / (17.07 GB/s × 70%) of each second; the busy fraction is charged IDD2N and the rest IDD2P.

**Boundaries.** Resident: SRAM macro (array + periphery) + bank-composition overhead + local wires + registers + amortized decode; leakage reported separately. LPDDR5X: DRAM device (E06) + SoC PHY/controller (A6); board idle excluded. Both systems move the output head and the KV cache externally at the same LPDDR energy (BF16 reference and the EXP-0004 candidate 4.156-bit head / 4.5-bit KV). Memory-subsystem energy and power only: no chip TDP, no joules per token, no battery life, no die area, no cost.

**Workloads.** Falcon-E 1B (396,361,728 B packed 2-bit; 440.4 MB provisioned with margin), Falcon-E 3B (729,808,896 B; 810.9 MB), BitNet 2B4T (521,011,200 B; 578.9 MB), all read once per token at batch 1 (EXP-0006 capacities reproduced by test). INT4 at 4.5 bits per weight for the fairness case (2.25× capacity and leakage).

## Hardware / Software Environment

Pure Python 3.12 on the host; runs in seconds; no hardware. The Micron datasheet and the Keckler paper were read with `pypdf` in WSL; the PDFs are not committed.

## Baseline

LPDDR5X streaming from one x16 channel (8533 MT/s, 17.07 GB/s, 70% utilization) with background power from the same datasheet, plus the SoC PHY bracket; the strongest current edge DRAM path for which a vendor datasheet exists. LPDDR6 (E09) would lower the streaming figure by roughly 20%, which is inside the LOW/HIGH range.

## Method

Verify head aef2ae0; collect and tier the sources; derive per-bit energies with explicit boundaries; build the resident energy model (macro + bank overhead + wire + fixed), the LPDDR model (device + PHY + background), the leakage model and the latency model; sweep read width, macro size, wire distance, token rate 1–250 tok/s, duty cycle 1–100% with retention factors; solve the break-even equations; rescore the three canonical models at 2-bit and 4.5-bit, with BF16 and candidate head/KV external; reconnect EXP-0004/0005/0006/0007; write 15 tests; independent completion review (`docs/reviews/REV-2026-10-07-exp-0008-completion-review.md`), whose eight corrections (LPDDR level symmetry, bank overhead, leakage planning figure, retention labels, LPDDR busy-fraction idle, Gate 4C framing, 4A precision, Q-023 disposition) were applied before commit; the pre-correction figures are kept in that review.

## Metrics

pJ/bit by boundary; mJ per token (weights only, and with head/KV external); static mW per provisioned MB; average memory power at duty cycle d; break-even token rate, duty cycle, leakage, SRAM pJ/bit, wire pJ/bit, LPDDR pJ/bit; first-word and steady-state latency; cold-start time and energy.

## Results

### R1. Source-derived per-bit energies (MEASURED sources, INFERRED derivations)

| Quantity | Value | Evidence |
|---|---|---|
| SRAM macro read, TSMC N5 at 0.4 V | 32.8 fJ/bit (write 54.1) | E01, T1 |
| SRAM macro read, 7 nm 256 Kb at nominal | 5.8 pJ/access = 23–91 fJ/bit for 256–64-bit widths | E02, T1 (width INFERRED) |
| E01 scaled to ~0.75 V by V² | ~115 fJ/bit | A2 |
| Keckler 10 nm projection | 28–37 fJ/bit | E04, T2 |
| Planning macro figure | 90 fJ/bit (LOW 33, HIGH 150) | A1 |
| Resident total at 1024-bit reads, 2 mm wires | 0.33 / 0.44 / 0.62 pJ/bit = macro 0.033/0.09/0.15 + bank 0.05/0.10/0.15 + delivery 0.25/0.25/0.32 + fixed ≤ 0.002 | A1, A5, A9, A12 |
| LPDDR5X device, Micron IDD4R maximum | 3.59 pJ/bit (typical assumed 2.5) | E06, T1; A6 |
| LPDDR5X full path to the SoC consumer | 3.5 / 4.0 / 6.1 pJ/bit | A6 |
| LPDDR5X background per x16 channel | self-refresh 0.77 (25 °C) – 9.8 mW (85 °C); power-down 5.4 mW; idle standby 32.1 mW | E06, T1 |
| SRAM leakage per provisioned MB | 0.05 / 0.60 / 1.0 mW/MB, always on | A4 (E11 T3, E05 28 nm T1 context); modern-node measurement NOT FOUND |

The per-bit resident total is flat across read widths 128–2048 bits (0.337 → 0.333 pJ/bit at LOW; 0.444 → 0.440 at PLANNING) because the per-access fixed cost amortizes; width is not where the result lives. The 6.19 MB logical bank decomposes into 25 × 256 KB, 13 × 512 KB, 7 × 1 MB or 4 × 2 MB macros (6.4–8.0 MB provisioned), one switching per access.

### R2. Dynamic energy per token, weights only, batch 1 (INFERRED)

| Model | Level | SRAM total pJ/bit | LPDDR pJ/bit | Resident mJ/token | LPDDR mJ/token | Ratio |
|---|---|---:|---:|---:|---:|---:|
| Falcon-E 1B | LOW / PLANNING / HIGH | 0.33 / 0.44 / 0.62 | 3.5 / 4.0 / 6.1 | 1.06 / 1.40 / 1.97 | 11.1 / 12.7 / 19.3 | 10.5× / 9.1× / 9.8× |
| Falcon-E 3B | same | same | same | 1.95 / 2.57 / 3.63 | 20.4 / 23.4 / 35.6 | same |
| BitNet 2B4T | same | same | same | 1.39 / 1.84 / 2.59 | 14.6 / 16.7 / 25.4 | same |

Conservative-to-optimistic (LPDDR LOW ÷ SRAM HIGH to LPDDR HIGH ÷ SRAM LOW): 5.6–18.3×. The ratio is level-independent across models because both systems read the same bytes. Two T1 points and one T2 projection support an order-of-magnitude statement, roughly 10×, not a three-figure ratio.

### R3. Static power and continuous break-even (INFERRED; leakage ASSUMPTION A4)

| Model | Provisioned MB | Static mW LOW / PLANNING / HIGH | Break-even tok/s vs LPDDR power-down, LOW / PLANNING / HIGH |
|---|---:|---|---|
| Falcon-E 1B | 440.4 | 22 / 264 / 440 | 1.7 / 22.9 / 25.0 |
| Falcon-E 3B | 810.9 | 40 / 486 / 811 | 1.9 / 23.2 / 25.2 |
| BitNet 2B4T | 578.9 | 29 / 347 / 579 | 1.8 / 23.1 / 25.1 |

Break-even: (leakage − DRAM power-down) / (LPDDR mJ/token − resident mJ/token). At the planning level residency saves ~11.3 mJ per token for Falcon-E 1B and pays ~259 mW of extra standing power, so it wins total memory power only above ~23 tok/s of continuous generation. The 4.5-bit case has 2.25× the leakage and the same break-even within 1%.

### R4. Token-rate sweep, Falcon-E 1B, planning level, 2-bit (INFERRED; LPDDR background by busy fraction, A13)

| tok/s | Resident dynamic + static mW (always on) | LPDDR total mW | LPDDR/resident | Resident with HYPOTHESIZED 0.1× per-macro retention between accesses | LPDDR channels at 70% |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.4 + 264 = 266 | 19 | 0.07× | 28 | 1 |
| 5 | 7 + 264 = 271 | 73 | 0.27× | 33 | 1 |
| 10 | 14 + 264 = 278 | 141 | 0.51× | 40 | 1 |
| 25 | 35 + 264 = 299 | 345 | 1.15× | 61 | 1 |
| 100 | 140 + 264 = 404 | 1,301 | 3.2× | 166 | 4 |
| 250 | 349 + 264 = 613 | 3,203 | 5.2× | 375 | 9 |

Leakage is 88% of resident memory power at 25 tok/s and 99% at 1 tok/s. 3B and BitNet scale with capacity (full tables in `token_rate_sweep.csv`).

### R5. Duty-cycle sweep, 25 tok/s when active, planning level, 2-bit (INFERRED; retention rows HYPOTHESIZED)

| Model | Duty | Resident avg mW, idle retention 1× | Resident avg mW, idle retention 0.1× | LPDDR avg mW | LPDDR/resident at 1× | at 0.1× |
|---|---:|---:|---:|---:|---:|---:|
| Falcon-E 1B | 1% | 265 | 29 | 9 | 0.03× | 0.30× |
| Falcon-E 1B | 10% | 268 | 54 | 39 | 0.15× | 0.73× |
| Falcon-E 1B | 25% | 273 | 95 | 90 | 0.33× | 0.95× |
| Falcon-E 1B | 50% | 282 | 163 | 175 | 0.62× | 1.08× |
| Falcon-E 1B | 100% | 299 | 299 | 345 | 1.15× | 1.15× |
| Falcon-E 3B | 25% | 503 | 174 | 158 | 0.31× | 0.91× |
| BitNet 2B4T | 25% | 359 | 124 | 116 | 0.32× | 0.93× |

### R6. Break-even thresholds, planning level (INFERRED)

| Threshold | Falcon-E 1B | Falcon-E 3B | BitNet 2B4T |
|---|---:|---:|---:|
| Continuous token rate | 22.9 tok/s | 23.2 | 23.1 |
| Duty cycle at 25 tok/s active, idle retention 1× / 0.2× / 0.1× | 85% / 51% / 32% | 88% / 59% / 40% | 86% / 54% / 35% |
| SRAM leakage at which residency loses total memory power at 5 / 25 / 100 tok/s continuous | 0.14 / 0.65 / 2.6 mW/MB | 0.14 / 0.65 / 2.6 | 0.14 / 0.65 / 2.6 |
| SRAM total pJ/bit at which the dynamic advantage vanishes | 4.0 pJ/bit | 4.0 | 4.0 |
| Local-wire pJ/bit that erases the dynamic advantage | 3.8 pJ/bit (~25 mm at 0.15 pJ/bit/mm) | 3.8 | 3.8 |
| LPDDR pJ/bit at which streaming wins on dynamic energy | 0.44 pJ/bit | 0.44 | 0.44 |

The planning leakage (0.60 mW/MB) sits just below the 25 tok/s threshold (0.65) and well above the 5 tok/s threshold (0.14): the whole Gate 4B verdict lives inside the unmeasured 0.05–1.0 mW/MB range.

### R7. Latency (INFERRED from T1/T3 timings and A8)

First word: LPDDR5X ~90 ns SoC-loaded (~38 ns device) versus 0.82 ns SRAM macro access. Bytes in flight to hide 90 ns at 17.07 GB/s: 1,536 B, inside the 4 KB prefetch, so the engine never waits after the first access and access latency does not differentiate the systems. Steady state is a bandwidth statement: one x16 channel at 70% utilization floors Falcon-E 1B at 33.2 ms per token (~30 tok/s), Falcon-E 3B at 61.1 ms (~16 tok/s), BitNet at 43.6 ms (~23 tok/s); above those rates streaming needs more channels (4 for 1B at 100 tok/s), each adding background power. The resident system at 8 TB/s aggregate bank bandwidth (EXP-0006 class) is never bandwidth-bound in this range.

### R8. Cold start (INFERRED, A10)

Loading the weights once from external storage at 1–4 GB/s takes 0.40–0.10 s (1B), 0.73–0.18 s (3B), 0.52–0.13 s (BitNet) and costs 13 / 24 / 17 mJ one-time, about 1.04 tokens of LPDDR streaming energy; negligible against any session longer than a few tokens, but it adds to the first-token latency of a cold device and the resident capacity must be re-filled after any loss of retention.

### R9. Head and KV external in both systems, planning dynamic energy, leakage excluded (INFERRED)

| Model | Context | Case | Resident total mJ/token | Streaming total mJ/token | Weight ratio | Total ratio | Head+KV share of resident total |
|---|---:|---|---:|---:|---:|---:|---:|
| Falcon-E 1B | 2k | BF16 head + BF16 KV | 7.3 | 18.6 | 9.1× | 2.55× | 81% |
| Falcon-E 1B | 2k | EXP-0004 candidate (4.156-bit head, 4.5-bit KV) | 3.0 | 14.3 | 9.1× | 4.81× | 53% |
| Falcon-E 1B | 8k | BF16 | 12.1 | 23.4 | 9.1× | 1.93× | 89% |
| Falcon-E 1B | 8k | candidate | 4.3 | 15.6 | 9.1× | 3.61× | 68% |
| Falcon-E 1B | 32k | BF16 / candidate | 31.5 / 9.8 | 42.7 / 21.0 | 9.1× | 1.36× / 2.16× | 96% / 86% |
| Falcon-E 3B | 2k | BF16 / candidate | 9.0 / 4.3 | 29.8 / 25.1 | 9.1× | 3.31× / 5.84× | 72% / 40% |
| BitNet 2B4T | 2k | BF16 / candidate | 27.9 / 8.7 | 42.7 / 23.5 | 9.1× | 1.53× / 2.70× | 93% / 79% |

With the weights resident, the external head and KV are the majority of resident-system memory energy in every cell; the edge form's total advantage is 1.4–3.3× with BF16 head/KV and 2.2–5.8× with the EXP-0004 candidate (EXP-0003/0005/0006 reconnection).

### R10. EXP-0006 reconnection (INFERRED)

| Model | EXP-0006 resident at 1.6 pJ/bit | EXP-0006 LPDDR at 5–15 pJ/bit | EXP-0006 ratio | EXP-0008 resident L/P/H | EXP-0008 LPDDR L/P/H | EXP-0008 ratio |
|---|---:|---:|---:|---:|---:|---:|
| Falcon-E 1B | 5.07 mJ | 15.9–47.6 mJ | 3.1–9.4× | 1.06 / 1.40 / 1.97 | 11.1 / 12.7 / 19.3 | 9.1× (5.6–18.3×) |
| Falcon-E 3B | 9.34 | 29.2–87.6 | 3.1–9.4× | 1.95 / 2.57 / 3.63 | 20.4 / 23.4 / 35.6 | 9.1× (5.6–18.3×) |
| BitNet 2B4T | 6.67 | 20.8–62.5 | 3.1–9.4× | 1.39 / 1.84 / 2.59 | 14.6 / 16.7 / 25.4 | 9.1× (5.6–18.3×) |

EXP-0006 overstated resident dynamic energy by ~3.6× (1.6 vs 0.44 pJ/bit) and overstated LPDDR by 1.25–2.5× (5–15 vs 4.0 planning); its energy-class framing stands; its omission of leakage is the material change. The EXP-0007 small-batch regime used the same 1.6 pJ/bit planning figure on the resident side and the 3.9–6 pJ/bit HBM range on the other, so its conservative movement-energy ratios at batch 1–2 move up by a similar ~3× factor on the dynamic side; its batching verdict (reuse arithmetic) does not depend on either number and is unchanged, and it never counted leakage either.

### R11. INT4 fairness (INFERRED)

At 4.5 bits per weight the same organization reads 2.25× the bytes at the same pJ/bit (ratio unchanged at 9.1×), provisions 991 MB (1B), 1.82 GB (3B), 1.30 GB (BitNet) and leaks 2.25× more (595 / 1,095 / 782 mW at planning). 3B INT4 provisioning is beyond one reticle at EXP-0006 densities. Ternary's edge advantage over INT4 residency is capacity and leakage, not pJ/bit.

## Interpretation

- **Dynamic energy (INFERRED on T1 sources):** modern SRAM at the macro-plus-bank-plus-wire boundary costs roughly 0.3–0.6 pJ/bit against 3.5–6 pJ/bit for the LPDDR5X device-plus-PHY path, so the resident weight read is roughly 10× cheaper per token (5.6–18×). The wire budget (3.8 pJ/bit before the advantage vanishes) and the width sweep show the result does not hinge on floorplan or read width.
- **Leakage (INFERRED on an ASSUMED range):** 440–811 MB of always-on SRAM at 0.6 mW/MB is 264–486 mW standing power, which exceeds the per-token dynamic saving unless the device generates continuously above ~23 tok/s or runs above ~85% duty at 25 tok/s. A retention mode that cuts idle leakage 10× (HYPOTHESIZED) moves the duty-cycle break-even to ~32–40%. The modern-node leakage per MB is unmeasured and its plausible range (0.05–1.0 mW/MB) spans the verdict.
- **Latency (INFERRED):** access latency is hidden by a 1.5 KB prefetch; what residency removes is the per-channel bandwidth ceiling (~30 / 16 / 23 tok/s per x16 channel) and the background power of every added channel.
- **Head and KV (INFERRED):** they dominate the resident system's remaining memory energy; the edge advantage with BF16 head/KV is 1.4–3.3× and needs the EXP-0004 candidate to reach 2.2–5.8×.
- **Against the prior experiments:** EXP-0006's ratio range is superseded upward on the dynamic side and qualified on the static side; EXP-0007's batching verdict is unchanged; EXP-0005's bank organization is the one modeled; EXP-0004's candidate is load-bearing for the edge total.

## Surprises

1. The resident energy is dominated by delivery (wire + registers, 0.25 of 0.44 pJ/bit at planning), not by the macro read (0.09): the macro is already cheap, the wire to the consumer is the cost.
2. Leakage is 88% of resident memory power at 25 tok/s continuous and 99% at 1 tok/s; the edge verdict is a standby-power verdict, not a read-energy verdict.
3. The LPDDR5X device figure derived from the datasheet (3.6 pJ/bit maximum, ~2.5 typical) is far below EXP-0006's 5–15 pJ/bit range; the streaming baseline got stronger.
4. The per-bit total is flat with read width because the fixed per-access cost amortizes; wide reads do not buy energy, only bandwidth.
5. A single x16 channel caps Falcon-E 3B streaming at ~16 tok/s at 70% utilization, which EXP-0006's "~120 tok/s" figure for 1B at full channel rate hid for the larger model.

## Limitations

- No modern-node measured leakage per MB was found; A4 is anchored on T3 cell-current statements and a 28 nm T1 point. The retention factors are hypotheses.
- E02's read width is inferred; E01's figure is at 0.4 V and was voltage-scaled by a physics rule.
- The bank-composition overhead (A12) and the wire distances (A5) are assumptions with no floorplan behind them.
- The LPDDR typical-device factor (0.7×) and the PHY bracket are assumptions and T3; SoC PHY idle power is not modeled, which favours streaming.
- Temperature enters only through the LOW/HIGH ends and the two self-refresh points; no measured SRAM leakage temperature coefficient.
- Memory-subsystem energy only: compute, activation movement, control and the non-ternary engines are excluded on both sides.
- Latency has no scheduler, bank-conflict or refresh-collision model.
- No RTL was changed or re-run; this is an analytical experiment on public sources.

## What This Proves

- INFERRED on T1 sources: resident SRAM weight reads at a modern node are roughly an order of magnitude cheaper per bit than LPDDR5X streaming through the SoC path (0.33–0.62 vs 3.5–6.1 pJ/bit).
- INFERRED: with leakage in the 0.05–1.0 mW/MB range, the total-memory-power break-even for a 1B–3B resident edge device is 1.7–25 tok/s continuous or 32–88% duty at 25 tok/s, i.e. the verdict is regime- and leakage-dependent.
- INFERRED: access latency is not a differentiator under prefetch; per-channel bandwidth is.

## What This Does NOT Prove

- Any chip-level power, TDP, joules per token, tokens per watt, battery life, area or cost.
- The leakage of a real modern HD SRAM macro, or that a retention mode with the assumed factors exists at that node.
- That the edge/resident form is commercially viable, or that edge is FlexTern's market.

## Gate verdicts

**Gate 4A — Dynamic energy: PASS (order of magnitude, T1-anchored).** Modern multi-bank SRAM reads at the macro + bank + wire boundary beat LPDDR5X streaming at the device + PHY boundary by roughly 10× per bit (5.6–18× across the ranges) for batch-1 edge decode of all three canonical models.

**Gate 4B — Static power survival: INCONCLUSIVE (regime-dependent; verdict spans the unmeasured range).** At the planning leakage of 0.6 mW/MB, residency wins total memory power only above ~23 tok/s continuous or ~85% duty at 25 tok/s without a retention mode (~32–40% with a HYPOTHESIZED 0.1× retention). At 0.05 mW/MB it wins almost everywhere; at 1.0 mW/MB it needs ≥ 25 tok/s continuous. Whether a modern HD macro leaks 0.05 or 1.0 mW/MB is the single number that decides this gate, and it is not published in any source found.

**Gate 4C — Latency: FAIL as a latency claim; the real difference is bandwidth.** First-word latency (90 vs 0.82 ns) is hidden by 1.5 KB of prefetch; residency's steady-state benefit is removing the per-channel bandwidth ceiling (~30 / 16 / 23 tok/s per x16 channel for 1B / 3B / BitNet) and the background power of added channels.

**Gate 4D — Edge residency: INCONCLUSIVE for hardware commitment; sufficient for continued analysis.** The dynamic advantage is real at order-of-magnitude; the total-power advantage depends on leakage and duty cycle (4B); the total-memory advantage with head and KV external is 1.4–3.3× at BF16 and 2.2–5.8× with the EXP-0004 candidate, which the founder has not adopted. No ADR selects SRAM and no decision commits FlexTern to edge.

**Q-023:** narrowed, not closed. The dynamic read-energy half is answered with T1 evidence; the leakage/retention half is open and now the load-bearing uncertainty.

## Artifacts / Logs

`results/sram_validation/`; `analysis/sram_validation/`; `tests/test_sram_validation.py`; review `docs/reviews/REV-2026-10-07-exp-0008-completion-review.md`. The EXP-0007 package's `sources.py` was renamed `dc_sources.py` in this commit to remove a module-name collision with `memory_residency/sources.py` when the suites run in one process; its outputs are unchanged (the regenerated `results/datacenter_gate/` differs only in the commit stamp).

## Follow-up Questions

- What does a modern (5–7 nm) HD SRAM macro leak per MB at 0.75 V and at a retention voltage, across 25–85 °C, and how fast does it wake? (The A4 range spans the Gate 4B verdict.)
- What duty cycle and token rate does the founder's target edge product actually run at? (The break-even tables answer the gate once that is known.)
- Does the EXP-0004 candidate head/KV become the baseline for the edge form? (Without it the total advantage is 1.4–3.3×.)
- Can per-macro retention between accesses (wake-up ≪ token period) be built at a modern node, and at what cost?
