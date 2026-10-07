# EXP-0006 — Model-resident memory feasibility for the Youssouf Architecture (Gate 2)

Status: Complete (awaiting founder/ChatGPT review)
Date: 2026-10-07
Commit: 32b954b ("research: evaluate model-resident memory feasibility (EXP-0006)"); results generated from that tree.

Primary data: `results/memory_residency/` (`summary.md`, `sources.csv`, `assumptions.csv`, `source_notes.md`, `capacity_comparison.csv`, `model_scaling.csv`, `residency_policies.csv`, `bank_sweep.csv`, `bandwidth_requirements.csv`, `streaming_limits.csv`, `area_ranges.csv`, `technology_comparison.csv`, `energy_ranges.csv`, `package_options.csv`, `economics_proxy.csv`, `int4_comparison.csv`, `system_reconnect.csv`, `sensitivity.csv`, `primary_comparison.csv`, `environment.json`). Labels as in `AGENTS.md` §7. **This is a research and analytical-modeling experiment on public technology evidence. No FlexTern silicon exists; nothing here is an ASIC PPA figure, a sustained bandwidth, a chip energy per token, or a product cost.** Decimal units (1 MB = 10⁶ B). Gate 2A and Gate 2B are scored separately at the end.

## Research Question

Gate 2A: is there a technically credible memory organization, tied to public technology evidence, that holds roughly 400–730 MB of packed ternary weights near compute with enough local bandwidth for the EXP-0005 bank-local dataflow, with manageable activation/result traffic and a scaling path?

Gate 2B: does 2-bit ternary materially improve that feasibility versus the same architecture holding INT4 weights (capacity, local bandwidth, bank count, memory tier), beyond the trivial 2× byte ratio?

## Hypothesis

The "model-resident inference" North Star (static state stays close to its consumer; compressed representations are consumed natively) survives a physical-memory sanity check at the 1B–3B scale on public SRAM-heavy die evidence, and ternary's halving of capacity crosses a real tier boundary (one reticle-limited die) that INT4 does not. The experiment was run to try to falsify this.

## Why It Matters

EXP-0005's strongest result (ternary weight stream removed from the global path) is conditional on whole-model residency. If no credible memory organization can hold 396–730 MB near compute, the Youssouf Architecture's advantage reduces to a datapath-size and bandwidth-multiplier story on a conventional streaming design.

## Setup

**Inputs (MEASURED, loaded programmatically):** the three canonical profiles through `roofline_models.Workload`; the EXP-0005 traffic levels (`hardware_gate/traffic_model.per_token`) and collector-bound cycle law (`hardware_gate/workload_mapping.nearmem_cycles`), imported, not re-derived; the EXP-0004 candidate head (4.156 bits/element) and KV (4.5 bits/element) as context.

**External evidence (22 sources, `sources.csv`):** TSMC N5/N3E SRAM bit cell 0.021 µm² and N3/N3E reported macro density 31.8–33.5 Mb/mm² (S01, S02); AMD 3D V-Cache 64 MiB on 41 mm² (gen 1, ISSCC 2022) and ~36 mm² (gen 2), 2–2.5 TB/s die-to-die (S03, S04); Graphcore GC200 900 MB SRAM with 1,472 cores on 823 mm² N7 (S05); Cerebras WSE-3 44 GB on 46,225 mm² N5 (S06); Tesla D1 440 MB on 645 mm² N7 (S07); Groq TSP 230 MB on 725 mm² 14 nm (S08); d-Matrix Corsair 2 GB SRAM across 8 N6 chiplets plus 256 GB LPDDR5 (S09); Intel 22 nm eDRAM 128 MiB on 77 mm² (S10); IBM eDRAM (S11, S22); HBM3E 24–36 GB and >1.2 TB/s per stack (S12); LPDDR5X 8533 Mb/s/pin (S13); HBM2 3.9 pJ/bit and GDDR5 14 pJ/bit modeled (S14); Horowitz ISSCC 2014 DRAM 1–2 nJ per access, >20 pJ/bit DRAM I/O, 10 pJ/bit with improved I/O, 10–20 pJ per cache access at 45 nm (S15); hybrid-bond I/O <0.05 pJ/bit vendor claim (S16); 3.4 TB/s/mm² bonding bandwidth density and SoIC pitch roadmap (S17); Samsung HBM-PIM (S18), SK hynix GDDR6-AiM (S19), UPMEM PIM ~1 GB/s per 64 MB bank (S20); EUV field 26 × 33 mm = 858 mm² (S21). Eight explicit assumptions (A1–A8) in `assumptions.csv`: density tiers, 5–30% usable-capacity margins (10% nominal), energy ranges, 50/70/80% bandwidth utilization (70% nominal), hypothetical model shapes, technology availability, unmeasured compute area, EXP-0005 collector/broadcast constraints.

**Density tiers (never mixed, A1):** raw bit cell 5.95 MB/mm² (not a chip density); N3E macro 3.8–4.2 (array plus local periphery); cache-only die 1.64–1.86 (V-Cache, includes TSVs, no compute); **whole SRAM-heavy die 0.68–1.09 MB/mm² (Dojo to Graphcore, N7, includes the compute)**, the tier a FlexTern die with Y-Core banks is assumed to land in; wafer-scale 0.95 (WSE-3, N5); eDRAM die 1.74 (22 nm, 2014).

**Memory classes studied:** (1) large on-die SRAM, (2) eDRAM, (3) 3D-stacked SRAM cache dies, (4) HBM with base-die/PIM compute, (5) DRAM-PIM DIMMs, (6) SRAM chiplets plus fabric, (7) hybrid hot-set residency; (8) external streaming as the baseline. Analog compute-in-memory is out of scope and not compared.

## Hardware / Software Environment

Pure Python 3.12 on the Windows host; no GPU, no FPGA, no synthesis. `analysis/memory_residency/` (`sources.py`, `capacity.py`, `residency.py`, `physical.py`, `report.py`), `tests/test_memory_residency.py` (17 tests). Runs in seconds.

## Baseline

Ternary streaming from external DRAM (LPDDR5X or HBM3E class) at the same target token rates, and INT4 in the same residency architecture with identical bank organization, activation precision, policy, token rate and utilization (the mandatory fairness comparison).

## Method

1. Verify remote main (3711041); read the canonical documents, EXP-0001..0005, the EXP-0005 handoff and review, and all result directories.
2. Collect sources by web search of vendor documents, conference reports and standards; record every physical constant with provenance and label; derive nothing from paywalled text.
3. Capacity for 2/4/8/16-bit weights, head separate (BF16 measured and EXP-0004 candidate), metadata, margins; hypothetical 7B/13B/30B/70B capacity scaling (A5).
4. Residency policies (full, layer-at-a-time, tiled 1–8 layers, hot set 25/50/75%) with preload, global and local bytes per token; bank sweep 16–256 with per-bank capacity (tail waste included), local bytes, activation buffer, delivered activations, result groups, collector-bound cycles and utilization.
5. Bandwidth at 10–250 tok/s with 50/70/80% utilization for streaming (external) and resident (local aggregate, per bank, activation delivery, global activation/output); streaming saturation points per interface class.
6. Area ranges per tier; tier crossings against the 858 mm² reticle, 41 mm² cache dies, 900 MB-class dies and 36 GB HBM stacks; energy ranges per movement; package organizations P1–P7; economics proxies; sensitivity thresholds; EXP-0003/0004/0005 reconnection at contexts 128/2k/8k/32k for BF16 and candidate head/KV.
7. Independent review on twenty attack points; record; docs.

## Metrics

Bytes (capacity, per-token traffic), GB/s (required, peak × utilization), mm² (ranges per tier), mJ per token per movement (ranges), die/stack counts, utilization, tier class, thresholds.

## Results

All ANALYTICAL from MEASURED counts unless labeled INFERRED (areas, energies) or HYPOTHESIZED (larger models). Falcon-E 1B figures quoted; the other models are in the tables.

### R1. Capacity (MEASURED counts)

| Model | 2-bit transformer | 4-bit | 8-bit | BF16 | Head BF16 | Head 4.156 b | Metadata | Physical at 10% margin (2-bit / 4-bit) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Falcon-E 1B | 396.4 MB | 792.7 | 1,585.4 | 3,170.9 | 134.2 MB (untied) | 34.9 MB | 0.2 MB | 440.4 / 880.8 MB |
| Falcon-E 3B | 729.8 MB | 1,459.6 | 2,919.2 | 5,838.5 | 134.2 MB (untied) | 34.9 MB | 0.3 MB | 810.9 / 1,621.8 MB |
| BitNet 2B4T | 521.0 MB | 1,042.0 | 2,084.0 | 4,168.1 | 656.7 MB (tied) | 170.6 MB | 0.9 MB | 578.9 / 1,157.8 MB |

2-bit totals equal EXP-0005's (`test_1_2`). Hypothetical (HYPOTHESIZED, 95% ternary fraction, no quality claim): 7B 1,662 MB, 13B 3,088 MB, 30B 7,125 MB, 70B 16,625 MB at 2 bits; double at 4 bits.

### R2. Residency policies, Falcon-E 1B, per token

| Policy | Resident capacity | Global weight bytes | Local weight bytes | Global total | Reduction | EXP-0005 advantage surviving |
|---|---:|---:|---:|---:|---:|---|
| streaming | 0 | 396.4 MB | 0 | 399.1 MB | 1.0× | none |
| full residency, steady state (EXP-0005 convention) | 396.4 MB | 0 | 396.4 MB | 2.8 MB | 144× | all; local reads remain |
| full residency, preload amortized over 1,000 tokens (A10) | 396.4 MB | 0.4 MB | 396.4 MB | 3.2 MB | 126× | all; local reads remain |
| layer-at-a-time | 16.5 MB | 396.4 MB | 396.4 MB | 399.1 MB | 1.0× | none (reload = streaming, plus the local read) |
| tiled 2/4/8 layers | 33–132 MB | 396.4 MB | 396.4 MB | 399.1 MB | 1.0× | none; only reload granularity changes |
| hot set 25 / 50 / 75% | 99 / 198 / 297 MB | 297 / 198 / 99 MB | 99 / 198 / 297 MB | 300 / 201 / 102 MB | 1.3 / 2.0 / 3.9× | proportional to the resident fraction |

The claim "whole-model residency is required" holds for the *full* benefit; the benefit is linear in the resident fraction, and any policy that reloads weights each token is streaming with an extra local read (`test_7_8`).

### R3. Bank organization, 2-bit, 16 B per bank per cycle (collector law MEASURED in EXP-0005)

| Model | Banks | Capacity/bank | Tail waste | Activations delivered/token | Result groups/token | Cycles/token | Utilization | Collector-bound |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Falcon-E 1B | 16 | 24.8 MB | 0 | 5.9 MB | 150,528 | 1,548,288 | 1.00 | no |
| Falcon-E 1B | 64 | 6.2 MB | 0 | 23.6 MB | 150,528 | 387,072 | 1.00 | no |
| Falcon-E 1B | 128 | 3.15 MB | 1.6% | 47.2 MB | 150,528 | 196,608 | 0.98 | no |
| Falcon-E 1B | 256 | 1.62 MB | 4.8% | 94.4 MB | 150,528 | 168,960 | 0.57 | yes |
| Falcon-E 3B | 64 / 256 | 11.4 / 2.95 MB | 0 / 3.5% | 39.8 / 159.4 MB | 266,240 | 712,704 / 307,200 | 1.00 / 0.58 | no / yes |
| BitNet 2B4T | 64 / 256 | 8.2 / 2.3 MB | 0.9 / 13.6% | 28.0 / 112.1 MB | 170,880 | 513,600 / 190,560 | 0.99 / 0.67 | no / yes |

Activation buffers: K × 8 bits per bank (13 KB at K = 13312). Beyond 128 banks the one-group-per-cycle collector, not capacity, limits the organization; delivered activation bytes reach 24% of the weight bytes at 256 banks.

### R4. Bandwidth at TARGET token rates (70% utilization), 2-bit

| Model | tok/s | Streaming external GB/s | LPDDR5X 64-bit channels | HBM3E stacks | Resident local aggregate GB/s | Per bank at 64 banks | Activation delivery at 64 banks | Global act+out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Falcon-E 1B | 10 | 5.7 | 1 | 1 | 5.7 | 0.09 | 0.34 | 0.04 |
| Falcon-E 1B | 100 | 57.0 | 1 | 1 | 56.6 | 0.89 | 3.4 | 0.40 |
| Falcon-E 1B | 250 | 142.6 | 3 | 1 | 141.6 | 2.2 | 8.4 | 0.99 |
| Falcon-E 3B | 100 | 105.0 | 2 | 1 | 104.3 | 1.6 | 5.7 | 0.70 |
| BitNet 2B4T | 100 | 74.9 | 2 | 1 | 74.4 | 1.2 | 4.0 | 0.45 |

Streaming saturation (70% utilization, batch 1): one 64-bit LPDDR5X channel sustains 120 / 65 / 91 tok/s (1B / 3B / BitNet, 2-bit) and half that at 4 bits; one HBM3E stack sustains 2,100 / 1,140 / 1,600 tok/s. Resident per-bank requirements (0.1–4 GB/s) are two to four orders of magnitude below V-Cache-class die bandwidth (2 TB/s per 64 MB die, S03). UPMEM-class DRAM-PIM (~1 GB/s per 64 MB bank, S20) fits only if the weights are spread thinly: at the capacity-minimal 7 banks each must deliver 8.1 GB/s at 100 tok/s, eight times what a UPMEM bank provides; at 64 banks (10% full each) 0.9 GB/s is within reach.

### R5. Area and tier crossings (INFERRED, 10% margin)

| Model | Bits | Physical | Chip tier 0.68–1.09 MB/mm² | Cache-die tier 1.64–1.86 | V-Cache dies | Organization class |
|---|---:|---:|---:|---:|---:|---|
| Falcon-E 1B | 2 | 440 MB | 404–648 mm² | 237–269 | 7 | one monolithic SRAM-heavy die |
| Falcon-E 1B | 4 | 881 MB | 808–1,295 | 474–537 | 14 | one reticle only at the top of the density range (marginal) |
| Falcon-E 3B | 2 | 811 MB | 744–1,192 | 436–494 | 13 | one reticle only at the top of the density range (marginal) |
| Falcon-E 3B | 4 | 1,622 MB | 1,488–2,385 | 872–989 | 25 | multi-die SRAM required |
| BitNet 2B4T | 2 | 579 MB | 531–851 | 311–353 | 9 | one monolithic SRAM-heavy die |
| BitNet 2B4T | 4 | 1,158 MB | 1,062–1,703 | 622–706 | 18 | exceeds one reticle at any whole-die density |
| hyp-7B | 2 | 1,847 MB | 1,695–2,717 | 993–1,126 | 28 | multi-die SRAM required |
| hyp-13B | 2 | 3,431 MB | 3,147–5,045 | 1,844–2,092 | 52 | multi-die SRAM required |
| hyp-30B / 70B | 2 | 7.9 / 18.5 GB | 7,263+ / 17,000+ | — | 118 / 276 | wafer-scale or DRAM-class |

Reticle limit 858 mm² (S21). Graphcore demonstrates 900 MB of SRAM with compute on 823 mm² at N7 (S05), so a 440 MB SRAM-plus-banks die is inside the demonstrated envelope, not an extrapolation. The eDRAM tier (1.74 MB/mm² at 22 nm in 2014) would place the 1B model at ~253 mm², but no leading-edge eDRAM offering was found (A6).

### R6. Energy per movement (INFERRED ranges, A3; never a chip J/token)

| Falcon-E 1B, per token | LPDDR5X streaming (5–15 pJ/bit) | HBM-class streaming (3.9–6) | SRAM-local resident (0.16–1.6, 45 nm anchor) | Act+out die crossing (≤0.3) | Resident inside an HBM stack (3.9–6) |
|---|---:|---:|---:|---:|---:|
| 2-bit | 16–48 mJ | 12–19 mJ | 0.5–5.1 mJ | 0.007 mJ | 12–19 mJ |
| 4-bit | 32–95 mJ | 25–38 mJ | 1.0–10.1 mJ | 0.007 mJ | 25–38 mJ |

The SRAM low end (0.16 pJ/bit) is Horowitz's 8 KB array; the resident banks here are 3–25 MB, at or above the 1 MB end of that anchor, so the planning figure is the TOP of the range: ~1.6 pJ/bit, 5.1 mJ/token for Falcon-E 1B. At 100 tok/s the 2-bit weight movement alone is then 1.6–4.8 W from LPDDR5X, 1.2–1.9 W from HBM-class and ~0.5 W from on-die SRAM: a 2.4–9.4× gap, 7–43 mJ/token, resting entirely on a 45 nm figure for the SRAM side. Placing weights inside an HBM stack with base-die compute does not change the weight-read energy class (it is still a DRAM read), only the fabric hop.

### R7. Package / 3D / chiplet organizations (ANALYTICAL)

| Organization | Weights | Y-Cores | Crosses a die boundary | Availability | Falcon-E 1B fit |
|---|---|---|---|---|---|
| P1 monolithic SRAM + banks (Graphcore/Dojo class) | on die | on die | nothing | commercial (S05, S07) | yes, 404–648 mm² |
| P2 compute die + hybrid-bonded SRAM cache dies (V-Cache class) | stacked dies | base die | every weight byte (2–2.5 TB/s per die) | one die commercial; 7-die SRAM stacks not shipping (A6) | marginal |
| P3 SRAM chiplets + fabric (d-Matrix class) | per chiplet | per chiplet | activations broadcast, results collected | commercial (S09) | yes, 2 × 256 MB-class |
| P4 HBM stack with base-die/PIM compute | DRAM dies | base die / banks | every weight byte, DRAM-to-logic | research prototype (S18) | capacity trivial; energy class unchanged |
| P5 DRAM-PIM DIMM (UPMEM class) | 64 MB DRAM banks | DRAM-process logic | weights local, ~1 GB/s per bank | commercial but not a ternary engine | 7 banks; rate-limited |
| P6 hybrid hot set + streamed remainder | split | on die | the streamed fraction | commercial parts | proportional benefit |
| P7 external streaming | DRAM | on die | every weight byte | commercial | baseline |

### R8. INT4 fairness (same architecture, same assumptions)

| Model | Ternary → INT4 physical | V-Cache dies | Class ternary → INT4 | Tier crossing |
|---|---:|---:|---|---|
| Falcon-E 1B | 440 → 881 MB | 7 → 14 | one monolithic die → marginal (fits only at the top of the density range) | yes |
| Falcon-E 3B | 811 → 1,622 MB | 13 → 25 | marginal → multi-die SRAM | yes |
| BitNet 2B4T | 579 → 1,158 MB | 9 → 18 | one monolithic die → exceeds one reticle at any whole-die density | yes |

Local bandwidth and local-read energy halve with ternary; compute logic is 2.7× smaller per term (EXP-0005); the near-memory global-traffic benefit is identical for both formats (EXP-0005). At ≥7B both formats need multi-die or DRAM-class capacity: no tier crossing.

### R9. System reconnection (ANALYTICAL, 100 GB/s model, 100 tok/s), Falcon-E 1B

| Context | Head/KV | Global streaming | Global resident | Local resident | Head | KV | Act+out | Dominant global term after residency | Head+KV share |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 128 | BF16 | 534 MB (53 GB/s) | 140 MB (14 GB/s) | 396 MB (40 GB/s) | 134 MB | 3 MB | 2.8 MB | head | 0.98 |
| 2,048 | BF16 | 581 MB (58) | 187 MB (19) | 396 MB (40) | 134 MB | 50 MB | 2.8 MB | head | 0.985 |
| 2,048 | candidate | 445 MB (45) | 52 MB (5.2) | 396 MB (40) | 35 MB | 14 MB | 2.8 MB | head | 0.95 |
| 32,768 | BF16 | 1,336 MB (134) | 942 MB (94) | 396 MB (40) | 134 MB | 805 MB | 2.8 MB | KV | 0.997 |
| 32,768 | candidate | 658 MB (66) | 264 MB (26) | 396 MB (40) | 35 MB | 227 MB | 2.8 MB | KV | 0.99 |

After residency the head and KV cache are 88–99.7% of the remaining global bytes for every model and context; BitNet's tied 657 MB BF16 head alone exceeds its ternary weights.

### R10. Sensitivity thresholds

- Minimum chip-level density for one reticle: Falcon-E 1B 2-bit 0.49–0.66 MB/mm² (5–30% margin), inside the 0.68–1.09 chip tier at every margin; INT4 1B 0.97–1.32, only at the top of the tier below 20% margin; ternary 3B 0.90–1.22, top of the tier; INT4 3B 1.79–2.43, above the tier at every margin and above the cache-die tier beyond 5% margin.
- Largest ternary-fraction parameter count on one reticle at the chip tier: 2.1–3.4 B at 2 bits (1.05–1.68 B at 4 bits); at the cache-die tier 5.1–5.7 B (2.5–2.9 B at 4 bits).
- Minimum per-bank local bandwidth for 100 tok/s: 0.9 GB/s at 64 banks, 0.2 GB/s at 256 (1B, 2-bit).
- Maximum bank count before the collector bound hits K = 2048 at 16 B per cycle: 128; utilization falls below 0.8 at 256 banks for all three models.
- Streaming saturation: 120 tok/s per 64-bit LPDDR5X channel for 1B 2-bit (60 at 4-bit), 2,100 per HBM3E stack.
- Model size where on-die residency stops at one reticle: between 3B (marginal) and 7B (multi-die) for ternary; between 1B (marginal) and 3B (no) for INT4.

## Interpretation

- **Capacity is feasible at 1B–3B on demonstrated-class dies (INFERRED):** 440 MB (1B, 10% margin) is half of what Graphcore shipped with compute on one N7 die; 811 MB (3B) is at the top of that envelope. This is the public-technology envelope, not a FlexTern design.
- **Bandwidth is never the limiter at these scales (ANALYTICAL):** 100 tok/s needs 57 GB/s external for streaming (one LPDDR5X channel) or 0.9 GB/s per bank for 64 resident banks. Residency therefore buys energy and latency, not throughput. The honest statement is that external LPDDR5X streaming already sustains ~120 tok/s batch 1 on Falcon-E 1B ternary, and HBM3E thousands.
- **Energy is the real argument for residency and it rests on the weakest evidence (INFERRED):** multi-MB on-die SRAM reads at ~1.6 pJ/bit (top of a 45 nm tutorial range; the 0.16 pJ/bit low end is an 8 KB array and does not apply to 3–25 MB banks) versus 3.9–15 pJ/bit for DRAM classes. The delta is 7–43 mJ/token for Falcon-E 1B, 0.7–4.3 W at 100 tok/s, a 2.4–9.4× ratio. Whether that is worth 400–650 mm² of SRAM-heavy silicon is an economics question this experiment cannot settle. The N5 → N3E HD bit cell did not shrink (0.021 µm² both, S01/S02), so node migration should not be expected to move the chip-tier density much either.
- **DRAM-class "near-memory" does not deliver the EXP-0005 benefit (INFERRED):** inside an HBM stack the local read is still a DRAM read at HBM energy; capacity is trivially sufficient but the only compute-in-stack evidence is a research prototype with FP16 units at 300 MHz (S18). UPMEM-class PIM gives ~1 GB/s per 64 MB bank, enough for ≤100 tok/s at 64 banks but with DRAM-process logic. INCONCLUSIVE as an architecture.
- **Partial residency (ANALYTICAL):** layer-at-a-time and tiling reload the whole model every token and keep nothing of the benefit; hot-set residency keeps a proportional share. A hierarchy helps only to the extent it is permanently resident.
- **Bank organization (ANALYTICAL):** 32–128 banks of 3–25 MB each keep utilization ≥ 0.98 on the real shapes with activation delivery ≤ 12% of weight bytes; 256+ banks are collector-bound (EXP-0005) and deliver 24% of the weight bytes as activations. The organization that fits the public SRAM evidence is tens of banks of several MB, which is the granularity of a cache slice, not thousands of tiny banks.
- **Ternary versus INT4 (INFERRED):** the 2× halves capacity, local bandwidth and local-read energy and moves every canonical model across one organization class: 1B from "one die" to "marginal"; 3B from "marginal" to "multi-die"; BitNet from "one die" to "exceeds one reticle". It does not change the technology class at ≥7B, where both need multi-die SRAM (≤13B) or DRAM-class capacity (≥30B). The near-memory global-traffic benefit is the same for both.
- **What remains after residency (ANALYTICAL):** the head and the KV cache are 88–99.7% of the remaining global bytes. The non-ternary consumer (Q-022) is now the dominant external traffic at every context, and for BitNet-class models the tied BF16 head is larger than the resident ternary weights.
- **Scaling (HYPOTHESIZED):** on-die residency has a ceiling near 2–3.4 B ternary parameters per reticle at the chip tier (5–6 B at cache-die density). Beyond that the organization is SRAM chiplets (d-Matrix class, 2 GB per card commercial) up to ~13B, then DRAM-class or wafer-scale.

### Competitive-baseline discipline

Near-memory and SRAM-resident inference are not FlexTern inventions: Graphcore, Tesla, Groq, Cerebras and d-Matrix ship SRAM-resident designs (category A for the organization). What is FlexTern-specific is only the 2-bit consumer (EXP-0005) and the capacity/bandwidth/energy halving it implies against INT4 in the same organization. No competitor performance is claimed or compared; bandwidth classes are used, not products.

## Surprises

1. Bandwidth is irrelevant to Gate 2 at 1B–3B: LPDDR5X streaming already sustains ~100 tok/s, so residency is an energy/latency choice, not a throughput necessity.
2. The strongest tier crossing is at 3B, not 1B: INT4 3B exceeds one reticle at every density anchor and margin, ternary 3B sits at the top of the demonstrated envelope.
3. Putting compute inside an HBM stack does not reduce the weight-read energy class at all; it only shortens the fabric hop. "Near-memory" and "resident in SRAM" are different claims.
4. The hypothetical scaling ceiling is low: one reticle holds 2–3.4 B ternary parameters at demonstrated chip-level density. Model-resident inference on a single die is a small-model architecture unless wafer-scale or multi-die.
5. The head and KV cache, not the ternary weights, dominate external traffic once the weights are resident; for BitNet the tied head (657 MB BF16) is larger than the ternary weights (521 MB).

## Limitations

- Densities are inferred from six public dies on two nodes (N7, N5) plus TSMC macro figures; none is a FlexTern design; compute area for Y-Core banks is unmeasured (A7) and is assumed to be absorbed in the "whole SRAM-heavy die" tier.
- Energy figures are ranges from a 45 nm tutorial (SRAM) and modeled DRAM papers; no modern-node SRAM pJ/bit measurement was available; nothing is summed into a chip J/token.
- Vendor bandwidths are peak; utilization is an assumption (50–80%).
- eDRAM availability at leading nodes and the absence of multi-die SRAM stacks are assumptions from absence of evidence (A6).
- Hypothetical models are capacity-only with an assumed 95% ternary fraction; no quality claim.
- Latency is not modeled; the residency benefit on time-to-first-token is unquantified.
- No cost model: economics are proxies (area, die count, stacks, package class).
- Residency is of the ternary weights only. The head (134 MB BF16 untied; 657 MB tied for BitNet) and the KV cache (805 MB BF16 at 32k context for Falcon-E 1B, 227 MB at the EXP-0004 candidate bits) are outside the resident budget and stay external; at long context the KV cache alone exceeds the resident footprint.
- The full-residency row that amortizes preload uses 1,000 tokens per preload (A10); EXP-0005's 144× figure is the steady-state convention, reported alongside.
- Source quality: six primary documents (S03, S09, S12, S13, S14, S15); the other sixteen rows are technical-press or WikiChip/SemiWiki summaries of primary disclosures, adequate for order-of-magnitude use only.

## What This Proves

- ANALYTICAL on MEASURED counts: the canonical models need 440 / 811 / 579 MB of physical capacity at 2 bits with 10% margin, 2× at INT4; local bandwidth per bank is 0.1–4 GB/s at 10–250 tok/s with 16–256 banks; activation delivery stays ≤ 12% of weight bytes up to 128 banks; layer and tiled residency keep none of EXP-0005's advantage, hot-set residency keeps a proportional share.
- INFERRED from public evidence: 440 MB of SRAM plus compute on one reticle-limited N7-class die is inside the demonstrated commercial envelope (Graphcore 900 MB / 823 mm²); 811 MB is at its top; INT4 at 3B is outside it at every anchor; ≥7B ternary is outside it.
- ANALYTICAL: after residency, the head and KV cache are 88–99.7% of the remaining global bytes.

## What This Does NOT Prove

- Any FlexTern area, frequency, power, energy per token, sustained bandwidth or cost. No design exists.
- That SRAM-resident inference is economical against LPDDR5X streaming, which already meets 100 tok/s at these model sizes; the energy delta rests on a 45 nm SRAM anchor.
- That compute inside DRAM stacks is a credible architecture for this dataflow (one research prototype, different precision).
- Anything about models above 3B beyond capacity arithmetic.

## Gate verdicts

**Gate 2A — PASS for Falcon-E 1B, PASS (marginal) for Falcon-E 3B and for BitNet 2B4T, at the published-technology envelope, for the ternary weights only:** a monolithic SRAM-heavy die of the Graphcore/Dojo class (N7, commercial) holds 440 MB comfortably (404–648 mm²), 811 MB (3B) and 579 MB (BitNet, 531–851 mm², within 7 mm² of the reticle at the low-density end) at the top of the demonstrated envelope; local bandwidth exceeds the requirement by orders of magnitude; activation/result traffic is 1–12% of weight bytes at 16–128 banks. The head (BitNet's tied 657 MB BF16 head is larger than its resident weights) and the KV cache (805 MB at 32k context for Falcon-E 1B) are outside the resident budget and remain external; cost, energy at modern nodes and product economics remain UNMEASURED. **FAIL for on-die residency at ≥7B** (multi-die or DRAM-class required). **INCONCLUSIVE for DRAM-class near-memory (HBM base-die / PIM):** capacity is trivial but public compute-in-stack evidence is one research prototype, and the weight-read energy class is unchanged.

**Gate 2B — PASS at the structural level for the 1B–3B class:** 2-bit ternary keeps every canonical model inside one reticle-limited die at demonstrated densities (1B comfortably, 3B at the top of the envelope) where INT4 does not (INT4 3B exceeds one reticle at every whole-die anchor and margin, and at the cache-only-die anchor fits only with ≤ 5% margin and no compute; INT4 1B fits only at the top of the whole-die density range), and halves local bandwidth, local-read energy and bank capacity in the same organization. **FAIL as a technology-class change at ≥7B** (both formats need multi-die SRAM or DRAM-class capacity), and the honest caveat that LPDDR5X streaming makes residency optional for throughput at these sizes, so the 2× buys energy and silicon, not feasibility of serving.

## Artifacts / Logs

`results/memory_residency/` as listed above; `analysis/memory_residency/`; `tests/test_memory_residency.py`.

## Follow-up Questions

- What is the measured read energy per bit of a multi-MB SRAM bank at N7/N5 (the single number the residency energy argument rests on)? Public ISSCC macro papers may hold it.
- Does a hierarchical collector (EXP-0005) let 256+ banks keep utilization, and what does it cost? Only relevant if the founder chooses the resident direction.
- How is the head consumed once the weights are resident (Q-022)? For BitNet-class models the tied BF16 head is the largest static object; the EXP-0004 candidate 4-bit head is the first thing a resident design would want.
- Would an eDRAM-class tier (if any foundry offers one at an advanced node) or a V-Cache-class multi-die SRAM stack change the 3B and 7B crossings?
- Latency: what does residency do to time-to-first-token and batch-1 decode latency, independent of bandwidth?
