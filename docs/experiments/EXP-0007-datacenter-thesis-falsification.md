# EXP-0007 — Datacenter thesis falsification: batching, HBM, KV and rack-scale reality (Gate 3)

Status: Complete (awaiting founder/ChatGPT review)
Date: 2026-10-07
Commit: the EXP-0007 commit ("research: stress-test FlexTern datacenter inference thesis"); results generated from that tree.

Primary data: `results/datacenter_gate/` (`summary.md`, `sources.csv`, `assumptions.csv`, `scenario_matrix.csv`, `batch_sweep.csv`, `context_sweep.csv`, `kv_sweep.csv`, `head_sweep.csv`, `model_scale_sweep.csv`, `topology_sweep.csv`, `pipeline_results.csv`, `pipeline_mode_sweep.csv`, `interconnect_results.csv`, `capacity_comparison.csv`, `traffic_comparison.csv`, `crossover_thresholds.csv`, `energy_ranges.csv`, `economic_proxies.csv`, `primary_comparison.csv`, `mlperf_anchor.json`, `environment.json`). Labels as in `AGENTS.md` §7. **Everything here is analytical: throughputs are bounds from peak × utilization, energies are movement-only INFERRED ranges (compute excluded), 7B/70B/400B properties are HYPOTHESIZED shapes with no quality claim. No tokens/MW is claimed.** Decimal units. Three gates are scored separately at the end.

## Research Question

Does a distributed 2-bit ternary model-resident inference system retain a material system-level opportunity against a strong conventional FP4-class HBM accelerator when both get fair datacenter-serving assumptions (batching, continuous serving, native low-bit KV, tensor and pipeline parallelism, current interconnect), across 1B–400B, batch 1–256, context 128–131k, prefill and decode?

## Hypothesis

The datacenter form of the thesis ("model-resident datacenter inference": static weights spatially resident across memory-compute tiles, activations flowing through) keeps a multi-fold movement-energy advantage after batching. The experiment was built to falsify this.

## Why It Matters

Every FlexTern result so far is batch 1 against a BF16 reference. A conventional accelerator amortizes weight traffic across concurrent requests; if that amortization erases the resident advantage, the datacenter ambition is not supported and the project's remaining opportunity is the edge/resident 1B–3B form of EXP-0006.

## Setup

**Systems.** A: conventional node of 8 B200-class GPUs (D01; MI355X cross-checks D02): 180 GB HBM3e and 8 TB/s per GPU, FP4 9 PFLOPS dense, NVLink 0.9 TB/s per direction, tensor-parallel within the node (weights read once per step for the whole batch, two all-reduces per layer), pipeline-parallel across nodes when capacity requires, NVFP4 weights charged at 4.5 bits (D13), native FP8 KV (D03; NVFP4 KV as a sensitivity), continuous batching granted (D12). B: ternary streaming on the same node with 2-bit weights consumed natively. C: ternary resident SRAM tiles (256 MB–2 GB, 10% margin; local read 8 TB/s per tile; a per-tile HBM3E-class DRAM holds KV when it does not fit the SRAM; tile links 0.9 TB/s NVLink-C2C class; TARGET compute 4,096 ternary terms per cycle per tile at 1 GHz, A6). D: INT4 resident tiles through the identical topology algorithm with 4.5-bit weights and the same term rate (no compute penalty applied, which favors D). Utilizations (A2): HBM 70%, compute 50%, links 70%; serving utilization 100/85/70/50%.

**Workloads.** Canonical Falcon-E 1B/3B from the profiles (MEASURED shapes); hypothetical 7B (Llama-2 shape), 70B and 400B (Llama-3.1 shapes, D11) with all linear projections ternary, untied head, vocab 32k or 128k (A8).

**Placement (A7).** Whole layers to tiles when a layer fits; otherwise a layer is split across k tiles (tensor-parallel, all-reduce bytes counted); head on its own tiles; pipeline stages = tiles / k; pipelined mode: microbatch b = floor(B / stages) with one microbatch per stage (weight reuse = b); batch-sequential mode: one microbatch of B with reuse B and bubble 1 − 1/stages. Interconnect tier by stage count: ≤ 4 package, ≤ 16 board, else rack.

**KV.** Per sequence, never batch-shared: L × 2 × nKV × hd × context × bits/8 read per token plus one token written; capacity gated first for every system (GPUs added by capacity; resident tiles put KV in per-tile DRAM when the SRAM is full).

**Energy tiers (A4, pJ/bit).** HBM/DRAM 3.9–6; SRAM local 0.16–1.6 (1.6 planning); chip 0.05–0.3; package 1.3 (D07); board 3–6 (D08); rack 3–8 (D09). Ratios are reported conservatively (A at its low end, C at its high end) where a single number is given.

## Hardware / Software Environment

Pure Python 3.12 on the host; `analysis/datacenter_gate/{sources,workloads,systems,report}.py`; 16 tests. Runs in seconds. No hardware.

## Baseline

System A as above. MLPerf Inference v5.1 (D04) 8×B200 Llama 2 70B offline 102,725 tok/s is used only as a sanity anchor: the model's System A bound at batch 128 and ~1.3k context is 10,799 tok/s per GPU (0.84× of the measured per-GPU figure), so the baseline is not undersized.

## Method

Verify head e59e9b4; collect sources; build the four systems; run the scenario matrix (5 models × 5 batches × 4 contexts × 2 phases × 4 systems = 640 feasible cells) and targeted sweeps (batch 1–256 with serving utilizations, context 128–131k with three KV precisions, head precision and vocabulary, tile capacity, pipeline mode); derive crossover thresholds; reconcile against EXP-0003 (batch-1 bytes), EXP-0005 (local reads) and EXP-0006 (capacities) in tests; independent review; record.

## Metrics

Per sequence-token bytes by level (global weights, local weights, KV read/write, head, interconnect, partial sums); capacity (static, KV per sequence, units); throughput and latency bounds; pipeline stages, microbatch, bubble; movement-energy ranges; economic proxies (units, HBM stacks, SRAM GB, racks); crossover thresholds.

## Results

### R1. Capacity and units (ANALYTICAL; 512 MB tiles unless stated)

| Model | Static weights 2-bit / 4.5-bit | A GPUs (weights + KV for B=16 at 8k) | C tiles (stages) | D tiles (stages) | C tiles at 2 GB | D tiles at 2 GB |
|---|---:|---:|---:|---:|---:|---:|
| Falcon-E 1B | 396 MB / 892 MB | 1 | 2 (1) | 3 (2) | 2 | 2 |
| Falcon-E 3B | 730 MB / 1.64 GB | 1 | 3 (2) | 5 (4) | 2 | 2 |
| hyp-7B | 1.62 GB / 3.64 GB | 1 | 5 (4) | 9 (8) | 2 | 4 |
| hyp-70B | 17.1 GB / 38.5 GB | 1 | 42 (40) | 162 (80, k = 2) | 11 (10) | 28 (27) |
| hyp-400B | 100.4 GB / 225.9 GB | 2 | 255 (126, k = 2) | 507 (126, k = 4) | 64 (63) | 127 (126) |

KV per sequence (FP8): Falcon-E 1B 25 MB at 2k; hyp-70B 336 MB at 2k, 1.34 GB at 8k, 5.37 GB at 32k, 21.5 GB at 128k. At 70B, 64 sequences at 128k do not fit one node (capacity forces 9 GPUs); resident tiles cannot hold KV beyond one or two sequences at 2k, so C and D keep KV in per-tile DRAM in every datacenter cell.

### R2. Weight reuse (ANALYTICAL; the heart of the result)

System A reads each shard once per step for the whole batch (reuse = B × serving utilization). A resident pipeline reads each stage's weights once per microbatch (ceil(B / stages), every sequence served): at 70B with 40 stages the microbatch is 1 for B ≤ 40, 2 at 64, 4 at 128, 7 at 256. Per sequence-token: A's global weight bytes fall from 38.5 GB (B = 1) to 2.41 GB (16) to 0.60 GB (64); C's local weight bytes stay at 17.1 GB until B exceeds the stage count, then 8.6 GB at 64 and 4.3 GB at 128. The frontier between the two modes: C ties A on weight energy only when the microbatch is at least (2.0 × 1.6) / (4.5 × 3.9) ≈ 0.18 of the batch, i.e. at most about 5 microbatches in flight on a 40-stage pipeline, under 14% stage utilization. Batch-sequential mode restores reuse B (267 MB per sequence-token at B = 64) but idles 39 of 40 stages (bubble 0.975) and bounds aggregate throughput at 60 tok/s against A's 5,916. Tile capacity does not escape this: at 2 GB tiles (10 stages) the microbatch at B = 64 is 7 and C's local weight bytes are 2.4 GB per sequence-token, about 31 mJ at the planning figure plus KV, against A's 30–45 mJ.

### R3. Decode batch sweep, movement energy per sequence-token (INFERRED ranges; A ÷ system, conservative = A low ÷ system high)

| Model | ctx | B | A mJ | B mJ | C mJ | D mJ | A/C conservative | C microbatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Falcon-E 1B | 2k | 1 | 30–46 | 14–22 | 0.6–5.9 | 1.2–12 | 5.1× | 1 |
| Falcon-E 1B | 2k | 4 | 8.0–12 | 4.2–6.4 | 0.9–2.6 | 1.4–7.2 | 3.1× | 4 |
| Falcon-E 1B | 2k | 16 | 2.6–4.0 | 1.6–2.5 | 0.8–1.6 | 0.9–2.7 | 1.7× | 16 |
| Falcon-E 1B | 2k | 64 | 1.2–1.9 | 1.0–1.5 | 0.8–1.3 | 0.8–1.6 | 0.96× | 64 |
| Falcon-E 1B | 32k | 1 | 42–64 | 26–40 | 13–25 | 14–31 | 1.7× | 1 |
| Falcon-E 1B | 32k | 16 | 14–22 | 13–21 | 13–20 | 13–21 | 0.73× | 16 |
| hyp-70B | 2k | 1 | 1,230–1,893 | 563–866 | 23–231 | 61–517 | 5.3× | 1 |
| hyp-70B | 2k | 4 | 315–485 | 149–229 | 33–243 | 61–517 | 1.3× | 1 |
| hyp-70B | 2k | 16 | 87–133 | 45–69 | 33–243 | 61–517 | 0.36× | 1 |
| hyp-70B | 2k | 64 | 30–45 | 19–29 | 22–129 | 61–517 | 0.23× | 2 |
| hyp-70B | 2k | 128 | 20–31 | 15–23 | 16–73 | 61–517 | 0.27× | 4 |
| hyp-70B | 32k | 1 | 1,387–2,134 | 720–1,107 | 190–484 | 218–758 | 2.9× | 1 |
| hyp-70B | 32k | 16 | 244–375 | 202–311 | 190–484 | 218–758 | 0.50× | 1 |
| hyp-70B | 128k | 16 | 1,467–2,255 | 1,377–2,120 | 1,369–2,213 | 1,397–2,487 | 0.66× | 1 |

Serving utilization below 100% raises A's figures (at 70B, B = 64, 50% utilization: 49–75 mJ) and leaves C's unchanged until the microbatch exceeds 1; it moves the 70B crossovers by one batch step at most. Latency is not a resident advantage at the TARGET density: C is slower than A in every cell (Falcon-E 1B 0.39 vs 0.17 ms per token at batch 1; 70B 16.7 vs 7.0 ms), and an interactive deployment of A would use 8-way tensor parallelism for roughly 8× lower step time at negligible energy cost, which this capacity-minimal baseline does not even claim.

### R4. Context and KV (ANALYTICAL)

KV share of System A's per-sequence bytes at batch 16, FP8: Falcon-E 1B 30% at 2k, 63% at 8k, 87% at 32k, 97% at 128k; hyp-70B 12% at 2k, 36% at 8k, 69% at 32k, 90% at 128k. The candidate 4.5-bit KV pushes the 50% crossover from 8k to 32k for Falcon-E 1B and leaves it at 2k–8k for the 7B–70B shapes. Because C and D keep KV in DRAM, their KV bytes and energy equal A's; above the crossover the four systems converge (A/C 0.66–1.6× at 128k for every model).

### R5. Head (ANALYTICAL)

At batch 1 the 70B 128k-vocabulary head is 2.1 GB per token at BF16 (5% of A's bytes, 10% of C's), 591 MB at 4.5 bits, 546 MB at the EXP-0004 candidate; a 32k vocabulary quarters it. At batch ≥ 16 the head is below 2% everywhere. The head never decides a datacenter cell; it matters for the batch-1 interactive case and for BitNet-class tied heads (EXP-0006).

### R6. Interconnect (ANALYTICAL)

Pipeline hidden-state hops: 70B, 40 stages: 319 kB per sequence-token (8 kB per hop), 0.0% of total bytes; tensor-parallel splits add 2–8 MB per sequence-token at 400B (k = 2–4). System A's 8-way all-reduce at 128k context: 1.7 MB per sequence-token. The interconnect pJ/bit at which C's activation movement would equal its weight-energy saving is in the thousands: interconnect never decides a cell at pipeline granularity. At 1,000 aggregate tok/s and batch 16, C's hop traffic is 5 GB/s over rack-tier links.

### R7. Pipeline (ANALYTICAL; 70B, ctx 8k, System C)

| B | tile cap | stages | microbatch | in flight | bubble | t_micro | aggregate tok/s bound | per-sequence tok/s | latency ms/token | bound |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 512 MB | 40 | 1 | 1 | 0.975 | 0.42 ms | 60 | 60 | 16.7 | compute (TARGET) |
| 16 | 512 MB | 40 | 1 | 16 | 0.60 | 0.42 ms | 957 | 60 | 16.7 | compute (TARGET) |
| 64 | 512 MB | 40 | 2 | 40 | 0.0 | 0.84 ms | 2,394 | 30 | 33.4 | compute (TARGET) |
| 128 | 512 MB | 40 | 4 | 40 | 0.0 | 1.67 ms | 2,394 | 15 | 66.8 | compute (TARGET) |
| 64 | 2 GB | 10 | 7 | 10 | 0.0 | 11.7 ms | 598 | 8.5 | 117 | compute (TARGET) |

Throughput and single-sequence latency differ as they must (2,394 aggregate vs 60 per sequence at B = 64). At the TARGET density the 42-tile resident 70B is bound by its own compute at 2,394 tok/s against the 8-GPU node's HBM-bound 5,916 at B = 64 (57 tok/s per tile vs 740 per GPU); the per-tile term rate would have to rise ~13× to match, i.e. ~50k terms per cycle per tile, roughly 5 M generic cells of EXP-0005 engines per tile.

### R8. Prefill (ANALYTICAL)

Prefill is compute-bound on every system. System A (one GPU, 70B): 2k prompt at batch 1 in 64 ms, 8k prompt at batch 16 in 4.3 s, 8k at batch 64 in 17.2 s (30–32k prompt-tokens/s). Resident ternary at the TARGET density: 35 s for one 2k prompt (545× slower), 148 s for 16 × 8k (34×), 295 s for 64 × 8k (17×). Two drivers: 42 tiles × 4.1 T terms/s is about 1/13 of one GPU's dense FP4 rate at 50% utilization, and at small batch only one or a few of the 40 stages work at a time. Resident weight reads are negligible in prefill (8 MB per prompt-token); the deficit is compute density and pipeline fill. A resident fabric would need conventional prefill (hybrid E) to be usable at all; this was not modeled further.

### R9. 70B dedicated section (HYPOTHESIZED shapes)

1. Static body 17.1 GB at 2 bits (68.4 B linear parameters; 17.5 GB would be exactly 2 bits on 70.0 B), 17.7 GB with the 4.5-bit 128k head; 13.7 GB at the 1.6-bit TARGET.
2. Tiles: 256 MB 84 (+2 head), 512 MB 42, 1 GB 20, 2 GB 11 (10 stages).
3. INT4: 512 MB 162 tiles (layers split k = 2 across tiles, 80 stages with partial sums), 2 GB 28.
4. Pipeline depth 40 stages at 512 MB (2 layers per tile), 10 at 2 GB.
5. Inter-tile activation traffic 319 kB per sequence-token at 40 stages (rack tier), negligible against 17 GB of local weight reads.
6. Advantage versus A: batch 1 5.3× conservative movement energy (23–231 mJ vs 1,230–1,893); batch 2 2.7×; batch 4 1.3×; batch 16 0.36×; batch 64 0.23× (microbatch 2); batch 128 0.27× (microbatch 4). The advantage is ≥ 2× only at batch 1–2 at 2k–8k, batch 1 at 32k, never at 128k.
7. Dominant term: at 2k the weights (A) or local weight reads (C); at 32k KV is 69% of A's bytes; at 128k KV is 90% and every system converges.
8. Compressed KV matters more than resident weights above 8k context: the candidate 4.5-bit KV halves A's bytes at 32k while residency changes nothing there.
9. Tier crossing: ternary needs 42 tiles (0.33 rack at 128 per rack) where INT4 needs 162 (1.27 racks) at 512 MB; at 2 GB, 11 vs 28. Real, but the 42-tile system is still 2.5× slower and 3–8× more energy per token than one 8-GPU node at batch 64.
10. Controlling assumption: the pipelined weight-reuse arithmetic (microbatch = B/stages); second, the TARGET compute density; third, the SRAM pJ/bit.

### R10. 400B dedicated section (capacity and system only)

Static body 100.4 GB at 2 bits (402 B linear parameters), 226 GB at 4.5 bits; 2 bits saves 1 of 9 GPUs' worth of HBM capacity at batch 128, 32k (B needs 8 GPUs, A 9) and 1 of 2 at batch 16, 8k, because KV, not weights, sets the capacity. Resident: 255 tiles at 512 MB (every layer split k = 2; 126 stages; 2 racks) or 64 tiles at 2 GB (63 stages); INT4 507 tiles (k = 4; 4 racks) or 127 at 2 GB. Pipeline depth 63–126, microbatch 1 at every batch to 128, per-sequence latency 98 ms per token, aggregate bound 1,285 tok/s on 255 tiles (5 tok/s per tile) against 4,927 on 9 GPUs. Activation traffic 6 MB per sequence-token including partial sums (0.03% of bytes). Energy: 196–1,402 mJ (C) against 509–783 (A) at batch 16, 8k; 394–1,707 against 707–1,087 at 32k. 2-bit versus 4-bit halves rack count (2 vs 4) and GPU-capacity count, but HBM capacity is simpler than SRAM residency at this scale: 2–9 GPUs hold the model and its KV; 255 tiles with 255 DRAM stacks do not beat them on any metric here.

### R11. Economic proxies (PROXY; no prices)

| Model | B | ctx | System | units | HBM stacks | SRAM GB | racks (proxy) | aggregate tok/s bound | tok/s per unit |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| hyp-70B | 128 | 32k | A | 5 GPUs | 40 | 0 | 0.07 | 4,935 | 987 |
| hyp-70B | 128 | 32k | B | 5 GPUs | 40 | 0 | 0.07 | 5,084 | 1,017 |
| hyp-70B | 128 | 32k | C | 42 tiles | 42 | 21.5 | 0.33 | 2,394 | 57 |
| hyp-70B | 128 | 32k | D | 162 tiles | 162 | 83 | 1.27 | 4,787 | 30 |
| hyp-400B | 128 | 32k | A | 9 GPUs | 72 | 0 | 0.13 | 4,927 | 547 |
| hyp-400B | 128 | 32k | C | 255 tiles | 255 | 131 | 1.99 | 1,285 | 5 |

### R12. Crossover thresholds (ANALYTICAL)

- Batch at which C's conservative movement-energy advantage first falls below 4 / 3 / 2 / 1.5 / 1.2×: Falcon-E 1B at 2k: 4 / 8 / 16 / 32 / 64 (so ≥ 2× holds to batch 8); at 8k: 1 / 2 / 4 / 8 / 16 (≥ 2× to batch 2); at 32k: below 2× at batch 1. hyp-7B at 2k: 1 / 1 / 2 / 4 / 4 (≥ 2× at batch 1 only). hyp-70B and hyp-400B at 2k and 8k: 2 / 2 / 4 / 4 / 4–8 (≥ 2× to batch 2); at 32k: 1 / 1 / 2 / 4 / 4 (70B) and 2 / 2 / 4 (400B); at 128k: below 2× at batch 1 (70B) or 2 (400B).
- Context at which KV exceeds 50% of A's bytes at batch 16: Falcon-E 1B 8k (BF16, FP8) or 32k (4.5-bit); 7B and 70B 2k–8k at every precision.
- Tile count INT4 / ternary: 1.5–2.0× at 7B–400B (integer effects at 1B–3B: 1.0–1.7×).
- HBM capacity versus bandwidth: at 1,000 aggregate tok/s, batch 64, 8k, System A needs 1 GPU by bandwidth for every model to 70B and 2 for 400B; capacity, not bandwidth, sets the GPU count from 70B upward and at long context.

## Interpretation

- **Batching (INFERRED):** the conventional node's weight amortization is the decisive mechanism. At 1B–3B residency keeps its full-batch reuse (one tile, one stage) and the advantage decays smoothly: ≥ 2× conservative only to batch 8 at 2k, batch 2 at 8k, none at 32k. At 7B and above, the resident pipeline's reuse is the microbatch, so the advantage is ≥ 2× only at batch 1–2 and residency is *worse* than HBM streaming from batch 4–8 upward at every context.
- **KV (INFERRED):** above 8k context for 7B–70B and 32k for 1B, KV dominates every system equally because resident SRAM cannot hold it; compressed KV (which the baseline has natively) is worth more than resident weights there.
- **Interconnect:** not a limiter at pipeline granularity; tensor-parallel splits add partial sums but remain under 0.1% of bytes. The threat to distributed residency is not communication, it is reuse and compute density.
- **INT4 residency (INFERRED):** gets the same global-traffic benefit and loses on the same reuse arithmetic; ternary halves its tile count, local bytes and local energy, which changes rack count (0.33 vs 1.27 racks at 70B) but not the verdict against the HBM node.
- **Prefill:** resident ternary tiles at the TARGET density are 50–150× slower than the baseline per prompt. Any datacenter form would be a hybrid with conventional prefill.
- **Ternary streaming (B) is the best datacenter use of ternary in this model:** 2–3× fewer HBM bytes and ~1.4–3.4× lower movement energy than A at batch ≤ 16 and short context, converging to parity above, on identical hardware, with no residency, no tiles, no reuse penalty. It is category A for the organization and depends only on a packed consumer, which EXP-0005 built.
- **Surviving regime:** small batch at ≤ 8k context, the interactive case: batch 1–2 for 7B–400B and batch ≤ 8 for single-tile 1B at 2k, where a resident system keeps 2–5× conservative movement energy. There is no latency advantage at the TARGET density: the resident system is slower than A in every cell, and A with 8-way tensor parallelism would be faster still.

### Competitive-baseline discipline

System A was given FP4 weights, 8 TB/s HBM, native FP8 KV, tensor parallelism, continuous batching and current NVLink; its bound reproduces MLPerf within 16%. No competitor claim is made. Cerebras (16-bit 70B across multiple wafers, 2,100 tok/s per user, D05) and Groq (INT8 70B on 576 chips, D06) are the precedents for SRAM-resident serving; both pay in physical units and both serve the small-batch latency regime this model also finds. Ternary would cut their resident bytes 8× and 4× respectively; it does not change the batching arithmetic.

## Surprises

1. Pipelined residency's weight reuse is the microbatch, not the batch: a 40-stage 70B pipeline at batch 64 re-reads every weight byte for every sequence. This single fact dominates the datacenter verdict and was not visible in EXP-0005/0006's batch-1 framing.
2. Resident tiles must put KV in DRAM at any useful context, so the "resident" system pays HBM-class energy for the term that dominates above 8k.
3. The interconnect, which the brief treated as the main threat, is never the limiter at pipeline granularity.
4. Ternary streaming on conventional HBM hardware is the strongest datacenter result for ternary in the whole model.
6. Giving every sequence a seat (microbatch = ceil(B / stages)) rather than truncating changed nothing material: the microbatch is 1–2 at any batch a 40-stage pipeline would plausibly see.
5. The baseline bound landed within 16% of a measured MLPerf result without tuning.

## Limitations

- Movement-only energy; compute energy is unsourced, so the energy ratios omit the one term where ternary's 2.7× cell advantage would appear.
- The resident compute density is a TARGET; a 13× higher density would equalize 70B decode throughput but would not change reuse or KV.
- Serving utilization and link tiers are assumptions with ranges; no scheduler is simulated.
- Hypothetical shapes only; no ternary 7B–400B model exists.
- Latency bounds ignore link latency and scheduling delay; System A's GPU count is capacity-minimal, so its small-batch latency is pessimistic by up to 8× (tensor parallelism), which only strengthens the negative result.
- Resident tiles are charged one HBM3E-class stack each for KV in the economic proxies, over-provisioned for the 100 MB–few GB of KV per stage; an LPDDR-class device would serve, which lowers the proxy count but not the energy class.
- Q-023 (modern SRAM read energy) remains unresolved; the SRAM side of every ratio rests on the 45 nm anchor. The founder considers its validation important for the edge/resident path.

## What This Proves

- ANALYTICAL on sourced parameters: a tensor-parallel FP4 HBM node amortizes weight traffic across batch so that a resident ternary pipeline keeps a ≥ 2× conservative movement-energy advantage only at batch 1–2 at ≤ 8k context for 7B–400B, and to batch 8 at 2k (2 at 8k) for single-tile 1B; above that, residency moves more bytes than the baseline.
- ANALYTICAL: KV traffic dominates all systems equally beyond 8k context for 7B–70B (32k for 1B), because resident SRAM cannot hold it.
- ANALYTICAL: ternary halves resident tile count and local traffic against INT4 in the same topology (42 vs 162 tiles at 70B, 255 vs 507 at 400B) and saves 1–4 GPUs of HBM capacity against FP4 at 400B, without changing the batching verdict.
- ANALYTICAL: interconnect traffic is < 0.1% of bytes in every resident cell.

## What This Does NOT Prove

- Any achieved throughput, latency, energy per token, tokens/MW or cost on any system.
- That a resident fabric cannot be built with a compute density far above the TARGET; only that the density assumed here loses.
- Anything about model quality at 7B–400B.
- That the small-batch interactive regime is commercially sufficient.

## Gate verdicts

**Gate 3A — Batching survival: FAIL beyond small batch.** With weight reuse granted to the conventional node, the resident ternary advantage in movement energy is ≥ 2× (conservative) only at batch 1–2 for 7B–400B at ≤ 8k context (batch 1 at 32k) and to batch 8 for single-tile 1B at 2k; it is below 1× from batch 8 to 256 at 70B and below 1× from batch 4 at 7B. A PASS based on batch 1 alone is not available by the prompt's rule.

**Gate 3B — Distributed residency: INCONCLUSIVE leaning FAIL.** Distribution is architecturally feasible: 42–255 tiles, 40–126 stages, < 0.1% interconnect bytes, no capacity wall with 2-bit weights. But the pipeline's microbatch reuse and the TARGET compute density leave the distributed system with 2–4× lower aggregate throughput and higher per-token movement energy than one HBM node from batch 8 upward, and KV must live in per-tile DRAM. Residency survives as an organization, not as an advantage, at 70B–400B.

**Gate 3C — Datacenter relevance: PASS only for two narrow regimes, FAIL as a general datacenter architecture.** (1) Small-batch, short-context interactive serving (batch 1–2 for 7B–400B, up to 8 for single-tile 1B, context ≤ 8k) keeps a 2–5× conservative movement-energy advantage with no latency advantage, the regime Cerebras and Groq occupy with far higher compute density than the TARGET assumed here. (2) Ternary streaming on conventional HBM hardware keeps 2–3× fewer weight bytes at small batch with no residency at all. Outside these, the remaining advantage under fair assumptions is 0–20% and often negative. The datacenter thesis as "model-resident racks" is not supported; the edge/resident 1B–3B form (EXP-0006) and ternary-as-a-format on conventional hardware are what the evidence leaves.

## Artifacts / Logs

`results/datacenter_gate/`; `analysis/datacenter_gate/`; `tests/test_datacenter_gate.py`.

## Follow-up Questions

- What is the measured modern-node multi-MB SRAM read energy and access latency (Q-023)? Every surviving advantage is a movement-energy claim on that number.
- Does a hybrid (conventional prefill, resident small-batch decode) have a defensible niche, and at what compute density per tile?
- What does ternary-as-a-format on conventional hardware (System B) need from the software stack: a packed-weight kernel at FP4-class throughput?
- Would a KV representation that resident tiles can hold in SRAM at ≥ 8k (sub-4-bit, sparse, or shared-prefix) change the long-context cells?
