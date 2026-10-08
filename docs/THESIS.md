# FlexTern Thesis

This is a research thesis, not a pitch. Every claim is labeled using the vocabulary in `../AGENTS.md` §7 (KNOWN / MEASURED / INFERRED / HYPOTHESIZED / TARGET / SPECULATIVE).

## Working thesis

> FlexTern is exploring whether AI hardware/software designed around ternary and ultra-low-precision models can deliver materially better inference economics than conventional accelerator architectures.

Status: **HYPOTHESIZED.** No FlexTern measurement supports or refutes it yet.

## Suspected mechanism

```
smaller weight representation
            +
lower memory traffic
            +
simpler ternary arithmetic
            +
zero skipping / sparsity
            +
adaptive precision
            +
hardware/software co-design
            ↓
potentially better:
   tokens / watt
   tokens / dollar
   memory efficiency
   inference density
```

Each line in the top half is a separate hypothesis. They may not all hold, and they may not compound.

## Established facts (KNOWN)

These are external facts, not FlexTern results.

- A ternary weight has three states and needs at most 2 bits to store, versus 16 for FP16/BF16 or 8 for INT8.
- Multiplying an activation by a weight in {-1, 0, +1} reduces to negate, pass-through, or skip. No multiplier is required.
- Ternary-weight language models have been published and shown competitive quality at some scales (e.g., Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," 2024, the BitNet b1.58 line of work). Whether this holds broadly and at frontier scale is not established. A copy or pointer should be added to `/reference/`.
- In many inference workloads, especially decode, memory bandwidth rather than arithmetic throughput is the limiting resource on conventional accelerators.

## FlexTern hypotheses (HYPOTHESIZED)

- **H1.** A dedicated ternary datapath (Y-Core) is materially smaller and/or lower-power than a conventional multiply-based datapath at the same activation precision on the same target. *First evidence (EXP-0001, 2026-10-05): MEASURED under generic synthesis, 5–12× fewer logic cells than INT8 multiply depending on scale and tool version. Not yet physical, not yet power.*
- **H2.** The advantage in H1 survives, or grows, when composed into dot products and arrays, because wide ternary accumulation avoids multiplier arrays entirely. *Evidence (EXP-0005, MEASURED generic synthesis): a 256-term packed consumer with 22-bit accumulators is 25.5k generic cells versus 69.5k for an INT8 × INT4 structural baseline at the same terms per cycle (2.7×), and 96–115 cells per term across 64–1024 terms/cycle; no multiplier anywhere. The advantage survives composition but shrinks from 12× (primitive) to under 3× because accumulation, decode and control are shared costs.*
- **H3.** 2-bit (or denser) weight storage reduces memory traffic enough to matter for decode-bound inference. *EXP-0007 (ANALYTICAL): under batching the per-sequence weight traffic of any system falls as 1/batch, so H3's payoff is confined to small batch; at batch ≥ 16 KV and compute, not weights, set the bytes and time for every system.* *EXP-0005 (MEASURED/ANALYTICAL): the stored 2-bit packing is consumed directly at initiation interval 1 by RTL that holds no N × K state, so the 2-bit traffic figure is a real consumer's figure, not a storage-only figure. A bank-local consumer removes the stream from the global path entirely (396 MB → 2.8 MB per token for Falcon-E 1B) if the whole model stays resident near compute; the local reads remain. EXP-0006 (ANALYTICAL/INFERRED on public data): at 1B–3B the traffic reduction is not needed for throughput (one LPDDR5X channel streams ~120 tok/s), so H3's payoff is energy and latency, bounded by an unmeasured modern-node SRAM read energy. EXP-0008 (INFERRED on T1 sources): the read-energy bound is now sourced, roughly 10× per bit in residency's favour at 5–7 nm against LPDDR5X; what bounds H3's edge payoff is instead the leakage of the resident megabytes, which is unmeasured at modern nodes and decides whether the advantage survives below ~23 tok/s or ~85% duty. EXP-0009 (INFERRED, T2-dependent; wake and reload figures corrected by EXP-0009A): with a modeled array-voltage retention mode and a two-layer active window the leakage term is beaten at room temperature from ~1.4 tok/s or ~3.5–6% duty, but not when hot: at 85 °C a retained 440 MB array leaks more than a self-refreshing DRAM, so H3's edge payoff is a function of operating temperature and of how the retention rail is supplied.* *Evidence (EXP-0002/EXP-0002B, INFERRED traffic model): 2-bit ternary layers are 44–73% of stream-once weight bytes depending on the model; the BF16 head takes 56% for BitNet (128k vocab) but 16–25% for Falcon-E (32k vocab), and at 32k context Falcon's KV cache read exceeds its weight stream. H3 is necessary but not sufficient; which non-ternary stream dominates is model- and context-dependent.*
- **H4.** Explicit zero-skipping yields useful savings on real ternary models, because those models contain exploitable sparsity. *Evidence (EXP-0002/EXP-0002B, MEASURED): 39.7–42.2% zeros in all three checkpoints, placement along K consistent with independence in all three. BitNet's dead MLP channels (4.76%) do not appear in Falcon-E (0). Per-element skipping faces unstructured zeros everywhere; structured sparsity is not a ternary property.*
- **H5.** Co-designing the compiler/runtime with the hardware captures gains that generic software on generic hardware cannot.
- **H6.** Co-designing ternary model weights, low-precision non-ternary components (output head, KV cache, activations), numerical representations, memory organization, and inference execution produces substantially better end-to-end inference economics than treating models and hardware separately. *Status: HYPOTHESIZED (ADR-0006, 2026-10-06). First test, EXP-0004 (MEASURED, one checkpoint, PILOT scale): Falcon-E 1B tolerates an INT4 head (+0.5–0.7% perplexity, task accuracies within one standard error) and a 4-bit KIVI-style KV cache (+0.17–1.07% perplexity depending on grouping and context) separately and together; one greedy-decoding sample at a 6k-token prompt degenerated earlier under quantization than at baseline. The model-side premise holds at this scale; the hardware-side benefit is unproven and the attainable savings on current runtimes are smaller than the logical ones.*

H1 has been tested at the generic-logic level (EXP-0001). H3 and H4 have workload evidence from two independent model families (EXP-0002, EXP-0002B) but no hardware evidence. EXP-0001 also showed the per-element advantage shrinking as accumulation is added, which bears on H2. The mixed-precision framing (ternary blocks plus non-ternary head, norms, attention, KV cache) is confirmed in both families; what differs is which non-ternary stream is largest.

EXP-0005 (MEASURED at RTL and generic-synthesis level, ANALYTICAL for traffic) tests the premise behind every EXP-0003 ceiling: a packed consumer exists, is small, runs at initiation interval 1 and needs no unpacking (Gate 1A PASS at the structural/cycle level; absolute frequency UNMEASURED). Bank-local placement removes the ternary weight stream from the global path at the cost of whole-model residency and NB-fold activation delivery (Gate 1B PASS at the dataflow level; physical energy and cost INCONCLUSIVE). The thesis has gained its first hardware dataflow evidence and its sharpest open requirement: hundreds of megabytes of resident near-compute memory. EXP-0006 (Gate 2, INFERRED from public evidence) finds that requirement inside the demonstrated commercial envelope for 1B–3B models (Graphcore-class 900 MB SRAM-with-compute dies; Falcon-E 1B needs 404–648 mm², 3B 744–1,192 mm² at 0.68–1.09 MB/mm²), finds that 2-bit storage is what keeps 3B inside one reticle where INT4 does not, and finds that nothing above ~3B fits one die. It also finds that bandwidth never forces residency at these sizes: the case for the Youssouf Architecture's resident form rests on energy and latency, not on feasibility of serving, and the head/KV consumer becomes the dominant external traffic once the weights are resident.

EXP-0007 (Gate 3, ANALYTICAL against a sourced FP4/HBM baseline) tests the datacenter form and does not support it: a tensor-parallel HBM node amortizes weights over the whole batch while a resident pipeline amortizes only over its microbatch, so the resident movement-energy advantage falls below 2× from batch 4 at 7B–400B (2.7× at batch 2) and is negative from batch 8; KV dominates every system equally beyond 8k context; prefill on resident tiles is 17–550× slower at the target density and the resident system is slower in every cell. What survives: small-batch interactive serving (batch 1–2 at 7B+, ≤ 8 for single-tile 1B, ≤ 8k context), ternary-as-a-format streaming on conventional HBM hardware (2–3× fewer weight bytes at small batch), and the edge/resident 1B–3B form. The thesis's datacenter ambition is therefore narrowed to those regimes; its edge form is unchanged and still rests on the unmeasured modern-node SRAM read energy (Q-023).

EXP-0008 (Gate 4, ANALYTICAL on T1 macro and datasheet sources) tests the edge form's memory energy: modern SRAM reads at the macro + bank + wire boundary cost 0.33–0.62 pJ/bit against 3.5–6.1 pJ/bit for LPDDR5X through the SoC path, roughly 10× (5.6–18×), which supersedes EXP-0006's 3–9× and passes Gate 4A at order of magnitude. Leakage, counted for the first time, is where the edge form lives: 440–811 MB of always-on SRAM at the planning 0.6 mW/MB is 264–486 mW, so residency wins total memory power only above ~23 tok/s continuous or ~85% duty at 25 tok/s without a retention mode (Gate 4B INCONCLUSIVE; the unmeasured 0.05–1.0 mW/MB range spans the verdict). Access latency is hidden by prefetch; per-channel bandwidth is the real streaming limit (Gate 4C FAIL as a latency claim). With head and KV external the total advantage is 1.4–3.3× at BF16 and 2.2–5.8× with the EXP-0004 candidate (Gate 4D INCONCLUSIVE for hardware commitment). The thesis's edge form is therefore a standby-power question, not a read-energy question.

EXP-0009 (Gate 5, ANALYTICAL on T1 standby figures and a T2 cell simulation) asks whether sequential layer execution lets most of the resident SRAM sit in retention: modern standby leakage is 0.21–0.52 mW/MB at 25 °C and ~1.6 mW/MB at 85 °C; an array-voltage retention mode (modeled; the modes modern papers document save only 10–30%) cuts it 13× cold and 4× hot; wake latency is hidden at every edge token rate and wake energy, once the EXP-0009A unit correction is applied (nJ per MB, not µJ; EXP-0009 had overstated it 1000×), is below 0.1% of the read energy; the result is that a resident Falcon-E 1B beats LPDDR5X at 25 °C from ~1.4 tok/s or ~3.5–6% duty (2.4–4.0× in total with head/KV) but at 85 °C only above ~17 tok/s (1.41×) and never at low duty. Long idle gaps are served by shutdown-and-reload only on the terms of the backing store (EXP-0009A): a flash-backed reload (~300 mJ) beats retention beyond ~29 s cool / ~1.8 s hot, a DRAM-backed one beyond ~3 s / ~0.2 s but never beats the streaming baseline during idle. Gate 5A INCONCLUSIVE, 5B PASS on latency and bounded on energy (INFERRED, unmeasured), 5C INCONCLUSIVE split by temperature, 5D INCONCLUSIVE. The edge form's thesis now turns on the product's operating temperature and on a foundry retention mode that no modern silicon paper reports, including whether its rail is supplied at V_ret or derived linearly (which halves the saving and flips the hot continuous case to 0.87×).

EXP-0003 (ANALYTICAL) bounds the whole thesis from above: with the non-ternary parts unchanged, ternary specialization alone can at most deliver an end-to-end decode speedup equal to total bytes over non-ternary bytes in the memory-bound regime. That ceiling is below 1.8× for BitNet 2B4T at any context (BF16 head), above 3× for Falcon-E at short context (≤ ~2.6k for 1B, ≤ ~7k for 3B), and below 1.6× for every model at 32k context. The thesis therefore depends on the system-level scope in ADR-0005: the ternary datapath is necessary but cannot carry multi-fold gains alone outside short-context, batch-1, modest-vocabulary decode. In bytes, the advantage 2-bit storage already realizes over an INT4-weight system with the same head and KV is 1.3–1.8×; over BF16-weight systems 3–7× (EXP-0003, bytes only, compute and energy unmodeled).

## Unknowns and risks

- Ternary model adoption may remain limited. If the model ecosystem stays at 4-bit and above, the hardware target shrinks.
- Model quality may constrain ternary use to small models or specific layers.
- Memory and communication may dominate arithmetic so completely that the Y-Core advantage is irrelevant at system level.
- Existing GPU/accelerator vendors may add competitive low-bit hardware, erasing the differentiation.
- The software ecosystem may matter more than raw silicon. A superior datapath nobody can target loses to an inferior one everyone can.
- Advantages seen at tiny scale (a single core, a small array) may disappear at transformer scale once control, memory, and interconnect are included.
- FPGA results may not predict ASIC results. The two must never be conflated.

## What would change the thesis

- H1 fails: the ternary datapath is not meaningfully smaller or cheaper than a well-designed low-precision multiplier. The project should then examine whether memory/encoding (H3) carries the thesis alone, or whether the thesis is wrong.
- H3 fails: weight-traffic reduction is small relative to activation/KV traffic. Focus would shift to activation handling.
- Ternary model quality stalls publicly: the thesis would need to broaden to "ultra-low precision" generally or narrow to specific niches.

The project follows the evidence. Pivoting is intentional, not failure.
