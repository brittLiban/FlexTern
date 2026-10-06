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
- **H2.** The advantage in H1 survives, or grows, when composed into dot products and arrays, because wide ternary accumulation avoids multiplier arrays entirely.
- **H3.** 2-bit (or denser) weight storage reduces memory traffic enough to matter for decode-bound inference. *Evidence (EXP-0002/EXP-0002B, INFERRED traffic model): 2-bit ternary layers are 44–73% of stream-once weight bytes depending on the model; the BF16 head takes 56% for BitNet (128k vocab) but 16–25% for Falcon-E (32k vocab), and at 32k context Falcon's KV cache read exceeds its weight stream. H3 is necessary but not sufficient; which non-ternary stream dominates is model- and context-dependent.*
- **H4.** Explicit zero-skipping yields useful savings on real ternary models, because those models contain exploitable sparsity. *Evidence (EXP-0002/EXP-0002B, MEASURED): 39.7–42.2% zeros in all three checkpoints, placement along K consistent with independence in all three. BitNet's dead MLP channels (4.76%) do not appear in Falcon-E (0). Per-element skipping faces unstructured zeros everywhere; structured sparsity is not a ternary property.*
- **H5.** Co-designing the compiler/runtime with the hardware captures gains that generic software on generic hardware cannot.
- **H6.** Co-designing ternary model weights, low-precision non-ternary components (output head, KV cache, activations), numerical representations, memory organization, and inference execution produces substantially better end-to-end inference economics than treating models and hardware separately. *Status: HYPOTHESIZED (ADR-0006, 2026-10-06). First test, EXP-0004 (MEASURED, one checkpoint, PILOT scale): Falcon-E 1B tolerates an INT4 head (+0.5–0.7% perplexity, task accuracies within one standard error) and a 4-bit KIVI-style KV cache (+0.17–1.07% perplexity depending on grouping and context) separately and together; one greedy-decoding sample at a 6k-token prompt degenerated earlier under quantization than at baseline. The model-side premise holds at this scale; the hardware-side benefit is unproven and the attainable savings on current runtimes are smaller than the logical ones.*

H1 has been tested at the generic-logic level (EXP-0001). H3 and H4 have workload evidence from two independent model families (EXP-0002, EXP-0002B) but no hardware evidence. EXP-0001 also showed the per-element advantage shrinking as accumulation is added, which bears on H2. The mixed-precision framing (ternary blocks plus non-ternary head, norms, attention, KV cache) is confirmed in both families; what differs is which non-ternary stream is largest.

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
