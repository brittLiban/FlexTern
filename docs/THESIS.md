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

- **H1.** A dedicated ternary datapath (Y-Core) is materially smaller and/or lower-power than a conventional multiply-based datapath at the same activation precision on the same target.
- **H2.** The advantage in H1 survives, or grows, when composed into dot products and arrays, because wide ternary accumulation avoids multiplier arrays entirely.
- **H3.** 2-bit (or denser) weight storage reduces memory traffic enough to matter for decode-bound inference.
- **H4.** Explicit zero-skipping yields useful savings on real ternary models, because those models contain exploitable sparsity.
- **H5.** Co-designing the compiler/runtime with the hardware captures gains that generic software on generic hardware cannot.

None of these has been tested in this repository. The current objective (`CURRENT_WORK.md`) addresses H1 only.

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
