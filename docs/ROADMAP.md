# FlexTern Roadmap

> **This roadmap provides direction, not a predetermined implementation sequence. Each phase may change based on experimental evidence.**

It contains no tickets. The only committed work is the single objective in `CURRENT_WORK.md`. Phases beyond the current one describe where the project *might* go if earlier phases produce supporting evidence.

```
PHASE 1   ◀── complete 2026-10-05 (EXP-0001)
Ternary primitive
Y-Core, Y-Dot16
controlled baseline

      ↓

PHASE 2   ◀── workload characterization complete 2026-10-06 (EXP-0002); next step not yet chosen
Real-workload characterization
ternary dot products / array architecture
memory / dataflow investigation

      ↓

PHASE 3
FPGA accelerator
real neural-network operations

      ↓

PHASE 4
End-to-end ternary transformer inference
defensible benchmarking

      ↓

PHASE 5
Compiler / runtime integration
developer-facing software

      ↓

PHASE 6
Design partners / commercialization

      ↓

PHASE 7
Custom silicon, if justified by evidence
```

## Model-side track (opened 2026-10-06, ADR-0006)

Alongside the hardware phases, a model-side track asks what numerical representation the hardware should be optimized for, one uncertainty at a time, on the pinned reference checkpoints. Its outputs are candidate formats, not requirements, until validated by a quality evaluation and recorded in an ADR. It is not a second roadmap; it has one objective at a time like everything else.

First result (EXP-0004, 2026-10-06, MEASURED on one checkpoint): Falcon-E 1B tolerates an INT4 output head and a 4-bit KV cache at small, measured perplexity cost. A candidate format exists in `../results/model_precision/candidate_format.md`; adopting it as a hardware assumption is a founder decision not yet taken.

## How phases advance

A phase does not end because its work is "done." It ends when the founder, after reviewing measured results with ChatGPT and any reviewer critiques, decides the evidence supports moving on, revising the phase, or pivoting.

Evidence from any phase may send the project sideways rather than forward. If Phase 1 shows that memory movement, encoding, or sparsity matters more than the elementary operation, Phase 2 will be reshaped accordingly.

## What this roadmap deliberately does not decide

Target FPGA, ASIC node, instruction set, compiler architecture, activation precision, weight packing, training approach, and product form (cloud / card / licensing). These are open and listed in `OPEN_QUESTIONS.md`.
