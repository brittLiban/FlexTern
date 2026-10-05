# FlexTern Roadmap

> **This roadmap provides direction, not a predetermined implementation sequence. Each phase may change based on experimental evidence.**

It contains no tickets. The only committed work is the single objective in `CURRENT_WORK.md`. Phases beyond the current one describe where the project *might* go if earlier phases produce supporting evidence.

```
PHASE 1   ◀── current
Ternary primitive
Y-Core
controlled baseline

      ↓

PHASE 2
Ternary dot products / array architecture
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

## How phases advance

A phase does not end because its work is "done." It ends when the founder, after reviewing measured results with ChatGPT and any reviewer critiques, decides the evidence supports moving on, revising the phase, or pivoting.

Evidence from any phase may send the project sideways rather than forward. If Phase 1 shows that memory movement, encoding, or sparsity matters more than the elementary operation, Phase 2 will be reshaped accordingly.

## What this roadmap deliberately does not decide

Target FPGA, ASIC node, instruction set, compiler architecture, activation precision, weight packing, training approach, and product form (cloud / card / licensing). These are open and listed in `OPEN_QUESTIONS.md`.
