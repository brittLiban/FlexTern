# Open Questions

Research and design questions that matter but are not yet decisions (ADRs) or defects (KNOWN_ISSUES). Some may stay open for months. Do not turn every question into immediate work.

When a question is resolved, mark it Closed, state the answer in one line, and point to the ADR, experiment, or document that holds it.

---

## Blocking the current objective

### Q-001
**Question:** Which FPGA board and toolchain is the measurement vehicle for Y-Core v0.1?
**Why it matters:** Resource and timing numbers are target-specific. Without a fixed target, no result is reproducible. This does *not* decide the long-term platform.
**Current evidence:** None.
**What would answer it:** Founder decision based on hardware and tools available.
**Status:** Open — blocks FT-CURRENT.

### Q-002
**Question:** What activation bit width does Y-Core v0.1 use?
**Why it matters:** Width drives both the Y-Core and the baseline size. The comparison is only fair if both use the same width. Final activation precision is a separate, later question.
**Current evidence:** None.
**What would answer it:** Founder decision for v0.1. Width sweeps could be a later experiment.
**Status:** Open — blocks FT-CURRENT.

### Q-003
**Question:** What is the ternary weight encoding at the Y-Core input boundary for v0.1?
**Why it matters:** The Y-Core must decode something. Encoding choice affects decode logic and, later, memory packing. For v0.1 it only needs to be explicit.
**Current evidence:** None.
**What would answer it:** Founder decision for v0.1. The *best* physical encoding is Q-011.
**Status:** Open — blocks FT-CURRENT.

### Q-004
**Question:** What exactly is the "conventional multiply-based datapath" baseline?
**Why it matters:** Benchmark fairness. A naive baseline inflates the result; an over-engineered one hides it. Operand widths, signedness, DSP usage, and whether accumulation is included must be fixed before synthesis.
**Current evidence:** None.
**What would answer it:** Founder decision, ideally after discussion with ChatGPT and/or a reviewer.
**Status:** Open — blocks FT-CURRENT.

## Thesis-level

### Q-005
**Question:** At scale, does arithmetic or memory movement dominate the benefit of ternary computation?
**Why it matters:** Determines whether the Y-Core or the memory/encoding subsystem is the real product.
**Current evidence:** None from FlexTern. External work suggests decode inference is memory-bound on conventional hardware (KNOWN), but that does not settle where FlexTern's advantage lies.
**What would answer it:** Phase 2+ experiments with realistic dataflow and memory traffic measurement.
**Status:** Open.

### Q-006
**Question:** How much exploitable sparsity (weight = 0) exists in target ternary models?
**Why it matters:** Zero-skipping hardware is only worth its complexity if zeros are common and clustered enough to skip.
**Current evidence:** None gathered.
**What would answer it:** Analyzing weight distributions of published ternary models.
**Status:** Open.

### Q-007
**Question:** Which activation precision is actually optimal for ternary-weight inference quality vs cost?
**Why it matters:** Activation width dominates datapath and traffic once weights are 2-bit.
**Current evidence:** None.
**What would answer it:** Model-quality studies plus hardware cost sweeps. Not a Phase 1 question.
**Status:** Open.

### Q-008
**Question:** Will ternary models be adopted widely enough for dedicated hardware to matter?
**Why it matters:** Existential for the thesis.
**Current evidence:** Published ternary LLMs exist; adoption is uncertain.
**What would answer it:** External ecosystem developments; cannot be answered inside this repo.
**Status:** Open.

## Architecture-level (not yet relevant)

### Q-009
**Question:** Should Y-Array use temporal or spatial accumulation?
**Status:** Open. Deferred until Phase 2.

### Q-010
**Question:** What accumulator width and dataflow does Y-Dot need?
**Status:** Open. Deferred until Phase 2.

### Q-011
**Question:** What is the best physical encoding/packing for ternary weights in memory?
**Status:** Open. Deferred.

## Strategic (founder-only; not for agents to decide)

### Q-012 — Final ASIC process node. Open. Deferred to Phase 7.
### Q-013 — Final instruction set / programming model. Open. Deferred to Phase 5.
### Q-014 — Compiler architecture. Open. Deferred to Phase 5.
### Q-015 — Product form: cloud service, add-in card, or IP licensing. Open. Deferred to Phase 6.
### Q-016 — Training architecture, if any. Open. Not currently in scope.
### Q-017 — IP / open-source posture for RTL, tools, and documentation. Open. Agents must ask before publishing or licensing anything.

## Closed

None.
