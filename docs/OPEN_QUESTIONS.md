# Open Questions

Research and design questions that matter but are not yet decisions (ADRs) or defects (KNOWN_ISSUES). Some may stay open for months. Do not turn every question into immediate work.

When a question is resolved, mark it Closed, state the answer in one line, and point to the ADR, experiment, or document that holds it.

---

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

- **Q-001** Measurement vehicle for v0.1 → generic Yosys synthesis, no physical target. Closed 2026-10-05 by ADR-0002.
- **Q-002** v0.1 activation width → signed 8-bit. Closed 2026-10-05 by ADR-0002. (Final precision remains Q-007.)
- **Q-003** v0.1 weight encoding → 2-bit, 00=0 / 01=+1 / 11=-1 / 10=reserved→0. Closed 2026-10-05 by ADR-0002. (Best physical packing remains Q-011.)
- **Q-004** Conventional baseline → signed INT8 × INT8 via `*`, 16-bit product, 20-bit dot16 accumulator. Closed 2026-10-05 by ADR-0002.
