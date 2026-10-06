# Open Questions

Research and design questions that matter but are not yet decisions (ADRs) or defects (KNOWN_ISSUES). Some may stay open for months. Do not turn every question into immediate work.

When a question is resolved, mark it Closed, state the answer in one line, and point to the ADR, experiment, or document that holds it.

---

## Thesis-level

### Q-005
**Question:** At scale, does arithmetic or memory movement dominate the benefit of ternary computation?
**Why it matters:** Determines whether the Y-Core or the memory/encoding subsystem is the real product.
**Current evidence:** EXP-0002 (INFERRED, analytical): for BitNet 2B4T batch-1 decode with weights streamed once, the 2-bit ternary layers are 521 MB/token but the BF16 tied output head is 657 MB/token and the KV cache at 4k context another 315 MB/token. Ternary arithmetic is 86% of linear MACs but only 35% of weight-plus-KV bytes. Not measured on any hardware.
**What would answer it:** A hardware experiment that models or measures the memory stream, not just the datapath.
**Status:** Open; evidence now points toward memory.

### Q-006
**Question:** How much exploitable sparsity (weight = 0) exists in target ternary models?
**Why it matters:** Zero-skipping hardware is only worth its complexity if zeros are common and clustered enough to skip.
**Current evidence:** EXP-0002 (MEASURED): 42.2% of ternary weights are zero (per-tensor 36–60%). Along the dot-product axis, zero placement matches independence once all-zero rows are removed, so there is no block structure to exploit at g = 4/8/16. What does exist is channel-level: 4.76% of MLP intermediate channels are entirely dead (gate and up rows both zero; layer 1 is 43.6% dead), which is skippable at the channel level with no per-element machinery.
**What would answer it:** Whether a second model shows the same pattern (Q-018), and whether unstructured 42% zeros can be exploited cheaply (e.g., by what the accumulator does with a zero term) versus the cost of detecting them.
**Status:** Open; partially answered for this model.

### Q-007
**Question:** Which activation precision is actually optimal for ternary-weight inference quality vs cost?
**Why it matters:** Activation width dominates datapath and traffic once weights are 2-bit.
**Current evidence:** EXP-0002 (KNOWN from reference code): the reference model uses per-token absmax symmetric INT8 with range [-127, 127]; -128 is unreachable. Whether narrower activations keep quality is untested.
**What would answer it:** Model-quality studies plus hardware cost sweeps.
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
**Current evidence:** EXP-0002: the reference checkpoint uses 2 bits/weight (4 per byte along the output dimension); 5-trits-per-byte would cut ternary bytes by 20% (521 → 417 MB) at the cost of a base-3 decoder. Whether that decoder is cheaper than the bandwidth it saves is unmeasured.
**Status:** Open. Deferred.

### Q-018
**Question:** Do the EXP-0002 findings (42% zeros, dead early-layer MLP channels, two dot lengths, per-tensor scale, INT8 per-token activations) hold for other native ternary models?
**Why it matters:** Every workload-driven hardware decision currently rests on one model.
**Current evidence:** One model only.
**What would answer it:** Re-running `analysis/ternary_model_profile.py` (generalized for other layouts) on a second native ternary release when one exists.
**Status:** Open.

### Q-019
**Question:** Is the non-ternary BF16 output head (328 M parameters, 657 MB, 13.6% of linear MACs) in FlexTern's scope, or an accepted fixed cost handled elsewhere?
**Why it matters:** Under a stream-once decode model it is the single largest weight stream per token, bigger than all ternary layers combined. Quantizing it to INT8/INT4 is a model-quality question FlexTern cannot answer alone.
**Current evidence:** EXP-0002 traffic model (INFERRED).
**What would answer it:** Founder decision on scope; a quality study if quantizing the head is considered.
**Status:** Open — founder-level.

### Q-020
**Question:** Can channel-level dead sparsity (whole-row zeros) be exploited at the compiler/packing level rather than in hardware, and does that make per-element zero-skipping unnecessary?
**Why it matters:** EXP-0002 shows the only structured sparsity is dead channels, which a compiler can simply drop (2.44% of weights, up to 43.6% of MLP channels in one layer).
**Current evidence:** EXP-0002 (MEASURED counts; no exploitation tried).
**What would answer it:** A packing/scheduling experiment, not RTL.
**Status:** Open.

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
