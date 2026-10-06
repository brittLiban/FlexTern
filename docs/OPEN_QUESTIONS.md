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
**Question:** Do the EXP-0002 findings hold for other native ternary models?
**Why it matters:** Every workload-driven hardware decision rested on one model.
**Current evidence:** EXP-0002B (MEASURED) across TII Falcon-E 1B and 3B: **generalize** — ternary {-1,0,+1} stored 2-bit; per-tensor scale applied after accumulation; INT8 per-token absmax activations with -128 unreachable; ±1 balanced within 0.1 pt; 39.7–42.2% zeros, unstructured along K; two dot lengths per model, all divisible by 256; BF16 non-ternary output head; q/k sparser than v/o. **Microsoft-specific** — dead MLP channels (4.76% vs 0 and 0), tied head, head dominating traffic (56% vs 25%/16%), exact K values, ReLU², sub-norms, stored-scale convention (β vs 1/β).
**What would answer it further:** A third independent family. Two families are evidence, not proof.
**Status:** Partially answered across two independent families; broader generalization remains open.

### Q-019
**Question:** Is the non-ternary BF16 output head in FlexTern's scope, or an accepted fixed cost handled elsewhere?
**Why it matters:** Under a stream-once decode model it can be the single largest weight stream per token.
**Current evidence:** EXP-0002/EXP-0002B (INFERRED traffic model): the head is BF16 in both families, but its share of weight bytes is vocabulary-driven: 56% for BitNet (128k vocab, tied), 25% for Falcon-E 1B and 16% for 3B (32k vocab, untied). Head *precision* generalizes; head *dominance* does not. At 32k context, Falcon's KV-cache read (805 MB/token for 1B) exceeds its weight stream.
**What would answer it:** Founder decision on scope; a quality study if quantizing the head is considered.
**Status:** Open — founder-level; the scope question now includes the KV cache at long context.

### Q-020
**Question:** Can channel-level dead sparsity be exploited at the compiler/packing level rather than in hardware?
**Why it matters:** EXP-0002 found whole dead MLP channels in BitNet (4.76%, 43.6% in layer 1).
**Current evidence:** EXP-0002B: Falcon-E 1B and 3B have **zero** all-zero rows in any tensor. Dead channels are a BitNet-specific artifact (ReLU² and/or training recipe), not a ternary property. Any exploitation would be model-conditional.
**What would answer it:** Nothing further needed for hardware; a compiler may handle it opportunistically if ever relevant.
**Status:** Open, deprioritized — evidence says do not build hardware for it.

### Q-021
**Question:** Should the Youssouf Architecture support parameterized activation precision rather than hard-coding INT8?
**Why it matters:** Both production-like native ternary families today use INT8 per-token absmax activations (EXP-0002, EXP-0002B), and Y-Core v0.1 matches them. Published research directions explore 4-bit (or lower) activations for ternary-weight models. If that becomes the production norm, a fixed-INT8 datapath wastes accumulator width, activation bandwidth, and (for popcount-style execution) half the bit-planes.
**Current evidence:** INT8 in both families (MEASURED from reference code). No lower-precision native release has been inspected; compatibility of lower-precision methods with these checkpoints is **not verified**.
**What would answer it:** A native ternary release with sub-8-bit activations, or a quality study on these checkpoints. Until then, treat activation width as an architectural parameter in analyses (roofline inputs, accumulator widths) and keep INT8 as the first implementation precision.
**Status:** Open — tracked risk. No redesign now.

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
