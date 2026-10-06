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

### Q-019 (reframed 2026-10-06 per ADR-0005)
**Question:** Which non-ternary components (output head, attention, KV cache, other) must eventually be optimized to preserve a material end-to-end advantage in the target workload? (Scope is settled: ternary blocks are core; head, attention, KV are system-level FlexTern scope; FlexTern is ternary-first, not ternary-only.)
**Why it matters:** End-to-end gain is capped by whatever is not improved. EXP-0003 quantifies the cap per workload region.
**Current evidence (EXP-0003, ANALYTICAL):** With BF16 head and KV at batch 1, the ideal ternary-only ceiling is below 1.8× for BitNet at every context (the 657 MB BF16 head alone exceeds the 521 MB ternary stream), while Falcon-E 1B/3B stay above 3× to roughly 2.6k/7k context and above 2× to roughly 10.7k/18k, then fall to 1.42×/1.60× at 32k as KV reads dominate. Head precision is the gating component for 128k-vocabulary models at any context; KV traffic is the gating component for every model beyond roughly 8–18k context and for any batched decode. Hypothetical INT8 head restores >2× for BitNet to ~2.5k context; hypothetical INT4 KV extends Falcon-E 1B's 2× region from ~10.7k context to beyond its supported 32k in the memory-bound regime, but only if attention compute keeps pace.
**What would answer it further:** Whether the head can be quantized without quality loss (model-side question), and the real cost of attention compute at long context, which this model treats as an abstract rate.
**Status:** Partially answered analytically; founder/ChatGPT to decide which component to pursue for which market.

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
**Status:** Open — tracked risk. No redesign now. EXP-0003 note: at the bandwidth-bound operating points studied, 4-bit activations change nothing in the token model (activations are on-chip); they would matter for accumulator width (K = 2048: 20 → 16 bits) and for any bit-serial engine's throughput, both only relevant once the ternary stage is compute-limited.

### Q-022
**Question:** What is the real cost of non-ternary compute (attention QKᵀ/AV, softmax, head MACs, norms) on a FlexTern system?
**Why it matters:** EXP-0003 models it as one abstract rate (1e12 op-eq/s default) with no evidence behind it. In the memory-bound regime it does not affect conclusions, but at ≥1 TB/s or with hypothetically compressed KV it becomes the limiter: Falcon-E 1B's ideal ceiling at 4k context drops from 2.69× (100 GB/s) to 1.73× (1 TB/s) purely because attention compute at the assumed rate takes over.
**Current evidence:** None; abstract parameter only.
**What would answer it:** A decision on how attention and the head execute in a FlexTern system (which engine, what precision), then a measurement or a defensible model of that engine.
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
