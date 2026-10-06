# ADR-0006 — Open a model-side track: numerical-representation research alongside hardware work

Status: Accepted
Date: 2026-10-06

## Context

EXP-0003 showed that for Falcon-E 1B the ternary engine leaves the BF16 output head (134 MB per token) and the BF16 KV cache (up to 805 MB per token at 32k context) untouched, and that these cap the end-to-end benefit of ternary specialization. Whether those components can be stored at lower precision is a property of the model, not of the hardware, and must be validated against real model behaviour before it becomes a hardware requirement.

## Decision

1. FlexTern opens a **model-side research track** whose purpose is to determine what numerical representation the Youssouf Architecture should eventually be optimized for, by testing low-precision representations of non-ternary components on the pinned reference checkpoints (ADR-0004).
2. The track runs **one uncertainty at a time**, like the hardware track, under the same one-objective rule (ADR-0001). It does not create a roadmap of model work.
3. Its outputs are **candidate formats**, never "FlexTern v1". A format becomes a hardware requirement only after a quality evaluation on a reference checkpoint and an ADR recording the decision.
4. The track does not train models from scratch, does not overwrite canonical checkpoints, and never commits weights.
5. Benefits of model-side quantization that any hardware can use are reported separately from benefits that would require ternary-native hardware or memory-centric dataflow.

## Alternatives Considered

- **Hardware-only track, treating head and KV precision as externally given.** Rejected: EXP-0003 shows those components decide the system ceiling; ignoring them leaves the thesis unverifiable.
- **Full co-training of a new low-bit model.** Rejected for now: costly, and the first question (can existing native ternary models tolerate low-bit head/KV) is answerable without it.

## Why

The co-design hypothesis (THESIS H6) can only be tested by changing the model side as well as the hardware side. EXP-0004 is its first test.

## Consequences

- The thesis gains H6 (HYPOTHESIZED). The roadmap notes the parallel model-side track.
- Quality evaluations become first-class experiment artifacts with pinned datasets, tokenizer, revision, framework versions, and seeds.
- Hardware experiments may now cite candidate formats, labeled as such.

## Evidence

Founder direction, 2026-10-06 (EXP-0004 prompt). EXP-0003 for the motivation.

## Supersedes

None.

## Superseded By

None.
