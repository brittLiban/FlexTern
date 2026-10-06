# ADR-0005 — System scope: FlexTern is ternary-first, not ternary-only

Status: Accepted
Date: 2026-10-06

## Context

EXP-0002 and EXP-0002B showed that both reference ternary families keep a BF16 output head, BF16 attention arithmetic, a BF16 KV cache, and small non-ternary work. Which of these dominates bytes is model- and context-dependent (head for BitNet's 128k vocabulary; KV cache for Falcon-E at 32k context). Q-019 had been framed as "optimize the head or the KV cache?", a binary that does not match how a system is built.

## Decision

1. The ternary transformer blocks (Y-Core, Y-Dot, and whatever composes them) are **core FlexTern scope**.
2. The output head, attention arithmetic, and KV-cache movement are **system-level FlexTern scope**: FlexTern must account for them in every end-to-end claim and may optimize them, but they do not necessarily execute on Y-Core.
3. FlexTern is **ternary-first, not ternary-only**.
4. Q-019 is reframed: *which non-ternary components must eventually be optimized to preserve a material end-to-end advantage in the target workload?* EXP-0003 quantifies when such work becomes necessary. No commitment to a KV-compression subsystem is made by this ADR.

## Alternatives Considered

- **Ternary-only scope.** Rejected: EXP-0002B shows non-ternary streams can exceed the ternary stream; a ternary-only product would have no end-to-end story.
- **Everything on Y-Core.** Rejected: attention, softmax, and BF16 head arithmetic are not ternary operations.
- **Decide now to build KV compression.** Rejected: premature; EXP-0003 exists to find when it matters.

## Why

End-to-end benefit is capped by the components that are not improved (Amdahl). Keeping them in scope, without presuming their implementation, is the only way to make honest system claims.

## Consequences

- Every experiment that reports an end-to-end number must report component-level contributions (ternary, head, attention, KV, other).
- Q-019 becomes an evidence question answered per workload region rather than a binary.
- Roadmap phases 3–5 may include non-ternary subsystems if evidence says they gate the advantage.

## Evidence

Founder direction, 2026-10-06 (EXP-0003 prompt). Workload evidence: EXP-0002, EXP-0002B.

## Supersedes

None.

## Superseded By

None.
