# ADR-0003 — Reference ternary workload: BitNet b1.58 2B4T

Status: Superseded (by ADR-0004, 2026-10-06; the BitNet 2B4T choice itself still stands as the first member of the reference set)
Date: 2026-10-06

## Context

Phase 1 showed the ternary primitive is cheap but that accumulation dilutes the advantage. Deciding what to build next requires knowing what a real ternary transformer actually demands. FlexTern needs one concrete, public, natively ternary model to characterize and to cite in later experiments.

## Decision

The reference workload is **`microsoft/bitnet-b1.58-2B-4T`**, Hugging Face revision **`04c3b9ad9361b824064a1f25ea60a8be9599b127`**, packed 2-bit checkpoint `model.safetensors` (1,178,623,988 bytes, SHA-256 `8143ae11…975b`), MIT license. Its quantization semantics are taken from `huggingface/transformers` `src/transformers/integrations/bitnet.py` at commit `d6abd133`.

Model files are never committed; the tool downloads to the Hugging Face cache and verifies size and hash.

## Alternatives Considered

- **`bitnet-b1.58-2B-4T-bf16`** (latent master weights). Rejected as primary: would require re-running absmean ternarization and could differ from the deployed packed weights. Useful only for cross-checking.
- **GGUF `i2_s` variant.** Rejected: bitnet.cpp-specific packing adds a layer of format reverse-engineering with no benefit.
- **Community ternary models or post-training-quantized ternary conversions.** Rejected: not natively trained ternary; would not represent the target workload.

## Why

It is the official, natively trained, publicly released ternary model closest to the scale FlexTern targets, with a documented reference implementation of both weight packing and activation quantization.

## Consequences

- Every later workload-dependent decision (dot lengths, zero rates, activation format, accumulator widths, head handling) cites this revision.
- If Microsoft or others release a larger or newer native ternary model, a new ADR must supersede this one and EXP-0002 must be re-run before conclusions are transferred.
- Findings are model-specific until a second model confirms them (Q-018).

## Evidence

Founder direction, 2026-10-06 (Phase 2 prompt names this model as the preferred target). Provenance verified via the Hugging Face API and model card.

## Supersedes

None.

## Superseded By

ADR-0004 — Reference workload set (adds Falcon-E 1B/3B; BitNet 2B4T remains the first reference).
