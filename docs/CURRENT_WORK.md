# Current Work

There is exactly one active engineering objective at a time. When it completes, this file is updated and the next objective is chosen by the founder with ChatGPT. "No active objective" is a valid state.

---

**Objective ID:** FT-CURRENT

**Title:** EXP-0004 — Low-precision output head and KV-cache feasibility for native ternary inference: can `tiiuae/Falcon-E-1B-Base` (ADR-0004 revision) preserve acceptable language-model quality with an INT4 output head, a ~4-bit KV cache, and both together?

**Status:** Complete. Awaiting founder/ChatGPT review. No next objective has been started.

**Owner / agent:** Claude (implementation)

**Started:** 2026-10-06
**Completed:** 2026-10-06

## Why we are doing this

EXP-0003 showed the ternary engine does not touch the two largest non-ternary streams of Falcon-E 1B (134 MB BF16 head per token; up to 805 MB BF16 KV reads per token at 32k context). Lower precision for those components must be validated against real model behaviour before it can become a hardware requirement. This is the first model-side experiment of FlexTern's model–hardware co-design direction (ternary-first, not ternary-only; ADR-0005, ADR-0006).

## Question we are trying to answer

Can the non-ternary memory footprint of an existing native ternary model be substantially reduced while preserving its useful language-model behaviour, and what does the real representation (packing, metadata, dequantization, cache semantics) look like?

## Outcome (one paragraph; details in EXP-0004)

MEASURED on one checkpoint at pilot scale: an INT4 RTN g128 head costs +0.67% WikiText-2 perplexity and no task accuracy outside one standard error (134 MB → 35 MB logical); a 4-bit KIVI-style KV cache costs +0.17–1.07% teacher-forced perplexity at 2k–4k depending on grouping (per-token grouping is the good one) and stores 3.5× fewer bytes; both together are additive. Resident memory shrinks, but on this runtime attention still reads BF16 and prefill is 4–8× slower. Candidate format written (`results/model_precision/candidate_format.md`), not adopted. Both techniques are category A (any GPU can use them).

## Scope (as executed)

- Feasibility audit; five configurations (baseline, A INT4 head, B/B1 4-bit KV in two groupings, C both); head RTN asymmetric g128 with explicit packing; KV via transformers `QuantizedCache` (hqq backend; quanto unusable, KI-006).
- Quality: lm-eval (WikiText-2, HellaSwag PILOT 2,000, PIQA, ARC-E/C, WinoGrande) for head configurations plus a KV-invariance control; teacher-forced incremental perplexity through the real cache at 2k and 4k; greedy generation at 11/1,500/6,144-token prompts.
- Memory sweep 128–16,384 (32k did not fit); latency with warmups and repeats; candidate-format report; EXP-0003 re-evaluation with measured bits.

## Explicitly out of scope

Training, fine-tuning, any RTL or FPGA work, compilers, compute-in-memory hardware, overwriting the canonical checkpoint, committing weights, selecting the final FlexTern format, performance claims from byte counts, competitor claims.

## Required validation

`tests/test_model_precision.py` (11 tests, all executed with the model loaded under `FLEXTERN_SLOW_TESTS=1`; 3 of them are gated behind that flag by default); all earlier suites (82 tests) still pass; executed evaluations recorded in `results/model_precision/manifest_*.json`.

## Relevant files

`analysis/model_precision/`, `tests/test_model_precision.py`, `scripts/run_model_precision.ps1`, `results/model_precision/`.

## Relevant ADRs

ADR-0004 (checkpoint), ADR-0005 (system scope), ADR-0006 (model-side track).

## Relevant experiments

EXP-0002B (workload), EXP-0003 (bottlenecks), EXP-0004 (this).

## Known uncertainties

One checkpoint; KV quality at pilot scale and ≤ 4k context; greedy-only generation; no path that reads 4-bit K/V directly, so traffic savings remain hypothetical (Q-022).

## Questions requiring founder decision

Acceptable-loss threshold; whether the candidate format becomes an ADR-level hardware assumption now or after a second checkpoint; whether FlexTern owns an attention/head engine that consumes compressed operands (Q-022). See the handoff `handoffs/2026-10-06-exp-0004-model-precision.md`.
