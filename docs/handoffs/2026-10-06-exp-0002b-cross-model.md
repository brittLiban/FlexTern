# Handoff

Date: 2026-10-06
Agent: Claude (Claude Code, model Fable 5.1), implementation role
Current commit: 73dc205 (EXP-0002B), followed by a docs-only fixup commit recording this hash

## Objective

FT-CURRENT, EXP-0002B: cross-model validation of EXP-0002 on TII Falcon-E 1B/3B. Complete.

## What I Completed

Generalized the profiler to a spec table, profiled Falcon-E 1B and 3B fully, cross-checked the 1B ternarization and scale convention on its latent branch (100% with BF16 emulation), verified the Microsoft outputs are numerically unchanged, generated a cross-model comparison, classified every property as generalizing / model-specific / uncertain, and documented implications for Y-Core and the planned roofline.

## What I Changed

- New: `analysis/model_specs.py`, `analysis/cross_model_compare.py`, `tests/test_analysis_cross_model.py`, `results/workload_profile/{falcon-e-1b,falcon-e-3b,cross_model}/`, ADR-0004, EXP-0002B, this handoff, two reviews.
- Rewritten: `analysis/ternary_model_profile.py` (spec-driven; same outputs for BitNet plus added columns), `analysis/ternarization_crosscheck.py` (`--model`, BF16 emulation), `scripts/run_workload_profile.ps1` (all models + compare), `analysis/README.md`.
- Updated: ADR-0003 (superseded), `OPEN_QUESTIONS.md` (Q-018 partial, Q-019, Q-020, new Q-021), `THESIS.md`, `ARCHITECTURE.md` (note only), `PROJECT_STATE.md`, `CURRENT_WORK.md`, `KNOWN_ISSUES.md` (KI-004), `README.md`.
- BitNet canonical outputs: all numeric values identical; columns added (divisibility, KV at max context, extra JSON fields); two JSON keys renamed (`embedding_bf16_tied_lm_head` → `embedding_bf16`, `pct_embedding` → `pct_embedding_and_head`).

## Tests / Validation Run

```
python -m unittest tests.test_analysis tests.test_analysis_cross_model     # 37 tests, OK
python analysis\ternary_model_profile.py --model bitnet-2b4t|falcon-e-1b|falcon-e-3b   # all asserts pass, SHA-256 OK
python analysis\ternarization_crosscheck.py --model falcon-e-1b             # scale 1/β ok; 100% bf16-emulated
python analysis\ternarization_crosscheck.py --model bitnet-2b4t             # scale β ok; 98-99.4% (KI-004)
python analysis\cross_model_compare.py
ad-hoc regression script: 10,929 committed BitNet numeric values vs regenerated, 0 differences
```

Phase 1 flow not re-run; nothing under `rtl/`, `tests/*.sv`, `synth/` changed.

## Actual Results

See EXP-0002B and `results/workload_profile/cross_model/comparison.md`. Headline: generalizing = ternary values, 2-bit packing, per-tensor post-accumulation scale, INT8 per-token activations (-128 unreachable), ±balance, ~40% unstructured zeros, two K per model all divisible by 256, BF16 head. Microsoft-specific = dead channels, tied head, head dominance, exact K, ReLU², sub-norms, β-vs-1/β convention.

## Remaining Problems

- None in the tooling. Falcon 3B's latent branch is sharded and was not cross-checked (not required).
- Phase 1 leftovers (`C:\oss-cad-suite`, 600 MB archive in the scratchpad) are still on disk.

## Important Context

- Scale convention is a loader concern: BitNet stores β and multiplies; Falcon stores 1/β and divides. Both via transformers, selected by `linear_class`.
- Both checkpoints use codes 00/01/10 for -1/0/+1 (code = q + 1); Y-Core v0.1's ADR-0002 encoding is 00/01/11. Either way a remap is needed at load; do not change the RTL for this.
- Falcon's `bfloat16` branch is a different, non-ternary model. Only `prequantized` is the latent of the packed branch.
- Falcon per-layer zero rate rises with depth (34% → 45%); BitNet spikes at layer 1. Neither has block structure along K.
- At 32k context the Falcon KV-cache read per token exceeds its entire weight stream (INFERRED).

## Questions for Founder / ChatGPT

1. Q-019 now has two faces: BitNet's BF16 head vs Falcon's long-context KV cache. Which non-ternary stream should the roofline treat as the primary system constraint, or should it sweep both?
2. Q-021: should the roofline (and any later RTL) carry activation bits as a parameter from the start, given 4-bit-activation research exists but no native release uses it yet?
3. Does the absence of dead channels in Falcon settle Q-020 as "compiler-only, if ever"?

## Recommended Immediate Next Consideration

Proceed to the planned compute-vs-memory roofline model (EXP-0003), but with the cross-model envelope from EXP-0002B as its input space: K 2048–13312 with 256-wide tiles, activation bits as a parameter (8 default), zero rate 38–47%, head precision and vocabulary as parameters, context 4k–32k with KV bytes/token/layer 1.0–2.6 KB, 2-bit and 1.6-bit packing. Read inputs from the three `model_summary.json` files rather than hand-entering one model's numbers. The accumulation RTL study recommended after EXP-0002 remains the natural datapath experiment afterward, now sized for K up to 13312 and 22-bit sums.
