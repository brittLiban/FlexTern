# Handoff

Date: 2026-10-06
Agent: Claude (Claude Code, model Fable 5.1), implementation role
Current commit: the Phase 2 commit ("research: characterize real ternary transformer workload")

## Objective

FT-CURRENT, Phase 2: characterize a real ternary transformer workload to determine the next hardware problem. Complete.

## What I Completed

Full workload profile of `microsoft/bitnet-b1.58-2B-4T` (revision 04c3b9ad): architecture, every linear shape, exact -1/0/+1 counts for all 2.08 B ternary weights, zero structure and dead channels, scale and activation schemes from reference code with a BF16 cross-check, operation inventory, storage and decode-traffic models, accumulator widths, evaluation of Y-Core v0.1 assumptions, and a ranked list of candidate next experiments with one recommendation. See EXP-0002.

## What I Changed

- New: `analysis/` (profile tool, safetensors reader, cross-check, README), `tests/test_analysis.py`, `scripts/run_workload_profile.ps1`, `results/workload_profile/` (15 generated files).
- New: ADR-0003 (reference workload), EXP-0002, this handoff, two reviews in `docs/reviews/`.
- Updated: `CURRENT_WORK.md`, `PROJECT_STATE.md`, `THESIS.md` (H3/H4 evidence), `ROADMAP.md`, `ARCHITECTURE.md` (workload-fit note only), `OPEN_QUESTIONS.md` (Q-005/6/7/11 evidence; Q-018/19/20 new), `KNOWN_ISSUES.md` (KI-004), `README.md`, `.gitignore`.
- Wording fix: "adder tree" → "adder chain" in `CURRENT_WORK.md`, `results/benchmark.md`, comments in `rtl/y_dot16.sv` and `rtl/int8_dot16_baseline.sv`, and `synth/y_dot16.ys`. Comment-only; no RTL logic touched.

## Tests / Validation Run

```
python -m unittest tests.test_analysis -v        # 16 tests, OK
python analysis\ternary_model_profile.py          # all in-tool asserts pass; SHA-256 OK
python analysis\ternarization_crosscheck.py       # 3 tensors; scale = absmean; 98-99.4% code agreement
```

```
powershell -File scripts\run_all.ps1 -SkipSynth   # Phase 1 simulations re-run after comment edits: all PASS
```

Synthesis was not re-run; no RTL logic changed, and the comment edits cannot alter a netlist.

## Actual Results

Headline numbers in EXP-0002 and `results/workload_profile/workload_summary.md`. Key: 2,084,044,800 ternary weights; 42.21% zero, 28.89% each ±1; K ∈ {2560, 6912}; INT8 per-token activations with -128 unreachable; per-tensor BF16 scale after accumulation; 4.76% dead MLP channels (43.6% in layer 1); head is 56% of per-token weight bytes under stream-once decode; accumulators need 20/21 bits.

## Remaining Problems

- The Hugging Face cache on this machine is in symlink-degraded mode (Windows, no Developer Mode). Harmless.
- `C:\oss-cad-suite` and the 600 MB archive from Phase 1 are still on disk; still dead weight.

## Important Context

- The checkpoint packs four ternary values per byte along the **output** dimension. One byte at column k holds weight[k] for four consecutive output rows. Any streaming datapath that consumes bytes directly feeds four dot products at once.
- The checkpoint's code assignment is 00/01/10 → -1/0/+1 (code = q + 1), not ADR-0002's 00/01/11. Remap is trivial, but do not assume they match.
- `weight_scale` is β = mean|W_latent| and multiplies the output. Values are 0.7–4.6 because the latent weights are large, not because the scale is inverted.
- Zero placement along K is independent once all-zero rows are removed. Do not design block-sparse hardware on the strength of the 42% figure.
- All traffic and MAC numbers are static. Nothing was run.

## Questions for Founder / ChatGPT

1. **Scope of the output head (Q-019).** It is 657 MB of BF16 per token under stream-once decode, more than all ternary layers. Is quantizing or otherwise handling it FlexTern's problem, or an accepted fixed cost? This decides whether "memory-feed architecture" can even be a FlexTern experiment.
2. **Is a second native ternary model available or expected** that would let Q-018 be answered before hardware decisions harden around one checkpoint?
3. **Does the dead-channel finding belong to a future FlexTern compiler/packing layer**, and should that layer start earlier in the roadmap than Phase 5?

## Recommended Immediate Next Consideration

EXP-0003, accumulation architecture at real dot lengths (chain vs tree vs carry-save vs dual-sign, 256-wide tiles to K = 2560/6912, 20/21-bit outputs, generic cells and depth). It is fully within FlexTern's control, it is already the dominant logic in Y-Dot16, and every Y-Array decision depends on it. The memory-stream question should be decided at the scope level (Q-019) in parallel, not built first.
