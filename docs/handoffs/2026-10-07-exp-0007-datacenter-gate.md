# Handoff

Date: 2026-10-07
Agent: Claude (Claude Code, model Fable 5.1), implementation and research role
Current commit: 925510b (EXP-0007), followed by a docs-only fixup commit recording this hash

## Objective

FT-CURRENT, EXP-0007: datacenter thesis falsification (Gate 3). Complete.

## What I Completed

A sourced current-generation conventional baseline (B200/MI355X class: 180 GB HBM3e, 8 TB/s, FP4 9 PFLOPS dense, NVLink, native FP8/NVFP4 KV, continuous batching) and a reproducible analytical package (`analysis/datacenter_gate/`) that scores four systems (A conventional FP4 HBM, B ternary streaming, C ternary resident tiles, D INT4 resident tiles) across 1B/3B (canonical) and 7B/70B/400B (hypothetical Llama-class shapes), batch 1–256, context 128–131k, prefill and decode, three KV precisions, two head precisions and two vocabularies, four tile capacities, two pipeline modes, serving utilizations, with KV-capacity gates, topology/pipeline/interconnect accounting, movement-energy ranges, economic proxies, crossover thresholds, an MLPerf sanity anchor and 16 tests; the EXP-0007 record with separate Gate 3A/3B/3C verdicts; docs; completion review.

## What I Changed

- New: `analysis/datacenter_gate/{sources,workloads,systems,report}.py`, `tests/test_datacenter_gate.py`, `results/datacenter_gate/` (19 files), EXP-0007, this handoff, the completion review.
- Updated: `PROJECT_STATE.md`, `CURRENT_WORK.md` (complete), `THESIS.md`, `ARCHITECTURE.md` (§4 note), `ROADMAP.md`, `OPEN_QUESTIONS.md` (Q-005, Q-015, Q-019, Q-022, Q-023 evidence; new Q-024 datacenter regime), `README.md`, `analysis/README.md`. `BENCHMARKS.md` and `KNOWN_ISSUES.md` unchanged. No ADR.
- Untouched: RTL, all earlier results.

## Tests / Validation Run

```
python -m unittest tests.test_datacenter_gate -v                         # 16 OK
python -m unittest tests.test_reference tests.test_analysis tests.test_analysis_cross_model tests.test_roofline tests.test_memory_residency tests.test_hardware_gate.TestPackingInterpretation tests.test_hardware_gate.TestWorkloadAndTrafficModels   # 115 OK
python analysis\datacenter_gate\report.py                                # seconds
```

EXP-0005 RTL test classes were not re-run (no RTL or testbench changed).

## Actual Results

See EXP-0007 R1–R12 and `results/datacenter_gate/summary.md`. Headline: batching and KV erase the resident advantage outside small batch and short context; at 70B a resident pipeline of 40 stages gets weight reuse of only the microbatch, so at batch ≥ 8 it moves more bytes than the HBM baseline and at the TARGET compute density it delivers a fraction of the baseline's throughput; prefill is 17–550× slower on resident tiles at that density; ternary halves tile count (42 vs 162 at 512 MB for 70B) but does not change the verdict. The surviving regime is batch 1–2 (7B–400B) or ≤ 8 (single-tile 1B) at ≤ 8k context, the interactive case, where the movement-energy advantage is 2–5× conservative and there is no latency advantage. Gate 3A FAIL beyond small batch; Gate 3B INCONCLUSIVE-to-FAIL (architecturally feasible, economically dominated); Gate 3C PASS only for the small-batch interactive regime and the edge/resident 1B–3B form.

## Remaining Problems

- Compute energy is unsourced: the energy proxy is movement-only, so the verdict is a movement verdict.
- The resident compute density is a TARGET (4.1 T terms/s per tile); a higher density changes throughput but not the reuse arithmetic.
- Serving utilization and interconnect energy tiers are assumptions with ranges.
- Modern-node SRAM read energy (Q-023) remains unresolved; the founder considers it important for the edge/resident commercialization path.

## Important Context

- The reuse arithmetic is the whole result: a tensor-parallel HBM node reads each weight shard once per step for the whole batch; a resident pipeline reads each stage's weights once per microbatch (B/stages). Batch-sequential mode restores reuse but idles all stages but one.
- KV capacity gates every long-context cell: resident SRAM tiles cannot hold KV beyond trivial contexts, so C/D pay DRAM-class energy for KV exactly like A.
- The 1B–3B single-tile case (stages = 1) is the only place residency keeps full-batch reuse; that is the edge/resident form, not a datacenter one.
- Prefill at the target compute density takes seconds to tens of seconds per 8k prompt on resident tiles versus tens of milliseconds on the baseline; a hybrid (conventional prefill, resident decode) would be the only datacenter form worth modeling further.

## Questions for Founder / ChatGPT

1. Does FlexTern pursue the datacenter form at all, given the evidence that only the batch-1–2 interactive regime survives at 7B+, or does it concentrate on the edge/resident 1B–3B form where EXP-0006 found the organization credible?
2. If the datacenter form continues, is the hybrid (conventional prefill + resident decode at small batch) the only configuration to study?
3. Is the modern-SRAM energy/latency validation (Q-023) now the commercialization-critical measurement for the edge form?

## Recommended Immediate Next Consideration

Resolve Q-023 for the edge/resident form: obtain a defensible modern-node multi-MB SRAM read-energy and access-latency figure (published ISSCC/VLSI macro measurements or a vendor datasheet) and re-score the EXP-0006 residency-versus-LPDDR5X delta and the EXP-0007 small-batch regime with it. Every surviving FlexTern advantage, edge and interactive-datacenter alike, is now a movement-energy claim that rests on that single unmeasured number; the datacenter batching result does not depend on it, so this is the one uncertainty whose resolution changes the decision.
