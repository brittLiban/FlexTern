# Handoff

Date: 2026-10-07
Agent: Claude (Claude Code, model Fable 5.1), implementation and research role
Current commit: 32b954b (EXP-0006), followed by a docs-only fixup commit recording this hash

## Objective

FT-CURRENT, EXP-0006: model-resident memory feasibility (Gate 2 of the co-design program). Complete.

## What I Completed

A source table of 22 public technology references with provenance and labels plus 8 explicit assumptions; a reproducible analysis package (`analysis/memory_residency/`) that loads the canonical profiles and the EXP-0005 traffic/collector laws and computes capacity at 2/4/8/16 bits with head and metadata separate, hypothetical 7B–70B scaling, residency policies, 16–256-bank organization, local/external bandwidth at 10–250 tok/s, area ranges per density tier with reticle/die tier crossings, energy ranges per movement, package organizations, economics proxies, sensitivity thresholds, INT4 fairness, and the EXP-0003/0004/0005 reconnection; 17 tests; the EXP-0006 record with separate Gate 2A / 2B verdicts; docs; completion review.

## What I Changed

- New: `analysis/memory_residency/{sources,capacity,residency,physical,report}.py`, `tests/test_memory_residency.py`, `results/memory_residency/` (20 files), EXP-0006, this handoff, the completion review.
- Updated: `PROJECT_STATE.md`, `CURRENT_WORK.md` (complete), `THESIS.md` (H3/H6 memory evidence), `ARCHITECTURE.md` (§4 exploratory note on residency organizations), `ROADMAP.md`, `OPEN_QUESTIONS.md` (Q-005, Q-011, Q-015 context, Q-022 evidence; new Q-023 memory technology), `README.md`, `analysis/README.md`. `BENCHMARKS.md` and `KNOWN_ISSUES.md` unchanged (no measurement, no defect). No ADR.
- Untouched: all RTL, all earlier results.

## Tests / Validation Run

```
python -m unittest tests.test_memory_residency -v                        # 17 OK
python -m unittest tests.test_reference tests.test_analysis tests.test_analysis_cross_model tests.test_roofline tests.test_hardware_gate.TestPackingInterpretation tests.test_hardware_gate.TestWorkloadAndTrafficModels   # 98 OK
python analysis\memory_residency\report.py                               # seconds
```

RTL test classes of EXP-0005 were not re-run (no RTL or testbench changed).

## Actual Results

See EXP-0006 R1–R10 and `results/memory_residency/summary.md`. Headline: Falcon-E 1B needs 440 MB physical at 2 bits (10% margin), 404–648 mm² at the demonstrated whole-SRAM-heavy-die tier (0.68–1.09 MB/mm², N7), inside the Graphcore 900 MB / 823 mm² envelope; INT4 doubles it to a marginal fit; INT4 3B exceeds one reticle at every anchor. Bandwidth never limits at 1B–3B (one LPDDR5X channel streams ~120 tok/s). Energy: multi-MB SRAM-local ~5 mJ/token (planning figure, top of the 45 nm range) vs DRAM-class 12–48 mJ/token for the weight movement (INFERRED ranges, 2.4–9.4×). After residency the head and KV are 88–99.7% of global bytes. Gate 2A PASS (1B; 3B and BitNet marginal; ternary weights only, head and KV external), FAIL on-die at ≥7B, INCONCLUSIVE for DRAM-class near-memory; Gate 2B PASS structurally at 1B–3B, FAIL as a class change at ≥7B.

## Remaining Problems

- No modern-node SRAM read-energy measurement; the energy argument for residency rests on a 45 nm anchor.
- Compute (Y-Core bank) area is unmeasured and assumed absorbed in the whole-die density tier.
- Latency benefits are unquantified.
- eDRAM availability and multi-die SRAM stacking are assumptions from absence of evidence.

## Important Context

- Density tiers must never be mixed: raw bit cell (5.95 MB/mm²) ≠ macro (~4) ≠ cache-only die (1.64–1.86) ≠ whole SRAM-heavy die with compute (0.68–1.09). The last is the only one usable for a chip estimate.
- "Near-memory in HBM" and "resident in SRAM" are different claims: the former keeps DRAM-class read energy.
- Residency does not raise throughput at these sizes; it is an energy/latency choice. Do not present the 126× global-byte reduction as speed.
- The collector bound from EXP-0005 limits useful bank counts to ≤128 at 16 B per cycle; thousands of tiny banks are not the organization the evidence supports.
- Every constant is in `sources.py`; `test_11` fails if a numeric constant appears there without an S/A tag.

## Questions for Founder / ChatGPT

1. Is an energy/latency advantage worth 400–650 mm² of SRAM-heavy silicon when LPDDR5X streaming already serves 100 tok/s at 1B? This is the product question Gate 2 cannot answer.
2. Is the target market 1B–3B models (where residency fits one die) or larger (where it does not)? Q-015 context.
3. Should Q-022 (head/KV consumer) be the next hardware question, given it dominates the remaining global traffic?

## Recommended Immediate Next Consideration

Obtain a defensible modern-node SRAM read-energy figure and a latency model for the resident organization, then re-score the residency-versus-streaming energy and latency delta with it (a sourcing and modeling task, no hardware). The entire case for residency at 1B–3B now rests on that single pJ/bit number and on latency, since bandwidth does not require it and the area is demonstrated-class. If the delta survives, the next hardware question is the non-ternary consumer (Q-022), which this experiment shows dominates global traffic after residency. If it does not, the Youssouf Architecture should be re-scoped to a streaming engine with the EXP-0005 datapath fed by external DRAM.
