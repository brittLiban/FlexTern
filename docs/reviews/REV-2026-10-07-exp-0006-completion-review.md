# Review — EXP-0006 model-resident memory feasibility (completion)

Reviewer: Fable 5.1 (advisor role, consulted by the implementing Claude agent)
Date: 2026-10-07
Scope: EXP-0006 and its artifacts (`analysis/memory_residency/`, `tests/test_memory_residency.py`, `results/memory_residency/`, the EXP-0006 record, the handoff, and the THESIS / OPEN_QUESTIONS / PROJECT_STATE / ARCHITECTURE wording) as they stood immediately before the EXP-0006 commit (32b954b). Per the founder's instruction that the reviewer receive a different prompt from the implementation, the following twenty points were put to the reviewer verbatim: attack (1) source quality, (2) process-node mixing, (3) SRAM bit-cell versus macro-density confusion, (4) DRAM versus SRAM comparability, (5) vendor peak versus sustainable bandwidth, (6) energy extrapolation, (7) the full-model residency assumption, (8) whether partial residency was fairly considered, (9) INT4 fairness, (10) whether the near-memory advantage is incorrectly attributed to ternary, (11) whether area ranges are defensible, (12) whether packaging assumptions are realistic, (13) whether 3D/chiplet bandwidth is treated as free, (14) activation broadcast, (15) collector scaling, (16) model-size scaling, (17) economic claims, (18) the Gate 2A verdict, (19) the Gate 2B verdict, (20) the recommended next objective.

## Overall Assessment

Capacities, margins, both density tiers, reticle classifications, the bank sweep, bandwidth figures, energy ranges and reconnection shares recompute from the artifacts. Seven defects were found and fixed before commit; none changed a headline number, two contradicted the experiment's own tables:

1. The Gate 2B verdict said INT4 at 3B "exceeds one reticle at every anchor and margin" while the sensitivity table showed it fits the cache-only-die tier at ≤ 5% margin. Reworded to "every whole-die anchor and margin; cache-only-die anchor only with ≤ 5% margin and no compute" in the record, CURRENT_WORK and PROJECT_STATE.
2. BitNet's Gate 2A "PASS" rested on 851 versus 858 mm² at the low-density end, and its tied 657 MB BF16 head stays external and is larger than the resident weights. Now "PASS (marginal)" with both facts stated.
3. The KV cache was absent from the resident capacity discussion. The record now states that residency is of the ternary weights only and that at 32k context the KV cache (805 MB BF16, 227 MB at the candidate bits) alone exceeds the resident footprint.
4. The SRAM energy low end (0.16 pJ/bit) is an 8 KB array figure that does not apply to 3–25 MB banks. The planning figure is now the top of the range (~1.6 pJ/bit, 5.1 mJ/token), narrowing the resident-versus-DRAM gap to 2.4–9.4× (7–43 mJ/token) from the 2.5–40× first stated.
5. The full-residency row amortized preload over an untagged 1,000 tokens (126×) while EXP-0005 used steady state (144×). Both rows are now reported and the amortization is assumption A10.
6. Two constants in `physical.py` (the multi-die SRAM ceiling) lacked provenance tags; now A9 anchored on d-Matrix's 2 GB per card and referenced from the code.
7. The record described the sources as "vendor documents, conference reports and standards"; only six of twenty-two are primary documents. `source_notes.md` and the record now name them and label the rest as technical-press summaries adequate for order-of-magnitude use.

Also fixed: the UPMEM sentence now states that the capacity-minimal 7-bank mapping needs 8× the per-bank bandwidth UPMEM provides; the N5 → N3E bit-cell stall (0.021 µm² both) is noted as sourced evidence that node migration will not move the chip-tier density much.

## What Looks Strong

- Four density tiers kept apart, raw bit-cell density marked unusable, and a test that fails on any untagged numeric constant in `sources.py`.
- The traffic levels and the collector bound are imported from EXP-0005, so the residency and bank tables cannot drift from the measured RTL laws.
- Residency policies tested in both directions: reload-per-token policies collapse to streaming, hot-set residency is linear, and the test suite asserts both.
- The DRAM-class near-memory organizations are refused the SRAM energy class explicitly: "resident in HBM" keeps a DRAM read.
- Every area, energy and economics row carries an INFERRED or PROXY label; nothing is summed into a chip J/token or a price.

## Concerns (verdicts on the twenty points)

1. **Source quality.** Adequate for order-of-magnitude conclusions; mostly secondary summaries of primary disclosures; now stated.
2. **Process-node mixing.** Avoided: the chip tier is N7-only (Dojo, Graphcore) and says so; the N5 wafer-scale figure lands inside it, consistent with the SRAM scaling stall.
3. **Bit-cell versus macro.** Four tiers, raw marked unusable, test-enforced. Sound.
4. **DRAM versus SRAM comparability.** DRAM kept as a separate energy class; HBM-resident organizations not given SRAM energy. Sound.
5. **Peak versus sustained.** 50/70/80% utilization applied everywhere; peaks labeled.
6. **Energy extrapolation.** Labeled ranges, never summed to a chip figure; low end corrected (fix 4). The SRAM side rests on a 45 nm tutorial and the record says so.
7. **Full-residency assumption.** Tested both ways; the policy table shows reload equals streaming.
8. **Partial residency.** Layer, tiled and hot-set modeled; the linear result is stated without spin.
9. **INT4 fairness.** Identical architecture; both bit widths through every table; the EXP-0005 cells ratio cited, not re-derived.
10. **Attribution.** Near-memory global-traffic benefit stated as format-neutral; ternary's gain confined to capacity, local bandwidth, local energy and cells.
11. **Area ranges.** Defensible as INFERRED bounds; the whole-die tier is conservative for an SRAM-dominant die, since the reference chips carry far more compute per MB than Y-Core banks would.
12. **Packaging.** Seven organizations with availability labels; deep SRAM stacks flagged as not shipping.
13. **3D/chiplet bandwidth.** Not treated as free: crossing energy range included; the weight-byte crossing is identified for the stacked-cache and HBM organizations.
14. **Broadcast.** EXP-0005 law reused: ≤ 12% of weight bytes to 128 banks, 24% at 256.
15. **Collector.** Bound preserved and now the stated limit on useful bank count.
16. **Model-size scaling.** Capacity only, labeled HYPOTHESIZED; 2–3.4 B ternary parameters per reticle at the chip tier is the right headline.
17. **Economics.** Proxies only, no prices.
18. **Gate 2A.** Follows with fixes 2 and 3.
19. **Gate 2B.** Follows with fix 1.
20. **Next objective.** Sourcing a modern-node SRAM read energy and a latency model is the single variable the verdict turns on; needs no hardware; pre-empts no founder decision.

## Assumptions Being Made

- Densities from six public dies on two nodes; compute area for Y-Core banks unmeasured and assumed absorbed in the whole-die tier.
- Energy: SRAM side from a 45 nm tutorial; DRAM side modeled papers and vendor figures; no modern-node SRAM measurement.
- eDRAM availability and multi-die SRAM stacking inferred from absence of public evidence.
- Hypothetical models: 95% ternary fraction, Llama-class shapes, no quality claim.

## Potential Failure Modes

- Reading "144× fewer global bytes" as throughput or energy; bandwidth does not require residency at these sizes.
- Quoting any mm² figure as a FlexTern die size.
- Treating "near-memory in HBM" as if it had SRAM energy.
- Forgetting that the head and KV cache stay external and dominate the remaining global traffic.

## Recommended Changes

All applied before commit: the seven fixes above, the UPMEM wording, the bit-cell stall note, the two-line commit trailer.
