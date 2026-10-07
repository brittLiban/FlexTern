# FlexTern Project State

**Last verified:** 2026-10-07 (EXP-0008 SRAM validation; 15 new tests PASS; all earlier non-RTL suites PASS in one process; EXP-0001..0007 artifacts and RTL unchanged apart from a module rename in the EXP-0007 package)

**Current phase:** Phase 2 — workload characterization and system bottleneck analysis — plus the model-side track (ADR-0006) and the three co-design gates: Gate 1 dataflow (EXP-0005), Gate 2 memory residency (EXP-0006), Gate 3 datacenter falsification (EXP-0007), Gate 4 modern SRAM energy and leakage (EXP-0008). Complete; awaiting founder/ChatGPT review.

**Current objective:** FT-CURRENT (EXP-0008, modern SRAM energy and latency validation) is **complete**. No next objective has been set. See `CURRENT_WORK.md`.

**Current architecture status:** Y-Core v0.1 and Y-Dot16 v0.1 (Phase 1) plus experimental packed consumers from EXP-0005; verified at RTL and generic-synthesis level; **not frozen, no ADR**. Memory organization for the bank-local form is exploratory (EXP-0006). The distributed, model-resident datacenter form is **not supported by the evidence** beyond small-batch interactive serving (EXP-0007). The edge/resident 1B–3B form has a sourced ~10× dynamic read-energy advantage over LPDDR5X and an unresolved standby-power question (EXP-0008). System scope fixed by ADR-0005: ternary-first, not ternary-only.

**What is working:**
- Phase 1 RTL, testbenches, synthesis flow (EXP-0001); workload profiler (EXP-0002/0002B); roofline (EXP-0003); GPU model-precision harness (EXP-0004); packed-ternary engines and traffic models (EXP-0005); memory-residency package (EXP-0006); datacenter serving model with a sourced FP4/HBM baseline, KV-capacity gates, pipeline reuse and provenance tests (EXP-0007); SRAM-versus-LPDDR5X energy, leakage, latency and break-even model on tiered primary sources (EXP-0008).

**What has been proven (MEASURED):**
- EXP-0001: ternary primitive needs no multiplier; 5–12× fewer generic cells than INT8 multiply.
- EXP-0002/0002B: across two families, ternary 2-bit storage, per-tensor post-accumulation scale, INT8 per-token activations, ~40% unstructured zeros, two dot lengths per model all divisible by 256, BF16 non-ternary head.
- EXP-0004 (one checkpoint, pilot scale): Falcon-E 1B tolerates an INT4 output head (+0.67% perplexity) and a 4-bit KV cache (+0.17–1.07%), additively.
- EXP-0005: the canonical 2-bit packing is consumed directly at initiation interval 1; 25,466 generic cells, no multiplier, versus 69,526 for an INT4 structural baseline; a bank-local engine gives bit-identical results reading every weight byte locally and none globally.

**What has been modeled (ANALYTICAL / INFERRED):**
- EXP-0003: ideal ternary-only ceilings ≤1.8× for BitNet at any context, ≥3× for Falcon-E below ~2.6k/7k context, 1.4–1.6× at 32k.
- EXP-0005: bank-local placement removes the ternary weight stream from the global path if 396–730 MB stay resident; useful bank counts ≤ 128.
- EXP-0006: 1B–3B residency fits one reticle-limited SRAM-heavy die at demonstrated densities; 2-bit keeps 3B inside where INT4 does not; ≥7B needs multi-die or DRAM-class capacity; bandwidth never forces residency at these sizes (one LPDDR5X channel streams ~120 tok/s); the resident case is energy and latency.
- EXP-0007 (against a sourced B200/MI355X-class FP4 node with native FP8 KV and continuous batching, bound within 16% of MLPerf): a tensor-parallel HBM node amortizes weights over the whole batch while a resident pipeline amortizes only over its microbatch; the resident movement-energy advantage is 5× at batch 1, 2.7× at batch 2, below 2× from batch 4 at 7B–400B, negative from batch 8 to 256 at 70B; single-tile 1B keeps ≥2× to batch 8 at 2k only; KV dominates every system beyond 8k context; interconnect never limits; ternary halves tiles versus INT4 (42 vs 162 at 70B) but at the TARGET compute density the resident 70B/400B systems deliver 2–4× lower throughput than one node; prefill on resident tiles is 17–550× slower; the resident system is slower than the node in every cell. Ternary streaming on conventional hardware keeps 2–3× fewer weight bytes at small batch.
- EXP-0008 (T1 SRAM-macro papers at 5–7 nm, Micron LPDDR5X datasheet IDD arithmetic): a resident bank read costs 0.33–0.62 pJ/bit (macro 0.03–0.15, bank overhead 0.05–0.15, wires 0.25–0.32) against 3.5–6.1 pJ/bit for LPDDR5X through the SoC path, roughly 10× (5.6–18×), superseding EXP-0006's 1.6 pJ/bit planning figure and 3–9× range; 440–811 MB of always-on SRAM at an assumed 0.05–1.0 mW/MB (planning 0.6; modern-node leakage unmeasured) is 22–811 mW standing power, so residency wins total memory power only above ~23 tok/s continuous or ~85% duty at 25 tok/s at the planning figure (32–40% with a hypothesized 0.1× retention mode); first-word latency is hidden by 1.5 KB of prefetch and one x16 channel caps streaming at ~30 / 16 / 23 tok/s for 1B / 3B / BitNet; with head and KV external the edge total advantage is 1.4–3.3× at BF16 and 2.2–5.8× with the EXP-0004 candidate.

**What has NOT been proven:**
- Any physical result: FPGA utilization, frequency, ASIC area, power, bandwidth, tokens/sec, energy per token, tokens/MW, cost. No FlexTern silicon or design exists.
- Modern-node SRAM leakage per MB and retention behaviour (Q-023, narrowed), the number the edge verdict now rests on; the read energy is sourced at order of magnitude (EXP-0008) but not measured on any FlexTern design; compute energy of ternary versus FP4 arithmetic (unsourced); Y-Core bank area.
- A commercially meaningful datacenter regime (Q-024); the real cost of the non-ternary consumer (Q-022); quality beyond one checkpoint (Q-019); generalization beyond two families (Q-018); INT8 activation persistence (Q-021).

**Current measured results:** `BENCHMARKS.md` (generic synthesis, EXP-0001 and EXP-0005); `../results/workload_profile/cross_model/comparison.md`; `../results/roofline/summary.md`; `../results/model_precision/quality_comparison.csv`; `../results/hardware_gate/summary.md`; `../results/memory_residency/summary.md`, `../results/datacenter_gate/summary.md` and `../results/sram_validation/summary.md` (analytical, not measured).

**Current blockers:** None technical. Awaiting direction. No FPGA for months (by founder statement).

**Current major uncertainty:** Whether any FlexTern form has a commercially meaningful regime: the edge/resident 1B–3B form wins ~10× on dynamic weight-read energy but its total memory power depends on an unmeasured modern-node SRAM leakage and on the product's duty cycle (break-even ~23 tok/s continuous or ~85% duty at the planning leakage); small-batch interactive serving and ternary-as-a-format on conventional hardware remain; the model-resident datacenter rack is not supported.

**Immediate next decision point:** Founder and ChatGPT review EXP-0008's four gate verdicts, fix the target edge duty cycle and token rate, decide whether the EXP-0004 candidate head/KV becomes the edge baseline, and decide whether to resolve the leakage term (Q-023, narrowed) next. The handoff `handoffs/2026-10-07-exp-0008-sram-validation.md` recommends one objective.
