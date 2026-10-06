# FlexTern Project State

**Last verified:** 2026-10-06 (EXP-0002B: three-model profile run, 37 analysis tests PASS, BitNet regression 10,929 values identical; Phase 1 flow unchanged since 2026-10-05)

**Current phase:** Phase 2 — workload characterization. Cross-model validation complete; awaiting founder/ChatGPT review.

**Current objective:** FT-CURRENT (EXP-0002B, validate EXP-0002 findings on Falcon-E) is **complete**. No next objective has been set. See `CURRENT_WORK.md`.

**Current architecture status:** Unchanged since Phase 1. Y-Core v0.1 and Y-Dot16 v0.1 implemented and verified. No hardware was built in Phase 2. See `ARCHITECTURE.md`.

**What is working:**
- Phase 1 RTL, testbenches, synthesis flow (EXP-0001).
- Spec-driven workload profiler covering three pinned checkpoints (ADR-0004), with byte-level reconciliation, BF16 latent cross-checks, and a cross-model comparison generator; one command reproduces all of it.

**What has been proven (MEASURED):**
- EXP-0001: ternary primitive needs no multiplier; 5–12× fewer generic cells than INT8 multiply.
- EXP-0002/EXP-0002B, across BitNet 2B4T and Falcon-E 1B/3B: ternary {-1,0,+1} stored 2-bit; one per-tensor scale applied after accumulation; INT8 per-token absmax activations with -128 unreachable; ±1 balanced within 0.1 pt; 39.7–42.2% zeros, unstructured along K; exactly two dot lengths per model (hidden, intermediate), all five observed values divisible by 256; BF16 non-ternary output head in both families.
- Microsoft-specific (not ternary properties): dead MLP channels, tied head, head dominating traffic, ReLU², sub-norms, stored-scale convention.

**What has NOT been proven:**
- Any physical result: FPGA utilization, frequency, ASIC area, power.
- Any runtime measurement; all arithmetic and traffic figures are static estimates.
- That findings hold beyond two families (Q-018), or that INT8 activations persist (Q-021).
- Any tokens/sec, tokens/watt, cost, or GPU comparison.

**Current measured results:** `BENCHMARKS.md` (hardware, EXP-0001); `../results/workload_profile/cross_model/comparison.md` (workloads, EXP-0002B).

**Current blockers:** None technical. Awaiting direction.

**Current major uncertainty:** Which non-ternary stream a system must feed first: the BF16 head (BitNet, vocabulary-driven) or the KV cache at long context (Falcon-E). This, more than the ternary datapath, now looks like the system-level question for the planned roofline model.

**Immediate next decision point:** Founder and ChatGPT review EXP-0002B's invariant classification and the revised roofline input envelope, then choose the next objective. The handoff `handoffs/2026-10-06-exp-0002b-cross-model.md` recommends one.
