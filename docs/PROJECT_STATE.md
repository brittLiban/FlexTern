# FlexTern Project State

**Last verified:** 2026-10-06 (EXP-0002 profile run; `scripts/run_workload_profile.ps1` all steps PASS; Phase 1 flow unchanged since 2026-10-05)

**Current phase:** Phase 2 — workload characterization. Complete; awaiting founder/ChatGPT review.

**Current objective:** FT-CURRENT (characterize BitNet b1.58 2B4T) is **complete**. No next objective has been set. See `CURRENT_WORK.md`.

**Current architecture status:** Unchanged since Phase 1. Y-Core v0.1 and Y-Dot16 v0.1 implemented and verified. No hardware was built in Phase 2. See `ARCHITECTURE.md`.

**What is working:**
- Phase 1 RTL, testbenches, synthesis flow (see EXP-0001).
- Phase 2 analysis tool: reproducible profile of the reference model in ~40 s from a cached checkpoint, with byte-level file reconciliation, 16 unit tests, and a BF16 cross-check of the official ternarization.

**What has been proven (MEASURED):**
- EXP-0001: ternary primitive needs no multiplier; 5–12× fewer generic cells than INT8 multiply depending on scale and tool version.
- EXP-0002, reference model `microsoft/bitnet-b1.58-2B-4T`: 2.084 B ternary weights; 42.2% zero, 28.9% each ±1; only two dot lengths, K = 2560 (74.5% of weights) and K = 6912 (25.5%); activations are per-token absmax INT8 with -128 unreachable; one BF16 scale per tensor applied after accumulation; 4.76% of MLP intermediate channels are entirely dead (43.6% in layer 1); the tied BF16 output head is 13.6% of parameters but 55.7% of file bytes and, under a stream-once model, 56% of per-token weight traffic.

**What has NOT been proven:**
- Any physical result: FPGA utilization, frequency, ASIC area, power.
- Any runtime measurement of the model; all arithmetic and traffic figures are static estimates.
- That findings from this one model generalize to other ternary models (Q-018).
- Any tokens/sec, tokens/watt, cost, or GPU comparison.

**Current measured results:** `BENCHMARKS.md` (hardware, EXP-0001); `../results/workload_profile/workload_summary.md` (workload, EXP-0002).

**Current blockers:** None technical. Awaiting direction.

**Current major uncertainty:** Whether FlexTern's next step should address the ternary datapath's accumulation (the Phase 1 finding) or the memory stream, where the un-ternarized output head and the activation-side work now look larger than the ternary arithmetic problem.

**Immediate next decision point:** Founder and ChatGPT review EXP-0002's ranking of candidate experiments and choose one. The handoff `handoffs/2026-10-06-phase2-workload-profile.md` recommends one.
