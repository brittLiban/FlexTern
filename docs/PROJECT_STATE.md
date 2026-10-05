# FlexTern Project State

**Last verified:** 2026-10-05 (full validation flow run via `scripts/run_all.ps1`; all steps PASS)

**Current phase:** Phase 1 — ternary primitive. Complete; awaiting founder/ChatGPT review.

**Current objective:** FT-CURRENT (Y-Core v0.1 + Y-Dot16 + generic synthesis comparison) is **complete**. No next objective has been set. See `CURRENT_WORK.md`.

**Current architecture status:** Y-Core v0.1 and Y-Dot16 v0.1 implemented, verified, and synthesized (generic). INT8 baselines exist for measurement only. Nothing above Y-Dot16 exists. See `ARCHITECTURE.md`.

**What is working:**
- Python golden reference with 19 passing unit tests.
- Y-Core RTL: exhaustive 768-case verification plus reserved-encoding check, PASS.
- Y-Dot16 RTL: deterministic + 10,000 random + 2,000 Python cross-check vectors, PASS.
- INT8 baselines: 65,536 exhaustive products + dot16 tests, PASS.
- Yosys generic synthesis of all four modules, logs in `results/`.
- One-command Windows reproduction (`scripts/run_all.ps1`, ~14 min, mostly one WASM-Yosys step).

**What has been proven (MEASURED):**
- The ternary primitive needs no multiplier; Yosys infers none.
- Under generic synthesis, Y-Core is 33 cells vs 407 for an INT8 multiplier (12.3×, Yosys 0.69), and Y-Dot16 is 1,330 vs 8,646 (6.5×). A second Yosys version gives 7.1× and 5.2×.
- The advantage shrinks with scale because accumulation is a shared cost.

**What has NOT been proven:**
- Any physical result: FPGA utilization, frequency, ASIC area, power.
- Anything at system scale: memory traffic, control, interconnect, real workloads.
- Any tokens/sec, tokens/watt, cost, or GPU comparison.
- That the ratio survives a fairer accounting (ternary models need more parameters for equal quality).

**Current measured results:** `BENCHMARKS.md` rows 1–4, sourced from EXP-0001.

**Current blockers:** None technical. Awaiting direction.

**Current major uncertainty:** Whether the element-level advantage matters once accumulation, memory, and workload are included. EXP-0001 already shows the ratio halving from element to 16-element scale.

**Immediate next decision point:** Founder and ChatGPT review EXP-0001 and choose the next single objective. The handoff `handoffs/2026-10-05-phase1-y-core.md` offers one recommendation.
