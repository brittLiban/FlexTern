# Current Work

There is exactly one active engineering objective at a time. This file is the single active objective. The founder and ChatGPT choose the next objective after review; do not automatically start a backlog item.

---

**Active: EXP-0013 (persistent predictive state, counterfactual planning, adaptive perception) — Phase 1.1 COMPLETE, STOPPED for founder review.** Branch `exp-0013-predictive-planning` (not merged). Records: `experiments/EXP-0013-audit.md` (Phase 0), `experiments/EXP-0013-phase1.md` (Phase 1), `experiments/EXP-0013-phase1-1.md` (Phase 1.1: decision memo in section 0). Phase 1.1 found: executed-work counters fixed and spy-verified; encoder skipping cuts MACs but gives no end-to-end speedup on batch-1 CPU or CUDA (batched CPU -9 to -19%); OccPick2D-LA v1 is solved by a non-learned observation-matched map+memory baseline (1.00), so it does **not** test the predictive-state architecture. Recommendation: Phase 2 as specified is BLOCKED; modify to a v2 benchmark with hidden dynamic state (software only first). No learned world model, no training, no quantization. Awaiting founder decision.

(Previous state, kept for context:) No other objective active; EXP-0012 awaiting review.

**Last completed objective:** EXP-0012 — ternary predictive-state architecture for robotics (algorithmic feasibility, simulation only). Completed 2026-11-07 on branch `exp-0012-predictive-state` (not merged to main). **Verdict under the locked rules: STOP** (G0 discrimination failed on DEV, G1 floor failed on TEST); questions A-D reported descriptively. Records: `experiments/EXP-0012-predictive-state.md`, `experiments/EXP-0012-literature-review.md`, `experiments/EXP-0012-hardware-feasibility.md`, `reviews/REV-2026-11-07-exp-0012-adversarial-review.md`, `handoffs/2026-11-07-exp-0012.md`, `analysis/predictive_state/` (protocol_v1.json, protocol_v1_amended.json), `results/predictive_state/`.

**Previous objective:** EXP-0011 Stage 1, verdict SPEED_LOW. Its open founder decisions (quantized head / TL2 kernel investigation, Stage 2 energy, second strategic report, commercial discovery thresholds) are unchanged.

**Not authorized / not started:** silicon, FPGA expansion, RTL redesign, hardware purchases, paid cloud, physical robots, Stage 2 energy measurement, any follow-up to EXP-0012 (see its handoff for options).
