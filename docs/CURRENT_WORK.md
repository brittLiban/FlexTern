# Current Work

There is exactly one active engineering objective at a time. This file is the single active objective. The founder and ChatGPT choose the next objective after review; do not automatically start a backlog item.

---

**Active: none authorized. EXP-0013 Phase 1.1 closeout COMPLETE, STOPPED for founder review.** The closeout branch `exp-0013-phase1-1-closeout` (from `main` @ `6cbd44b`) is **not merged** and awaits explicit founder approval to merge. Record: `experiments/EXP-0013-phase1-1-closeout.md` (gates A-F, Phase 1.2 readiness memo in section 14).

Status of EXP-0013: Phase 0 (audit), Phase 1 and Phase 1.1 are **merged into `main`** at `6cbd44b` (together with EXP-0012, which was an ancestor of that branch); their records carry dated correction notes. Closeout findings: the historical paired comparisons were not truly paired (shared RNG; measured); under the new paired-replay protocol `obs_B_map_memory` still solves OccPick2D-LA v1 (200/200 in all three conditions; 87 unique layouts) and `obs_A_single_frame` gets 0.995 / 0.970 / 0.560, so v1 still does not test the predictive-state architecture; TEST layouts were used for Phase 1.1 *timing traces* (not task scoring) and are now guarded; inference execution re-verified (34 tests); AC-power latency repeat confirms the Phase 1.1 conclusions; energy unmeasured.

**Locked:** OccPick2D-LA v1 protocol (LA_T_PICK 65, cap 80, dilation 1, controllers `obs_A`/`obs_B`/privileged controls), all historical result files under `results/predictive_state*`. **Stopped:** EXP-0012 (verdict STOP), EXP-0013 Phase 1/1.1 (v1 benchmark saturated), Phase 2 as specified (BLOCKED).

**Authorized work: none.** **Blocked:** any neural-model training, learned world model, GPU sweep, Phase 1.2 / v2 implementation, new hidden-state environments, new TEST evaluations on existing ranges, paid cloud, hardware. **Pending founder approvals:** (1) merge of the closeout branch; (2) whether to authorize Phase 1.2 (v2-B action-dependent hidden consequence, software only, with v2-A as control) and the reservation of a fresh held-out layout range; (3) the unchanged EXP-0011 and EXP-0012 decisions below.

**Governance rule (founder, 2026-11-07):** all experimental work stops before merging; no agent merges an experiment branch into `main` without explicit founder approval (AGENTS.md §10).

**Last completed objective:** EXP-0012 — ternary predictive-state architecture for robotics (algorithmic feasibility, simulation only). Completed 2026-11-07 on branch `exp-0012-predictive-state` (merged to `main` as an ancestor of `6cbd44b`). **Verdict under the locked rules: STOP** (G0 discrimination failed on DEV, G1 floor failed on TEST); questions A-D reported descriptively. Records: `experiments/EXP-0012-predictive-state.md`, `experiments/EXP-0012-literature-review.md`, `experiments/EXP-0012-hardware-feasibility.md`, `reviews/REV-2026-11-07-exp-0012-adversarial-review.md`, `handoffs/2026-11-07-exp-0012.md`, `analysis/predictive_state/` (protocol_v1.json, protocol_v1_amended.json), `results/predictive_state/`.

**Previous objective:** EXP-0011 Stage 1, verdict SPEED_LOW. Its open founder decisions (quantized head / TL2 kernel investigation, Stage 2 energy, second strategic report, commercial discovery thresholds) are unchanged.

**Not authorized / not started:** silicon, FPGA expansion, RTL redesign, hardware purchases, paid cloud, physical robots, Stage 2 energy measurement, any follow-up to EXP-0012 (see its handoff for options).
