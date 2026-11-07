# Review: EXP-0012 (ternary predictive-state architecture) — adversarial review of the completed study

Date: 2026-11-07
Reviewer: the `advisor` tool (a stronger reviewer model that receives the full session transcript). **This is not an independently prompted agent**: the advisor tool takes no parameters, so the review prompt could not be varied from the implementation context. Three consultations are summarized: before approach (orientation), mid-study before any TEST measurement, and after results (this file's main content). A review is advice, not an accepted change.

## Consultation 1 — before the approach was fixed (adopted)
Keep the scope small and tiny-model; isolate code and put new packages in a separate venv (`~/ftvenv` untouched); time-box the simulator probe and fall back to a custom 2D benchmark if rendering is not clean; verify every citation and label unverified ones; compute the Amdahl ceilings before training; lock the protocol with numeric gates before any measurement; require a discrimination check and stop if the baselines do not separate; give INT8/INT4 the same QAT as ternary; make JEPA an auxiliary loss with an EMA target and monitor collapse; use a plain-PyTorch recurrent reference instead of building Mamba kernels; keep adaptive-perception triggers free of the current frame's expensive encoding and charge their cost; do not present analytical savings as measured; commit to the experiment branch only. All adopted (see the protocol and the experiment record).

## Consultation 2 — mid-study, before TEST (adopted unless noted)
Amendment 4 sound (one budget for all arms, chosen on dev). Fix before test: (a) fixed-rate trigger must be calibrated empirically on the measured encoder fraction, not k = 1/budget — **fixed** (amendment 5a); (b) validate the probe with positive controls, standardized features, CV ridge, more rollouts — **fixed** (5b); (c) LONG metric mislabeled/confounded; add per-pick conditional success — **fixed** (5c); (d) plan adaptive evals for the INT4 controls, restrict adaptive specs to NOM/OCC/DIST/HARD, pull final z_std — **done**; (e) commit an amended lock with data/checkpoint hashes and pre-state the decision rules — **done** (`a6cccb4`, `f75b95a`); (f) frame T16 vs G honestly, report deltas over persistence — **done**.

## Consultation 3 — after results: findings and dispositions
| # | Finding | Disposition |
|---|---|---|
| 1 | "Ternary lost no measurable success vs FP16" compares against a baseline weakened by the JEPA loss (G .452 vs G0 .596); every quantized arm carries JEPA, so precision and JEPA are confounded; INT8 below FP16 marks the noise floor. | **Fixed.** R3-C, PROJECT_STATE, handoff and hardware note now say: among JEPA-trained arms FP16/INT8/INT4/ternary lie within ~0.12 macro, inside the seed spread; no precision effect detected; against G0, G-Ta is 0.90x and G-Tu 0.85x; the no-JEPA precision effect was not measured. No ordering of ternary over FP16/INT4 is claimed. A no-JEPA ternary arm was **not run**. |
| 2 | State that STOP does not depend on the JEPA recipe. | **Fixed.** Sentence added under the verdict: G0 also fails DEV discrimination (+0.124 < 0.15) and the TEST floor (macro .596 < .60, NOM .664 < .70); descriptive, not a substitute gate. |
| 3 | G4: the 3.09x pair is not a valid locked comparison (G-I4u eligibility used a 2-seed mean against G's 3-seed mean; on matched seeds G-I4u fails; G-I4a misses by 0.007; scopes differ). | **Fixed.** `report.py` now tests eligibility on matched seeds (result: no eligible control); the record reports same-scope G-Ta vs G-I4a (2.01x bytes, 1.02x MACs) and the other pairs as non-eligible; G4 fails on MACs under any pairing. |
| 4 | `report.py` bug: the full-perception encoder fraction averaged in BLK40/BLK40D (valid fraction ~0.52), biasing it low and understating the G3 reduction. | **Fixed and regenerated.** Full-perception encoder fraction is 0.71-0.72 (was 0.65); encoder reduction at the learned 0.33 budget is 62-64% (was 59-60%); no gate outcome changes. |
| 5 | The learned trigger lands ~0.02 lower in encoder fraction than the controls at each budget. | **Acknowledged, documented, not rerun.** One line added in R3-D with the slope-based bound (~0.003 macro) versus the 0.03-0.05 gap. |
| 6a | "Predictor beats persistence from k >= 4" is false for G seed 2. | **Fixed** ("in 2 of 3 seeds; at k = 8 in all three"). |
| 6b | Literature review labelled the recurrent-JEPA gap "KNOWN". | **Fixed** (relabelled as not found in a limited search). |
| 6c | Stage A ceiling paragraph (831k MACs, 85%) predates amendment 1. | **Fixed** (pointer to `cost_table.md`: 89% of MACs). |
| 6d | "S4 underperforms R" note superseded by amendment 4. | **Fixed** (annotated where it appears). |
| 7 | Write this review file. | **Done** (this file). |
| 8 | "Main-branch artifacts unchanged" in PROJECT_STATE is unverified. | **Addressed by a direct check:** `git diff main...HEAD` shows only `docs/CURRENT_WORK.md`, `docs/KNOWN_ISSUES.md` and `docs/PROJECT_STATE.md` modified outside the new EXP-0012 files (no change to `rtl/`, earlier `results/` or earlier `analysis/`). A full host-Python run of the earlier suites was started but its RTL-simulation tests are slow; it had not finished when this was written, so the earlier suites were **not** re-verified as passing in this session. |
| 9 | The 28 final checkpoints exist only in the WSL home directory. | **Done.** Committed to `results/predictive_state/checkpoints/` (45 MB total) with SHA-256 in `protocol_v1_amended.json`; the 1.0 GB training set is regenerated deterministically by `train.gen_train(0)` (hash recorded). |
| 10 | Commit and push the branch only; do not merge. | **Done** (see final report). |
| 11 | Final report must answer the closing question directly and list every out-of-repo side effect. | **Done** in the final message. |

## Reviewer's standing objections that remain open
- Seed variance (3 seeds, 0.34-0.60 per-seed spread) limits every ranking below ~0.1 macro (KI-022).
- INT4/INT8 QAT recipes and the JEPA weight were not tuned; both "JEPA hurts" and "ternary is within noise of INT4" are conditional on untuned recipes.
- The benchmark is a toy whose encoder dominates cost; nothing here says anything about real robots, real encoders, energy, or hardware.
