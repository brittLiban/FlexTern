# EXP-0012 — Ternary predictive-state architecture for robotics (algorithmic feasibility)

Status: **in progress** (Stage A done, protocol locked; see "Lock" below). Branch `exp-0012-predictive-state`, not merged to main.
Authorized by the founder directive of 2026-11-07. Owner: Claude (implementation); founder + ChatGPT (direction).
Literature review: `EXP-0012-literature-review.md`. Protocol (machine-readable): `analysis/predictive_state/protocol_v1.json`.

## Research question
Does persistent state + JEPA-style latent prediction + ternary-weight computation give a meaningful, *separable* advantage in inference cost for a partially observable manipulation task, against competent FP16, INT8, INT4 and non-recurrent controls? Four sub-questions (A persistent state, B JEPA, C ternary, D adaptive perception) are tested separately (`protocol_v1.json: questions`).

## Starting evidence from earlier experiments (read before designing; not reinterpreted)
- EXP-0011 (MEASURED, CPU): native ternary was **not** faster than quality-matched INT4 at the runtime level (SPEED_LOW, geomean end-to-end 0.73). Ternary hardware benefit must therefore be argued from bytes and logic, not from existing CPU kernels.
- EXP-0003 (ANALYTICAL): ternary specialization is bounded by the non-ternary share of per-step cost (Amdahl). The same logic is applied here before any training: see "Analytical ceilings".
- EXP-0001/0005 (MEASURED, generic synthesis): ternary dot products need no multiplier (~2.7x fewer cells than INT8xINT4 at 256 terms). EXP-0008/0009 (ANALYTICAL): resident-SRAM read energy and retention leakage are temperature-dependent open questions. No hardware work is authorized here.

## Stage A

### Environment audit (MEASURED on this machine, 2026-11-07)
AMD Ryzen 7 260 (8C/16T), 8 GB RTX 5060 laptop GPU, host ~15 GB RAM with WSL2 capped at ~5-6 GB, 566 GB free disk. WSL Python 3.14.4 venv `~/ftvenv` (torch 2.11.0+cu128, numpy 2.5.3); **it was not modified**. A throw-away venv `~/simcheck/simvenv` (uv, Python 3.14) holds mujoco for the feasibility probe only.

### Simulator feasibility (time-boxed probe; what was and was not done)
| Option | Result |
|---|---|
| ManiSkill 3.0.1 / SAPIEN 3.0.3 | `pip download --only-binary` found cp314 wheels. **Not installed and not run.** SAPIEN rendering needs a working Vulkan stack; WSL has only software/other ICDs listed (lavapipe among them); GPU Vulkan under WSL2 was not tested. |
| MuJoCo 4.0.0 | Installed in the throw-away venv; EGL offscreen rendering works only with `PYOPENGL_PLATFORM=egl MUJOCO_GL=egl` (without it: PyOpenGL error), **11.4 ms per 64x64 frame** for a trivial scene (renderer backend not inspected). |
| gymnasium-robotics 1.4.2, pymunk 7.3.1 | wheels exist; not used. PyBullet: no wheel for cp314. LIBERO: not tried (robosuite stack, not probed). |

Decision (INFERRED, not a measurement): the study needs on the order of 10^6 rendered frames per arm-seed and ~30 trained models plus closed-loop evaluation. At 11 ms/frame, rendering alone makes a 3D simulator several hours per data set before training, and ManiSkill/LIBERO were unverified here. The directive explicitly allows "a clearly defined partially observable robotic ... pick-and-place environment" when a full simulator is too expensive. A fast custom benchmark was built. A MuJoCo EGL port is a possible follow-up if the 2D result warrants it.

### Benchmark: OccPick2D (`analysis/predictive_state/env.py`)
Point end-effector, continuous 2D action + gripper, 32x32 RGB top-down render with pixel noise, proprioception (position, held flag, gripper state) and a task code (target colour). Picks the target among distractors and places it in a fixed goal zone.
- Partial observability: 2-3 static occluders hide objects; **camera blackouts** (frames lost, flagged); the target is always visible at t=0, so remembering it is possible.
- Disturbance: **unexpected displacement** of the target by 0.25-0.40 units at a random step (drops it if held), possibly during a blackout.
- Held-out generalization: layout ids (occluders + goal) in disjoint ranges for train/dev/test; test HARD uses 4 distractors (train has 1-3). LONG: five sequential picks (5x the single-pick horizon, up to ~400 steps) with the recurrent state never reset.
- Success: target released inside the goal zone before the 80-step pick limit; wrong object picked ends the episode.
- Validation before any learned policy (MEASURED on 200 test-split episodes per condition): the privileged expert succeeds 100% in NOM, OCC, DIST, HARD and LONG (5/5 picks); a do-nothing policy scores 0% (unit tests).

**What it cannot establish:** real contact dynamics, 3D grasping, camera/lighting realism, real encoders (a 0.83 M-MAC CNN stands in for a vision backbone, so the encoder/updater cost ratio is a property of this toy, not of real VLAs), or any real-robot result. Simulated results are not real-robot validation.

### Baselines and candidates (`protocol_v1.json: arms`)
R (single frame) and S4 (4-frame stack) are the conventional reactive controllers; T16 is a causal transformer over a 16-step latent history; G0 is a GRU without JEPA; G is the FP16 GRU + JEPA (this is both Baseline B and Baseline D: recurrent state with the same encoder, run at every step). INT8/INT4 QAT recurrent controls and the ternary candidates (updater-only E1, encoder+updater E2) are trained with QAT under the same budget. The adaptive Candidate F and its FP16/INT4/INT8 controls are inference-time triggers on those checkpoints (the random observation-dropout used in training makes every recurrent arm able to run on skipped frames). All arms train by behaviour cloning on the same demonstrations, because a fair comparison needs one training regime; RL or latent-space planning is **not** attempted (a CEM/MPC planner on the predictor is left as a possible follow-up, not a part of this prototype).

### Analytical ceilings (ANALYTICAL; computed before any training; `cost_model.py`)
Per control step at full perception: the encoder is 85% of MACs (831k of ~977k) for the recurrent arms; the GRU updater + heads are ~12%. A ternary updater therefore changes MAC count by 0 (a ternary MAC is an add, same count) and reduces weight bytes only; skipping the encoder is the only route to a >1.2x MAC reduction. In weight+state bytes the encoder is 55-91% of the step depending on how the encoder is quantized. Ternary vs INT4 on the same scope can give at most ~2x on weight bytes (2 vs 4 bits) and less once state, scales and the trigger head are included, so the G4 ">=2x vs the strongest conventional quantized control" bar is near the arithmetic limit before any model quality is considered. Exact table: `results/predictive_state/cost_table.md` (written at the end).

## Lock
`protocol_v1.json` fixes constants, gates, verdict rules and code hashes; it was written before any learned-policy measurement. Founder-provided gates were turned into numbers there (absolute floor, retention CI rule, 2x encoder reduction, long-horizon probe drift, cost metric hierarchy). Changes after the lock are recorded below as numbered amendments; old results are kept.

## Results
(to be filled)

## Amendments
(none yet)
