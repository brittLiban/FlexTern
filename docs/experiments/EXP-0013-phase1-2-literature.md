# EXP-0013 Phase 1.2: literature comparison (world models for control and JEPA-style prediction)

Date: 2026-11-07. Written from web searches run this session plus the Stage A review (`EXP-0012-literature-review.md`).

**Citation hygiene.** Everything below was taken from search-result abstracts and secondary summaries; **I did not read these papers in full**, and I did not open their benchmark definitions. Claims about what a paper evaluates are therefore INFERRED from abstracts and are labelled so; numbers are the authors' claims (KNOWN as claims, not reproduced). arXiv IDs are those returned by the search.

## 1. Papers

| work | what it is (per abstract/summary) | idea worth borrowing | what it does not address that matters here |
|---|---|---|---|
| TD-MPC2, arXiv 2310.16828 (ICLR 2024) | decoder-free latent world model, local trajectory planning in latent space, SimNorm latent normalisation, Q-function ensemble; 104 online RL tasks with one hyperparameter set; a 317M-parameter multi-task agent | (1) **plan with a learned terminal value** so short horizons suffice: relevant because Phase 1.1 found horizon is not the bottleneck on v1 and a value term is what a short-horizon planner needs for delayed consequences; (2) SimNorm-style bounded latent for stable recurrent rollouts (a ternary-state risk, G5) | tasks are state-observation or fully visible pixel control; no designed *hidden delayed consequence under observation dropout*, and no accounting of perception cost |
| DINO-WM, 2411.04983 (ICML 2025) | world model on frozen DINOv2 patch features, trained offline from trajectories; planning by optimising action sequences toward a goal observation; maze, push, particle manipulation | frozen perceptual encoder plus a small predictor (matches FlexTern's "encoder is the expensive part" finding); evaluation by success versus planning budget | the dynamics it learns are visible in each frame; goals are observations; per-decision planning cost is large (LeWM reports a roughly 48x planning-time gap against it) |
| V-JEPA 2, 2506.09985 | action-free pretraining on >1M hours of video, then an action-conditioned predictor (V-JEPA 2-AC) post-trained on <62 h of DROID robot video; CEM planning with image goals; zero-shot Franka pick-and-place (authors' claim) | **small amount of action-conditioned data on top of a frozen representation is enough for planning (claimed)**: supports a staged recipe, and CEM-in-latent-space as the planning baseline | transformer predictor over frames, no persistent compact state; hidden state that is not visible in the context window is not the target; no energy/latency budget |
| V-JEPA 2.1, 2603.14482 (March 2026) | dense predictive loss over masked and unmasked tokens, better dense features; distilled 80M/300M variants; summary mentions improved grasping over the 2-AC variant | dense (spatial) targets and **distilled small encoders**: the only direction here that targets the encoder cost FlexTern cares about | still an image-feature model; says nothing about timers or event memory |
| LeWorldModel, 2603.19312 | 15M-parameter JEPA trained end-to-end from pixels with two losses (next-embedding prediction and a Gaussian-prior regulariser, SIGReg), CEM planning; claims up to 48x faster planning than DINO-WM and probes showing physical quantities in the latent | the **minimal two-term objective** and the probing/surprise tests as a template for "does the latent hold physical state"; compact scale is on FlexTern's side | evaluated mostly in clean simulation (INFERRED from abstract-level reading); no hidden delayed state |
| Fast LeWorldModel, 2606.26217 | replaces repeated one-step rollout with **action-prefix prediction**: encode prefixes of a candidate action sequence and predict the future latents in parallel; reported lower planning time and higher average success (exact speedups not retrieved) | rollout cost is the lever for the *planning computation* metric, and parallel prefix prediction avoids compounding error that would hit a ternary recurrent state hardest | still assumes the relevant state is observable in the current latent |
| POPGym, arXiv 2303.01859 (ICLR 2023); "Mastering Memory Tasks with World Models" (R2I, Memory Maze), arXiv 2403.04253 (ICLR 2024); "When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment", arXiv 2307.03864 (NeurIPS 2023) | benchmark suites and diagnostic tasks for partially observable RL: POPGym (15 environments with difficulty levels, 13 memory-model baselines), Memory Maze (long-term memory for world-model agents; R2I adds state-space models), Ni et al. (configurable tasks that measure memory length and credit-assignment length separately) | **the memory-versus-credit-assignment question is already an established research topic**; v2 should reuse the idea (separate controls for remembering a fact vs forecasting), not claim it. They are generic POMDPs without perception-cost accounting or hand-coded timer baselines (INFERRED from abstracts; I did not open their task definitions) |
| Physically grounded JEPA: PSG-JEPA 2608.06799 (ties latents to proprioceptive state, training-only), Liu et al. 2609.03565 (inverse dynamics plus state alignment; workshop paper), Sub-JEPA 2605.09241, Flow-JEPA 2608.29029, Delta-JEPA 2606.31232 (action-sensitive latent via difference decoding) | auxiliary objectives that make the latent action-sensitive and tied to physical quantities | **grounding the latent in proprioception, and testing action sensitivity directly**: both are in the preregistered metrics (action-sensitivity score, hidden-state probe) | grounding uses *observable* physical state; none is aimed at a quantity that is invisible by construction, which is the v2-B situation |

*Search coverage:* standard web searches this session; POPGym Arcade (2503.01450) appeared in a POPGym result and was not opened. I did not search MiniGrid-memory or other memory-task suites, so the memory-benchmark row is a sample, not a survey.

## 2. What the field already provides (so FlexTern should not claim it)

- KNOWN (claims): latent action-conditioned prediction plus MPC works for fully or mostly observed manipulation and navigation (V-JEPA 2-AC, DINO-WM, LeWM, TD-MPC2).
- KNOWN (claims): collapse-avoidance and grounding objectives improve stability and planning success (SIGReg, state alignment, dense losses).
- KNOWN: planning cost is a recognised bottleneck (LeWM versus DINO-WM; Fast-LeWM).
- INFERRED: none of these evaluate a *designed hidden, delayed, action-dependent* variable whose value must be carried through observation dropout, nor report encoder-invocation fractions, nor compare against a hand-coded finite-state predictor having the same information.

## 3. Ideas to borrow

1. **Decoder-free latent plus short-horizon value** (TD-MPC2) for any later C6 planner; do not use long open-loop rollouts.
2. **CEM-in-latent MPC as the standard planning baseline** (V-JEPA 2-AC, LeWM), with success reported against planning budget.
3. **Action-prefix prediction** (Fast-LeWM) if C6 plans; it is also the natural unit for the "planning computation" metric.
4. **Probes and surprise tests on the latent** (LeWM) against *true hidden variables* (`g`, `phase`, `k`): linear probe accuracy and the action-sensitivity score.
5. **Proprioceptive grounding** (PSG-JEPA): v2 gives exact proprio, so grounding is free and should be a switch in the C6 ablations, not assumed.
6. **Small distilled encoders** (V-JEPA 2.1 distillation) as the realistic encoder if C6 ever uses a non-trivial vision backbone.

## 4. What would actually differentiate FlexTern

Only claims that survive the table above:

1. **Not a differentiator by itself:** separating memory from credit assignment or prediction is the stated aim of existing POMDP benchmarks (POPGym, Memory Maze, Ni et al.); the C1 / C1+ / C2 ladder applies that idea and is not new. What may be new in combination (INFERRED, from abstracts only): a hand-coded finite-state predictor holding the same information as the learned agent as the explicit bar, **together with** observation dropout on a perception-limited robot-style loop and perception-cost accounting.
2. **Perception-cost accounting as a first-class metric** (encoder invocation fraction under a trigger, blackout versus chosen skip, bytes and latency), which none of the surveyed papers appears to make central (INFERRED).
3. **Evidence about carrying a timer in a compact (possibly ternary) recurrent state** across observation dropout: stability of low-bit recurrent state is unresolved at this scale (Stage A review, section 3). Hypothesized, not measured.

Not a differentiator: using JEPA, ternary weights or recurrence by themselves (all published), and not a differentiator before E3/E4 in the design document's evidence ladder.

## 5. Consequence for the design

The literature does not change v2: it motivates the metrics (planning budget, action sensitivity, probes) and the staged recipe (frozen/small encoder plus small action-conditioned predictor), and it reinforces that the discriminating test must be a variable that is hidden by construction. It also warns that a positive v2-B result will be read as less than it is: the surveyed systems would likely solve v2-B only if given a recurrent or event-memory component, which is exactly what C2 is.
