# EXP-0012 — Literature review (Stage A)

Scope: what is already demonstrated, what is unresolved, and what FlexTern could add that is not a recombination of known pieces. Written 2026-11-07 from web searches run that day plus prior knowledge.

**Citation hygiene.** `[S]` = the arXiv ID/title was returned by a web search this session (abstract-level confirmation only; I did not read full papers). `[M]` = cited from memory; the ID was searched but no clean abstract hit came back, so treat the ID as unverified until opened. Numbers quoted from papers are the authors' claims, not reproduced here. Labels follow AGENTS.md section 7 (KNOWN = published and external to FlexTern).

## 1. JEPA and latent predictive learning
- I-JEPA, arXiv 2301.08243 [S]: predicts embeddings of masked image blocks in representation space, no pixel reconstruction, no hand-crafted augmentations. KNOWN.
- V-JEPA, arXiv 2404.08471 [M] (video feature prediction); **V-JEPA 2, arXiv 2506.09985 [S]**: encoder pre-trained on >1M hours of video, then a ~300M-parameter action-conditioned predictor (V-JEPA 2-AC) trained on ~62 hours of unlabeled DROID robot video; planning by model-predictive control in latent space; zero-shot Franka pick-and-place from image goals reported by the authors. KNOWN (authors' claims). The predictor is a transformer over frames, **not a recurrent compact-state model**, and the cost per planning step is large.
- DINO-WM, arXiv 2411.04983 [S]: world model on frozen DINOv2 patch features, planning by action-sequence optimization. LeWorldModel, arXiv 2603.19312 [S] (15M parameters, end-to-end from pixels with a Gaussian-prior regularizer in place of EMA/stop-grad; evaluated mostly in clean simulation). Sub-JEPA 2605.09241, VJEPA 2601.14354, and "Toward Physically Grounded JEPA World Models..." 2609.03565 [S] all concern collapse avoidance or goal-conditioned planning.
- **Unresolved (not found in a limited search; absence of evidence, not a KNOWN fact):** I found no paper that combines an EMA-target JEPA objective with a *recurrent persistent state* on partially observable manipulation (search said so explicitly). JEPA is a training objective; recurrence is an architecture. This experiment keeps them separate (arms G vs G0).
- Collapse: EMA target + stop-grad is the standard guard (I-JEPA/V-JEPA family). Here BC gradients also shape the encoder, so collapse is monitored (embedding std) instead of assumed absent.

## 2. Recurrent world models and persistent state
- RSSM/Dreamer line: DreamerV3, arXiv 2301.04104 [S] (recurrent state-space model, 150+ tasks, single configuration). PlaNet / TD-MPC2 from memory, not cited with IDs. KNOWN.
- UF-RNN, arXiv 2510.10217 [S]: recurrent robot-motion model using uncertainty-driven foresight; EASE, arXiv 2506.17516 [S]: recurrent generative perception with prediction error as uncertainty signal. Neither skips encoder computation.
- **FlexTern angle:** nothing new in "keep a compact state"; the open question is whether it is *cheaper than re-perceiving* once an honest encoder budget is charged. That is Q-A/Q-D here.

## 3. State-space models, Mamba, and quantizing recurrence
- Mamba, arXiv 2312.00752 [S]; Mamba-2 (SSD), arXiv 2405.21060 [S]. Step-wise inference is a constant-size state update, which is why a plain diagonal-SSM/GRU reference is adequate for a batch-1 robotics loop (no custom scan kernel needed; `mamba-ssm` kernels were deliberately not built on torch 2.11 / sm_120).
- Quantizing SSMs: Bi-Mamba (1-bit) arXiv 2411.11843 [S, via snippet]; Slender-Mamba (ternary Mamba-2 170M, COLING 2025) [S]; **Ternary Mamba: Grouped QAT of W1.58A16 SSMs, arXiv 2606.18114 [S]**. The last reports a 1.3B Mamba-2 at ~3.6x compression close to Bi-Mamba accuracy, a *zero-ratio collapse* instability with learnable scales when doing QAT from pretrained checkpoints, and that post-hoc corrections that work for transformers fail for SSMs because error accumulates through the recurrence. KNOWN (authors' claims, language modelling, not control). These are weights-only (A16) results; **whether the recurrent state itself can be low-bit over long control horizons is not established by what I found.** That is a place where this experiment can produce evidence (arms G-Ta vs G-Ta-s8, LONG, probe drift).
- HYPOTHESIZED consequence for this work: ternary gate matrices in a GRU/SSM feed their own error back every step, so long-horizon drift is the specific risk (gate G5).

## 4. Ternary / low-bit networks
- BitNet b1.58 (arXiv 2402.17764 [M]) and the 2B4T report, arXiv 2504.12285 [S]. FlexTern's own EXP-0011 MEASURED that native ternary was not faster than quality-matched INT4 on a CPU. KNOWN/MEASURED.
- BitVLA, arXiv 2506.07530 [S] (v2 March 2026): a fully ternary VLA on BitNet 2B4T, vision encoder distilled to 1.58 bits (Quantize-then-Distill); authors report parity with OpenVLA-OFT on LIBERO with ~11x less memory and ~4.4x lower end-to-end latency. KNOWN (authors' claims): this is the main prior art for "ternary robotics". It ternarizes a *feed-forward* VLM backbone; it has no persistent state and no skipped perception.

## 5. Vision-language-action models and robotics inference
- OpenVLA (arXiv 2406.09246 [M]); LIBERO (2306.03310 [M]); ManiSkill3 (2410.00425 [M]) as established benchmarks. Characterization: "Characterizing VLA Models across XPUs", arXiv 2604.24447 [S] (VLM backbone compute-bound, action expert memory-bound; DP-Cache/V-AEFusion up to 2.9x on GPU, 6x on edge NPUs by the authors); vla2npu (FCCM 2026) [S] ports SmolVLA to an NPU. Vendor claims (Quadric, Hailo, DEEPX) came only from news aggregators; not cited as evidence.

## 6. Adaptive computation and selective perception
- DISK (arXiv 2602.00440) [S]: training-free per-step compute/skip for autoregressive world models in driving; Adaptive Computation Time (1603.08983 [M]) and Skip-RNN (1708.06834 [M]) are the classic recurrent skipping references. EA-WM (2606.13053) [S] event-aware manipulation world model, scores futures rather than skipping perception.
- **Trap to avoid (design rule adopted):** a trigger based on prediction error against the new observation needs the expensive encoder on that observation, defeating the skip. Gates here may use only proprioception, task code, recurrent state and 8x8 pooled raw pixels, and their cost is charged.

## 7. Hardware for SSMs / recurrent models
- EpochCore, arXiv 2507.21394 [S] (systolic array for structured SSMs; authors report large gains over a GPU); a KU Leuven fusion-aware SSM accelerator arXiv 2504.17333 [S]; eMamba (edge Mamba, FPGA/22nm) [S]. I found **no ternary SSM or ternary-recurrent robotics hardware** in these searches (absence in a limited search, not proof).

## 8. What FlexTern could contribute that is not a recombination (assessment)
| Candidate | Status |
|---|---|
| Ternary weights + SSM/GRU recurrence | Published at LM scale (Ternary Mamba et al.); **not novel**. |
| JEPA objective | Published; **not novel**. |
| Ternary VLA | BitVLA; **not novel**. |
| Evidence on long-horizon stability of *low-bit recurrent state in closed-loop control* (state drift under QAT, probe retention) | Unresolved in what I found; **measurable here, small-scale, simulated**. |
| Controlled separation of ternary vs INT4 vs skipping contributions under one training regime, with trigger cost charged | Not found as a controlled study; **measurable here**. |
| Execution scheduling: encoder-gating policy co-designed with a state updater whose cost is bytes-bound and tiny | **HYPOTHESIZED**; the Amdahl ceilings (analytical, `EXP-0012-predictive-state.md`) say the encoder, not the updater, dominates in this toy, so the scheduling idea is where a >2x claim would have to live. |
| Hardware: ternary recurrent array + small resident state SRAM | **SPECULATIVE**, no hardware work authorized; EXP-0008/0009 caveats (temperature, retention) carry over. |

Honest summary: the combination is a recombination of known parts. Any real contribution must come from measured evidence on stability, separability of gains, and scheduling, not from the idea itself.
