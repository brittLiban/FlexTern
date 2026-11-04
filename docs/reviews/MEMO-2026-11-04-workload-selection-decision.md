# Decision memo — FlexTern workload selection (draft for founder review)

Date: 2026-11-04
Author: Claude (Sonnet 5.5); independent challenge by an advisor agent (Opus 5.5) before drafting (disposition at the end)
File: `docs/reviews/MEMO-2026-11-04-workload-selection-decision.md` (`docs/reviews/` normally holds REV- critiques; I put this memo there because AGENTS.md has no folder for decision memos; move it where you prefer).
Status: **Proposal for founder review. Nothing here is a decision, an ADR, or an implemented change.** No repository architecture, canonical document or code was modified; this file is the only new artifact and is not committed.
Labels follow AGENTS.md §7 (KNOWN / MEASURED / INFERRED / HYPOTHESIZED / TARGET / SPECULATIVE). I interviewed no customers, and I state no market sizes, prices or customer spending; where a source gave one and I could not verify it against a primary source I left it out.

---

## 0. Bottom line

1. **Ranking** (criteria scored separately in §6): **1. On-device industrial AI assistants / small language models (SLM)**, **2. robotics and autonomous-machine inference**, **3. industrial edge computer vision.** The SLM ranking is driven mainly by *compatibility with the research FlexTern has already done* and by *how cheaply it can be tested*. It is **weakest on buyer pain and defensibility**, and if "compatibility with existing research" is removed from the scoring the three markets are within my scoring error of each other (§6). That is an honest reading, not a strong win.
2. **Recommended workload:** an **offline technician question-answering assistant over equipment documentation** (retrieved context of roughly 0.5–4k tokens, 128–256-token answers, batch 1) on a constrained industrial edge device, using a 1–3B-class model.
3. **Recommended falsifiable experiment ("EXP-0011", proposed, not started):** at **matched quality on standard benchmarks** and with INT4 comparators fixed by a written rule, measure whether the ternary format (BitNet b1.58 2B4T on Microsoft's MIT-licensed bitnet.cpp) gives lower **energy per answered query** and lower time-to-first-token than the best INT4 model on identical commodity hardware, across a context sweep and a duty sweep. Outcomes are pre-stated over a device × context grid (§7.4). **My prior, stated before any run (INFERRED from EXP-0003's 1.3–1.8× byte band): the most likely result is the middle band, whose founder action is "do not invest in silicon on this evidence." Even a ≥ 2.0× pass would not justify silicon for a low-duty assistant (§4.4); it would first require a buyer with a high-duty workload.**
4. **What this experiment can and cannot tell you.** It tests the ternary *format on commodity kernels*, which anyone can copy for free. It does **not** test FlexTern silicon. A pass shows an advantage that is easy to copy; a fail removes the SLM-edge path for ternary and leaves silicon carrying the whole case, which EXP-0003 already bounds.
5. **A fact you should have before choosing this workload:** FlexTern's own modeled edge result (EXP-0008/0009/0009A, ANALYTICAL) says the resident-SRAM form **loses at low duty cycles**, and an on-demand assistant is a low-duty workload. So this recommendation is a decision about the *format and software* path (the regime that survived EXP-0007), not an endorsement of the resident-SRAM silicon for this application (§4.4).
6. **Decisions I need from you:** (a) accept or reject the workload and experiment; (b) approve the minimum kit: a commodity wall-power meter plus **one battery-less x86 mini-PC as the primary device**, and optionally **one rail-monitored Jetson Orin Nano-class board as a second tier** (without two tiers the "cheaper device class" claim is dropped); I did not find reliable primary prices, so do not rely on any number from me; (c) set or confirm the latency TARGETs in §7, since there is no customer requirement.

---

## 1. Evidence ledger

**FlexTern MEASURED** (reproducible artifacts in the repository):
- EXP-0001: a ternary multiply-free primitive is 5–12× smaller in generic synthesized logic cells than an INT8 multiplier (generic Yosys, no target device, no power).
- EXP-0002 / 0002B: across two native ternary families (BitNet b1.58 2B4T and Falcon-E), ~40% unstructured zeros, 2-bit storage, INT8 per-token activations, a BF16 output head.
- EXP-0004 (one checkpoint, pilot scale): Falcon-E 1B tolerates an INT4 head (+0.67% perplexity) and a 4-bit KV cache.
- EXP-0005: a packed-ternary consumer at initiation interval 1; 25,466 generic cells against 69,526 for an INT8×INT4 structural baseline at 256 terms/cycle (2.7×), generic synthesis only.
- EXP-0010 / 0010B / 0010C (synthetic warehouse video, one laptop CPU under WSL2): ORT sustains 30 fps; the network is under half of per-frame compute (decode, preprocess and postprocess are the rest); no adequate quality control for person monitoring was established (YOLOX-Tiny meets the gating pair on one of two fresh layouts for one policy; preregistered decision CHANGE_WORKLOAD); **no energy was ever measured.**

**FlexTern ANALYTICAL / INFERRED** (models on public data, not measurements):
- EXP-0003: with the head and KV cache unchanged, ternary specialization alone is bounded in decode: ≤ 1.8× for BitNet 2B4T at any context; ≥ 3× for Falcon-E only at short context (≤ ~2.6k for 1B, ≤ ~7k for 3B); ≤ 1.6× at 32k. The byte advantage of 2-bit storage over an INT4-weight system with the same head/KV is **1.3–1.8×**.
- EXP-0007: the model-resident datacenter form does not survive batching and KV; surviving regimes are small-batch interactive serving, **ternary as a format on conventional hardware (2–3× fewer weight bytes at small batch)**, and the edge 1B–3B form.
- EXP-0006/0008/0009/0009A: the edge resident-SRAM form reads ~10× cheaper than LPDDR5X per bit, but leakage decides: at 25 °C it wins from ~1.4 tok/s or ~3.5–6% duty; at 85 °C only above ~17 tok/s continuous (1.41×) and **never at low duty**; thermal envelope and retention-rail supply are unmeasured inputs (Q-023).

**External evidence used** (reliability is stated where it matters; full list in §9):
- Primary or near-primary, fetched: the BitNet b1.58 2B4T report and model card, the bitnet.cpp report, the NVIDIA Jetson Orin and Thor product pages, the NVIDIA Jetson AI Lab GR00T N1.7 tutorial, the Albireo arXiv abstract, the Falcon-E model card, the Microsoft BitNet repository.
- Secondary (search summaries, press): Hailo-10H figures, T-MAC figures, TeLLMe v2 figures, BitVLA, Jetson AI Lab LLM benchmark page (marked work-in-progress), Cognex In-Sight 3900 and Siemens Industrial Copilot announcements, small-model landscape blogs.
- **Not verified, therefore omitted or flagged:** market-size figures, vendor market shares, device prices, Taalas funding/AMD-acquisition claims (the Taalas site I could fetch states only the concept and a marketing efficiency claim).

---

## 2. What FlexTern's own results imply for the choice

- **The strongest, most defensible FlexTern fact is a bytes fact.** Ternary storage is 1.3–1.8× fewer bytes than INT4 (INFERRED, EXP-0003), and BitNet's own paper reports non-embedding memory of 0.4 GB against 0.7 GB for INT4 Qwen2.5-1.5B (authors' measurement). Anything FlexTern claims beyond that must come from compute, memory placement or execution (all HYPOTHESIZED).
- **A workload decision must state its duty cycle and temperature.** The edge silicon results are regime-dependent (§1). A workload with low duty or a hot enclosure is in the regime where FlexTern's modeled form loses.
- **The vision work taught what a workload decision needs before research starts:** a founder-set application requirement, a control that is adequate on fresh data, and an energy measurement boundary. EXP-0010C found none of these were in place; this memo asks for them up front.
- **Datacenter is closed (EXP-0007); edge 1B–3B models are the open region.** Hence the three markets below.

---

## 3. Market 1 — Industrial edge computer vision

**Customer problem and buyer (KNOWN in general, vendor evidence).** Manufacturers need automated inspection, code reading, assembly verification and safety/occupancy monitoring at line speed. The people who buy *compute* are vision-system vendors and integrators who design a processor into a product, not the plant itself. Cognex's May 2026 In-Sight 3900 is an example: a PC-free embedded AI vision system on Qualcomm Dragonwing silicon, claimed (by Cognex) to run inspections up to 4× faster than the previous generation at up to 25 megapixels, with models trained centrally and executed locally (vendor press release reprints; no independent benchmarks or power figures found). **INFERRED:** incumbents already standardize on commodity SoCs and NVIDIA modules rather than custom silicon.

**Representative models and checkpoints.** COCO-pretrained detectors (SSDLite, YOLOX under Apache-2.0, other YOLO families under other licenses), transformer detectors (RF-DETR), anomaly-detection models. FlexTern used YOLOX-Tiny and SSDLite320 in EXP-0010. **I found no pretrained ternary detector** (search was not exhaustive), and Y-Core was designed for transformer matrix multiplies (EXP-0002), not convolutions (HYPOTHESIZED that it could map).

**Hardware and constraints.** NVIDIA Orin Nano 7–25 W with 102 GB/s; Orin NX 10–25 W; AGX Orin 15–60 W (204.8 GB/s); AGX Thor 40–130 W (273 GB/s) (NVIDIA product pages). Hailo-10H: a typical 2.5 W and 40 INT4 TOPS (vendor claims relayed by press). Fanless enclosures, 30 fps real-time loops and thermal limits are the usual constraints. **Thermal: I did not source enclosure or thermal limits for any device in this market.**

**Current optimized solutions.** TensorRT and INT8 post-training quantization on Jetson, vendor NPUs, and *conventional temporal skipping*: Albireo (arXiv 2609.29648, accepted at SEC 2026) wraps a detector with per-object Kalman filters and runs the detector only when uncertainty exceeds a threshold, on YOLO11x/YOLO26x/RF-DETR-Large on AGX Thor and AGX Orin, keeping AP@50 within ±1.2 pp of per-frame inference while reducing total energy by **12.1–17.6%**; a fixed skip-every-other-frame baseline loses 8.6 pp AP@50 (abstract; the energy-measurement method was not stated in the text I retrieved).

**Where FlexTern could plausibly help.** (a) Temporal execution: but the published conventional method already captures 12–18% at equal accuracy, and FlexTern's own Stage A found the network is under half of per-frame compute on a CPU pipeline (MEASURED), so system energy is dominated by non-network stages (INFERRED). (b) Ternary or low-precision detectors: unsupported by any checkpoint I found. **The EXP-0010 series also showed that the cost of building a trustworthy control for this workload is high** (three objectives, no adequate control).

**How to test without custom hardware.** Real (non-synthetic) footage under a license, the Stage A/B/C harness, and an external power meter. A cheap *oracle bound* is available first: measure on real footage what fraction of frames a perfect event scheduler could skip at preserved event recall; if that bound is small the opportunity is small regardless of method.

**What would falsify it.** The oracle temporal bound allows less than ~2× skipping at preserved event recall on real footage, **or** a conventional Albireo-class gate reaches most of the oracle bound (so there is no room for a new method), **or** system energy is dominated by sensor, ISP and decode rather than inference.

**How FlexTern would make money (HYPOTHESIZED).** Sell compute IP or modules to camera and vision-system vendors who already buy from NVIDIA and Qualcomm. That needs at least 2× system-level advantage, a mature compiler and runtime, and customer qualification cycles; the barrier is software support and incumbency, not the datapath.

---

## 4. Market 2 — On-device industrial AI assistants and small language models

**4.1 Customer problem and buyer.** Maintenance staff and operators need answers from equipment manuals, alarm codes and procedures where connectivity is poor or data may not leave the site. Siemens' press release of 12 November 2024 says an on-premises version of the Industrial Copilot for Operations "is planned to be offered" as a hardware-software bundle on a SIMATIC Industrial PC (IPC 1047E; distributor listings describe it as a 4U 19-inch rack PC), uses NVIDIA NIM microservices, needs no internet connection and keeps data on the shopfloor; the use cases are shopfloor questions about operational and document data, error handling and maintenance. **KNOWN:** at least one incumbent announced an offline assistant on a rack-class PC with an NVIDIA software stack. **UNKNOWN:** whether it ships, its model size, its hardware cost, and customers' willingness to pay; I have no data on any of these. Likely buyers are machine builders and industrial-software vendors embedding assistants, and plant maintenance organizations; I did not verify either. **INFERRED:** the incumbent's hardware class is large and GPU-based, which is where a smaller, cheaper or lower-power assistant could matter; I have no evidence that buyers want one.

**4.2 Models and checkpoints.**
- *Ternary:* BitNet b1.58 2B4T (about 2B parameters, 4T training tokens, MIT license, 4,096-token context; the model card says it is intended for research and development and does not recommend commercial or real-world use without further testing); Falcon-E 1B–3B (Falcon-LLM license; the Falcon-E-3B-Instruct card shows 955 MB but a benchmark average of 22.6 against 27.2 for Qwen2.5-3B-Instruct on the authors' own normalized suite, i.e. lower quality than same-nominal-size INT4-class peers); community BitNet variants.
- *Conventional comparators:* Qwen2.5-1.5B INT4 (0.7 GB non-embedding in BitNet's own table), Llama 3.2 1B/3B, Gemma and Phi small models, and newer 2026 small models (their quality is only blog-sourced here and must be checked on model cards).

**4.3 Typical hardware and what is already achieved.** CPU-only x86 industrial PCs; Jetson Orin Nano 8 GB (102 GB/s, 7–25 W): NVIDIA's work-in-progress benchmark page lists Llama 3.2 3B at about 43 tok/s (MLC, INT4) and about 52.6 tok/s (vLLM W4A16) on the Orin Nano Super; Hailo-10H (vendor claims): about 10 tok/s on 2B models at about 2.5 W typical, with 8 GB LPDDR4 on the M.2 module. **INFERRED:** decode speed on INT4 is already above comfortable reading speed on commodity edge devices, so *speed alone is probably not the buyer's pain.* The likelier pain points are idle power, time-to-first-token on retrieval-heavy prompts, RAM footprint (co-hosting with other software), the cheapest hardware tier that meets a latency target, fanless operation, and answer quality. **Thermal constraints for these devices were not sourced (see §4.4).**

**4.4 Duty cycle and temperature, checked against FlexTern's own modeled regime (required, because the workload is low duty).**
- An on-demand assistant is idle most of the time. The query rate is unknown and is a parameter, not an assumption: the experiment reports average power across duty points (§7).
- **What "duty" means in the FlexTern records (checked against EXP-0009/0009A):** the fraction of time the model is actively generating at 25 tok/s; the cool break-even is about 3.5–6% and the hot (85 °C) case needs duty above about 61–65%. **Illustrative arithmetic, not data:** a 128-token answer at 25 tok/s occupies about 5 s, so 3.5% duty at 25 °C corresponds to roughly 25 answered queries per hour; a lower-rate assistant is below the cool break-even, and at 85 °C a low-rate assistant never wins.
- EXP-0009/0009A (ANALYTICAL): the resident-SRAM edge form beats LPDDR5X from ~3.5–6% duty at 25 °C and **never at low duty at 85 °C**; shutdown-and-reload during idle depends on the backing store (a DRAM-backed one never beats the streaming baseline during idle). I did not source the thermal envelope of industrial enclosures (the Siemens rack PC cited above is listed, via a distributor, with 24/7 operation up to 50 °C ambient; unverified against the Siemens data sheet).
- **Plain statement:** *if* the assistant's duty cycle is below a few percent, FlexTern's modeled resident-SRAM silicon is in its losing regime for this workload. What remains is the format-level story (fewer bytes per query on commodity hardware) and idle-power engineering, which is what the recommended experiment measures. If a duty regime of continuous or high-rate generation (for example a streaming voice agent or continuous alarm-flood triage) is what a buyer actually has, the silicon analysis becomes relevant again; I have no evidence for such a buyer.

**4.5 Current optimized solutions.** llama.cpp, MLC, vLLM with INT4/W4A16; **T-MAC** (Microsoft, table-lookup kernels): reported up to 4× throughput and 70% lower energy than llama.cpp on low-bit models, and on a Jetson AGX Orin 0.66 vs 2.12 J/token for Llama-2-7B at 2-bit (project README and arXiv abstract relayed by summaries; the project itself notes baseline-version caveats); **bitnet.cpp**: 2.37–6.17× speedups on x86 and 1.37–5.07× on ARM, with 71.9–82.2% (x86) and 55.4–70.0% (ARM) energy reductions (Microsoft report, authors' benchmarks). **Critical reading (KNOWN from the paper text I fetched):** bitnet.cpp's speedups are against **llama.cpp fp16 only; the paper has no 4-bit baseline, and it does not describe how energy was measured**. BitNet 2B4T's headline 0.028 J/token is an *estimated* energy, and its CPU latency table compares against full-precision models. The report's INT4 comparison covers **memory and quality only** (0.4 GB vs 0.7 GB; suite average 55.0 vs 52.2 and 51.2 for GPTQ and AWQ Qwen2.5-1.5B; the models differ in size and training data). **The decisive comparison, ternary against a good INT4 model on speed and energy at matched quality, is therefore not in the literature I found.** That gap is the experiment.

**4.6 Where ternary, low precision or memory work might plausibly give an advantage (all HYPOTHESIZED unless stated).**
- Weight bytes: 1.3–1.8× fewer than INT4 (INFERRED), which helps a bandwidth-bound decode and RAM footprint; the BF16 embedding and head limit the total (EXP-0003: below 1.8× for BitNet at any context).
- CPU kernels without multipliers (lookup-table kernels) may lower energy per operation; T-MAC suggests table-lookup kernels also help 2–4-bit models, so the advantage may not be specific to ternary.
- Quality at matched bytes is the real unknown: ternary models may need more parameters to match INT4 quality (the Falcon-E card shows a deficit; BitNet 2B4T is competitive on the authors' benchmarks).
- Temporal execution is irrelevant here; memory placement matters only if the duty regime favors it (§4.4).

**4.7 How to test without custom hardware.** All software: commodity devices, standard open benchmarks, existing runtimes, and a power meter. The laptop in the repository cannot produce credible joules (WSL2, a battery, no meter; Q-026): the experiment needs a battery-less host with a commodity wall-power meter, or a Jetson-class board whose onboard INA3221 reports module input power (VDD_IN) and should be cross-checked against an external meter (forum sources note it is a sampled module-level estimate). That is a purchase decision for you.

**4.8 What would falsify the opportunity.** (a) At matched quality, the median ratio R_med of active energy per answered query (best INT4 comparator ÷ ternary) is below 1.3× on every tested device (the §7.4 rule; then neither bytes nor kernels deliver a usable edge); (b) quality cannot be matched at equal or fewer bytes; (c) INT4 on already-shipping NPUs or small GPUs already meets a latency target at lower power than any ternary-on-CPU configuration, so there is no buyer gap; (d) buyers' actual requirement is answer quality that only larger or cloud models reach.

**4.9 How FlexTern would make money (HYPOTHESIZED; the moat is the weak point).** Options: (i) a ternary model-and-runtime offering for device makers (the runtime, bitnet.cpp, and the BitNet 2B4T weights are MIT-licensed, so the software is free for competitors to use; the defensible part would have to be model quality, domain tuning, tooling or support); (ii) hardware IP or a chip, only if silicon beats INT4 NPUs in a duty regime customers have (§4.4) and the 2× target holds against T-MAC-class software; (iii) services (domain fine-tuning with open tools such as the onebitllms package in the Falcon-E card). **Moat consequence of the recommended experiment:** a pass on commodity kernels demonstrates an advantage competitors can copy immediately; a fail leaves silicon as the only differentiator, and EXP-0003 and §4.4 already bound that case.

---

## 5. Market 3 — Robotics and autonomous-machine inference

**Customer problem and buyer.** Robot makers (humanoids, mobile manipulators, AMRs) need on-board perception and policy inference with limited battery and thermal budgets. The buyers are the robot makers' compute and software teams, a concentrated group. (KNOWN in general; I did not verify the buyer structure.) NVIDIA's Jetson AGX Thor page positions humanoid robotics as its primary focus.

**Representative models.** GR00T N1.6/N1.7 (about 3B parameters, 6.5 GB checkpoint for N1.7 with a Cosmos-Reason2-2B backbone trimmed to 16 of 28 layers), π0/π0.5, OpenVLA and OpenVLA-OFT (7B-class), SmolVLA (450M, small enough to run on CPU per Hugging Face), Gemini Robotics On-Device (private preview, trusted testers only). **BitVLA (arXiv 2506.07530, work in progress)** is a ternary VLA built on BitNet b1.58 2B4T: 3.0B parameters in 1.4 GB, compared with 4.7 GB for INT4 OpenVLA-OFT (about 29.8%), with comparable LIBERO performance as claimed by its authors (I did not retrieve the paper's tables; the figure is relayed by summaries).

**Hardware and constraints.** Jetson AGX Thor: 2,070 FP4 TFLOPS (sparse), 128 GB LPDDR5X at 273 GB/s, 40–130 W; T4000 variant 64 GB at 40–70 W (NVIDIA). AGX Orin 15–60 W. NVIDIA's GR00T repository reports about 28 Hz for N1.7 with TensorRT on Jetson AGX Thor (batch 1, LIBERO, MAXN) and does not state a required control frequency; a search summary attributed a "10+ Hz is sufficient" guideline to a documentation mirror that I could not confirm, so I do not rely on it. **Thermal: I did not source thermal or battery budgets for any robot platform.**

**Current optimized solutions and the latency breakdown (primary source: NVIDIA Jetson AI Lab tutorial, GR00T N1.7 on Thor, MAXN, 4 denoising steps).** End-to-end 126 ms (8.0 Hz) in PyTorch eager, 81 ms with TensorRT bf16, 44 ms with FP8, **about 40 ms (25.1 Hz) with mixed NVFP4**, a 3.1× speedup from precision and compilation alone; cosine similarity to the PyTorch outputs 0.9997 for NVFP4; closed-loop task success stated "on par" but without numbers on that page. In the optimized configuration the **action head takes 17.2 ms and the backbone 13.6 ms**; in the unoptimized PyTorch runs the action head was the larger share (68 ms vs 48 ms).

**Where FlexTern could plausibly help.** The backbone is a transformer, so ternary weights could cut its weight bytes (BitVLA claims this). But the action head's iterative denoising is the larger latency component and is not a weight-bandwidth-bound transformer decode, so end-to-end gain is Amdahl-limited: at most the backbone's share of the latency (about a third of 40 ms in the optimized configuration; INFERRED from the tutorial's numbers). NVIDIA's own stack already extracts 3.1× from precision, and a vertically integrated vendor sets the baseline.

**How to test without custom hardware.** Simulation (LIBERO) for quality and closed-loop success of a ternary VLA against an NVFP4 baseline; latency and energy need a Jetson-class board and a meter (a purchase decision; I do not have a reliable primary price for Thor).

**What would falsify it.** The ternary VLA loses closed-loop task success against an equal-bytes INT4/NVFP4 VLA; or the action head (not the backbone) dominates end-to-end latency and energy on the real board; or the board's policy rate is already adequate at the robot's power budget.

**How FlexTern would make money (HYPOTHESIZED).** Sell compute or IP to a small number of robot makers who currently buy NVIDIA platforms with a deep software stack; qualification and integration cycles are long; defensibility would rest on a measured advantage against Thor-class NVFP4 and a software path as good as TensorRT.

---

## 6. Ranking: four criteria scored separately

Scores are 1 (poor) to 5 (strong). They are **my judgment from the evidence above, not measurements**; the reason for each is given. Because FlexTern's revenue story is silicon while the experiment tests software, the SLM column is split into the two businesses.

| Criterion | Industrial vision | SLM: format/software path | SLM: FlexTern silicon path | Robotics |
|---|---|---|---|---|
| **Technical feasibility** (a FlexTern technique plausibly yields an advantage and can be tested) | **2.** No ternary detector checkpoint found; conventional temporal skipping already captures 12–18%; network under half of CPU pipeline compute; three vision objectives did not yield an adequate control. | **3.** Byte advantage over INT4 is real in principle (1.3–1.8× INFERRED; 0.4 vs 0.7 GB reported); speed and energy against INT4 are **unmeasured** in the literature I found; quality at matched bytes unknown. | **1.5.** The modeled resident-SRAM form loses at low duty and when hot (EXP-0009A, ANALYTICAL); EXP-0003 bounds the ternary-only gain; no silicon exists. | **2.** Backbone-only gain (the action head dominates); ternary VLA is a work-in-progress paper; testing latency/energy needs a costly board. |
| **Commercial value** (buyer pain and a path to revenue; evidence-limited) | **4.** Large, mature buyer base with active launches (e.g., Cognex 2026), but buyers are incumbents who design in commodity SoCs. | **2.5.** An incumbent has announced an offline assistant, so demand exists, but speed is probably not the pain, willingness to pay is unknown, and BitNet 2B4T's card says research use only. | **2.** Needs a high-duty buyer I have not found, plus the cost of building a chip. | **3.** Active investment narrative, concentrated buyers, early market; I have no evidence of what they would pay. |
| **Defensibility** | **1.5.** NVIDIA, Qualcomm, Hailo and the vision incumbents; software ecosystems dominate. | **1.5.** Runtime (bitnet.cpp) and weights are MIT-licensed; T-MAC covers low-bit kernels; only model quality or tooling could differentiate. | **3.** Silicon is harder to copy than MIT software, but is unproven and faces NPUs and hardwired-model chips. | **2.** NVIDIA's integrated stack and precision gains (3.1×) set the baseline; BitVLA-type ideas are published. |
| **Compatibility with existing research** | **1.** Different model class; FlexTern's vision work measured latency and quality only. | **5.** Everything FlexTern measured is about ternary LMs; EXP-0003/0007 give the quantitative frame. | **5.** The silicon work (EXP-0001/0005/0006–0009A) is exactly this. | **3.** Backbone is LM-like; BitVLA is built on BitNet; the action head is not. |
| **Sum (unweighted)** | 8.5 | **12** | 11.5 | 10 |
| **Sum without compatibility** | 7.5 | 7 | 6.5 | 7 |

**Honest reading.** SLM ranks first **because of compatibility and testability**; its weakest criteria are buyer pain (2.5) and defensibility (1.5 for software, 3 for unproven silicon). Without the compatibility criterion the four columns are within my scoring error of each other. The silicon path scores lowest on feasibility because of FlexTern's own modeled low-duty result (§4.4). The case for choosing SLM is therefore not "it is the best market" but "it is the cheapest, cleanest place to find out whether the ternary format gives *any* edge advantage over INT4 before FlexTern spends on silicon." If you weight commercial value most heavily, industrial vision has the highest score on that criterion alone and the lowest on every FlexTern-specific one.

---

## 7. Recommendation: one workload, one falsifiable experiment

### 7.1 Workload (specific)
**Offline technician Q&A over equipment documentation**, running on a constrained industrial edge device, batch 1, 1–3B-class model, greedy decoding. Prompt lengths **512, 1,024, 2,048 and 4,096 tokens** (BitNet 2B4T's maximum is 4,096; retrieval prompts are prefill-heavy, so I do not fix the length at the value that favors ternary; the sweep reports time-to-first-token and decode at every length), 128-token answers. The query rate is a parameter, not an assumption. **Primary device: one battery-less x86 mini-PC (CPU-only, at least 8 threads, 16 GB RAM) with a commodity wall-power meter.** Optional second tier: one rail-monitored Jetson Orin Nano-class board.

### 7.2 The claim, in buyer terms (HYPOTHESIZED)
At matched quality on standard benchmarks, the ternary format (i) lowers **active energy per answered query** (the ratio R below, which is intrinsic to the format), (ii) does not worsen **time-to-first-token**, and therefore (iii) lowers **average power at a stated duty cycle** (the ratio A_d below). The claim that ternary meets a latency target on a **cheaper or smaller device class** is tested only if at least two device tiers are run; with one device it is dropped. Decode tokens per joule is a secondary number.

### 7.3 Protocol (to be frozen before any run; a proposed EXP-0011, not started)
- **Order of work.** (1) Quality runs on all candidate models; (2) the comparator C2 is then locked by the written rule; (3) only then the energy and latency runs.
- **Models and comparators, fixed by a written rule before any run.** Ternary: BitNet b1.58 2B4T (bitnet.cpp I2_S and the best available lookup-table kernel); Falcon-E-3B as an optional second ternary model (license differs). INT4 comparators: (C1) Qwen2.5-1.5B-Instruct INT4 (GPTQ/AWQ, converted to the best available runtime format), as in BitNet's own table; (C2) the smallest INT4 model, from a candidate list written down before the run, whose suite average under our harness is at least the ternary model's minus one point; (C3) an **equal total resident bytes** comparator. Both equal-quality and equal-byte comparisons are reported.
- **Quality.** The existing open evaluation harness (the repository already uses lm-eval) on standard tasks (for example ARC, WinoGrande, PIQA, GSM8K, MMLU, IFEval); **no custom domain suite is built in this experiment**, and domain fit to a real maintenance task is explicitly **untested**.
- **Runtime fairness.** Each model on the best available kernels for its format (llama.cpp, T-MAC where applicable, bitnet.cpp, and on a GPU board the INT4 W4A16 and BitNet GPU kernels), identical prompts, identical thread counts, interleaved run order, fixed ambient conditions, greedy decoding.
- **Minimum kit (a purchase decision for the founder).** A commodity wall-power meter plus a battery-less mini-PC; optionally a Jetson Orin Nano-class board using VDD_IN from its onboard monitor, cross-checked with the external meter. The current laptop is unsuitable (WSL2, battery, no meter, Q-026), so a meter alone is not enough.
- **Measured per configuration:** time-to-first-token and prefill throughput; decode tokens/s; idle power; **active energy per answered query above idle**, integrated from meter samples; resident memory; achieved memory bandwidth as a fraction of device peak (to tell bandwidth-bound from compute-bound).
- **Latency TARGETs** (no customer basis; you must set or confirm them): time-to-first-token ≤ T1 s at 2k tokens and decode ≥ D tok/s, with T1 and D set before the run.

### 7.4 Pre-stated decision rule (one rule over the device × context grid)
Cells are (device, context length). For each cell: **R** = (active energy per answered query of the best matched INT4 comparator) ÷ (that of the ternary model); **T** = (time-to-first-token of the comparator) ÷ (that of the ternary model), so T < 1 means ternary is slower. Summaries: **R_med** (median over all cells), **R_min**, **T_min** (minimum over all cells). **Duty-weighted ratio** A_d = (P_idle + q·E_int4) ÷ (P_idle + q·E_tern) for duty points d = 0.1%, 1%, 10%, 50%, 100%, with q the query rate implied by d, computed from the measured idle power and per-query energies, so the result can be laid over the EXP-0009A break-evens.
**Pre-statement on idle power:** idle power is a property of the device, not of the weight format, so A_d falls toward 1 as duty falls even when R is large. A claim about a **low-duty buyer** therefore rests on idle power, time-to-first-token and footprint, not on R.

| Outcome | Condition (quality matched at no more bytes) | Founder action |
|---|---|---|
| **PASS: advantage beyond bytes** | R_med ≥ 2.0 **and** R_min ≥ 1.3 **and** T_min ≥ 0.8 (ternary first-token latency no worse than 1.25× the comparator's) on every tested device | Examine compute versus bandwidth and what silicon could add. **Silicon only if a buyer with a high-duty workload exists (A_10% or continuous matters);** otherwise software/tooling only. The advantage is free to copy. |
| **MIDDLE: format-level advantage** | not PASS, and R_med ≥ 1.3 on at least one tested device | Consistent with the EXP-0003 byte prediction and free to copy: do not invest in silicon on this evidence; consider software/tooling only if quality parity holds on stronger domain tests. A device split (R_med ≥ 1.3 on one device, < 1.3 on another) is MIDDLE with that split reported. |
| **MIXED: energy win, latency loss** | R_med ≥ 1.3 but T_min < 0.8 at the 2k and 4k contexts (prefill is compute-bound) | Report as a split and claim no win: this workload is prefill-heavy, so first-token latency governs buyer-visible latency. |
| **FALSIFIED** | quality cannot be matched at no more bytes, **or** R_med < 1.3 on every tested device | Stop the ternary-SLM-edge path on commodity hardware. Silicon would need an independent case that EXP-0003 and §4.4 already bound. |

(2.0× is the TARGET used in EXP-0010 and exceeds the EXP-0003 byte band, so a pass requires gains beyond bytes; 1.3× is the low end of that byte band.) If INT4 and ternary both run far below the device's peak memory bandwidth, R mostly reflects software maturity rather than format; report it as such and do not read it as a format result.

### 7.5 What the experiment cannot show
FlexTern silicon or any FlexTern datapath; any customer's actual requirement or willingness to pay; models beyond the 1–3B class, contexts above 4k, concurrency or batching; behavior at 85 °C; domain quality. It uses published models that BitNet's own card labels research-use.

### 7.6 Why this experiment first
It answers the one question that gates everything else in the SLM path and that the literature I found does not answer (ternary against a good INT4 model on speed and energy at matched quality), it needs no custom hardware and no new model training, and each outcome maps to a different founder action (§7.4). It is software-only work of the same size class as the earlier objectives.

### 7.7 Decisions needed from the founder (AGENTS §6 format)

```
QUESTION: Accept the offline technician Q&A workload and the matched-quality energy-per-query experiment as the next objective?
WHY IT MATTERS: It is the cheapest test of whether the ternary format beats INT4 on edge energy and latency at matched quality; the most likely result is the middle band (INFERRED), which would argue against silicon for this workload.
OPTIONS: (a) accept as written; (b) accept with a different workload; (c) reject and choose another market; (d) stop the ternary edge path without testing.
MY RECOMMENDATION: (a), knowing the likely answer is "format-level advantage, free to copy".
WHAT I NEED FROM YOU: a decision, and whether a high-duty buyer workload (streaming voice, continuous alarm triage) should be added as a second workload so the silicon question is not answered by default.
```
```
QUESTION: Approve purchasing the minimum measurement kit (a commodity wall-power meter and a battery-less mini-PC, optionally one Jetson Orin Nano-class board)?
WHY IT MATTERS: Without a meter on a battery-less host there is no credible energy number (Q-026); the laptop cannot provide one.
OPTIONS: (a) meter + mini-PC; (b) (a) + Jetson board for a second tier; (c) CPU-only latency study with no energy claim.
MY RECOMMENDATION: (b), because only two tiers can test a "cheaper device class" claim; I have no reliable primary prices.
WHAT I NEED FROM YOU: approval and a budget ceiling.
```
```
QUESTION: What time-to-first-token and decode-rate targets, and what query-rate range, should the experiment use?
WHY IT MATTERS: There is no customer requirement; the outcomes depend on a latency target and on the duty regime.
OPTIONS: you set T1 and D; or I propose values and label them TARGET; or run with a sweep and no pass/fail on latency.
MY RECOMMENDATION: you set them; if not, I propose values and label them TARGET.
WHAT I NEED FROM YOU: the numbers, or permission to propose them.
```

---

## 8. Risks and what I did not do

- **No customer interviews, no spending or market-size figures, no performance results of my own.** All external numbers are cited to their source and flagged by reliability; many are secondary summaries and need confirmation before anyone quotes them.
- **Hardwired-model silicon** (Taalas, a startup whose site states the concept of turning a model into custom silicon with a marketing efficiency claim; I could not verify specifics or funding from a primary source) is a different competitor class from NPUs and GPUs and is a risk to any fixed-function edge chip.
- **Model ecosystem risk (THESIS):** if INT4 small models keep improving, ternary's quality-per-byte lead may close; the experiment fixes its comparators before any run to avoid choosing a convenient one.
- **I did not evaluate** a 4th market (industrial anomaly detection specifically, voice interfaces, automotive), and the three markets were taken as given.
- **Provenance of my scores:** one author plus an advisor; no independent scoring panel.
- **A documentation defect I noticed while reading (AGENTS §5 requires flagging it):** the EXP-0010B bullet in `docs/PROJECT_STATE.md` contains a corrupted fragment ("...still far above the 0.10 target.79–0.83, still below the bar.") left by an earlier docs edit of mine. I have not touched it because you asked for no repository changes; I can fix it in a docs-only commit after your review.

## 9. Sources

Fetched and read (primary): [Siemens press release](https://press.siemens.com/global/en/node/7042); BitNet b1.58 2B4T report ([arXiv 2504.12285](https://arxiv.org/abs/2504.12285), HTML tables 1–2) and [model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T); bitnet.cpp report ([arXiv 2410.16144](https://arxiv.org/html/2410.16144v2)) and [repository](https://github.com/microsoft/BitNet); [Falcon-E-3B-Instruct card](https://huggingface.co/tiiuae/Falcon-E-3B-Instruct); [TeLLMe (arXiv 2504.16266)](https://arxiv.org/abs/2504.16266); [NVIDIA Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/); [NVIDIA Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/); [Jetson AI Lab, GR00T N1.7 on Thor](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/); [Albireo (arXiv 2609.29648)](https://arxiv.org/abs/2609.29648); [Taalas site](https://taalas.com).

Search summaries (secondary; verify before quoting): Hailo-10H ([Hackster](https://hackster.io/news/hailo-s-latest-accelerator-the-hailo-10h-promises-on-device-gen-ai-in-a-sub-5w-power-envelope-77b9a9eca624), [SiliconANGLE](https://siliconangle.com/2025/07/22/hailo-launches-hailo-10h-chip-support-generative-ai-edge/)); T-MAC ([Microsoft Research](https://www.microsoft.com/en-us/research/?p=1112340)); BitVLA ([arXiv 2506.07530](https://arxiv.org/pdf/2506.07530)); SmolVLA ([Hugging Face blog](https://huggingface.co/blog/smolvla)); Jetson AI Lab LLM benchmarks ([page](https://jetson-ai-lab.com/benchmarks.html), marked work-in-progress); Cognex In-Sight 3900 ([press release reprint](https://www.nasdaq.com/press-release/cognex-launches-highest-performance-embedded-ai-vision-system-powered-qualcomm-2026)); Siemens Industrial Copilot ([Siemens press release, 12 November 2024](https://press.siemens.com/global/en/node/7042), fetched: on-premises bundle "planned", IPC 1047E, NVIDIA NIM, no internet required); GR00T throughput ([NVIDIA Isaac-GR00T repository](https://github.com/NVIDIA/Isaac-GR00T), fetched: about 28 Hz on Thor with TensorRT, no required frequency stated); Jetson power monitoring ([NVIDIA forums](https://forums.developer.nvidia.com/t/jetson-agx-orin-ina3221-power-monitor-layout/223111)).

Repository: THESIS.md, PROJECT_STATE.md, ADR-0005, EXP-0001–0010C, handoffs of 2026-11-04.

## 10. Disposition of the advisor's review

The advisor's review before drafting changed this memo as follows. (1) The workload's duty cycle and temperature are checked against EXP-0009A and the losing regime is stated plainly (§4.4, §0 item 5). (2) The claim is stated in buyer terms (energy per answered query, time-to-first-token, cheaper device class) rather than continuous decode tokens per joule (§7.2). (3) Context length is swept and justified from the workload rather than taken from EXP-0003's favorable region (§7.1). (4) Quality matching uses existing standard benchmarks, comparators are fixed by a written rule, and domain fit is declared untested (§7.3). (5) Thresholds are explained, outcomes are mapped to founder actions, a bandwidth-bound check is required, the result is stated to test the format on free commodity kernels rather than silicon, and the moat consequence is spelled out (§4.9, §7.4–7.5). (6) The need for a meter or a rail-monitored board is stated as a purchase decision, without secondary prices (§4.7, §0). (7) Unverified market sizes, shares, prices and the AMD–Taalas claim are omitted; vendor figures are labelled; the BitNet energy estimate, the undescribed bitnet.cpp energy method and the research-use statement are noted; measured and analytical FlexTern results are separated (§1). The ranking now shows every criterion separately and states that SLM ranks first mainly on compatibility and testability (§6).

**Second advisor review (after drafting)** led to these further changes: the decision rule now tests the buyer-visible claims (first-token latency and a duty-weighted ratio, with the idle-power pre-statement) and uses one aggregation rule over the device x context grid (7.4); the prior (middle band most likely) and the high-duty-buyer condition for silicon are stated in section 0 and 7.4; the SLM column is split into the format/software and silicon businesses (section 6); the minimum purchase kit and a primary device are named (7.1, 7.3); the duty definition and break-evens were re-checked against EXP-0009/0009A (4.4); the Siemens claim now cites the primary press release (planned, not shipping) and the GR00T 10+ Hz claim was dropped as unverified; thermal gaps are stated per market; the decisions are listed in AGENTS section 6 format (7.7); a corrupted PROJECT_STATE fragment is flagged (section 8).
