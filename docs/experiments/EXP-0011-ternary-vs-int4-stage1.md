# EXP-0011 Stage 1 — native ternary vs optimized INT4 on a CPU (quality, latency, throughput, memory)

Status: **in progress** (protocol locked before measurement; see the lock commit). Stage 2 (physical energy) is **not approved and not started**: `EXP-0011-stage2-energy-plan.md`.
Authorized by the founder directive of 2026-11-05. Owner: Claude (implementation); founder + ChatGPT (direction).

## Research question
At matched model quality, can native ternary inference substantially outperform strong, optimized INT4 inference on commercially relevant edge workloads? Stage 1 answers the **speed and memory** part on one x86 CPU. It makes no energy claim and says nothing about FlexTern hardware.

## Corrections to the directive (recorded before implementation)
1. BitNet 2B4T has a 4,096-token context, so the nominal "4,096 prompt + 128/256 output" cells overflow; prompt+output is capped at 4,096 (p_eff = 4,096 - G).
2. The founder's 2x / 1.3x thresholds are energy-efficiency thresholds. Stage 1 cannot measure energy, so it has its own preregistered speed/memory classification (SPEED_PASS / SPEED_MIDDLE / SPEED_LOW) at the same numeric cut-points; the energy meaning is reserved for Stage 2. A speed ratio above the bytes-per-token ratio of the actual files is flagged for audit (decode is mostly memory-bound; EXP-0003 puts the byte-driven decode ceiling at roughly 1.3-1.8x vs INT4, INFERRED).
3. The models differ in training data and architecture, so "quality-matched" is a tolerance rule (macro-average within 3.0 points; sensitivities 0 and 5) on four log-likelihood tasks, not an equality. The comparator is the fastest INT4 inside the matched pool, which is conservative for ternary.
4. Results hold for one laptop CPU under WSL2, not for ARM or Jetson-class targets.
5. The second strategic report named in the directive is not in the repository and was not read.

## Environment and methods
Locked in `analysis/ternary_vs_int4/protocol_v1.json` (constants, model/data SHA-256, runtime commits, code hashes, environment, repository state). Harness: `analysis/ternary_vs_int4/` (`config.py`, `perf.py` around `llama-bench -pg`, `quality.py` around `llama-perplexity`, `summarize.py`, `lock.py`, `envinfo.py`); tests: `tests/test_ternary_vs_int4.py`.

- Runtimes: bitnet.cpp (BitNet repo `0b341e5`, built with clang, `BITNET_X86_TL2=OFF`, I2_S kernels) and upstream llama.cpp (`e30ad80`). Upstream cannot load the BitNet I2_S file (`TYPE_IQ4_NL_4_4 REMOVED` tensor error), so the ternary model runs only under bitnet.cpp; INT4 models run under both runtimes.
- Models: BitNet b1.58 2B4T I2_S GGUF (official Microsoft release); Qwen2.5-1.5B-Instruct Q4_K_M and Q4_0, Qwen2.5-3B-Instruct Q4_K_M, Llama-3.2-1B-Instruct Q4_K_M, SmolLM2-1.7B-Instruct Q4_K_M (GGUF files from the Qwen and bartowski repositories). Selection rationale is in `config.py`; the "card average" values are Microsoft's, not ours. The 3B model was first dropped for slow download, then **re-added after the pre-lock review** (from the bartowski mirror) because the 1.5B-class models alone could leave no INT4 model at or above BitNet's quality.
- Quality: HellaSwag (400), Winogrande (500), ARC-Easy (500), ARC-Challenge (299), MMLU (500; added after review as a knowledge task nearer to Q&A use) via `llama-perplexity` built-in tasks, ctx 2048, 8 threads, same runtime for all models. Standard errors are binomial; differences below about 3 points are noise.
- Performance: `llama-bench -pg P,1` (time to first token) and `-pg P,G` (total) at P in {512, 1024, 2048, 4096-G}, G in {128, 256}; 3 repetitions; peak RSS from `/usr/bin/time -v`; threads per (model, runtime, regime) chosen from sweeps at (512,128) and (2048,128) before the main run; flash-attn pinned on for both runtimes; timing refused unless on AC with >= 30% charge; seeded shuffled order; 10 s cooldown between invocations.
- Not measured: energy, power, CPU frequency and temperature (unobservable in WSL2), network or tokenization overheads, accuracy on application tasks.

## Pre-lock review (independent advisor) and revisions
Before the lock the protocol was reviewed and revised (nothing had been measured): (1) the original task basket and 1.5B-class models could leave an empty or borderline matched pool, so the 3B model was re-added, MMLU added and the empty-pool fallback removed (an empty pool now yields NO_MATCHED_COMPARATOR); (2) the "bytes ceiling" based on file size was replaced by tensor-level accounting; (3) per-regime thread sweeps, explicit flash-attn, power refusal rule, shorter foreground budget. **Static finding recorded in the lock (MEASURED from GGUF headers):** BitNet 2B4T's GGUF streams about 1.18 GB per decoded token, of which about 0.66 GB is an F16 tied embedding/head and only about 0.52 GB is ternary I2_S weights; Qwen2.5-1.5B Q4_K_M streams about 0.98 GB, so a bandwidth-bound decode is predicted NOT to favour BitNet (HYPOTHESIZED; smoke runs showing a similar direction were visible before the lock, so the prediction is not blind).

## Results
**Status of the results: quality and static accounting complete; speed/TTFT/memory timing NOT complete** (paused when the laptop went onto battery; see KI-021). No speed or memory ratio is reported here because none has been measured under the locked conditions. Every failed or anomalous run is retained in `results/ternary_vs_int4/`.

### R1. The first BitNet runtime was numerically broken (negative result; amendment 3, KI-020)
The first BitNet quality run (official GGUF under bitnet.cpp main `0b341e5`, the build the repository's own instructions produce) scored macro 39.8 (HellaSwag 39.8, Winogrande 56.2, ARC-E 46.0, ARC-C 29.8, MMLU 27.4), far below every INT4 model and far below the model card. Diagnostics (post-hoc, not part of the decision rule):

| Check | Result |
|---|---|
| wikitext perplexity (20 chunks of 512), official HF checkpoint, bf16 on GPU | 16.88 |
| same, official GGUF under bitnet.cpp main | 118.6 (AVX2-only build 118.7; GGUF reconverted with main's converter 96.8; llama-bpe pre-tokenizer override 96.9; batch 512/64/8/1 identical; forced BOS identical) |
| same, official GGUF under bitnet.cpp 765741d (2025-05-21) | **19.64** (+16% vs HF) |
| Qwen2.5-1.5B Q4_K_M under the same pipeline (different tokenizer, sanity only) | 11.7 |
| lm-eval on the HF checkpoint, 400 tasks each | ARC-E 78.0, Winogrande 75.5, HellaSwag acc 48.8 / acc_norm 60.5 |

Conclusion (INFERRED from the table): the failure is in bitnet.cpp main as built here, not in the model; root cause not determined. The valid ternary runtime is bitnet.cpp 765741d. A ternary runtime is now accepted only if its perplexity is within 25% of the HF checkpoint's. The invalid scores must not be cited as evidence about ternary quality.

### R2. Quality at the valid runtime (MEASURED; 5 zero-shot log-likelihood tasks)
| Model | HellaSwag (400) | Winogrande (500) | ARC-E (500) | ARC-C (299) | MMLU (500) | macro |
|---|---|---|---|---|---|---|
| BitNet b1.58 2B4T (bitnet.cpp 765741d) | 61.5 | 72.6 | 66.6 | 43.1 | 33.6 | **55.5** |
| Qwen2.5-3B Q4_K_M | 72.5 | 70.8 | 58.0 | 42.8 | 38.0 | 56.4 |
| SmolLM2-1.7B Q4_K_M | 70.8 | 68.4 | 61.6 | 40.5 | 35.2 | 55.3 |
| Qwen2.5-1.5B Q4_K_M | 65.5 | 62.8 | 62.4 | 41.1 | 36.2 | 53.6 |
| Qwen2.5-1.5B Q4_0 | 65.2 | 64.6 | 57.6 | 37.8 | 34.2 | 51.9 |
| Llama-3.2-1B Q4_K_M | 59.0 | 63.8 | 60.0 | 34.1 | 34.4 | 50.3 |

Binomial standard error is about 2.4-2.5 points per task (2.9 for ARC-C) and about 1.1 points on the macro, so macro differences under about 2 points are noise. INT4 scores were produced under bitnet.cpp main (their perplexity pipeline behaves normally); BitNet's under 765741d. MMLU is near chance (25) for every model in this zero-shot format, so it adds little discrimination. BitNet is stronger than the 1.5B-class INT4 models on Winogrande and ARC, weaker on HellaSwag against SmolLM2/3B and on MMLU. Matched pools (INT4 macro >= BitNet minus tolerance): tolerance 0: {Qwen2.5-3B}; 3: {Qwen2.5-1.5B Q4_K_M, Qwen2.5-3B, SmolLM2}; 5 adds Qwen2.5-1.5B Q4_0. **So, on this evaluation, BitNet's quality is comparable to a 1.5B-3B INT4 model, not to the 1B one.**

### R3. Bytes streamed per decoded token (MEASURED from GGUF tensor types; bandwidth model INFERRED)
| Model | weights streamed per token | of which F16 head/embedding | KV bytes per token (f16) | KV at 4,096 tokens |
|---|---|---|---|---|
| BitNet 2B4T I2_S | 1.179 GB | 0.657 GB (tied embedding/head) | 76.8 KB (30 layers x 5 KV heads) | 315 MB |
| Qwen2.5-1.5B Q4_K_M | 0.980 GB | 0.354 GB (Q6_K head) | 28.7 KB | 117 MB |
| Qwen2.5-1.5B Q4_0 | 0.929 GB | 0.191 GB (Q6_K head) | 28.7 KB | 117 MB |
| Qwen2.5-3B Q4_K_M | 1.924 GB | tied | 36.9 KB | 151 MB |
| SmolLM2-1.7B Q4_K_M | 1.054 GB | tied | 196.6 KB | 805 MB |

Only 0.52 GB of BitNet's 1.18 GB per token is ternary; 56% is a 128k-vocabulary F16 head. If decode were purely bandwidth-bound, BitNet would not be faster than Qwen2.5-1.5B Q4_K_M (weights 1.2x, KV 2.7x). This is the preregistered prediction; it is not yet tested by timing.

### R4. Timing (INCOMPLETE)
The thread sweep and main grid started under the locked harness. Nineteen sweep invocations completed (all with the laptop on AC, 100% charge) and the run then stopped because the harness detected battery power. Partial data are in `results/ternary_vs_int4/perf/sweep_interrupted_battery_event/`. They are **not** used: each cell has one or two repetitions and no complete comparison. One anomaly is retained: bitnet.cpp 765741d on Qwen2.5-1.5B Q4_K_M at 8 threads, prompt 2048, first-token run, reported a 281 s median against a 34 s total-latency run; this coincided with the power-state transition (KI-021) and must be re-measured.

## Provisional reading (nothing below is a timing result)
1. At matched quality the quality-matched INT4 comparators are 1.5B-3B models, and the byte accounting says the ternary model does not obviously stream fewer bytes per token than they do, because of its F16 head and larger KV cache. A ≥2x speed advantage would therefore exceed what bandwidth alone predicts (HYPOTHESIZED to be checked by timing).
2. The ternary format's own advantage sits in the ternary matmul weights (0.52 GB vs about 0.6-0.8 GB for INT4 body weights of comparable models), not in the whole model; a future ternary model with a quantized head and fewer KV heads would change this and is a model-design question, not a bitnet.cpp question.
3. bitnet.cpp's published speedups were not reproduced or refuted: the latest main failed numerically, and the release-era runtime was used.

## Limitations (known before measurement)
Laptop under WSL2 (virtualization, shared power/thermal budget, no frequency/temperature visibility); AC state recorded but power plan Balanced; one CPU microarchitecture (Zen 4 with AVX-512 available); multiple-choice log-likelihood tasks are a proxy for application quality; instruction-tuned models are scored in plain-completion mode; the ternary model's published activation and KV settings are used as released; INT4 models use their K-quant or block-quant defaults; no hand-tuning of either side.
