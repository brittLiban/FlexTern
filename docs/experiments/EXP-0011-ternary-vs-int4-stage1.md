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
_Pending._ Quality, thread sweep, main grid and the preregistered classification will be added here, with every failed or anomalous run retained in `results/ternary_vs_int4/`.

## Limitations (known before measurement)
Laptop under WSL2 (virtualization, shared power/thermal budget, no frequency/temperature visibility); AC state recorded but power plan Balanced; one CPU microarchitecture (Zen 4 with AVX-512 available); multiple-choice log-likelihood tasks are a proxy for application quality; instruction-tuned models are scored in plain-completion mode; the ternary model's published activation and KV settings are used as released; INT4 models use their K-quant or block-quant defaults; no hand-tuning of either side.
