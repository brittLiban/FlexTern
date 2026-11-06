# EXP-0011 Stage 1 — native ternary vs optimized INT4 on a CPU (quality, latency, throughput, memory)

Status: **Stage 1 complete (verdict SPEED_LOW; adversarial review pending in REV file)**; protocol locked before measurement (lock commit 6a6cac7) with seven documented amendments. Stage 2 (physical energy) is **not approved and not started**: `EXP-0011-stage2-energy-plan.md`.
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
**Status of the results: Stage 1 complete. Verdict SPEED_LOW** (R4). Every failed, interrupted, flagged or superseded run is retained in `results/ternary_vs_int4/`.

### R1. The ternary runtime had to be repaired before its quality meant anything (negative results; amendments 3 and 5, KI-020)
Three successive BitNet quality runs, each later shown invalid or degraded, are retained: (a) official GGUF under bitnet.cpp main `0b341e5`, built by hand with CMake (not through `setup_env.py`): macro 39.8; (b) same GGUF under the release-era bitnet.cpp `765741d` with the file's missing pre-tokenizer type left at llama.cpp's default: macro 55.5; (c) the same with the Llama-3 BPE pre-tokenizer set: macro 56.7, the **valid** result. Diagnostics (post-hoc; not part of the decision rule):

| Check (wikitext perplexity, 20 chunks of 512 unless stated) | Result |
|---|---|
| official HF checkpoint, bf16, GPU | 16.88 |
| official GGUF, bitnet.cpp main | 118.6 (AVX2-only build 118.7; GGUF reconverted with main's converter 96.8; plus pre-tokenizer override 96.9; batch size 512/64/8/1 and forced BOS made no difference) |
| official GGUF, bitnet.cpp 765741d, default pre-tokenizer | 19.64 (+16% vs HF) |
| official GGUF, bitnet.cpp 765741d, `--override-kv tokenizer.ggml.pre=str:llama-bpe` | **16.80** (-0.5% vs HF) |
| Qwen2.5-1.5B Q4_K_M, bitnet.cpp main (different tokenizer; sanity only) | 11.7 |
| lm-eval on the HF checkpoint, 400 tasks each | ARC-E 78.0, Winogrande 75.5, HellaSwag acc 48.8 / acc_norm 60.5 |

The official BitNet 2B4T GGUF carries no `tokenizer.ggml.pre` entry, so llama.cpp falls back to its default pre-tokenizer; this degrades perplexity and task scores noticeably. Separately, bitnet.cpp main as built here has a further numerical defect (perplexity stays 97 even with the tokenizer fixed); its cause is undetermined and the claim is limited to "as built here by hand". The amendment-3 validity gate (+25% of HF) was chosen after seeing +16% and is post-hoc; the corrected runtime is within 0.5%.

### R2. Quality at the valid configuration (MEASURED; 5 zero-shot log-likelihood tasks)
| Model | HellaSwag (400) | Winogrande (500) | ARC-E (500) | ARC-C (299) | MMLU (500) | macro-5 | macro-4 (no MMLU, post-hoc) |
|---|---|---|---|---|---|---|---|
| BitNet b1.58 2B4T (bitnet.cpp 765741d, llama-bpe) | 65.5 | 72.2 | 68.0 | 44.1 | 33.6 | **56.7** | 62.5 |
| Qwen2.5-3B Q4_K_M | 72.5 | 70.8 | 58.0 | 42.8 | 38.0 | 56.4 | 61.0 |
| SmolLM2-1.7B Q4_K_M | 70.8 | 68.4 | 61.6 | 40.5 | 35.2 | 55.3 | 60.3 |
| Qwen2.5-1.5B Q4_K_M | 65.5 | 62.8 | 62.4 | 41.1 | 36.2 | 53.6 | 58.0 |
| Qwen2.5-1.5B Q4_0 | 65.2 | 64.6 | 57.6 | 37.8 | 34.2 | 51.9 | 56.3 |
| Llama-3.2-1B Q4_K_M | 59.0 | 63.8 | 60.0 | 34.1 | 34.4 | 50.3 | 54.2 |

Binomial standard error is about 2.4-2.5 points per task (2.9 for ARC-C), about 1.1 on the macro: differences under about 2 points are noise. INT4 scores come from the bitnet.cpp-main build of `llama-perplexity`, BitNet's from the May-2025 build; **tool-version cross-check:** Qwen2.5-1.5B Q4_K_M re-scored on the May-2025 build gives macro 53.58 (HellaSwag 65.75, Winogrande 64.2, ARC-E 60.6, ARC-C 42.1, MMLU 35.2) against 53.61, so the tool-version offset is about 0.03 points on the macro (per-task offsets up to 1.4). MMLU is near chance (25) for every model in this zero-shot format and adds little; the locked 5-task macro was kept. BitNet's earlier HellaSwag file (default tokenizer) had `acc` null because the first parser expected a percent sign; its 61.5 was read by a fallback reparse of `stdout_tail`, and that run is superseded anyway.

Matched pools (INT4 macro >= BitNet minus tolerance): tolerance 0: **empty** (Qwen2.5-3B is 0.27 below BitNet: a tie within noise, but the rule yields NO_MATCHED_COMPARATOR); tolerance 3: {Qwen2.5-3B, SmolLM2-1.7B}; tolerance 5 adds Qwen2.5-1.5B Q4_K_M (3.1 below BitNet) and Q4_0. Without MMLU (post-hoc) the tolerance-5 pool loses Q4_0 and the tolerance-3 pool is unchanged. **On this evaluation, BitNet's quality is comparable to Qwen2.5-3B and SmolLM2-1.7B INT4 and about 3 points above Qwen2.5-1.5B INT4**, so the headline comparators are the larger INT4 models, which stream more bytes per token than BitNet (R3).

### R3. Bytes streamed per decoded token (MEASURED from GGUF tensor types; bandwidth model INFERRED)
| Model | weights streamed per token | of which F16 head/embedding | KV bytes per token (f16) | KV at 4,096 tokens |
|---|---|---|---|---|
| BitNet 2B4T I2_S | 1.179 GB | 0.657 GB (tied embedding/head) | 76.8 KB (30 layers x 5 KV heads) | 315 MB |
| Qwen2.5-1.5B Q4_K_M | 0.980 GB | 0.354 GB (Q6_K head) | 28.7 KB | 117 MB |
| Qwen2.5-1.5B Q4_0 | 0.929 GB | 0.191 GB (Q6_K head) | 28.7 KB | 117 MB |
| Qwen2.5-3B Q4_K_M | 1.924 GB | tied | 36.9 KB | 151 MB |
| SmolLM2-1.7B Q4_K_M | 1.054 GB | tied | 196.6 KB | 805 MB |

Only 0.52 GB of BitNet's 1.18 GB per token is ternary; 56% is a 128k-vocabulary F16 head. If decode were purely bandwidth-bound, BitNet would not be faster than Qwen2.5-1.5B Q4_K_M (weights 1.2x, KV 2.7x); INT4-bytes / BitNet-bytes per decoded token (weights plus f16 KV at the stated depth): Qwen2.5-1.5B Q4_K_M 0.83 at depth 0, 0.78 at 2,048; Qwen2.5-3B 1.63 and 1.50; SmolLM2-1.7B 0.89 and 1.09 (its 32 KV heads make long contexts expensive). So bandwidth alone favours BitNet only against the 3B comparator (and SmolLM2 at long context), not against the 1.5B-class model. This is the preregistered prediction; it is not yet tested by timing.

### R4. Timing (MEASURED; complete) — locked protocol, 96 main-grid invocations, no failed runs
Laptop (AMD Ryzen 7 260, 8 cores / 16 threads, AVX-512), WSL2, on AC; thread counts chosen per (model, runtime, regime) from the 96-invocation sweep (`thread_choice.json`); 3 repetitions per invocation; TTFT from `llama-bench -pg P,1`, total from `-pg P,G`; decode = (G-1)/(total-TTFT) (derived from two runs, hence noisier). Full per-cell tables with min-max ranges: `results/ternary_vs_int4/report.md`. Representative cells (median; ranges are in the report):

| model / runtime | P=512,G=128 TTFT s / total s | P=2048,G=128 TTFT / total | P=3968,G=128 TTFT / total | decode tok/s (P=512,G=256) | peak RSS GB (P=512 / 3968) |
|---|---|---|---|---|---|
| BitNet 2B4T, bitnet.cpp 765741d | 3.60 / 11.8 | 17.1 / 26.0 | 44.4 / 57.3 | 15.4 | 1.33 / 1.59 |
| Qwen2.5-3B Q4_K_M, llama.cpp | 3.45 / 19.4 | 14.0 / 28.4 | 31.3 / 44.7 | 9.0 | 3.36 / 3.48 |
| SmolLM2-1.7B Q4_K_M, llama.cpp | 2.29 / 9.5 | 9.0 / 20.7 | 20.6 / 33.0 | 17.3 | 2.03 / 2.67 |
| Qwen2.5-1.5B Q4_K_M, llama.cpp | 1.58 / 9.1 | 6.9 / 14.4 | 15.8 / 22.6 | 19.3 | 1.85 / 1.95 |
| Qwen2.5-1.5B Q4_0, llama.cpp | 2.41 / 8.5 | 10.6 / 17.2 | 24.2 / 32.2 | 21.5 | 1.91 / 2.01 |
| Qwen2.5-1.5B Q4_K_M, bitnet.cpp 765741d (same-vintage control) | 5.49 / 11.6 | 18.3 / 26.0 | 41.3 / 50.0 | 19.5 | 1.22 / 1.32 |

**Ratios (INT4 time / BitNet time, so > 1 means BitNet is faster), geometric mean over the 8 cells (4 contexts x 2 output lengths):**

| comparator (quality macro vs BitNet 56.7) | TTFT | decode | end-to-end | worst / best cell (end-to-end) | peak-RSS ratio (INT4 / BitNet) |
|---|---|---|---|---|---|
| Qwen2.5-3B Q4_K_M, llama.cpp (56.4) | 0.79 | 1.37 | **1.12** | 0.74 (P3840,G256) / 1.64 (P512,G128) | 2.38x |
| SmolLM2-1.7B Q4_K_M, llama.cpp (55.3) | 0.55 | 0.93 | **0.74** | 0.51 / 0.91 | 1.59x |
| Qwen2.5-1.5B Q4_K_M, llama.cpp (53.6) | 0.39 | 0.73 | **0.58** | 0.35 / 0.80 | 1.32x |
| Qwen2.5-1.5B Q4_0, llama.cpp (51.9) | 0.60 | 0.64 | **0.63** | 0.41 / 0.82 | 1.36x |
| Qwen2.5-1.5B Q4_K_M, bitnet.cpp 765741d (53.6) | 1.17 | 0.69 | **0.89** | 0.64 / 1.06 | 0.88x |

By context (end-to-end ratio vs Qwen2.5-3B): 1.64 / 1.13 / 1.09 / 0.78 at P = 512 / 1,024 / 2,048 / ~3,900 (G=128), 1.55 / 1.36 / 1.01 / 0.74 (G=256): BitNet's advantage over the 3B model shrinks with context and reverses, because BitNet's prefill (88-142 tok/s) is slower than the 3B model's (123-159 tok/s) at long context and far slower than modern INT4 prefill of the 1.5B-class models (250-330 tok/s).

**Preregistered classification (locked rule):**
- tolerance 3.0 (headline; matched pool {Qwen2.5-3B, SmolLM2-1.7B}, fastest member per metric per cell): geometric-mean end-to-end ratio **0.74 -> SPEED_LOW** (TTFT 0.55, decode 0.93; worst cell 0.51). No audit flag fired.
- tolerance 5.0 (pool also contains the two Qwen2.5-1.5B models): 0.57 -> SPEED_LOW.
- tolerance 0.0: the pool is empty (Qwen2.5-3B is 0.27 points below BitNet, a tie within noise) -> NO_MATCHED_COMPARATOR under the rule; against Qwen2.5-3B alone the ratio is 1.12, still below 1.3.
- Fixed primary comparator (Qwen2.5-1.5B Q4_K_M): end-to-end 0.58.
**Verdict: SPEED_LOW.** BitNet 2B4T on bitnet.cpp 765741d is not faster than the quality-matched INT4 models on this CPU; it is faster only than the larger Qwen2.5-3B (decode 1.37x, end-to-end 1.12x, below the 1.3x bar).

### R5. Variability, anomalies, flagged samples (MEASURED)
- Run-to-run: the spread (max-min over 3 repetitions, relative to the median) has a median of 4.1% across the 96 invocations, is under 10% in 83% of them, and reaches 36% at most; three invocations were flagged and **re-measured with 6 further repetitions each** (originals kept, classification uses the originals): BitNet P3840/G256 total 90.4 s (82-98) vs repeat 69.8 s (68.7-76.0), the original ended with the laptop reporting battery status 1 (an AC-adapter drop during the run): contaminated; Qwen2.5-3B P3840/G256 66.7 s (62-86) vs 61.3 s; Qwen2.5-1.5B Q4_K_M upstream P1024/G256 20.3 s (17.7-24.4) vs 18.0 s. Replacing the flagged cells by their repeats moves the tolerance-3 headline from 0.74 to 0.76 and the tolerance-5 value from 0.57 to 0.58: the verdict does not change.
- The earlier 537 s first-token sample (bitnet.cpp 765741d, Qwen2.5-1.5B Q4_K_M, 8 threads, P=2048, AC) **did not reproduce**: nine repeat samples of that exact cell were 24.3-26.2 s (`perf/anomaly_repeat/`). Its cause is unattributed: a one-off stall of the laptop, WSL2 or the system, not the runtime and not a power-state transition (it was recorded on AC). No stall of this kind occurred in the 96 main-grid invocations or the 96-invocation sweep.
- AC adapter dropped twice during the resumed run (batteries read 98% and 94%); the guard waited, then refused, and amendments 6 and 7 made it wait up to 30 minutes and restart automatically. Only one measured cell (above) was caught mid-drop by the post-run check; earlier sweep invocations lack the post-run check (their pre-run checks were ok).
- Not observable under WSL2: CPU frequency, temperature, throttling. Thermal state is therefore an unmeasured confounder, bounded only by the repeat agreement above.

### R6. What explains the result (labels: MEASURED / INFERRED)
1. **Model size / footprint (INFERRED from R3 and R4):** BitNet's whole-model footprint is 1.18 GB because ternary body weights are 0.52 GB; the quality-matched Qwen2.5-3B needs 1.92 GB and is therefore slower in decode (1.37x measured, 1.50x bandwidth-implied) and uses 2.4x the memory. But SmolLM2-1.7B (macro 55.3) is 1.05 GB, and Qwen2.5-1.5B (53.6) 0.98 GB, so the footprint advantage exists only against the 3B comparator. Whether ternary representation or model design is responsible for the gap to SmolLM2 cannot be separated here: BitNet spends 0.66 GB on an F16 head that a quantized head would nearly remove.
2. **Decode matches bandwidth (MEASURED vs INFERRED):** measured decode ratios (1.37, 0.93, 0.73, 0.64) track the bandwidth-implied ratios (1.50, 1.07, 0.78, 0.74) within 10-15%; no decode effect beyond bytes moved is visible, and none of the audit flags fired.
3. **Prefill / TTFT is where BitNet loses (MEASURED), attributable to runtime maturity or kernel design (INFERRED, not isolated):** on the same bitnet.cpp 765741d build, BitNet's TTFT is 1.17x better than Qwen2.5-1.5B Q4_K_M, whereas modern llama.cpp runs that INT4 model 2.6-3.5x faster than the old build (TTFT) and 2.3-2.8x faster than BitNet. This is the old-runtime effect the directive asked to keep separate: the data cannot say whether a modern ternary kernel would close the gap, and nothing here shows ternary arithmetic is inherently slower.
4. **Memory (MEASURED, runtime-dependent):** the same Qwen2.5-1.5B Q4_K_M file has 1.22-1.32 GB peak RSS under bitnet.cpp 765741d and 1.85-1.95 GB under modern llama.cpp (which repacks weights), so the "memory advantage" of BitNet over that model flips from 1.32x to 0.88x depending on the INT4 runtime. Peak RSS is process-level, includes mmap'd weights and runtime buffers, and is not a minimum-footprint measurement.
5. **Unattributed:** the 3B-vs-BitNet end-to-end gain at short context (1.55-1.64x at P=512) combines the model-size effect with BitNet's tolerable short-context prefill; it vanishes by P~3,900.

### R7. Limitations
One laptop CPU under WSL2 (no frequency/temperature visibility, AC drops, an unexplained earlier stall); `llama-bench` synthetic prompts (no tokenization, sampling or server overhead); the ternary kernel is the May-2025 I2_S (TL2 and the newest kernels not validated); quality is a zero-shot log-likelihood proxy at 300-500 tasks; MMLU is near chance for all models; INT4 quality scores used a different `llama-perplexity` build than BitNet's (cross-check 0.03 macro points); the matched pool is judged on macro differences of about the noise level (1.1 points); only two quantizations (Q4_K_M, Q4_0) were timed for the Qwen 1.5B model and one for the others; no energy, no power. The documented `setup_env.py` flow on bitnet.cpp main was not executed: by code reading it performs the same CMake configuration as the by-hand build and no model conversion, so it is expected (INFERRED, untested) to hit the same defect.

## Verdict and reading (Stage 1)
**SPEED_LOW** under the locked rule. Ternary did not demonstrate a credible speed advantage at matched quality on this CPU and runtime pair; it demonstrated a memory/decode advantage only against the larger Qwen2.5-3B model. EXP-0011 has shown (MEASURED): the corrected quality of BitNet 2B4T is comparable to 3B-class INT4; its decode speed follows the bytes it streams, which are dominated by an F16 head and a large KV cache rather than by ternary weights; the old ternary kernel's prefill is 2-3x slower than modern INT4 prefill. It has not shown (UNMEASURED): whether a modern or optimized ternary kernel would win, any energy effect, behaviour on ARM/Jetson/NPU targets, or application-level accuracy.

## Limitations (known before measurement)
Laptop under WSL2 (virtualization, shared power/thermal budget, no frequency/temperature visibility); AC state recorded but power plan Balanced; one CPU microarchitecture (Zen 4 with AVX-512 available); multiple-choice log-likelihood tasks are a proxy for application quality; instruction-tuned models are scored in plain-completion mode; the ternary model's published activation and KV settings are used as released; INT4 models use their K-quant or block-quant defaults; no hand-tuning of either side.
