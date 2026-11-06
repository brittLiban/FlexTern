# EXP-0011 Stage 1 attribution analysis (post-hoc; INFERRED unless stated)

## A. Runtime-independent footprint (MEASURED from GGUF tensors; f16 KV cache)

| model | all tensors GB (resident once loaded) | + KV at 0 | + KV at 2,048 | + KV at 4,096 | weights streamed per token GB |
|---|---|---|---|---|---|
| BitNet 2B4T (ternary) | 1.18 | 1.18 | 1.34 | 1.49 | 1.18 |
| Qwen2.5-3B Q4_K_M | 1.92 | 1.92 | 2.00 | 2.07 | 1.92 |
| SmolLM2-1.7B Q4_K_M | 1.05 | 1.05 | 1.46 | 1.86 | 1.05 |
| Qwen2.5-1.5B Q4_K_M | 1.11 | 1.11 | 1.17 | 1.23 | 0.98 |
| Qwen2.5-1.5B Q4_0 | 1.06 | 1.06 | 1.12 | 1.18 | 0.93 |

Static footprint (all tensors + KV) is runtime-independent; measured peak RSS (report.md) additionally depends on the runtime (the same Qwen2.5-1.5B Q4_K_M file peaks at 1.22-1.32 GB under bitnet.cpp 765741d and 1.85-1.95 GB under modern llama.cpp). Against the 3B model BitNet's footprint advantage is real (1.18 vs 1.93 GB static); against SmolLM2 it holds only at long context (SmolLM2 has 32 KV heads); against the 1.5B models it does not exist.

## B. Effective memory bandwidth in decode (INFERRED): decode tok/s x bytes streamed per token (weights + f16 KV at depth P+G/2), GB/s

| model / runtime | min | median | max over the 8 cells |
|---|---|---|---|
| BitNet 2B4T (ternary) / bitnet_old | 8.1 | 18.9 | 19.3 |
| Qwen2.5-1.5B Q4_0 / upstream | 14.6 | 19.3 | 20.4 |
| Qwen2.5-1.5B Q4_K_M / bitnet_old | 15.0 | 18.4 | 20.8 |
| Qwen2.5-1.5B Q4_K_M / upstream | 15.0 | 16.6 | 20.3 |
| Qwen2.5-3B Q4_K_M / upstream | 14.8 | 17.7 | 20.3 |
| SmolLM2-1.7B Q4_K_M / upstream | 15.9 | 18.5 | 20.6 |

(Derived decode throughput is noisy; the P=3840,G=256 BitNet cell is affected by an AC-adapter drop in the original run.)

## C. Prefill throughput per body parameter (INFERRED): prefill tok/s x block-matrix parameters (G param-tokens/s), by prompt length

| model / runtime | P=512 | P=1024 | P=2048 | P~3.9k | body params (B) |
|---|---|---|---|---|---|
| BitNet 2B4T (ternary) / bitnet_old | 297 | 256 | 250 | 186 | 2.08 |
| Qwen2.5-1.5B Q4_0 / upstream | 278 | 254 | 254 | 215 | 1.31 |
| Qwen2.5-1.5B Q4_K_M / bitnet_old | 122 | 119 | 146 | 126 | 1.31 |
| Qwen2.5-1.5B Q4_K_M / upstream | 423 | 419 | 387 | 329 | 1.31 |
| Qwen2.5-3B Q4_K_M / upstream | 412 | 442 | 407 | 352 | 2.77 |
| SmolLM2-1.7B Q4_K_M / upstream | 360 | 319 | 367 | 310 | 1.61 |

If these throughputs are similar across models on the same runtime, prefill time scales with body size rather than with weight format.

## D. Sensitivity to the contaminated cell (BitNet P3840,G256; AC adapter dropped during the original run)

| comparator | context (G=128): P=512 / 1024 / 2048 / 3968 | context (G=256), repeat-substituted: P=512 / 1024 / 2048 / 3840 |
|---|---|---|
| Qwen2.5-3B Q4_K_M / upstream | 1.64 / 1.13 / 1.09 / 0.78 | 1.55 / 1.36 / 1.01 / 0.88 |
| SmolLM2-1.7B Q4_K_M / upstream | 0.80 / 0.79 / 0.80 / 0.58 | 0.84 / 0.91 / 0.79 / 0.67 |
| Qwen2.5-1.5B Q4_K_M / upstream | 0.77 / 0.68 / 0.55 / 0.40 | 0.72 / 0.71 / 0.59 / 0.45 |

The G=256 column's last entry uses the repeat-measured values for both sides where available; the context trend against the 3B model is carried by the G=128 column, whose last cell (P=3968) was not flagged.

## E. Uncertainty on the headline (bootstrap of repetitions, 2,000 draws, seed 1; medians recomputed; fastest pool member per cell)

- tolerance 3.0 (pool ['QWEN25_3B_Q4KM', 'SMOLLM2_1P7B_Q4KM']): geometric-mean end-to-end ratio 0.733, 95% bootstrap range [0.709, 0.755] (locked point estimate in report.md). This range reflects only run-to-run noise in 3 repetitions per cell, not systematic error (WSL2, thermal state, runtime vintage).
- tolerance 5.0 (pool ['QWEN25_1P5B_Q4KM', 'QWEN25_1P5B_Q40', 'QWEN25_3B_Q4KM', 'SMOLLM2_1P7B_Q4KM']): geometric-mean end-to-end ratio 0.564, 95% bootstrap range [0.549, 0.582] (locked point estimate in report.md). This range reflects only run-to-run noise in 3 repetitions per cell, not systematic error (WSL2, thermal state, runtime vintage).

## F. Head quantization, HYPOTHESIZED (not measured; quality untested)
BitNet's tied 128,256 x 2,560 head is F16 (0.657 GB). A Q6_K-class head (6.5625 bits/weight) would be about 0.269 GB, lowering the weights streamed per token from 1.18 to about 0.79 GB (bandwidth-implied decode vs Qwen2.5-1.5B Q4_K_M would change from 0.78 to roughly 1.1-1.2 at short context). Whether BitNet keeps its quality with a quantized head is unknown here; EXP-0004 found a Falcon-E 1B model tolerated an INT4 head, which is relevant precedent only.
