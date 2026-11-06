# EXP-0011 Stage 1 timing report (auto-generated; speed/memory only, no energy)

Invocations: 96; failed: 0; flagged: 3.

## Per-cell measurements (median of 3 repetitions; ranges are min-max; RSS is peak resident set of the process)

| model / runtime / threads | prompt | output | TTFT s | prefill tok/s | decode tok/s | total s | peak RSS GB | flags |
|---|---|---|---|---|---|---|---|---|
| BitNet 2B4T (ternary) / bitnet_old t8 | 512 | 128 | 3.60 (3.53-3.72) | 142 | 15.5 | 11.79 (11.74-12.34) | 1.33 | - |
| BitNet 2B4T (ternary) / bitnet_old t8 | 512 | 256 | 3.92 (3.82-4.20) | 130 | 15.4 | 20.47 (20.13-21.74) | 1.33 | - |
| BitNet 2B4T (ternary) / bitnet_old t8 | 1024 | 128 | 8.34 (8.26-8.40) | 123 | 15.3 | 16.63 (16.53-17.16) | 1.37 | - |
| BitNet 2B4T (ternary) / bitnet_old t8 | 1024 | 256 | 8.55 (8.51-8.60) | 120 | 15.1 | 25.48 (25.11-25.63) | 1.37 | - |
| BitNet 2B4T (ternary) / bitnet_old t12 | 2048 | 128 | 17.06 (16.55-17.40) | 120 | 14.2 | 26.01 (25.85-27.07) | 1.45 | - |
| BitNet 2B4T (ternary) / bitnet_old t12 | 2048 | 256 | 17.37 (16.25-17.99) | 118 | 10.7 | 41.26 (39.53-44.97) | 1.45 | - |
| BitNet 2B4T (ternary) / bitnet_old t12 | 3840 | 256 | 43.82 (42.12-44.59) | 88 | 5.5 | 90.39 (82.37-97.65) | 1.59 | POWER_AFTER_NOT_OK |
| BitNet 2B4T (ternary) / bitnet_old t12 | 3968 | 128 | 44.44 (44.37-45.77) | 89 | 9.9 | 57.27 (55.18-57.43) | 1.59 | - |
| Qwen2.5-1.5B Q4_0 / upstream t8 | 512 | 128 | 2.41 (2.40-2.43) | 212 | 21.0 | 8.46 (8.42-8.62) | 1.91 | - |
| Qwen2.5-1.5B Q4_0 / upstream t8 | 512 | 256 | 2.45 (2.42-2.46) | 209 | 21.5 | 14.30 (14.25-14.45) | 1.91 | - |
| Qwen2.5-1.5B Q4_0 / upstream t8 | 1024 | 128 | 5.29 (5.22-5.56) | 194 | 15.2 | 13.63 (13.55-13.83) | 1.92 | - |
| Qwen2.5-1.5B Q4_0 / upstream t8 | 1024 | 256 | 5.17 (4.95-5.31) | 198 | 20.3 | 17.71 (17.67-18.06) | 1.92 | - |
| Qwen2.5-1.5B Q4_0 / upstream t8 | 2048 | 128 | 10.56 (10.47-10.79) | 194 | 19.2 | 17.16 (17.03-17.30) | 1.95 | - |
| Qwen2.5-1.5B Q4_0 / upstream t8 | 2048 | 256 | 10.54 (10.43-10.59) | 194 | 18.4 | 24.37 (23.64-24.51) | 1.96 | - |
| Qwen2.5-1.5B Q4_0 / upstream t8 | 3840 | 256 | 23.80 (23.64-24.73) | 161 | 19.3 | 37.00 (36.99-37.21) | 2.01 | - |
| Qwen2.5-1.5B Q4_0 / upstream t8 | 3968 | 128 | 24.23 (24.11-24.34) | 164 | 15.9 | 32.21 (31.34-33.21) | 2.01 | - |
| Qwen2.5-1.5B Q4_K_M / bitnet_old t8 | 512 | 128 | 5.49 (5.32-5.59) | 93 | 20.9 | 11.57 (11.46-11.58) | 1.22 | - |
| Qwen2.5-1.5B Q4_K_M / bitnet_old t8 | 512 | 256 | 5.26 (5.24-5.27) | 97 | 19.5 | 18.37 (17.88-20.79) | 1.22 | - |
| Qwen2.5-1.5B Q4_K_M / bitnet_old t8 | 1024 | 128 | 11.29 (11.21-11.31) | 91 | 19.8 | 17.69 (17.68-17.84) | 1.24 | - |
| Qwen2.5-1.5B Q4_K_M / bitnet_old t8 | 1024 | 256 | 11.54 (11.27-11.61) | 89 | 19.5 | 24.64 (24.56-24.68) | 1.24 | - |
| Qwen2.5-1.5B Q4_K_M / bitnet_old t12 | 2048 | 128 | 18.34 (17.69-18.81) | 112 | 16.6 | 25.99 (25.24-27.60) | 1.26 | - |
| Qwen2.5-1.5B Q4_K_M / bitnet_old t12 | 2048 | 256 | 17.90 (17.90-18.61) | 114 | 16.8 | 33.12 (32.50-33.96) | 1.26 | - |
| Qwen2.5-1.5B Q4_K_M / bitnet_old t12 | 3840 | 256 | 39.65 (38.71-42.85) | 97 | 13.7 | 58.23 (56.80-58.39) | 1.32 | - |
| Qwen2.5-1.5B Q4_K_M / bitnet_old t12 | 3968 | 128 | 41.31 (40.54-42.18) | 96 | 14.6 | 50.01 (49.74-51.83) | 1.32 | - |
| Qwen2.5-1.5B Q4_K_M / upstream t12 | 512 | 128 | 1.58 (1.54-1.60) | 323 | 16.9 | 9.10 (8.82-9.95) | 1.85 | - |
| Qwen2.5-1.5B Q4_K_M / upstream t12 | 512 | 256 | 1.62 (1.58-1.67) | 316 | 19.3 | 14.83 (14.79-14.94) | 1.85 | - |
| Qwen2.5-1.5B Q4_K_M / upstream t12 | 1024 | 128 | 3.20 (3.12-3.43) | 320 | 15.8 | 11.23 (10.41-11.33) | 1.87 | - |
| Qwen2.5-1.5B Q4_K_M / upstream t12 | 1024 | 256 | 3.07 (3.07-3.25) | 333 | 14.8 | 20.31 (17.72-24.36) | 1.87 | SPREAD>25% |
| Qwen2.5-1.5B Q4_K_M / upstream t12 | 2048 | 128 | 6.94 (6.92-7.05) | 295 | 17.0 | 14.40 (14.17-16.83) | 1.90 | - |
| Qwen2.5-1.5B Q4_K_M / upstream t12 | 2048 | 256 | 7.13 (7.12-7.35) | 287 | 14.8 | 24.32 (23.85-26.77) | 1.90 | - |
| Qwen2.5-1.5B Q4_K_M / upstream t12 | 3840 | 256 | 14.33 (14.12-15.02) | 268 | 15.0 | 31.35 (30.99-31.54) | 1.95 | - |
| Qwen2.5-1.5B Q4_K_M / upstream t12 | 3968 | 128 | 15.79 (15.72-15.94) | 251 | 18.6 | 22.64 (22.44-22.81) | 1.95 | - |
| Qwen2.5-3B Q4_K_M / upstream t12 | 512 | 128 | 3.45 (3.44-3.98) | 148 | 8.0 | 19.38 (16.70-20.19) | 3.36 | - |
| Qwen2.5-3B Q4_K_M / upstream t12 | 512 | 256 | 3.26 (3.19-3.29) | 157 | 9.0 | 31.67 (30.59-31.95) | 3.36 | - |
| Qwen2.5-3B Q4_K_M / upstream t12 | 1024 | 128 | 6.43 (6.41-6.43) | 159 | 10.3 | 18.71 (18.57-18.75) | 3.38 | - |
| Qwen2.5-3B Q4_K_M / upstream t12 | 1024 | 256 | 6.57 (6.45-6.76) | 156 | 9.1 | 34.65 (34.04-35.80) | 3.38 | - |
| Qwen2.5-3B Q4_K_M / upstream t12 | 2048 | 128 | 13.97 (13.85-13.99) | 147 | 8.8 | 28.44 (27.82-29.69) | 3.42 | - |
| Qwen2.5-3B Q4_K_M / upstream t12 | 2048 | 256 | 13.55 (13.53-13.58) | 151 | 9.1 | 41.52 (40.94-42.60) | 3.42 | - |
| Qwen2.5-3B Q4_K_M / upstream t12 | 3840 | 256 | 31.15 (30.93-32.55) | 123 | 7.2 | 66.73 (62.20-86.25) | 3.48 | SPREAD>25% |
| Qwen2.5-3B Q4_K_M / upstream t12 | 3968 | 128 | 31.29 (31.15-31.54) | 127 | 9.4 | 44.74 (44.38-49.42) | 3.48 | - |
| SmolLM2-1.7B Q4_K_M / upstream t8 | 512 | 128 | 2.29 (2.27-2.29) | 223 | 17.7 | 9.49 (9.39-9.65) | 2.03 | - |
| SmolLM2-1.7B Q4_K_M / upstream t8 | 512 | 256 | 2.44 (2.30-2.52) | 210 | 17.3 | 17.19 (17.15-17.47) | 2.03 | - |
| SmolLM2-1.7B Q4_K_M / upstream t8 | 1024 | 128 | 5.18 (5.17-5.22) | 198 | 16.0 | 13.10 (12.98-13.21) | 2.13 | - |
| SmolLM2-1.7B Q4_K_M / upstream t8 | 1024 | 256 | 5.10 (5.03-5.16) | 201 | 14.1 | 23.13 (21.87-23.29) | 2.13 | - |
| SmolLM2-1.7B Q4_K_M / upstream t12 | 2048 | 128 | 8.98 (8.49-9.08) | 228 | 10.8 | 20.71 (18.16-20.79) | 2.33 | - |
| SmolLM2-1.7B Q4_K_M / upstream t12 | 2048 | 256 | 8.95 (8.41-9.26) | 229 | 10.9 | 32.43 (29.40-35.39) | 2.33 | - |
| SmolLM2-1.7B Q4_K_M / upstream t12 | 3840 | 256 | 19.68 (19.13-20.42) | 195 | 9.5 | 46.52 (45.78-46.86) | 2.67 | - |
| SmolLM2-1.7B Q4_K_M / upstream t12 | 3968 | 128 | 20.61 (19.05-21.34) | 193 | 10.2 | 33.03 (31.38-34.48) | 2.67 | - |

## Ratios against each INT4 comparator (INT4 time / BitNet time; >1 = BitNet faster)

| comparator | geomean TTFT | geomean decode | geomean total | min total (cell) | max total (cell) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B Q4_K_M / upstream | 0.39 | 0.73 | 0.58 | 0.35 (P3840,G256) | 0.80 (P1024,G256) |
| Qwen2.5-1.5B Q4_K_M / bitnet_old | 1.17 | 0.69 | 0.89 | 0.64 (P3840,G256) | 1.06 (P1024,G128) |
| Qwen2.5-1.5B Q4_0 / upstream | 0.60 | 0.64 | 0.63 | 0.41 (P3840,G256) | 0.82 (P1024,G128) |
| Qwen2.5-3B Q4_K_M / upstream | 0.79 | 1.37 | 1.12 | 0.74 (P3840,G256) | 1.64 (P512,G128) |
| SmolLM2-1.7B Q4_K_M / upstream | 0.55 | 0.93 | 0.74 | 0.51 (P3840,G256) | 0.91 (P1024,G256) |

## Per-cell total-latency ratio (INT4 total / BitNet total) by context and output length

| comparator | G | P=512 | P=1024 | P=2048 | P=3968/3840 |
|---|---|---|---|---|---|
| Qwen2.5-1.5B Q4_K_M / upstream | 128 | 0.77 | 0.68 | 0.55 | 0.40 |
| Qwen2.5-1.5B Q4_K_M / upstream | 256 | 0.72 | 0.80 | 0.59 | 0.35 |
| Qwen2.5-1.5B Q4_K_M / bitnet_old | 128 | 0.98 | 1.06 | 1.00 | 0.87 |
| Qwen2.5-1.5B Q4_K_M / bitnet_old | 256 | 0.90 | 0.97 | 0.80 | 0.64 |
| Qwen2.5-1.5B Q4_0 / upstream | 128 | 0.72 | 0.82 | 0.66 | 0.56 |
| Qwen2.5-1.5B Q4_0 / upstream | 256 | 0.70 | 0.69 | 0.59 | 0.41 |
| Qwen2.5-3B Q4_K_M / upstream | 128 | 1.64 | 1.13 | 1.09 | 0.78 |
| Qwen2.5-3B Q4_K_M / upstream | 256 | 1.55 | 1.36 | 1.01 | 0.74 |
| SmolLM2-1.7B Q4_K_M / upstream | 128 | 0.80 | 0.79 | 0.80 | 0.58 |
| SmolLM2-1.7B Q4_K_M / upstream | 256 | 0.84 | 0.91 | 0.79 | 0.51 |

## Peak resident memory and bandwidth-implied decode ratio

| comparator | RSS ratio (INT4 / BitNet) geomean [min-max] | measured decode ratio (BitNet / INT4) geomean | bandwidth-implied decode ratio (INT4 bytes / BitNet bytes, weights + f16 KV) geomean |
|---|---|---|---|
| Qwen2.5-1.5B Q4_K_M / upstream | 1.32 [1.23-1.40] | 0.73 | 0.78 |
| Qwen2.5-1.5B Q4_K_M / bitnet_old | 0.88 [0.83-0.92] | 0.69 | 0.78 |
| Qwen2.5-1.5B Q4_0 / upstream | 1.36 [1.26-1.44] | 0.64 | 0.74 |
| Qwen2.5-3B Q4_K_M / upstream | 2.38 [2.19-2.52] | 1.37 | 1.50 |
| SmolLM2-1.7B Q4_K_M / upstream | 1.59 [1.52-1.68] | 0.93 | 1.07 |

(Decode throughput is derived as (G-1)/(total - TTFT) from two separate runs, so it is noisier than TTFT and total latency; use it with the ranges above.)

## Preregistered classification (locked rule)

- tolerance 3.0: matched INT4 = ['QWEN25_3B_Q4KM', 'SMOLLM2_1P7B_Q4KM']; outcome = `{"label": "SPEED_LOW", "geomean_total": 0.7398256070587872, "geomean_decode": 0.9328368703060553, "geomean_ttft": 0.5492765116603441, "geomean_bandwidth_implied_decode": 1.0696317276956424, "worst_cell_total": 0.5146430500011013, "flags": []}`
  - fixed primary comparator (QWEN25_1P5B_Q4KM) geomeans: {"decode": 0.661, "ttft": 0.385, "total": 0.584}
- tolerance 0.0: matched INT4 = []; outcome = `{"label": "NO_MATCHED_COMPARATOR", "flags": [], "note": "no INT4 model reaches the ternary model's quality within tolerance: no efficiency ratio is computed; see the quality-vs-speed table"}`
- tolerance 5.0: matched INT4 = ['QWEN25_1P5B_Q4KM', 'QWEN25_1P5B_Q40', 'QWEN25_3B_Q4KM', 'SMOLLM2_1P7B_Q4KM']; outcome = `{"label": "SPEED_LOW", "geomean_total": 0.5659577887271544, "geomean_decode": 0.6115362746531778, "geomean_ttft": 0.3854263947660237, "geomean_bandwidth_implied_decode": 0.7513972887922773, "worst_cell_total": 0.34678709762350396, "flags": []}`
  - fixed primary comparator (QWEN25_1P5B_Q4KM) geomeans: {"decode": 0.661, "ttft": 0.385, "total": 0.584}

## Re-measurement of the flagged cells (originals are kept; the classification above uses the ORIGINAL medians)

| cell | original total s (samples) | repeat total s (6 samples) | original flags |
|---|---|---|---|
| BitNet 2B4T (ternary) / bitnet_old P3840 G256 | 90.4 (82.4, 90.4, 97.7) | 69.8 (76.0, 70.0, 70.8, 69.6, 68.7, 69.7) | POWER_AFTER_NOT_OK |
| Qwen2.5-1.5B Q4_K_M / upstream P1024 G256 | 20.3 (17.7, 20.3, 24.4) | 18.0 (18.6, 18.1, 18.2, 18.0, 17.9, 18.0) | SPREAD>25% |
| Qwen2.5-3B Q4_K_M / upstream P3840 G256 | 66.7 (66.7, 62.2, 86.2) | 61.3 (63.4, 60.7, 60.7, 61.9, 62.6, 60.5) | SPREAD>25% |

Sensitivity (flagged cells replaced by their repeat medians), tolerance 3.0: geomean end-to-end ratio = 0.764.

Sensitivity (flagged cells replaced by their repeat medians), tolerance 5.0: geomean end-to-end ratio = 0.584.