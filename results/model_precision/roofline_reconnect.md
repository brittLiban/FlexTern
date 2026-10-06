# EXP-0003 scenarios re-evaluated with EXP-0004 measured representations (Falcon-E 1B, batch 1, 100 GB/s; ANALYTICAL)

Measured INT4 head: 4.156 bits/element logical (fp16 scale + 4-bit zero per group of 128); 4.188 bits/element as stored here (zero in uint8). Measured int4 KV cache (hqq, g=64, axis 0, 128-token BF16 residual): effective bits/element by context: 128: 4.59, 512: 4.52, 2048: 4.51, 4096: 4.50, 8192: 4.50, 16384: 4.50.

| ctx | scenario | head bits | KV bits | ternary MB | head MB | KV MB | total MB | ternary frac | limiter | ideal ceiling (mem-bound) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 512 | EXP-0003 reference (BF16 head, BF16 KV) | 16.00 | 16.00 | 396.4 | 134.2 | 12.6 | 543.4 | 0.729 | ternary | 3.70x |
| 512 | A: measured INT4 head bytes, BF16 KV | 4.16 | 16.00 | 396.4 | 34.9 | 12.6 | 444.0 | 0.893 | ternary | 9.31x |
| 512 | B: BF16 head, measured int4 KV bytes (logical) | 16.00 | 4.52 | 396.4 | 134.2 | 3.6 | 534.3 | 0.742 | ternary | 3.87x |
| 512 | C: measured INT4 head + int4 KV (logical) | 4.16 | 4.52 | 396.4 | 34.9 | 3.6 | 435.0 | 0.911 | ternary | 11.26x |
| 512 | C attainable on this runtime (KV dequantized to BF16 before attention) | 4.16 | 16.00 | 396.4 | 34.9 | 12.6 | 444.0 | 0.893 | ternary | 9.31x |
| 2048 | EXP-0003 reference (BF16 head, BF16 KV) | 16.00 | 16.00 | 396.4 | 134.2 | 50.4 | 581.1 | 0.682 | ternary | 3.15x |
| 2048 | A: measured INT4 head bytes, BF16 KV | 4.16 | 16.00 | 396.4 | 34.9 | 50.4 | 481.8 | 0.823 | ternary | 5.64x |
| 2048 | B: BF16 head, measured int4 KV bytes (logical) | 16.00 | 4.51 | 396.4 | 134.2 | 14.2 | 545.0 | 0.727 | ternary | 3.67x |
| 2048 | C: measured INT4 head + int4 KV (logical) | 4.16 | 4.51 | 396.4 | 34.9 | 14.2 | 445.6 | 0.889 | ternary | 9.05x |
| 2048 | C attainable on this runtime (KV dequantized to BF16 before attention) | 4.16 | 16.00 | 396.4 | 34.9 | 50.4 | 481.8 | 0.823 | ternary | 5.64x |
| 4096 | EXP-0003 reference (BF16 head, BF16 KV) | 16.00 | 16.00 | 396.4 | 134.2 | 100.7 | 631.5 | 0.628 | ternary | 2.69x |
| 4096 | A: measured INT4 head bytes, BF16 KV | 4.16 | 16.00 | 396.4 | 34.9 | 100.7 | 532.1 | 0.745 | ternary | 3.92x |
| 4096 | B: BF16 head, measured int4 KV bytes (logical) | 16.00 | 4.50 | 396.4 | 134.2 | 28.3 | 559.1 | 0.709 | ternary | 3.44x |
| 4096 | C: measured INT4 head + int4 KV (logical) | 4.16 | 4.50 | 396.4 | 34.9 | 28.3 | 459.8 | 0.862 | ternary | 7.25x |
| 4096 | C attainable on this runtime (KV dequantized to BF16 before attention) | 4.16 | 16.00 | 396.4 | 34.9 | 100.7 | 532.1 | 0.745 | ternary | 3.92x |
| 8192 | EXP-0003 reference (BF16 head, BF16 KV) | 16.00 | 16.00 | 396.4 | 134.2 | 201.4 | 732.1 | 0.541 | ternary | 2.18x |
| 8192 | A: measured INT4 head bytes, BF16 KV | 4.16 | 16.00 | 396.4 | 34.9 | 201.4 | 632.8 | 0.626 | ternary | 2.68x |
| 8192 | B: BF16 head, measured int4 KV bytes (logical) | 16.00 | 4.50 | 396.4 | 134.2 | 56.6 | 587.4 | 0.675 | ternary | 3.07x |
| 8192 | C: measured INT4 head + int4 KV (logical) | 4.16 | 4.50 | 396.4 | 34.9 | 56.6 | 488.1 | 0.812 | ternary | 5.32x |
| 8192 | C attainable on this runtime (KV dequantized to BF16 before attention) | 4.16 | 16.00 | 396.4 | 34.9 | 201.4 | 632.8 | 0.626 | ternary | 2.68x |
| 16384 | EXP-0003 reference (BF16 head, BF16 KV) | 16.00 | 16.00 | 396.4 | 134.2 | 402.7 | 933.5 | 0.425 | attention_kv | 1.74x |
| 16384 | A: measured INT4 head bytes, BF16 KV | 4.16 | 16.00 | 396.4 | 34.9 | 402.7 | 834.1 | 0.475 | attention_kv | 1.91x |
| 16384 | B: BF16 head, measured int4 KV bytes (logical) | 16.00 | 4.50 | 396.4 | 134.2 | 113.3 | 644.1 | 0.615 | ternary | 2.60x |
| 16384 | C: measured INT4 head + int4 KV (logical) | 4.16 | 4.50 | 396.4 | 34.9 | 113.3 | 544.7 | 0.728 | ternary | 3.67x |
| 16384 | C attainable on this runtime (KV dequantized to BF16 before attention) | 4.16 | 16.00 | 396.4 | 34.9 | 402.7 | 834.1 | 0.475 | attention_kv | 1.91x |
| 32768 | EXP-0003 reference (BF16 head, BF16 KV) | 16.00 | 16.00 | 396.4 | 134.2 | 805.3 | 1,336.1 | 0.297 | attention_kv | 1.42x |
| 32768 | A: measured INT4 head bytes, BF16 KV | 4.16 | 16.00 | 396.4 | 34.9 | 805.3 | 1,236.8 | 0.320 | attention_kv | 1.47x |
| 32768 | B: BF16 head, measured int4 KV bytes (logical) | 16.00 | 4.50 | 396.4 | 134.2 | 226.5 | 757.3 | 0.523 | ternary | 2.10x |
| 32768 | C: measured INT4 head + int4 KV (logical) | 4.16 | 4.50 | 396.4 | 34.9 | 226.5 | 658.0 | 0.602 | ternary | 2.52x |
| 32768 | C attainable on this runtime (KV dequantized to BF16 before attention) | 4.16 | 16.00 | 396.4 | 34.9 | 805.3 | 1,236.8 | 0.320 | attention_kv | 1.47x |

Reading: 'logical' rows assume hardware that reads the compressed form directly (HYPOTHESIZED). The 'attainable' row is what this PyTorch runtime actually streams: the compressed head (dequantized on-chip per forward) but a BF16 KV temporary every step. Quality of each representation is reported separately in EXP-0004.
