# EXP-0012 analytical cost table (ANALYTICAL, not measured)

Per control step, batch 1. Weight bytes: fp=16 bit, int8=8, int4=4 (+2 B scale/out-channel), ternary=2 (+2 B/tensor); first conv, norms and biases not counted as quantized. State read+write at 16 bit (8 bit for -s8). The trigger head (and 8x8 pooling) is charged for recurrent arms in the adaptive rows. Predictor and EMA encoder are excluded (training/analysis only). Ternary MACs are adds.

| arm | encoder fraction | trigger charged | MACs (k) | bytes (kB) | encoder share of MACs | encoder share of bytes |
|---|---|---|---|---|---|---|
| R | 1.00 | no | 1145 | 348.5 | 0.98 | 0.89 |
| S4 | 1.00 | no | 1477 | 351.9 | 0.99 | 0.89 |
| T16 | 1.00 | no | 1433 | 908.1 | 0.79 | 0.34 |
| G | 1.00 | no | 1265 | 588.1 | 0.89 | 0.53 |
| G | 0.33 | yes | 534 | 421.0 | 0.70 | 0.24 |
| G | 0.50 | yes | 725 | 473.9 | 0.78 | 0.33 |
| G-I8a | 1.00 | no | 1265 | 298.5 | 0.89 | 0.53 |
| G-I8a | 0.33 | yes | 534 | 213.8 | 0.70 | 0.24 |
| G-I8a | 0.50 | yes | 725 | 240.6 | 0.78 | 0.33 |
| G-I4u | 1.00 | no | 1265 | 383.3 | 0.89 | 0.81 |
| G-I4u | 0.33 | yes | 534 | 185.3 | 0.70 | 0.55 |
| G-I4u | 0.50 | yes | 725 | 238.2 | 0.78 | 0.65 |
| G-I4a | 1.00 | no | 1265 | 152.6 | 0.89 | 0.53 |
| G-I4a | 0.33 | yes | 534 | 109.1 | 0.70 | 0.24 |
| G-I4a | 0.50 | yes | 725 | 122.9 | 0.78 | 0.33 |
| G-Tu | 1.00 | no | 1265 | 347.1 | 0.89 | 0.90 |
| G-Tu | 0.33 | yes | 534 | 143.8 | 0.70 | 0.71 |
| G-Tu | 0.50 | yes | 725 | 196.7 | 0.78 | 0.79 |
| G-Ta | 1.00 | no | 1265 | 77.4 | 0.89 | 0.54 |
| G-Ta | 0.33 | yes | 534 | 54.8 | 0.70 | 0.25 |
| G-Ta | 0.50 | yes | 725 | 61.9 | 0.78 | 0.34 |
