# Candidate numerical representation (EXP-0004) — CANDIDATE FORMAT, not FlexTern v1

Status labels: MEASURED (observed on this machine with this software), INFERRED (derived), HYPOTHESIZED (not demonstrated), UNRESOLVED.

## Ternary transformer weights (unchanged, MEASURED in EXP-0002B)

- Packing: 2-bit codes, code = q + 1, four per byte along the output dimension (transformers BitNet packing).
- Accumulation: INT8 activations × ternary; 19–22 signed bits at K = 2048–13312 (INFERRED from format).
- Scale: one BF16 scalar per tensor, applied after accumulation (stored as 1/β for Falcon-E).

## INT4 output head candidate

- Method: RTN asymmetric groupwise, fp16 scale, 4-bit zero (stored uint8), codes packed 2/byte along K; dequantize-to-BF16 per forward (MEASURED).
- Group size 128; 524,288 groups; logical 34.9 MB = 4.156 bits/element incl. metadata (1.3 MB); stored here as 35.1 MB (zero kept in uint8).
- Weight reconstruction error: relative Frobenius 0.1057, max abs 0.03125 (MEASURED).
- Execution requirement: this runtime dequantizes to BF16 (134.2 MB temporary) every forward and runs a BF16 matmul. A hardware consumer would need an INT4-weight × activation path with per-group scale/zero (HYPOTHESIZED).
- Accuracy impact: see quality_comparison.csv (lm_eval tasks and incremental perplexity), MEASURED.

## int4 KV-cache candidate

- Implementation: transformers `QuantizedCache`, hqq backend, 4-bit, group 64, residual 128 tokens kept in BF16 (MEASURED).
- Stored tensors (layer 0 after a ~400-token generation): data torch.uint8 [32, 1348], scale torch.bfloat16 [1, 1348], zero torch.bfloat16 [1, 1348]; effective 4.50 bits/element for the quantized part (MEASURED).
- Cache organization: lazy init quantizes the prefill; new tokens accumulate in a BF16 residual; when it reaches 128 tokens the whole history (dequantized old + residual + new) is re-quantized (MEASURED from source and exercised).
- Decode requirement on this runtime: the entire layer cache is dequantized into a BF16 temporary before every attention call, so resident memory shrinks but the bytes read by attention do not (MEASURED behaviour). A hardware consumer would need attention that reads 4-bit K/V with per-group scale/zero directly (HYPOTHESIZED).
- Update requirement: periodic re-quantization of the history; whether that is idempotent depends on grouping (see test_requantization_drift and the per-position NLL buckets) — MEASURED in this experiment.
- Accuracy impact: incremental perplexity vs the BF16-cache baseline, MEASURED; generation samples, qualitative.

## What is UNRESOLVED

- Whether an INT4-weight head path or a 4-bit-KV attention path can be built without the dequantization temporaries (no kernel measured).
- Whether the quality results transfer to other checkpoints (one model evaluated).
- Acceptable-loss threshold (founder decision; not predeclared).
- Benefit attribution: both representations run on any GPU (category A); hardware-specific benefit (B/C) is unresolved.
