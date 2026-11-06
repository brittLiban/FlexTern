#!/bin/bash
# POST-HOC diagnostic (not part of the preregistered decision): is BitNet's low score in the llama.cpp-style eval a runtime/format effect?
# Scores the official packed HF checkpoint with lm-evaluation-harness on the GPU. GPU/CPU are NOT used for any timing here.
cd /mnt/c/Users/liban/Documents/flextern
source ~/ftvenv/bin/activate
export HF_HOME=/mnt/c/Users/liban/.cache/huggingface TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p results/ternary_vs_int4/diagnostic
lm_eval --model hf --model_args pretrained=microsoft/bitnet-b1.58-2B-4T,dtype=bfloat16 --tasks hellaswag,arc_easy,winogrande \
  --limit 400 --device cuda --batch_size 16 --output_path results/ternary_vs_int4/diagnostic/lm_eval_hf_bitnet > ~/exp0011/diag_lmeval.log 2>&1
echo DIAG_DONE >> ~/exp0011/diag_lmeval.log
