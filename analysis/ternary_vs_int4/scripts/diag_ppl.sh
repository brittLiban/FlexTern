#!/bin/bash
cd /mnt/c/Users/liban/Documents/flextern
source ~/ftvenv/bin/activate
export HF_HOME=/mnt/c/Users/liban/.cache/huggingface TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
python -m analysis.ternary_vs_int4.diag_ppl > ~/exp0011/diag_ppl.log 2>&1
echo PPL_DONE >> ~/exp0011/diag_ppl.log
