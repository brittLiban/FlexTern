#!/bin/bash
cd ~/exp0011/data
rm -f winogrande-debiased-eval.csv
curl -sL --fail -o winogrande-debiased-eval.csv https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/resolve/main/winogrande-debiased-eval.csv || echo FAIL winogrande
for f in arc-easy-validation.bin arc-challenge-validation.bin; do
  curl -sL --fail -o $f https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/$f || echo FAIL $f
done
ls -l
sha256sum hellaswag_val_full.txt winogrande-debiased-eval.csv arc-*.bin > SHA256SUMS.txt
