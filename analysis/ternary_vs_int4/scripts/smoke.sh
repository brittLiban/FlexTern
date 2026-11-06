#!/bin/bash
B=~/exp0011/BitNet/build/bin; U=~/exp0011/llama.cpp/build/bin; M=~/exp0011/models
$B/llama-bench --help 2>&1 | grep -E '^\s+-(pg|d|r|t|ngl|o|p|n|fa|mmp|delay)[ ,]' | head -20
echo "--- smoke bitnet.cpp runtime on BitNet I2_S"
$B/llama-bench -m $M/bitnet-b1.58-2B-4T-i2_s.gguf -p 16 -n 8 -t 8 -r 1 -o json 2>&1 | tail -30 | cut -c1-200
echo "--- smoke upstream on BitNet I2_S"
$U/llama-bench -m $M/bitnet-b1.58-2B-4T-i2_s.gguf -p 16 -n 8 -t 8 -r 1 -o json 2>&1 | tail -6 | cut -c1-200
