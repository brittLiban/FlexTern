#!/bin/bash
# post-hoc: does the pre-tokenizer warning explain the remaining +16% perplexity gap under bitnet.cpp 765741d?
cd /tmp
B=~/exp0011/BitNet_old/build/bin/llama-perplexity
M=~/exp0011/models/bitnet-b1.58-2B-4T-i2_s.gguf
echo "--- baseline (as scored)"; $B -m $M -f ~/exp0011/data/wiki.test.raw -c 512 --chunks 20 -t 8 -ngl 0 2>&1 | grep -E "pre-token|Final estimate"
echo "--- override tokenizer.ggml.pre=llama-bpe"; $B -m $M -f ~/exp0011/data/wiki.test.raw -c 512 --chunks 20 -t 8 -ngl 0 --override-kv tokenizer.ggml.pre=str:llama-bpe 2>&1 | grep -E "pre-token|Final estimate|error|invalid"
echo PREOLD_DONE
