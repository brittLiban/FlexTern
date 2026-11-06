#!/bin/bash
cd ~/exp0011/BitNet
source ~/ftvenv/bin/activate
export HF_HOME=/mnt/c/Users/liban/.cache/huggingface PYTHONPATH=$HOME/exp0011/BitNet/3rdparty/llama.cpp/gguf-py PYTHONDONTWRITEBYTECODE=1
SNAP=$(ls -d /mnt/c/Users/liban/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/*/ | head -1)
python utils/convert-hf-to-gguf-bitnet.py "$SNAP" --outtype i2_s --outfile ~/exp0011/models/bitnet-b1.58-2B-4T-i2_s-reconv.gguf 2>&1 | tail -15
ls -l ~/exp0011/models/bitnet-b1.58-2B-4T-i2_s-reconv.gguf
cd /tmp
~/exp0011/BitNet/build/bin/llama-perplexity -m ~/exp0011/models/bitnet-b1.58-2B-4T-i2_s-reconv.gguf -f ~/exp0011/data/wiki.test.raw -c 512 --chunks 20 -t 8 -ngl 0 2>&1 | grep -E "Final estimate|error|failed" | head
echo CONV_DONE
