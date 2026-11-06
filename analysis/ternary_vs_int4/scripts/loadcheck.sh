#!/bin/bash
# load-only smoke check (no data kept): every candidate loads in each runtime it is assigned
cd ~/exp0011/models
for f in qwen2.5-3b-instruct-q4_k_m; do
  for rt in BitNet/build/bin llama.cpp/build/bin; do
    if ~/exp0011/$rt/llama-bench -m $f.gguf -p 8 -n 1 -r 1 -t 8 -ngl 0 -o csv >/dev/null 2>/tmp/lc.err; then echo "OK   $f $rt"; else echo "FAIL $f $rt: $(tail -2 /tmp/lc.err | tr '\n' ' ' | cut -c1-200)"; fi
  done
done
