#!/bin/bash
# detached, resumable download of the stronger comparator (no timing is done here)
cd ~/exp0011/models
until [ -s .done3b ]; do
  curl -sL --fail -C - --retry 30 --retry-delay 3 --retry-all-errors -o qwen2.5-3b-instruct-q4_k_m.gguf \
    https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf && echo ok > .done3b
  sleep 3
done
