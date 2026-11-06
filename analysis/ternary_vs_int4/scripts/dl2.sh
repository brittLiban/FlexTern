#!/bin/bash
# parallel resumable downloads of the remaining candidate GGUFs (no timing is done here)
cd ~/exp0011/models
get() { curl -sL --fail -C - -o "$1" "$2" || echo "FAIL $1"; }
get qwen2.5-3b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf &
get llama-3.2-1b-instruct-q4_k_m.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf &
get smollm2-1.7b-instruct-q4_k_m.gguf https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf &
wait
echo DL2_DONE
