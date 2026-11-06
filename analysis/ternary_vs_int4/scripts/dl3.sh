#!/bin/bash
# resumable, retrying downloads; stops after ~8 minutes so the caller can re-invoke (no timing is done here)
cd ~/exp0011/models
get() { timeout 480 curl -sL --fail -C - --retry 20 --retry-delay 3 --retry-all-errors -o "$1" "$2" || echo "INCOMPLETE $1"; }
get qwen2.5-1.5b-instruct-q4_0.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf &
get llama-3.2-1b-instruct-q4_k_m.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf &
get smollm2-1.7b-instruct-q4_k_m.gguf https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf &
wait
ls -l | cut -c20-120
