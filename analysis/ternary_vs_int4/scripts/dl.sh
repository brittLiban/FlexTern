#!/bin/bash
# download candidate GGUFs (no timing is done here); records sha256
mkdir -p ~/exp0011/models && cd ~/exp0011/models
get() { # name url
  if [ ! -s "$1" ]; then curl -sL --fail -C - -o "$1" "$2" || { echo "FAIL $1"; rm -f "$1"; return; }; fi
  ls -l "$1" | awk '{print $5, $9}'
}
get bitnet-b1.58-2B-4T-i2_s.gguf https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf
get qwen2.5-1.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
get qwen2.5-1.5b-instruct-q4_0.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf
get qwen2.5-3b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
get llama-3.2-1b-instruct-q4_k_m.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
get smollm2-1.7b-instruct-q4_k_m.gguf https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf
sha256sum *.gguf > SHA256SUMS.txt
cat SHA256SUMS.txt
