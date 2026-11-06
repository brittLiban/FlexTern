# EXP-0011 run scripts (as run; reproducibility record)

These are the shell/PowerShell scripts used to build, download, validate and run EXP-0011 Stage 1 on the founder's laptop (WSL2 Ubuntu, user `liban`, working area `~/exp0011`, venv `~/ftvenv`). They are kept verbatim, including one-off diagnostics; the locked parameters live in `../config.py` and `../protocol_v1*.json`. Models, builds and data are NOT in git.

Order of use:
1. `setup1.sh` — clone Microsoft/BitNet (recursive; commit `0b341e5`) and ggml-org/llama.cpp (`e30ad80`).
2. `build1.sh upstream` — llama.cpp: `cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++`.
   bitnet.cpp main (UNUSABLE for BitNet 2B4T here, KI-020): `cmake -B build -DBITNET_X86_TL2=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_CURL=OFF`.
3. `old.sh`, `old2.sh` — the validated ternary runtime: copy BitNet to `BitNet_old`, `git checkout 765741d` + submodules, copy a preset TL2 header (`preset_kernels/Llama3-8B-1.58-100B-tokens/bitnet-lut-kernels-tl2.h` and its `kernel_config_tl2.ini`, needed only to satisfy CMake; the I2_S kernel is used), `cmake -B build -DBITNET_X86_TL2=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release`, build targets `llama-perplexity llama-bench`.
4. `dl.sh`, `dl2.sh`, `dl3.sh`, `dl4.sh`, `data.sh` — model and dataset downloads (URLs inside; Qwen2.5-3B from the bartowski mirror because the Qwen repo download was too slow). SHA-256 values are in `../protocol_v1.json`.
5. `loadcheck.sh`, `smoke.sh`, `pgsmoke.py` — load-only and 1-repetition harness checks (not data).
6. `runq.sh`, `runq_hs.sh`, `runq5.sh` — detached quality runs (resumable). `pre_old.sh`, `diag_ppl.sh`, `diag_lmeval.sh`, `conv2.sh` — post-hoc diagnostics (perplexity parity with the HF checkpoint, lm-eval on the HF checkpoint, GGUF reconversion with bitnet.cpp's converter).
7. `keepawake.ps1` — process-scoped Windows keep-awake (`SetThreadExecutionState`), no persistent power setting changed; stopped by creating `keepawake.stop` next to it.
8. `runperf.sh`, `runperf2.sh` — select, thread sweep, choose threads, main grid with automatic restart after power-guard refusals (amendments 6-7). `anom.sh`, `flagrep.sh` — anomaly and flagged-cell repeats.
9. Analysis: `python -m analysis.ternary_vs_int4.report` then `python -m analysis.ternary_vs_int4.attribution` (WSL, `PYTHONPATH=.`).
