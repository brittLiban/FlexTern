# Known Issues

Unresolved technical defects and limitations. When an issue is resolved, move it to the Closed section with a one-line resolution and commit reference. Keep enough context to understand what happened; drop everything else.

## Open

```
KI-001
Status:           Workaround
Severity:         Minor
First observed:   2026-10-05
Relevant commit:  823902e
Description:      The YoWASP (WebAssembly) Yosys 0.69 build takes ~13 minutes on
                  synth/int8_dot16_baseline.ys. Native Yosys 0.52 under WSL does
                  the same script in <3 s. With `synth -noabc` the WASM build
                  finishes in 2 s, so ABC under WASM is the slow stage on the
                  16-multiplier netlist. The other three scripts take ~1-2 s each.
Impact:           scripts/run_all.ps1 takes ~14 minutes end to end on Windows.
                  Results are unaffected.
Workaround:       Accept the wait, or set $env:FLEXTERN_YOSYS to a native Yosys
                  (not possible on this machine, see KI-002), or run under WSL.
```

```
KI-002
Status:           Workaround
Severity:         Minor
First observed:   2026-10-05
Relevant commit:  823902e
Description:      Windows Smart App Control (enabled on the founder's machine)
                  blocks the unsigned OSS CAD Suite yosys.exe with "An
                  Application Control policy has blocked this file". Icarus
                  Verilog's winget installer was allowed, presumably on
                  installer reputation.
Impact:           Native Windows Yosys cannot be used. The extracted suite at
                  C:\oss-cad-suite and the 600 MB archive in the session
                  scratchpad are dead weight.
Workaround:       `pip install yowasp-yosys` (runs inside signed Python, allowed).
                  run_all.ps1 probes candidates and uses the first that runs.
                  WSL Ubuntu with apt yosys is a second fallback.
```

```
KI-003
Status:           Open
Severity:         Minor (affects interpretation, not correctness)
First observed:   2026-10-05
Relevant commit:  823902e
Description:      Generic cell counts for small designs vary ~2x between Yosys
                  versions (Y-Core: 33 cells in 0.69, 60 in 0.52). Large
                  designs vary ~3%.
Impact:           Single-element ratios (7x-12x) are not stable facts. Only
                  the 16-element ratio (5.2x-6.5x) should be treated as
                  semi-stable, and even that is tool-dependent.
Workaround:       Always report the Yosys version alongside any count. Prefer
                  larger designs for ratio claims. A physical target would
                  give more meaningful numbers.
```

```
KI-004
Status:           Open (informational)
Severity:         Minor
First observed:   2026-10-06
Relevant commit:  dd27adb
Description:      Applying the official absmean ternarization (WeightQuant) to
                  the BF16 latent repo microsoft/bitnet-b1.58-2B-4T-bf16
                  reproduces the packed repo's 2-bit codes at 98.0-99.4%, not
                  100%, on three tensors checked. The stored weight_scale does
                  match mean|W_latent| to BF16 precision. The two repos appear
                  to be different training snapshots or were ternarized with
                  different rounding.
Impact:           Anyone deriving ternary statistics from the BF16 repo would
                  get numbers that differ by ~1-2 percentage points from the
                  deployed packed weights. EXP-0002 uses the packed repo only.
Workaround:       Treat the packed repo as authoritative. Do not mix the two.
Update 2026-10-06 (EXP-0002B): emulating BF16 arithmetic in the ternarization
                  (as a bf16-loaded model would compute) makes the TII Falcon-E
                  1B packed branch reproduce from its 'prequantized' branch at
                  exactly 100.000% on all tensors checked, but makes the
                  Microsoft agreement worse (94.8-98.8%). The Microsoft gap is
                  therefore a snapshot difference, not an arithmetic artifact.
                  Falcon-E has no such issue.
```

```
KI-005
Status:           Workaround
Severity:         Major (blocks any PyTorch work on the Windows host)
First observed:   2026-10-06
Relevant commit:  EXP-0004 commit
Description:      Windows Smart App Control blocks PyTorch's unsigned DLLs
                  (shm.dll, torch_cpu.dll, c10.dll, torch_cuda.dll are all
                  NotSigned) at `import torch` with WinError 4551, even for the
                  official cu128 wheel. Same policy that blocked Yosys (KI-002).
Impact:           EXP-0004 cannot run natively on Windows. pip install succeeds;
                  import fails.
Workaround:       Run under WSL2 Ubuntu, which sees the RTX 5060 through the
                  Windows driver (/usr/lib/wsl/lib/libcuda.so) and reuses the
                  Windows Hugging Face cache via /mnt/c. scripts/
                  run_model_precision.ps1 -Wsl does this. The Windows PyTorch
                  install is left in place but unusable.
```

```
KI-006
Status:           Workaround
Severity:         Minor
First observed:   2026-10-06
Relevant commit:  EXP-0004 commit
Description:      optimum-quanto 0.2.7 registers CUDA/C++ extensions at import
                  whenever CUDA is visible and JIT-builds them on first use with
                  torch.utils.cpp_extension.load, which needs nvcc/CUDA_HOME.
                  The WSL environment has the NVIDIA driver but no CUDA toolkit,
                  so transformers' QuantizedCache(backend="quanto") fails at the
                  first dequantize with "CUDA_HOME environment variable is not
                  set". Unregistering the extensions does not help because the
                  op implementations are already bound.
Impact:           The quanto KV backend cannot be used here.
Workaround:       EXP-0004 uses backend="hqq" (pure PyTorch) for the same
                  KIVI-style QuantizedCache. FLEXTERN_KV_BACKEND=quanto re-enables
                  quanto on a machine with a CUDA toolkit.
```

```
KI-007
Status:           Workaround
Severity:         Minor
First observed:   2026-10-06
Relevant commit:  EXP-0004 commit
Description:      Long single-process GPU runs under WSL2 (memory capped at
                  6 GB by .wslconfig on a 15 GB host) were killed by host
                  memory pressure twice during EXP-0004 (a 73-minute lm-eval
                  run and an 18-minute incremental-decode run), losing all
                  unsaved results.
Impact:           scripts/run_model_precision.ps1 -Wsl, which chains every
                  phase, may not finish in one invocation on this machine.
Workaround:       analysis/model_precision/run_experiment.py saves after each
                  (config, task) and skips completed cells on rerun; run one
                  phase (or one config) per invocation, close memory-heavy
                  host applications, and keep the WSL cap. manifest_*.json
                  records each chunk's arguments.
```

### Template

```
KI-XXX
Status:           Open | Investigating | Workaround | Closed
Severity:         Blocker | Major | Minor | Cosmetic
First observed:   YYYY-MM-DD
Relevant commit:
Description:
Impact:
Workaround:
```

## Closed

None.
