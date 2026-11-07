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
Relevant commit:  438da28
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
Relevant commit:  438da28
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
Relevant commit:  438da28
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

```
KI-008
Status:           Workaround
Severity:         Minor
First observed:   2026-10-07
Relevant commit:  eef813f
Description:      The canonical YoWASP Yosys 0.69 asserts
                  (`modules_.count(module->name) == 0`, rtlil_bufnorm.cc:681)
                  in `hierarchy -check -top <m> -chparam ...` when the design
                  contains recursively instantiated parameterized modules
                  (rtl/sum_tree.sv). It is also roughly 20x slower than the
                  native Yosys 0.52 in WSL on the EXP-0005 engines (ABC under
                  WebAssembly).
Impact:           Parameter points cannot be set with hierarchy -chparam in the
                  canonical tool; large sweeps are impractical in it.
Workaround:       synth/exp0005/*.ys use `chparam -set ... <module>`,
                  `hierarchy -check -auto-top`, `rename -top <point>`. The
                  EXP-0005 sweep ran on Yosys 0.52 (WSL, native) with the
                  256-term streaming point cross-checked on Yosys 0.69; the
                  tool version is a column in results/hardware_gate/synthesis.csv.
KI-009
Status:           Workaround
Severity:         Minor
First observed:   2026-10-07
Relevant commit:  899976e
Description:      The Debian/Ubuntu ngspice package (45.2) is built without the
                  BSIM-CMG FinFET model (MOS levels 1-6, 8-10, 14, 49, 54-58, 60,
                  68, 73 only), so the ASAP7 predictive PDK model cards
                  (`level = 72`) cannot be loaded directly.
Impact:           A Tier-2 SPICE check of FinFET SRAM cell leakage needs an
                  extra compile step.
Workaround:       Compile BSIM-CMG 110 from Verilog-A with OpenVAF 23.5.0 into
                  an OSDI module, load it with `pre_osdi`, and rewrite the model
                  cards to `.model <name> bsimcmg devtype=1|0` (drop
                  `version = 107`); see analysis/sram_retention/spice/README.md.
                  `coremod`/`capmod` are reported as unrecognized and ignored.
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

```
KI-011
Status:           Workaround
Severity:         Minor
First observed:   2026-11-04
Relevant commit:  EXP-0010 Stage A commit series
Description:      In the WSL venv (torch 2.11.0+cu128) `pip install torchvision`
                  from PyPI installs a build whose C++ ops do not register:
                  `RuntimeError: operator torchvision::nms does not exist` on
                  import. The wheel from the PyTorch cu128 index matches.
Impact:           torchvision (and any detection model) is unusable until fixed.
Workaround:       pip install --no-deps --force-reinstall torchvision==0.26.0
                  --index-url https://download.pytorch.org/whl/cu128
```

```
KI-012
Status:           Open
Severity:         Moderate (timing validity)
First observed:   2026-11-04
Relevant commit:  EXP-0010 Stage A commit series
Description:      EXP-0010 latency runs show (a) a power-state confound: all first-
                  set replays started on battery (Windows BatteryStatus 1) under
                  the Balanced plan because the first host-state parser mislabeled
                  the state; an AC repeat set was added and AC vs battery is not
                  cleanly separable; (b) rare unexplained stalls: 7 of 12 B1
                  replays contain a frame >100 ms (max 839 ms, 28 drops in one
                  run) while stage costs are ~20 ms and harness feeder slip stayed
                  small. Cause unknown (WSL2/Hyper-V scheduling, thermal, Windows
                  background activity, GC are untested hypotheses).
Impact:           p99/max latency and drop counts have wide repeat variance and
                  are not a property of the software alone; p50 is stable (19.7-
                  24.3 ms). Latency must be quoted as ranges.
Workaround:       Repeat runs (>= 3) and report ranges; use a dedicated plugged-in
                  Linux host for any comparison where tails matter.
```

```
KI-013
Status:           Open
Severity:         Moderate (baseline validity)
First observed:   2026-11-04
Relevant commit:  EXP-0010 Stage A commit series
Description:      Pretrained COCO SSDLite320-MobileNetV3 scores poorly on the
                  selected synthetic warehouse clips at the frozen IoU 0.5: holdout
                  AP50 0.130, F1 0.218 at the calibrated threshold; medium
                  (32-96 px) persons almost never matched. A post-hoc diagnostic
                  (EXP-0010 R2b) shows mixed causes: holdout AP rises to 0.34 at
                  IoU 0.1 and 68% of its false positives are near-misses on real
                  people (visible-extent labels vs whole-person predictions,
                  squash, or localization: not separated); on the calibration clip
                  88% of false positives are on forklifts/pallet trucks. The
                  calibrated threshold does not transfer across layouts
                  (precision 0.48/0.59/0.23).
Impact:           Any quality-retention claim for a temporal-execution method
                  measured against this detector is confounded by its misses.
                  Stage B should not rely on this control unchanged.
Workaround:       None yet; see the handoff's recommended control-adequacy gate.
```

```
KI-014
Status:           Open
Severity:         Minor
First observed:   2026-11-04
Relevant commit:  EXP-0010 Stage A commit series
Description:      MTMC_Tracking_2026 documentation/data mismatches: ground_truth.json
                  keys contain spaces ("object type", "2d bounding box visible")
                  while the README shows underscores; nominal validation scenes
                  020/021 share a layout (identical camera geometry and map.png)
                  with training scenes 000-007; the two real-world test scenes
                  (026/027) have no ground truth.
Impact:           Naive scene-name splits leak layout; the real-world clips cannot
                  score quality.
Workaround:       gt.py accepts both key styles; Stage A splits by layout group
                  (results/edge_inference/dataset_audit.json).
```

```
KI-015
Status:           Mitigated in EXP-0010C (metric v3 treats humanoids by policy, handles truncation and uses symmetric tolerance); the EXP-0010B record keeps the flawed metric by design
Severity:         Major (affects validity of any quality claim on the new fresh holdouts)
First observed:   2026-11-04
Relevant commit:  EXP-0010B commit
Description:      The EXP-0010B zone-occupancy metric (and the layout rule for the fresh
                  holdout) did not carry Stage A's policy of treating humanoid robots
                  (FourierGR1T2, AgilityDigit) as ignore regions. The development layouts
                  contain no humanoids; both fresh holdouts (L08, L02) contain many.
                  Review also found one-sided boundary tolerance (detections get a 32 px
                  margin, NEG uses the same margin) and ignored persons (small, foot outside
                  the frame) leaving zones NEG.
Impact:           Humanoid zones are about 40% of YOLOX-Tiny's cell-level FPR on the holdouts
                  (80% of its confident false-alarm detections); people cut off at the frame
                  bottom account for nearly all person-matched false alarms. The
                  preregistered verdict was scored as written and is reported unchanged,
                  with labelled post-hoc rescoring. The holdout is spent for this protocol.
Workaround:       none yet. A corrected protocol (ADR-0007 point 3) plus a NEW fresh holdout
                  whose layout rule screens class mix is needed before any pass claim.
```

```
KI-016
Status:           Open
Severity:         Minor
First observed:   2026-11-04
Relevant commit:  EXP-0010B commit
Description:      Ground-truth foot points come from projecting the dataset's 3D boxes with
                  per-scene calibration. The projection reproduces visible boxes on L10/L11 and
                  both fresh layouts (rule in protocol_v2) but fails on L09 (visible/expected
                  height median 0.69), and the 3D cuboids are loose (width ~1.5x visible), so a
                  projected 3D box is not a whole-person label. Foot error tails reach tens of px.
Impact:           Zone metrics exist for 4 clips only; some zone errors are label noise; the
                  INFERRED whole-person-box AP is unreliable on L10.
Workaround:       scenes failing the consistency rule are excluded from zone metrics.
```

```
KI-017
Status:           Open
Severity:         Minor
First observed:   2026-11-04
Relevant commit:  EXP-0010B commit
Description:      Windows memory protection killed two long background jobs (development
                  detection loop; L02 holdout run) while WSL held ~1.5 GB used / 4 GB cache and
                  the host had ~7.8 GB free. A third partial download (YOLOX-S ONNX) stalled at
                  about 11 of 35.9 MB. Related to KI-007.
Impact:           Jobs must be run one process at a time and restarted; YOLOX-S was never
                  evaluated.
Workaround:       per-process foreground-sized runs; results are saved per run.
```

```
KI-018
Status:           Open
Severity:         Minor
First observed:   2026-11-04
Relevant commit:  EXP-0010C lock commit 97cbd9b
Description:      The EXP-0010C operating-threshold rule (macro balanced accuracy over development
                  clips) chose the lowest grid value (tau 0.02): balanced accuracy is diluted by
                  empty cells, so it rewards recall at the cost of precision. At tau 0.02 low-score
                  clutter dominates the false alarms on several clips (L10 88%, L02 75%).
Impact:           The locked fresh-holdout result used tau 0.02 as locked; a post-hoc sweep shows no
                  threshold meets the gating pair on all six clips, so the verdict does not depend on it.
Workaround:       ADR-0008 point 6: choose thresholds with a precision-aware criterion in any new protocol.
```

```
KI-019
Status:           Open (mitigated in new code)
Severity:         Minor
First observed:   2026-11-04
Relevant commit:  EXP-0010C
Description:      Git run from WSL against the CRLF working tree reports every file as modified
                  unless core.autocrlf=true is set; EXP-0010B manifests also recorded the commit as
                  clean because common.git_info only checks analysis/edge_inference.
Impact:           EXP-0010B manifests (commit b056ace, dirty=false) do not describe the code that ran;
                  Stage A's git_info is unchanged.
Workaround:       analysis/control_semantics/provenance.py records whole-repository state with
                  -c core.autocrlf=true; EXP-0010B code state is documented in its record.
```

```
KI-020
Status:           Open (worked around by pinning an older runtime and a tokenizer override)
Severity:         Major (for any BitNet 2B4T result obtained with the affected configurations)
First observed:   2026-11-05
Relevant commit:  EXP-0011
Description:      (1) The official BitNet 2B4T GGUF has no tokenizer.ggml.pre entry; llama.cpp then uses its
                  default pre-tokenizer instead of Llama-3 BPE, raising wikitext perplexity from 16.80 to
                  19.64 under bitnet.cpp 765741d (HF reference 16.88). (2) bitnet.cpp main (0b341e5, Jul 2026),
                  built by hand with CMake (setup_env.py was not run), gives perplexity 118.6 on the same GGUF
                  and 96.8-96.9 on a reconverted GGUF or with the pre-tokenizer override; independent of
                  AVX-512, batch size and BOS. Its submodule tracks release-bitnet-embedding-0.6b-270m and the
                  Nov 2025 commit changed the I2_S kernels. Cause undetermined; the claim is limited to the
                  build made here.
Impact:           Two earlier BitNet quality runs (macro 39.8 and 55.5; valid macro 56.7) are retained as
                  negative results under results/ternary_vs_int4/quality/failed_*. Published bitnet.cpp
                  speed claims are not tested against a numerically valid main build.
Workaround:       Use bitnet.cpp 765741d with --override-kv tokenizer.ggml.pre=str:llama-bpe; gate any
                  ternary runtime on perplexity parity with the HF checkpoint
                  (analysis/ternary_vs_int4/diag_ppl.py). A time-boxed run of the documented setup_env.py flow
                  on main was not done.
```

```
KI-021
Status:           Open (mitigated)
Severity:         Minor
First observed:   2026-11-05
Relevant commit:  EXP-0011
Description:      The benchmark laptop is not a controlled timing platform. Observed: (1) battery status 1
                  (discharging) twice during the resumed run (an AC-adapter drop at 98% and at 94% charge); one
                  main-grid cell (BitNet P3840/G256) was measured during a drop (90.4 s vs 69.8 s on repeat);
                  (2) one 537 s first-token sample on AC in the earlier sweep that did NOT reproduce in nine
                  repeats (24.3-26.2 s), cause unattributed; (3) a 32,432 s wall time for a quality task that
                  normally takes about 10 minutes while on battery (sleep/hibernate plausible, unproven).
                  WSL2 cannot read CPU frequency or temperature.
Impact:           Timing is trusted only with AC present; flagged cells must be repeated; thermal state is an
                  unmeasured confounder.
Workaround:       Harness refuses to time off AC, waits up to 30 minutes for AC, restarts automatically, records
                  power before and after each invocation (amendments 6-7); a process-scoped keep-awake was used;
                  flagged and interrupted results are retained.
```

```
KI-022
Status:           Open
Severity:         Moderate
First observed:   2026-11-07
Relevant commit:  EXP-0012 branch exp-0012-predictive-state
Description:      Closed-loop behaviour-cloning success on OccPick2D is highly seed-dependent at small training budgets: at 3000 steps x batch
                  24 the same architecture and data gave NOM success 0.63 / 0.22 / 0.32 (R, seeds 0/1/2) and 0.72 / 0.16 / 0.46 (T16). At
                  6000 x 48 the spread shrank but was not eliminated (EXP-0012 record, amendment 4). Only three training seeds per arm.
Impact:           Differences below ~0.1 absolute macro success between arms should not be read as ranking; episode-bootstrap CIs
                  (+-0.03 to 0.05) understate seed variance.
Workaround:       Report per-seed values next to means; more seeds or longer training would be needed to resolve small gaps.
```

```
KI-023
Status:           Open
Severity:         Minor
First observed:   2026-11-07
Relevant commit:  EXP-0012 branch exp-0012-predictive-state
Description:      Hashes recorded in protocol_v1.json / protocol_v1_amended.json are SHA-256 of the files as they sat in the working tree
                  when `lock.py` ran under WSL (LF line endings). git on Windows may check the same files out with CRLF, so a hash recomputed
                  from a Windows checkout can differ although content is identical.
Impact:           Code-hash verification must be done on LF content (`git show <commit>:<path> | sha256sum`).
Workaround:       Compare against `git show` output, not the working copy.
```

```
KI-024
Status:           Open
Severity:         Moderate
First observed:   2026-11-07
Relevant commit:  EXP-0012 branch exp-0012-predictive-state
Description:      GPU training of the recurrent arms is launch-bound (a Python loop of ~100-160 GRU steps with fake-quant ops). With 8 GB of GPU
                  memory, three concurrent training processes ran ~2.3x slower in aggregate than two (7.8 GB used, 90% utilization), and
                  background launches started from the Windows side (nohup of a `wsl` process) die when the session ends; launching from inside
                  WSL (`nohup setsid ... &` in a script run by `wsl -- bash -s`) survives.
Impact:           Wall-clock cost of a 3-seed, 9-arm study was ~7 hours on this machine; mistakes in orchestration cost hours.
Workaround:       Two concurrent workers; launch detached jobs from inside WSL; consider CUDA graphs or a fused cell if more arms are added.
```

```
KI-025
Status:           Open
Severity:         Moderate
First observed:   2026-11-07
Relevant commit:  EXP-0013 Phase 1 (branch exp-0013-predictive-planning)
Description:      EXP-0012 reported "encoder runs" were LOGICAL counts: evaluate.py executed the encoder on every row at every step and
                  masked the output (G_s0, 32 DEV episodes: 2560 images executed vs 331 requested at fixed:0.25). Success numbers remain
                  valid (outputs were masked identically; verified by per-episode equality of the new gated path), but every "encoder cut",
                  cost-saving or latency statement in EXP-0012 refers to requested, not executed, work. The batch-1 latency path (latency.py)
                  did skip, but with different trigger code (no fixed:K). Fixed in Phase 1 (infer.RecurrentRunner); the verdict is not changed.
                  Open part: the rebuilt batch-1 path is ~0.1-0.3 ms/step slower in skip modes from counter/gating bookkeeping (p50 ~0.4-0.6 ms
                  baseline) and the batched GPU evaluation is not faster in wall time; training still encodes all frames.
Impact:           No latency or energy speed-up claim is supported by either path yet.
Workaround:       Quote executed counts from infer.Counters; add a counters-off timing mode before any latency claim.
```

```
KI-026
Status:           Open
Severity:         Minor
First observed:   2026-11-07
Relevant commit:  EXP-0013 Phase 1
Description:      OccPick2D-LA v1: a one-step lookahead with a map-derived value function solves the task (oracle H=1), so it does not prove
                  multi-step rollouts are necessary; random-shooting oracle results degrade with horizon because of sampler quality (N=64
                  random-walk plans). The reactive baselines were hand-built (tangent direction tailored to the divider) and not tuned beyond
                  a single memory variant. Step budget (65) was derived from oracle path lengths only, but baseline outcomes on TRAIN layouts
                  were seen while debugging.
Impact:           Gate C is "PASS (narrow)"; do not cite LA results as evidence that planning is necessary.
Workaround:       See EXP-0013-phase1.md section 3 and the Phase 2 proposal (value head; CEM; harder variants).
```

## Closed

```
KI-010
Status:           Closed (2026-10-08, EXP-0009A)
Severity:         Major (affected EXP-0009's wake-energy and idle-policy conclusions)
First observed:   2026-10-08
Relevant commit:  899976e (defect); fixed in 8ab0a24 (EXP-0009A)
Description:      EXP-0009 assumption B7 stated E = 4 nF/MB x 0.75 V x 0.40 V
                  = 1.2 nJ/MB but the model applied 1.2 uJ/MB (and derived its
                  PLANNING 0.15 / LOW 0.02 uJ/MB from it): a 1000x unit error
                  that would have required ~2 A of in-rush per 256 KB macro.
                  Assumption B14 priced shutdown-and-reload with no standby
                  cost for the weights' persistent copy, at LPDDR read energy
                  but flash-class bandwidth. P2+S sessions charged one extra
                  whole-model wake per session.
Impact:           Wake energy per token 0.53 mJ (38% of read energy) -> 1.3 uJ
                  (< 0.1%); P2 ratio at 25 tok/s 4.90x -> 6.03x (25 C),
                  1.34x -> 1.41x (85 C); shutdown-vs-retention break-even
                  1.2 s / 0.08 s -> 2.8 s / 0.18 s (DRAM-backed) or 29 s /
                  1.8 s (flash-backed); hot long-idle "shutdown wins" became
                  backing-store dependent. Gate verdicts: 5B energy half
                  INCONCLUSIVE -> bounded PASS (unmeasured); 5A/5C/5D unchanged
                  in kind. Leakage, P0, latency, EXP-0008 reconciliation
                  unaffected (byte-identical artifacts).
Resolution:       wake.py rebuilt from rail capacitances with a dimensional
                  test and in-rush check; policy.py prices PX-DRAM (B16) and
                  PX-FLASH (B17) with backing standby, reload energy and
                  latency; tests/test_sram_retention_correction.py (15 tests);
                  original artifacts kept in
                  results/sram_retention/superseded_899976e/; record
                  docs/experiments/EXP-0009A-energy-accounting-correction.md.
```
