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
