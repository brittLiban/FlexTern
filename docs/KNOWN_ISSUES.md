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
