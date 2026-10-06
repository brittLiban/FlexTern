<#
.SYNOPSIS
    FlexTern EXP-0005: packed ternary streaming engine vs banked near-memory dataflow.

.DESCRIPTION
    1. Locates a real Python 3 (skips the Microsoft Store alias).
    2. Runs tests/test_hardware_gate.py: packing interpretation, Python reference, and the
       Icarus RTL testbenches on generated vectors (checkpoint slices used only if the pinned
       files are already in the local Hugging Face cache; never downloaded).
    3. Generic Yosys synthesis sweep (analysis/hardware_gate/synth_points.py).  -SynthTool wsl
       uses the native Yosys in WSL Ubuntu (faster); the default is the YoWASP build
       (canonical in this repository).  -SkipSynth reuses the committed logs.
    4. Workload mapping, traffic model, capacity, EXP-0003/0004 reconnection, summary.
    Stops at the first failure.  No FPGA tool is involved; nothing physical is measured.

.PARAMETER SynthTool   yowasp (default) | wsl
.PARAMETER SynthPoints comma-separated point names, or 'all' (default)
#>
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipSynth,
    [switch]$FullRtl,
    [ValidateSet('yowasp', 'wsl')][string]$SynthTool = 'yowasp',
    [string]$SynthPoints = 'all'
)

$ErrorActionPreference = 'Continue'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = $null
if ($env:FLEXTERN_PYTHON -and (Test-Path $env:FLEXTERN_PYTHON)) { $Python = $env:FLEXTERN_PYTHON }
if (-not $Python) {
    foreach ($c in (Get-Command python -All -ErrorAction SilentlyContinue)) {
        if ($c.Source -like '*\WindowsApps\*') { continue }
        $Python = $c.Source; break
    }
}
if (-not $Python) {
    $fb = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($fb) { $Python = $fb.FullName }
}
if (-not $Python) { Write-Host 'Python 3 not found.' -ForegroundColor Red; exit 1 }
Write-Host "python: $Python"

if (-not $SkipTests) {
    Write-Host "`n== EXP-0005 tests (reference + RTL simulation) ==" -ForegroundColor Cyan
    if ($FullRtl) { $env:FLEXTERN_FULL_RTL = '1' }
    & $Python -m unittest tests.test_hardware_gate -v
    if ($LASTEXITCODE -ne 0) { Write-Host 'EXP-0005 tests FAILED.' -ForegroundColor Red; exit 1 }
    Write-Host 'EXP-0005 tests: PASS' -ForegroundColor Green
}

if (-not $SkipSynth) {
    Write-Host "`n== Generic synthesis sweep ($SynthTool) ==" -ForegroundColor Cyan
    & $Python analysis\hardware_gate\synth_points.py --tool $SynthTool --points $SynthPoints
    if ($LASTEXITCODE -ne 0) { Write-Host 'Synthesis sweep FAILED.' -ForegroundColor Red; exit 1 }
} else {
    & $Python analysis\hardware_gate\synth_points.py --parse-only
}

Write-Host "`n== Workload mapping, traffic model, reconnection, summary ==" -ForegroundColor Cyan
& $Python analysis\hardware_gate\run_gate.py
if ($LASTEXITCODE -ne 0) { Write-Host 'Analysis FAILED.' -ForegroundColor Red; exit 1 }
Write-Host "`nEXP-0005 flow complete. See results\hardware_gate\summary.md" -ForegroundColor Green
exit 0
