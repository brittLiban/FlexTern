<#
.SYNOPSIS
    FlexTern EXP-0002: run the analysis tests, then the ternary workload profile.

.DESCRIPTION
    1. Locates a real Python 3 (skips the Microsoft Store alias).
    2. Runs tests/test_analysis.py (synthetic tensors; no download).
    3. Runs analysis/ternary_model_profile.py, which downloads the pinned
       microsoft/bitnet-b1.58-2B-4T checkpoint (1.18 GB) into the Hugging
       Face cache on first use and writes results/workload_profile/.
    Stops at the first failure.

.PARAMETER SkipTests
    Skip the unit tests.
.PARAMETER SkipHash
    Skip the SHA-256 of the checkpoint (saves ~10 s).
#>
[CmdletBinding()]
param([switch]$SkipTests, [switch]$SkipHash, [switch]$SkipCrosscheck)

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
    Write-Host "`n== Analysis unit tests ==" -ForegroundColor Cyan
    & $Python -m unittest tests.test_analysis -v
    if ($LASTEXITCODE -ne 0) { Write-Host 'Analysis tests FAILED.' -ForegroundColor Red; exit 1 }
    Write-Host 'Analysis tests: PASS' -ForegroundColor Green
}

Write-Host "`n== Workload profile ==" -ForegroundColor Cyan
$args_ = @('analysis\ternary_model_profile.py')
if ($SkipHash) { $args_ += '--skip-hash' }
& $Python @args_
if ($LASTEXITCODE -ne 0) { Write-Host 'Workload profile FAILED.' -ForegroundColor Red; exit 1 }
Write-Host "`nWorkload profile: PASS (results\workload_profile\)" -ForegroundColor Green

if (-not $SkipCrosscheck) {
    Write-Host "`n== BF16 ternarization cross-check (HTTP range fetch of 3 latent tensors, ~63 MB) ==" -ForegroundColor Cyan
    & $Python analysis\ternarization_crosscheck.py
    if ($LASTEXITCODE -ne 0) { Write-Host 'Cross-check FAILED.' -ForegroundColor Red; exit 1 }
    Write-Host 'Cross-check: PASS' -ForegroundColor Green
}
exit 0
