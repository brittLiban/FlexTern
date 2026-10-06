<#
.SYNOPSIS
    FlexTern EXP-0002 / EXP-0002B: run the analysis tests, then the ternary workload profiles.

.DESCRIPTION
    1. Locates a real Python 3 (skips the Microsoft Store alias).
    2. Runs tests/test_analysis.py and tests/test_analysis_cross_model.py (synthetic tensors; no download).
    3. Runs analysis/ternary_model_profile.py for each model in -Models (default: all three),
       downloading each pinned checkpoint into the Hugging Face cache on first use
       (bitnet-2b4t 1.18 GB, falcon-e-1b 0.67 GB, falcon-e-3b 1.00 GB).
    4. Runs the BF16 latent ternarization cross-check for models that declare one.
    5. Runs analysis/cross_model_compare.py when all three profiles exist.
    Stops at the first failure.

.PARAMETER Models
    Model slugs to profile. Default: bitnet-2b4t, falcon-e-1b, falcon-e-3b.
.PARAMETER SkipTests
    Skip the unit tests.
.PARAMETER SkipHash
    Skip the SHA-256 of each checkpoint.
.PARAMETER SkipCrosscheck
    Skip the HTTP-range latent cross-check.
#>
[CmdletBinding()]
param(
    [string[]]$Models = @('bitnet-2b4t', 'falcon-e-1b', 'falcon-e-3b'),
    [switch]$SkipTests, [switch]$SkipHash, [switch]$SkipCrosscheck
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
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

if (-not $SkipTests) {
    Write-Host "`n== Analysis unit tests ==" -ForegroundColor Cyan
    & $Python -m unittest tests.test_analysis tests.test_analysis_cross_model -v
    if ($LASTEXITCODE -ne 0) { Write-Host 'Analysis tests FAILED.' -ForegroundColor Red; exit 1 }
    Write-Host 'Analysis tests: PASS' -ForegroundColor Green
}

foreach ($m in $Models) {
    Write-Host "`n== Workload profile: $m ==" -ForegroundColor Cyan
    $args_ = @('analysis\ternary_model_profile.py', '--model', $m)
    if ($SkipHash) { $args_ += '--skip-hash' }
    & $Python @args_
    if ($LASTEXITCODE -ne 0) { Write-Host "Workload profile FAILED for $m." -ForegroundColor Red; exit 1 }
    Write-Host "Workload profile ${m}: PASS" -ForegroundColor Green

    if (-not $SkipCrosscheck -and ($m -eq 'bitnet-2b4t' -or $m -eq 'falcon-e-1b')) {
        Write-Host "`n== Latent ternarization cross-check: $m (HTTP range fetch of 3 tensors) ==" -ForegroundColor Cyan
        & $Python analysis\ternarization_crosscheck.py --model $m
        if ($LASTEXITCODE -ne 0) { Write-Host "Cross-check FAILED for $m." -ForegroundColor Red; exit 1 }
        Write-Host "Cross-check ${m}: PASS" -ForegroundColor Green
    }
}

$all = @('bitnet-2b4t', 'falcon-e-1b', 'falcon-e-3b') | ForEach-Object { Test-Path (Join-Path $RepoRoot ("results\workload_profile" + $(if ($_ -eq 'bitnet-2b4t') { '' } else { "\$_" }) + "\model_summary.json")) }
if ($all -notcontains $false) {
    Write-Host "`n== Cross-model comparison ==" -ForegroundColor Cyan
    & $Python analysis\cross_model_compare.py
    if ($LASTEXITCODE -ne 0) { Write-Host 'Cross-model comparison FAILED.' -ForegroundColor Red; exit 1 }
    Write-Host 'Cross-model comparison: PASS (results\workload_profile\cross_model\)' -ForegroundColor Green
} else {
    Write-Host "`nCross-model comparison skipped: not all three profiles present." -ForegroundColor Yellow
}
exit 0
