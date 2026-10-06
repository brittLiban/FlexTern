<#
.SYNOPSIS
    FlexTern EXP-0003: run the roofline validation tests, then the analytical bottleneck model.

.DESCRIPTION
    1. Locates a real Python 3 (skips the Microsoft Store alias).
    2. Runs tests/test_roofline.py (reads the committed canonical profiles; no download).
    3. Runs analysis/flextern_roofline.py and writes results/roofline/.
    Stops at the first failure. No model download is needed.
#>
[CmdletBinding()]
param([switch]$SkipTests)

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
    Write-Host "`n== Roofline tests ==" -ForegroundColor Cyan
    & $Python -m unittest tests.test_roofline -v
    if ($LASTEXITCODE -ne 0) { Write-Host 'Roofline tests FAILED.' -ForegroundColor Red; exit 1 }
    Write-Host 'Roofline tests: PASS' -ForegroundColor Green
}

Write-Host "`n== Roofline model ==" -ForegroundColor Cyan
& $Python analysis\flextern_roofline.py
if ($LASTEXITCODE -ne 0) { Write-Host 'Roofline model FAILED.' -ForegroundColor Red; exit 1 }
Write-Host "`nRoofline: PASS (results\roofline\)" -ForegroundColor Green
exit 0
