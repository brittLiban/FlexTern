<#
.SYNOPSIS
    FlexTern EXP-0004: tests, feasibility audit, and the low-precision head / KV evaluation phases.

.DESCRIPTION
    Requires CUDA PyTorch (cu128 for Blackwell GPUs), transformers >= 5, optimum-quanto, lm_eval, datasets.
    Sets TORCHDYNAMO_DISABLE=1 (no C++ compiler for torch.compile on this machine).
    Phases run in order; each writes results/model_precision/*.json. Stops at the first failure.

.PARAMETER Phases
    Comma list for run_experiment.py (default: lmeval,incremental,generation,memory,latency).
.PARAMETER Limit
    lm_eval per-task sample limit (PILOT). Omit for full tasks.
.PARAMETER Windows / Ctx
    Incremental-decode evaluation windows and context length.
.PARAMETER SkipTests / SkipAudit
#>
[CmdletBinding()]
param(
    [string]$Phases = 'lmeval,incremental,generation,memory,latency',
    [int]$Limit = 0, [int]$Windows = 8, [int]$Ctx = 2048,
    [switch]$SkipTests, [switch]$SkipAudit,
    [switch]$Wsl, [string]$WslDistro = 'Ubuntu', [string]$WslVenv = '~/ftvenv'
)

$ErrorActionPreference = 'Continue'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:TORCHDYNAMO_DISABLE = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

if ($Wsl) {
    # Windows Smart App Control blocks PyTorch's unsigned DLLs (KI-005). Run the same phases inside WSL2,
    # which sees the NVIDIA GPU through the Windows driver, reusing the Windows Hugging Face cache.
    $repoWsl = '/mnt/' + $RepoRoot.Substring(0, 1).ToLower() + ($RepoRoot.Substring(2) -replace '\\', '/')
    $hf = '/mnt/c/Users/' + $env:USERNAME + '/.cache/huggingface'
    $cmd = "cd '$repoWsl' && source $WslVenv/bin/activate && export TORCHDYNAMO_DISABLE=1 HF_HOME='$hf' HF_HUB_DISABLE_SYMLINKS_WARNING=1 && "
    if (-not $SkipTests) { $cmd += "python -m unittest tests.test_model_precision -v && " }
    if (-not $SkipAudit) { $cmd += "python analysis/model_precision/audit.py && " }
    $cmd += "python analysis/model_precision/run_experiment.py --phases $Phases --windows $Windows --ctx $Ctx"
    if ($Limit -gt 0) { $cmd += " --limit $Limit" }
    if ($Phases -match 'memory') { $cmd += " && python analysis/model_precision/roofline_reconnect.py" }
    $cmd += " && python analysis/model_precision/report.py"
    Write-Host "wsl -d $WslDistro -- bash -lc `"$cmd`"" -ForegroundColor DarkGray
    & wsl -d $WslDistro -- bash -lc $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host 'EXP-0004 (WSL) FAILED.' -ForegroundColor Red; exit 1 }
    Write-Host "`nEXP-0004 phases complete (results\model_precision\)" -ForegroundColor Green
    exit 0
}

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
    Write-Host "`n== Model-precision unit tests ==" -ForegroundColor Cyan
    & $Python -m unittest tests.test_model_precision -v
    if ($LASTEXITCODE -ne 0) { Write-Host 'Tests FAILED.' -ForegroundColor Red; exit 1 }
}
if (-not $SkipAudit) {
    Write-Host "`n== Feasibility audit ==" -ForegroundColor Cyan
    & $Python analysis\model_precision\audit.py
    if ($LASTEXITCODE -ne 0) { Write-Host 'Audit FAILED.' -ForegroundColor Red; exit 1 }
}
Write-Host "`n== Experiment phases: $Phases ==" -ForegroundColor Cyan
$args_ = @('analysis\model_precision\run_experiment.py', '--phases', $Phases, '--windows', $Windows, '--ctx', $Ctx)
if ($Limit -gt 0) { $args_ += @('--limit', $Limit) }
& $Python @args_
if ($LASTEXITCODE -ne 0) { Write-Host 'Experiment FAILED.' -ForegroundColor Red; exit 1 }
if ($Phases -match 'memory') {
    Write-Host "`n== Roofline reconnection ==" -ForegroundColor Cyan
    & $Python analysis\model_precision\roofline_reconnect.py
    if ($LASTEXITCODE -ne 0) { Write-Host 'Roofline reconnection FAILED.' -ForegroundColor Red; exit 1 }
}
Write-Host "`nEXP-0004 phases complete (results\model_precision\)" -ForegroundColor Green
exit 0
