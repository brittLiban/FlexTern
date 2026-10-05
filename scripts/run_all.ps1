<#
.SYNOPSIS
    FlexTern Phase 1 validation runner (Windows PowerShell 5.1 compatible).

.DESCRIPTION
    Runs, in order, stopping at the first failure:
      1. Python golden reference unit tests
      2. Cross-check vector generation from the Python reference
      3. Y-Core exhaustive RTL verification (Icarus Verilog)
      4. Y-Dot16 RTL verification (deterministic + 10,000 random + Python cross-check)
      5. INT8 baseline RTL verification
      6. Yosys generic synthesis of y_core, y_dot16, int8_mul_baseline, int8_dot16_baseline
      7. Writes results/synth_environment.txt with tool versions and commit

    A step is reported PASS only if the tool's exit code is 0 AND, for
    simulations, the testbench printed its unique success token. vvp returns
    0 after $finish regardless of what was printed, so the token check is
    what actually proves the testbench passed.

    Simulation logs go to build/ (ignored by git). Synthesis logs go to
    results/ (committed).

.PARAMETER SkipSynth
    Run only the Python and simulation steps (useful if Yosys is not installed).

.NOTES
    Tool discovery:
      python   : first 'python' on PATH that is a real interpreter (not the
                 Microsoft Store alias). Override with $env:FLEXTERN_PYTHON.
      iverilog : PATH, then C:\iverilog\bin. Override with $env:FLEXTERN_IVERILOG_DIR.
      yosys    : $env:FLEXTERN_YOSYS, then 'yosys' on PATH, then
                 C:\oss-cad-suite\bin\yosys.exe, then the YoWASP build
                 (yowasp-yosys.exe next to python, from `pip install yowasp-yosys`).
                 The first candidate that actually runs `-V` wins.
#>
[CmdletBinding()]
param(
    [switch]$SkipSynth
)

$ErrorActionPreference = 'Continue'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot 'build') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot 'results') | Out-Null

$Summary = New-Object System.Collections.ArrayList
$script:Failed = $false

function Write-Header($text) {
    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor DarkGray
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor DarkGray
}

# Run a native executable, capture stdout+stderr to a log file, return
# an object with ExitCode and Output (string). No 2>&1 (breaks in PS 5.1).
function Invoke-Tool {
    param(
        [string]$Exe,
        [string[]]$Arguments,
        [string]$LogPath
    )
    $errPath = "$LogPath.stderr"
    $p = Start-Process -FilePath $Exe -ArgumentList $Arguments -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $LogPath -RedirectStandardError $errPath `
            -WorkingDirectory $RepoRoot
    $out = ''
    if (Test-Path $LogPath) { $out = Get-Content $LogPath -Raw }
    $err = ''
    if (Test-Path $errPath) {
        $err = Get-Content $errPath -Raw
        if ($err) { Add-Content -Path $LogPath -Value "`n--- stderr ---`n$err" }
        Remove-Item $errPath -Force
    }
    return [pscustomobject]@{ ExitCode = $p.ExitCode; Output = "$out`n$err" }
}

# Record a step result. On failure, print the summary so far and exit 1.
function Record-Step {
    param([string]$Name, [bool]$Pass, [string]$Detail = '')
    $status = if ($Pass) { 'PASS' } else { 'FAIL' }
    [void]$Summary.Add([pscustomobject]@{ Step = $Name; Status = $status; Detail = $Detail })
    $color = if ($Pass) { 'Green' } else { 'Red' }
    Write-Host ("  -> {0}  {1}" -f $status, $Detail) -ForegroundColor $color
    if (-not $Pass) {
        $script:Failed = $true
        Write-Summary
        exit 1
    }
}

function Write-Summary {
    Write-Host ''
    Write-Host 'FlexTern Phase 1 Validation' -ForegroundColor White
    Write-Host '---------------------------'
    foreach ($s in $Summary) {
        $color = if ($s.Status -eq 'PASS') { 'Green' } else { 'Red' }
        Write-Host ("{0,-42} {1}" -f ($s.Step + ':'), $s.Status) -ForegroundColor $color
    }
    Write-Host ''
    if ($script:Failed) {
        Write-Host 'Phase 1 validation FAILED. See build/ and results/ logs.' -ForegroundColor Red
    } elseif ($SkipSynth) {
        Write-Host 'Phase 1 simulation steps passed. Synthesis was SKIPPED, so Phase 1 is not complete.' -ForegroundColor Yellow
    } else {
        Write-Host 'Phase 1 completed successfully.' -ForegroundColor Green
    }
}

# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------
Write-Header 'Tool discovery'

# Python: reject the Microsoft Store alias, which exits non-zero with a hint.
$Python = $null
if ($env:FLEXTERN_PYTHON -and (Test-Path $env:FLEXTERN_PYTHON)) { $Python = $env:FLEXTERN_PYTHON }
if (-not $Python) {
    foreach ($cand in (Get-Command python -All -ErrorAction SilentlyContinue)) {
        if ($cand.Source -like '*\WindowsApps\*') { continue }
        $Python = $cand.Source; break
    }
}
if (-not $Python) {
    $fallback = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($fallback) { $Python = $fallback.FullName }
}
if (-not $Python) {
    Write-Host 'Python 3 not found. Install with:  winget install --id Python.Python.3.12 --exact' -ForegroundColor Red
    Record-Step 'Python available' $false 'not found'
}
$pyVer = Invoke-Tool $Python @('--version') (Join-Path $RepoRoot 'build\python_version.txt')
if ($pyVer.ExitCode -ne 0 -or $pyVer.Output -notmatch 'Python 3\.') {
    Record-Step 'Python available' $false "unusable: $Python"
}
Write-Host "  python   : $Python  ($($pyVer.Output.Trim()))"

# Icarus
$IverilogDir = $null
if ($env:FLEXTERN_IVERILOG_DIR -and (Test-Path (Join-Path $env:FLEXTERN_IVERILOG_DIR 'iverilog.exe'))) { $IverilogDir = $env:FLEXTERN_IVERILOG_DIR }
if (-not $IverilogDir) {
    $c = Get-Command iverilog -ErrorAction SilentlyContinue
    if ($c) { $IverilogDir = Split-Path -Parent $c.Source }
}
if (-not $IverilogDir -and (Test-Path 'C:\iverilog\bin\iverilog.exe')) { $IverilogDir = 'C:\iverilog\bin' }
if (-not $IverilogDir) {
    Write-Host 'Icarus Verilog not found. Install with:  winget install --id Icarus.Verilog --exact' -ForegroundColor Red
    Record-Step 'Icarus Verilog available' $false 'not found'
}
$Iverilog = Join-Path $IverilogDir 'iverilog.exe'
$Vvp      = Join-Path $IverilogDir 'vvp.exe'
$ivVer = Invoke-Tool $Iverilog @('-V') (Join-Path $RepoRoot 'build\iverilog_version.txt')
Write-Host "  iverilog : $Iverilog  ($(($ivVer.Output -split "`n")[0].Trim()))"

# Yosys. Candidates are tried in order until one actually runs `-V`
# successfully. A candidate can exist on disk yet be unrunnable, for example
# when Windows Smart App Control blocks an unsigned binary, so existence is
# not enough.
$Yosys = $null
$ysVer = $null
if (-not $SkipSynth) {
    $candidates = New-Object System.Collections.ArrayList
    if ($env:FLEXTERN_YOSYS) { [void]$candidates.Add($env:FLEXTERN_YOSYS) }
    $c = Get-Command yosys -ErrorAction SilentlyContinue
    if ($c) { [void]$candidates.Add($c.Source) }
    [void]$candidates.Add('C:\oss-cad-suite\bin\yosys.exe')
    # YoWASP: Yosys compiled to WebAssembly, installed with `pip install yowasp-yosys`.
    # Runs inside the (signed) Python process, so it is not subject to Smart App Control.
    [void]$candidates.Add((Join-Path (Split-Path -Parent $Python) 'Scripts\yowasp-yosys.exe'))
    $c = Get-Command yowasp-yosys -ErrorAction SilentlyContinue
    if ($c) { [void]$candidates.Add($c.Source) }

    foreach ($cand in $candidates) {
        if (-not (Test-Path $cand)) { continue }
        try {
            $v = Invoke-Tool $cand @('-V') (Join-Path $RepoRoot 'build\yosys_version.txt')
        } catch {
            Write-Host "  yosys candidate blocked or failed to launch: $cand" -ForegroundColor Yellow
            continue
        }
        if ($v.ExitCode -eq 0 -and $v.Output -match 'Yosys') {
            $Yosys = $cand
            $ysVer = $v
            break
        }
        Write-Host "  yosys candidate did not run: $cand" -ForegroundColor Yellow
    }
    if (-not $Yosys) {
        Write-Host 'No runnable Yosys found. Easiest fix:  python -m pip install yowasp-yosys   (see README).' -ForegroundColor Red
        Write-Host 'Or set $env:FLEXTERN_YOSYS to a yosys executable. Use -SkipSynth to run tests only.' -ForegroundColor Red
        Record-Step 'Yosys available' $false 'not found or not runnable'
    }
    $ysLine = ($ysVer.Output -split "`n" | Where-Object { $_ -match 'Yosys' } | Select-Object -First 1).Trim()
    Write-Host "  yosys    : $Yosys  ($ysLine)"
}

# ---------------------------------------------------------------------------
# 1. Python reference tests
# ---------------------------------------------------------------------------
Write-Header '1. Python golden reference unit tests'
$r = Invoke-Tool $Python @('-m', 'unittest', 'tests.test_reference', '-v') (Join-Path $RepoRoot 'build\python_tests.txt')
$ok = ($r.ExitCode -eq 0) -and ($r.Output -match '(?m)^OK\s*$')
$ran = if ($r.Output -match 'Ran (\d+) tests') { "$($Matches[1]) tests" } else { '' }
Record-Step 'Python reference' $ok $ran

# ---------------------------------------------------------------------------
# 2. Cross-check vectors
# ---------------------------------------------------------------------------
Write-Header '2. Generate Python-derived cross-check vectors'
$r = Invoke-Tool $Python @('scripts\gen_vectors.py') (Join-Path $RepoRoot 'build\gen_vectors.txt')
$ok = ($r.ExitCode -eq 0) -and (Test-Path 'tests\vectors\y_dot16_vectors.hex') -and ($r.Output -match 'wrote (\d+) vectors')
Record-Step 'Cross-check vector generation' $ok $r.Output.Trim()

# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------
function Run-Sim {
    param([string]$Name, [string]$Label, [string[]]$Sources, [string]$Token)
    Write-Header $Label
    $vvpFile = Join-Path $RepoRoot "build\$Name.vvp"
    $c = Invoke-Tool $Iverilog (@('-g2012', '-Wall', '-o', $vvpFile) + $Sources) (Join-Path $RepoRoot "build\${Name}_compile.txt")
    if ($c.ExitCode -ne 0) { Record-Step $Label $false "iverilog compile failed (build\${Name}_compile.txt)" }
    if ($c.Output -match 'warning') { Write-Host "  compile warnings present, see build\${Name}_compile.txt" -ForegroundColor Yellow }
    $s = Invoke-Tool $Vvp @($vvpFile) (Join-Path $RepoRoot "build\${Name}_sim.txt")
    $ok = ($s.ExitCode -eq 0) -and ($s.Output -match [regex]::Escape($Token)) -and ($s.Output -notmatch 'MISMATCH|FAIL')
    $counts = ($s.Output -split "`n" | Where-Object { $_ -match 'cases passed' } | ForEach-Object { $_.Trim() }) -join '; '
    Record-Step $Label $ok $counts
}

Run-Sim 'y_core_tb' 'Y-Core exhaustive RTL verification' @('rtl\y_core.sv', 'tests\y_core_tb.sv') 'PASS: Y-Core exhaustive verification completed.'
Run-Sim 'y_dot16_tb' 'Y-Dot16 RTL verification' @('rtl\y_core.sv', 'rtl\y_dot16.sv', 'tests\y_dot16_tb.sv') 'PASS: Y-Dot16 verification completed.'
Run-Sim 'int8_baseline_tb' 'INT8 baseline RTL verification' @('rtl\int8_mul_baseline.sv', 'rtl\int8_dot16_baseline.sv', 'tests\int8_baseline_tb.sv') 'PASS: INT8 baseline verification completed.'

# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
if ($SkipSynth) {
    Write-Host ''
    Write-Host '  Synthesis skipped (-SkipSynth).' -ForegroundColor Yellow
} else {
    function Run-Synth {
        param([string]$Name, [string]$Label)
        Write-Header $Label
        $script = "synth\$Name.ys"
        $r = Invoke-Tool $Yosys @('-q', $script) (Join-Path $RepoRoot "build\${Name}_yosys_console.txt")
        $resultFile = "results\${Name}_synth.txt"
        $ok = ($r.ExitCode -eq 0) -and (Test-Path $resultFile) -and ((Get-Content $resultFile -Raw) -match 'STAGE 2')
        $detail = ''
        if ($ok) {
            $txt = Get-Content $resultFile -Raw
            $stage2 = ($txt -split 'STAGE 2')[1]
            if ($stage2 -match '(\d+)\s+cells') { $detail = "$($Matches[1]) generic cells" }
            $warn = ($r.Output -split "`n" | Where-Object { $_ -match '(?i)warning' }).Count
            if ($warn -gt 0) { $detail += "; $warn warning line(s), see build\${Name}_yosys_console.txt" }
        } else {
            $detail = "see build\${Name}_yosys_console.txt"
        }
        Record-Step $Label $ok $detail
    }

    Run-Synth 'y_core'              'Y-Core synthesis'
    Run-Synth 'y_dot16'             'Y-Dot16 synthesis'
    Run-Synth 'int8_mul_baseline'   'INT8 multiplier baseline synthesis'
    Run-Synth 'int8_dot16_baseline' 'INT8 dot16 baseline synthesis'

    # Environment record for the results directory
    $commit = (& git rev-parse --short HEAD 2>$null)
    if (-not $commit) { $commit = 'unknown' }
    $env_txt = @(
        "FlexTern Phase 1 synthesis environment",
        "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')",
        "Git commit (HEAD at time of run): $commit",
        "",
        "Yosys    : $Yosys",
        "           $ysLine",
        "Icarus   : $Iverilog",
        "           $(($ivVer.Output -split "`n")[0].Trim())",
        "Python   : $Python",
        "           $($pyVer.Output.Trim())",
        "",
        "Synthesis type: generic Yosys synthesis (synth -flatten, internal cell library via ABC).",
        "No FPGA target, no standard-cell library, no timing constraints.",
        "Cell counts are technology-independent structure measurements, not area/power/frequency."
    )
    Set-Content -Path 'results\synth_environment.txt' -Value ($env_txt -join "`n") -Encoding utf8
}

Write-Summary
exit 0
