# run_robustness_extensions.ps1
#
# Runs the three robustness extensions of the ATLAS 80030 paper, in order:
#   1. feature_response.py  - how each input moves under esup/esdown/cer/cpos (minutes)
#   2. tagger_variants.py   - 6 training seeds x 3 input sets, all uncertainties (about 2 hours)
#   3. stat_bootstrap.py    - 100 bootstrap replicas of every frozen metric (about 3 hours)
#
# Nothing frozen is changed. Every result goes to a NEW folder in 04_Results:
#   feature_response_round1, tagger_variants_round1, stat_bootstrap_round1
#
# Each script first checks itself against the frozen results and stops with
# an error if anything does not match.
#
# Run from PowerShell in the project folder:
#   powershell -ExecutionPolicy Bypass -File .\run_robustness_extensions.ps1
#
# If the bootstrap is interrupted, continue it with:
#   .\.venv\Scripts\python.exe 03_Code_files\stat_bootstrap.py --resume 04_Results\stat_bootstrap_round1

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { throw "Python environment not found: $py" }
$code = Join-Path $PSScriptRoot '03_Code_files'

$steps = @('feature_response.py', 'tagger_variants.py', 'stat_bootstrap.py')
$start = Get-Date
foreach ($s in $steps) {
    $t0 = Get-Date
    Write-Host ""
    Write-Host "=================================================================="
    Write-Host " $s   (started $($t0.ToString('HH:mm:ss')))"
    Write-Host "=================================================================="
    & $py (Join-Path $code $s)
    if ($LASTEXITCODE -ne 0) {
        throw "$s stopped with exit code $LASTEXITCODE. Nothing frozen was changed; the lines above give the cause."
    }
    $m = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Write-Host " $s finished in $m minutes"
}
$h = [math]::Round(((Get-Date) - $start).TotalHours, 2)
Write-Host ""
Write-Host "ALL THREE FINISHED in $h hours. New result folders:"
Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot '04_Results') -Directory |
    Where-Object { $_.Name -match '^(feature_response|tagger_variants|stat_bootstrap)_round\d+$' } |
    ForEach-Object { Write-Host "  $($_.FullName)" }
