# Run a Python script with venv312 (CUDA). Usage: .\run.ps1 evaluate_ensemble.py [--args]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Script,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$Python = Join-Path $PSScriptRoot "venv312\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "venv312 not found at $Python"
    exit 1
}

$ScriptPath = Join-Path $PSScriptRoot $Script
& $Python $ScriptPath @Args
