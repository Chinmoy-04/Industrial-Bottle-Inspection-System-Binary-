# Request a graceful pause — training stops after the current epoch.
$flag = Join-Path $PSScriptRoot "pause_training.flag"
New-Item -Path $flag -ItemType File -Force | Out-Null
Write-Host "Pause requested: $flag"
Write-Host "Training will stop after the current epoch and save a checkpoint."
