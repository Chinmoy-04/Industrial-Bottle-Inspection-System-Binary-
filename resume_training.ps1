# Clear pause flag so training can continue (re-run train_model_v17.py).
$flag = Join-Path $PSScriptRoot "pause_training.flag"
if (Test-Path $flag) {
    Remove-Item $flag -Force
    Write-Host "Pause flag removed. Start training with:"
    Write-Host "  python train_model_v17.py"
} else {
    Write-Host "No pause flag present — nothing to clear."
}
