# Auto-run for new Cursor/VS Code terminals in this workspace.
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
. (Join-Path $PWD "venv312\Scripts\Activate.ps1")
