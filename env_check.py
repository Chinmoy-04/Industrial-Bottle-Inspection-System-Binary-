"""Ensure scripts run under venv312 (CUDA). Auto-relaunches if another venv is selected."""
import os
import sys
from pathlib import Path

EXPECTED = "venv312"
PROJECT_ROOT = Path(__file__).resolve().parent
VENV312_PYTHON = PROJECT_ROOT / "venv312" / "Scripts" / "python.exe"


def _is_venv312() -> bool:
    exe = Path(sys.executable).resolve()
    prefix = Path(sys.prefix).resolve()
    return EXPECTED in exe.parts or EXPECTED in prefix.parts


def relaunch_with_venv312() -> None:
    """Re-exec this script with venv312 Python (does not return)."""
    if _is_venv312() or not VENV312_PYTHON.exists():
        return
    print(f"[env] Wrong interpreter: {sys.executable}")
    print(f"[env] Relaunching with: {VENV312_PYTHON}")
    os.execv(str(VENV312_PYTHON), [str(VENV312_PYTHON), *sys.argv])


def require_venv312(*, require_cuda: bool = False) -> None:
    relaunch_with_venv312()
    if not _is_venv312():
        raise RuntimeError(
            f"Wrong Python environment.\n"
            f"  Current: {sys.executable}\n"
            f"  Required: {VENV312_PYTHON}\n"
            f"  Or run: .\\run.ps1 your_script.py"
        )
    if require_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available (CPU-only PyTorch?).\n"
                f"Use: {VENV312_PYTHON}"
            )


if __name__ == "__main__":
    require_venv312(require_cuda=True)
    import torch
    print("OK:", sys.executable)
    print("CUDA:", torch.cuda.is_available(), torch.__version__)
