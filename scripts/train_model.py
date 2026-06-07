"""
Entry point + re-exports for training.

This wrapper keeps `python scripts/train_model.py` working while the core trainer
currently lives at the repository root (`train_model.py`).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ROOT_IMPL = PROJECT_ROOT / "train_model.py"
_spec = importlib.util.spec_from_file_location("train_model_root", _ROOT_IMPL)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load training implementation at {_ROOT_IMPL}")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]

# Re-export public API
ComprehensiveTrainer = _mod.ComprehensiveTrainer
main = _mod.main


if __name__ == "__main__":
    main()

