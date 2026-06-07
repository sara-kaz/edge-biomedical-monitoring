"""
Entry point + re-exports for 4-case accuracy testing.

This wrapper keeps `python scripts/test_accuracy_4_cases.py` working while the core
implementation currently lives at the repository root (`test_accuracy_4_cases.py`).
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

_ROOT_IMPL = PROJECT_ROOT / "test_accuracy_4_cases.py"
_spec = importlib.util.spec_from_file_location("test_accuracy_4_cases_root", _ROOT_IMPL)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load testing implementation at {_ROOT_IMPL}")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]

# Re-export public API
AccuracyTester4Cases = _mod.AccuracyTester4Cases
main = _mod.main


if __name__ == "__main__":
    main()

