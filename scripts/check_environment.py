#!/usr/bin/env python3
"""Minimal environment check for the energy workspace."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    checks = {
        "python": sys.version.split()[0],
        "repo_root": str(REPO_ROOT),
        "modules": {
            "cv2": module_available("cv2"),
            "numpy": module_available("numpy"),
            "pandas": module_available("pandas"),
            "sklearn": module_available("sklearn"),
            "ultralytics": module_available("ultralytics"),
            "optuna": module_available("optuna"),
            "torch": module_available("torch"),
        },
    }

    print(json.dumps(checks, ensure_ascii=False, indent=2))
    missing = [name for name, ok in checks["modules"].items() if not ok]
    if missing:
        print(f"missing modules: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
