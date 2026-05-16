#!/usr/bin/env python3
"""Run the existing maesyori_fast YOLO preprocessing pipeline from a config file."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_ROOT = REPO_ROOT / "src" / "library"
RUNS_ROOT = REPO_ROOT / "runs"


def add_local_libraries() -> None:
    for package_dir in LIBRARY_ROOT.iterdir():
        if package_dir.is_dir():
            sys.path.insert(0, str(package_dir))


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_config(config: dict) -> None:
    required = ["base_dir", "model_path", "target_folders"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"config missing required keys: {', '.join(missing)}")

    if not isinstance(config["target_folders"], list) or not config["target_folders"]:
        raise ValueError("target_folders must be a non-empty list")


def create_run_dir(config_path: Path, config: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"preprocess_yolo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(config_path, run_dir / "config.json")
    metadata = {
        "created_at": timestamp,
        "command": " ".join(sys.argv),
        "base_dir": config["base_dir"],
        "model_path": config["model_path"],
        "target_folders": config["target_folders"],
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "preprocess_yolo.json",
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print resolved settings without running preprocessing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_config(config)

    base_dir = Path(config["base_dir"]).expanduser()
    model_path = Path(config["model_path"]).expanduser()
    target_folders = config["target_folders"]
    area_threshold = int(config.get("area_threshold", 50000))
    max_workers = int(config.get("max_workers", 2))

    resolved = {
        "config": str(config_path),
        "base_dir": str(base_dir),
        "model_path": str(model_path),
        "target_folders": target_folders,
        "area_threshold": area_threshold,
        "max_workers": max_workers,
    }
    print(json.dumps(resolved, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir does not exist: {base_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")

    add_local_libraries()
    import maesyori_fast

    run_dir = create_run_dir(config_path, resolved)
    print(f"run_dir: {run_dir}")

    maesyori_fast.run(
        base_dir=str(base_dir),
        model_path=str(model_path),
        target_folders=target_folders,
        area_threshold=area_threshold,
        max_workers=max_workers,
    )

    with (run_dir / "status.json").open("w", encoding="utf-8") as f:
        json.dump({"status": "completed"}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
