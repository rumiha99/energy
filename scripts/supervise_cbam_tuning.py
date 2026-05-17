#!/usr/bin/env python3
"""Supervise CBAM tuning and switch from focused to exploit search after review."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def result_files(path: Path) -> list[Path]:
    return sorted(path.glob("trial_*/result.json"))


def load_results(path: Path) -> list[dict]:
    rows = []
    for file_path in result_files(path):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            data["result_path"] = str(file_path)
            rows.append(data)
        except Exception:
            continue
    return rows


def best_result(*dirs: Path) -> dict | None:
    rows: list[dict] = []
    for directory in dirs:
        rows.extend(load_results(directory))
    if not rows:
        return None
    return max(rows, key=lambda row: row.get("test", {}).get("macro_f1", -1.0))


def pid_is_running(pid_path: Path) -> bool:
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def stop_pid(pid_path: Path) -> None:
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return


def start_tuning(output_dir: Path, phase: str, deadline: str, study_name: str, min_epochs: int, max_epochs: int) -> subprocess.Popen:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tuner.log"
    cmd = [
        "python3",
        str(REPO_ROOT / "scripts" / "tune_cbam_full_from_images.py"),
        "--phase",
        phase,
        "--deadline",
        deadline,
        "--output-dir",
        str(output_dir),
        "--study-name",
        study_name,
        "--min-epochs",
        str(min_epochs),
        "--max-epochs",
        str(max_epochs),
    ]
    log_file = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    (output_dir / "runner.pid").write_text(str(proc.pid), encoding="utf-8")
    return proc


def parse_deadline_timestamp(value: str) -> float:
    text = value.strip().replace("/", "-")
    if "T" not in text and " " not in text:
        text = f"{text}T00:00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
    return dt.timestamp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--broad-dir", type=Path, default=REPO_ROOT / "runs" / "cbam_tuning_20260518_0800")
    parser.add_argument("--focused-dir", type=Path, default=REPO_ROOT / "runs" / "cbam_tuning_focused_20260518_0800")
    parser.add_argument("--exploit-dir", type=Path, default=REPO_ROOT / "runs" / "cbam_tuning_exploit_20260518_0800")
    parser.add_argument("--review-interval-seconds", type=int, default=1800)
    parser.add_argument("--focused-min-results", type=int, default=3)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events_path = args.exploit_dir.parent / "cbam_tuning_supervisor_events.jsonl"
    append_jsonl(events_path, {"event": "supervisor_start", "time": datetime.now().isoformat(timespec="seconds"), "args": vars(args)})
    exploit_started = pid_is_running(args.exploit_dir / "runner.pid") or bool(result_files(args.exploit_dir))

    deadline_ts = parse_deadline_timestamp(args.deadline)
    while time.time() < deadline_ts:
        focused_results = load_results(args.focused_dir)
        best = best_result(args.broad_dir, args.focused_dir, args.exploit_dir)
        append_jsonl(
            events_path,
            {
                "event": "review",
                "time": datetime.now().isoformat(timespec="seconds"),
                "focused_results": len(focused_results),
                "exploit_started": exploit_started,
                "best_macro_f1": best.get("test", {}).get("macro_f1") if best else None,
                "best_path": best.get("result_path") if best else None,
            },
        )
        if not exploit_started and len(focused_results) >= args.focused_min_results:
            stop_pid(args.focused_dir / "runner.pid")
            time.sleep(5)
            proc = start_tuning(
                args.exploit_dir,
                "exploit",
                args.deadline,
                "cbam_exploit_after_auto_review",
                args.min_epochs,
                args.max_epochs,
            )
            exploit_started = True
            append_jsonl(events_path, {"event": "exploit_start", "time": datetime.now().isoformat(timespec="seconds"), "pid": proc.pid})
        time.sleep(args.review_interval_seconds)
    append_jsonl(events_path, {"event": "supervisor_finish", "time": datetime.now().isoformat(timespec="seconds")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
