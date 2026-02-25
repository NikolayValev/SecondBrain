#!/usr/bin/env python3
"""Append a structured agent run record to agents/memory/runs.jsonl."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

AGENT_CHOICES = ("codex", "copilot", "antigravity", "other")
STATUS_CHOICES = ("success", "partial", "failed")


def now_utc_iso() -> str:
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return stamp.replace("+00:00", "Z")


def dedupe_clean(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        clean = value.strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def resolve_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def normalize_paths(raw_paths: Iterable[str], root: Path) -> List[str]:
    normalized: List[str] = []
    for raw in raw_paths:
        text = raw.strip()
        if not text:
            continue
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            relative = resolved.relative_to(root).as_posix()
            normalized.append(relative)
        except ValueError:
            normalized.append(text.replace("\\", "/"))
    return dedupe_clean(normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log an agent run to agents/memory/runs.jsonl."
    )
    parser.add_argument("--agent", required=True, choices=AGENT_CHOICES)
    parser.add_argument("--task", required=True, help="Short task title.")
    parser.add_argument("--status", required=True, choices=STATUS_CHOICES)
    parser.add_argument("--summary", required=True, help="Outcome summary.")
    parser.add_argument(
        "--lesson",
        action="append",
        default=[],
        help="Reusable lesson. Repeat for multiple lessons.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Tag for domain/risk tracking. Repeatable.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Touched file path (absolute or repo-relative). Repeatable.",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Command executed during the run. Repeatable.",
    )
    parser.add_argument(
        "--next-step",
        default="",
        help="Required for partial/failed runs; optional for success.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=None,
        help="Optional execution duration in seconds.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repo root override. Defaults to current working directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_repo_root(Path(args.root))
    memory_dir = root / "agents" / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    lessons = dedupe_clean(args.lesson)
    tags = dedupe_clean(args.tag)
    commands = dedupe_clean(args.command)
    files = normalize_paths(args.file, root)

    if args.status in {"partial", "failed"} and not args.next_step.strip():
        raise SystemExit("--next-step is required when --status is partial or failed.")

    timestamp = now_utc_iso()
    compact_stamp = timestamp.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    run_id = f"{compact_stamp}_{args.agent}"

    payload = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "agent": args.agent,
        "task": args.task.strip(),
        "status": args.status,
        "summary": args.summary.strip(),
        "lessons": lessons,
        "tags": tags,
        "files": files,
        "commands": commands,
        "next_step": args.next_step.strip() or None,
        "duration_seconds": args.duration_seconds,
    }

    runs_path = memory_dir / "runs.jsonl"
    with runs_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True))
        fh.write("\n")

    latest_path = memory_dir / "last_run.json"
    latest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Logged run: {run_id}")
    print(f"Runs file: {runs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
