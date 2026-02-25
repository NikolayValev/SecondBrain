#!/usr/bin/env python3
"""Build agents/memory/LESSONS.md from agents/memory/runs.jsonl."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


def now_utc_iso() -> str:
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return stamp.replace("+00:00", "Z")


def resolve_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def safe_summary_cell(value: str, max_len: int = 120) -> str:
    clean = value.replace("|", r"\|").replace("\n", " ").strip()
    return clean if len(clean) <= max_len else clean[: max_len - 3] + "..."


def load_runs(path: Path) -> List[Dict]:
    runs: List[Dict] = []
    if not path.exists():
        return runs
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs


def top_counter_lines(counter: Counter, minimum_count: int, limit: int) -> List[str]:
    out: List[str] = []
    for text, count in counter.most_common(limit):
        if count < minimum_count:
            continue
        out.append(f"{text} ({count})")
    return out


def build_process_updates(status_counter: Counter, tag_counter: Counter) -> List[str]:
    updates: List[str] = []
    total = sum(status_counter.values())
    failed = status_counter.get("failed", 0)
    partial = status_counter.get("partial", 0)
    risky = failed + partial

    if total > 0 and risky / total >= 0.3:
        updates.append("Add an earlier checkpoint with a focused test before broader edits.")
    if tag_counter.get("tests", 0) == 0 and total > 0:
        updates.append("Attach at least one test-focused tag and run command in each run log.")
    if tag_counter.get("db", 0) >= 3:
        updates.append("Run schema-impact checks before and after database changes.")
    if tag_counter.get("rag", 0) >= 3 or tag_counter.get("embeddings", 0) >= 3:
        updates.append("Run a manual semantic query sanity check after RAG or embedding changes.")
    if not updates:
        updates.append("Keep current workflow; continue logging granular lessons.")
    return updates


def markdown_list(lines: Iterable[str]) -> str:
    formatted = [f"- {line}" for line in lines]
    return "\n".join(formatted) if formatted else "- None yet."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize reusable lessons from agents/memory/runs.jsonl."
    )
    parser.add_argument("--root", default=".", help="Repo root override.")
    parser.add_argument(
        "--runs-file",
        default="agents/memory/runs.jsonl",
        help="Path to runs.jsonl relative to repo root.",
    )
    parser.add_argument(
        "--output",
        default="agents/memory/LESSONS.md",
        help="Output markdown file relative to repo root.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Analyze only the most recent N runs.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum frequency to include repeated lessons/patterns.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_repo_root(Path(args.root))
    runs_path = (root / args.runs_file).resolve(strict=False)
    output_path = (root / args.output).resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runs = load_runs(runs_path)
    if args.window > 0:
        runs = runs[-args.window :]

    status_counter: Counter = Counter()
    tag_counter: Counter = Counter()
    lessons_counter: Counter = Counter()
    failures_counter: Counter = Counter()

    for run in runs:
        status = str(run.get("status", "")).strip()
        if status:
            status_counter[status] += 1

        for tag in run.get("tags", []) or []:
            clean = normalize_text(str(tag))
            if clean:
                tag_counter[clean] += 1

        for lesson in run.get("lessons", []) or []:
            clean = normalize_text(str(lesson))
            if clean:
                lessons_counter[clean] += 1

        if status in {"failed", "partial"}:
            summary = normalize_text(str(run.get("summary", "")))
            if summary:
                failures_counter[summary] += 1

    repeated_lessons = top_counter_lines(lessons_counter, args.min_count, limit=10)
    repeated_failures = top_counter_lines(failures_counter, args.min_count, limit=10)
    common_tags = top_counter_lines(tag_counter, minimum_count=1, limit=8)
    process_updates = build_process_updates(status_counter, tag_counter)

    recent_rows: List[str] = []
    for run in runs[-10:]:
        recent_rows.append(
            "| {timestamp} | {agent} | {status} | {summary} |".format(
                timestamp=safe_summary_cell(str(run.get("timestamp_utc", "")), 24),
                agent=safe_summary_cell(str(run.get("agent", "")), 16),
                status=safe_summary_cell(str(run.get("status", "")), 10),
                summary=safe_summary_cell(str(run.get("summary", "")), 120),
            )
        )

    markdown = f"""# Agent Lessons
_Auto-generated from `{args.runs_file}` on {now_utc_iso()}._

## Snapshot
- Runs analyzed: {len(runs)}
- Success: {status_counter.get("success", 0)}
- Partial: {status_counter.get("partial", 0)}
- Failed: {status_counter.get("failed", 0)}

## Common Tags
{markdown_list(common_tags)}

## Repeated Lessons
{markdown_list(repeated_lessons)}

## Repeated Failure Patterns
{markdown_list(repeated_failures)}

## Suggested Process Updates
{markdown_list(process_updates)}

## Recent Runs
| Timestamp (UTC) | Agent | Status | Summary |
|---|---|---|---|
{chr(10).join(recent_rows) if recent_rows else "| - | - | - | No runs logged yet. |"}
"""

    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote lessons: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
