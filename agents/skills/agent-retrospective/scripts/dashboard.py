#!/usr/bin/env python3
"""Show trend metrics from agents/memory/runs.jsonl."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_iso_utc(value: str) -> Optional[datetime]:
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def resolve_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


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
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_iso_utc(str(record.get("timestamp_utc", "")))
            record["_timestamp"] = ts
            runs.append(record)
    runs.sort(key=lambda x: x.get("_timestamp") or datetime.min.replace(tzinfo=timezone.utc))
    return runs


def clamp_pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((part / total) * 100.0, 1)


def normalize_summary(summary: str) -> str:
    clean = " ".join(summary.strip().split())
    return clean.lower()


def summarize_status(runs: Iterable[Dict]) -> Dict[str, int]:
    counter: Counter = Counter()
    for run in runs:
        status = str(run.get("status", "")).strip().lower()
        if status in {"success", "partial", "failed"}:
            counter[status] += 1
    return {
        "success": counter.get("success", 0),
        "partial": counter.get("partial", 0),
        "failed": counter.get("failed", 0),
        "total": sum(counter.values()),
    }


def print_kv(label: str, value: str) -> None:
    print(f"{label:<26}{value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show retrospective dashboard metrics.")
    parser.add_argument("--root", default=".", help="Repo root override.")
    parser.add_argument(
        "--runs-file",
        default="agents/memory/runs.jsonl",
        help="Path to runs.jsonl relative to repo root.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Use only the most recent N runs for dashboard metrics.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Show daily trend rows for last N days.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Maximum top failure patterns and tags to print.",
    )
    parser.add_argument(
        "--agent",
        choices=["codex", "copilot", "antigravity", "other"],
        default="",
        help="Filter to a single agent.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_repo_root(Path(args.root))
    runs_path = (root / args.runs_file).resolve(strict=False)
    runs = load_runs(runs_path)

    if args.agent:
        runs = [r for r in runs if str(r.get("agent", "")).strip().lower() == args.agent]
    if args.window > 0:
        runs = runs[-args.window :]

    overall = summarize_status(runs)
    success_rate = clamp_pct(overall["success"], overall["total"])

    now = datetime.now(timezone.utc)
    last_7_cutoff = now - timedelta(days=7)
    recent_7 = [r for r in runs if r.get("_timestamp") and r["_timestamp"] >= last_7_cutoff]
    recent_7_status = summarize_status(recent_7)
    recent_7_rate = clamp_pct(recent_7_status["success"], recent_7_status["total"])

    failure_patterns: Counter = Counter()
    tag_counter: Counter = Counter()
    agent_counter: Counter = Counter()
    daily_status: Dict[str, Counter] = defaultdict(Counter)

    for run in runs:
        agent = str(run.get("agent", "")).strip().lower()
        if agent:
            agent_counter[agent] += 1

        for tag in run.get("tags", []) or []:
            tag_text = str(tag).strip().lower()
            if tag_text:
                tag_counter[tag_text] += 1

        status = str(run.get("status", "")).strip().lower()
        if status in {"failed", "partial"}:
            summary = normalize_summary(str(run.get("summary", "")))
            if summary:
                failure_patterns[summary] += 1

        ts = run.get("_timestamp")
        if ts and status in {"success", "partial", "failed"}:
            day = ts.date().isoformat()
            daily_status[day][status] += 1

    print("Agent Retrospective Dashboard")
    print("=" * 30)
    print_kv("Runs file", str(runs_path))
    print_kv("Agent filter", args.agent or "all")
    print_kv("Runs considered", str(overall["total"]))
    print_kv("Success rate", f"{success_rate}%")
    print_kv("Last 7d success rate", f"{recent_7_rate}%")
    print_kv(
        "Status counts",
        f"success={overall['success']}, partial={overall['partial']}, failed={overall['failed']}",
    )

    print("\nTop failure patterns")
    if failure_patterns:
        for text, count in failure_patterns.most_common(args.top):
            preview = text if len(text) <= 100 else text[:97] + "..."
            print(f"- {preview} ({count})")
    else:
        print("- None")

    print("\nTop tags")
    if tag_counter:
        for tag, count in tag_counter.most_common(args.top):
            print(f"- {tag} ({count})")
    else:
        print("- None")

    print("\nRuns by agent")
    if agent_counter:
        for agent, count in agent_counter.most_common():
            print(f"- {agent}: {count}")
    else:
        print("- None")

    print(f"\nDaily trend (last {args.days} days)")
    print("date        total success partial failed success_rate")
    start_day = (now - timedelta(days=max(args.days - 1, 0))).date()
    for offset in range(max(args.days, 1)):
        day = start_day + timedelta(days=offset)
        key = day.isoformat()
        counts = daily_status.get(key, Counter())
        total = counts.get("success", 0) + counts.get("partial", 0) + counts.get("failed", 0)
        rate = clamp_pct(counts.get("success", 0), total)
        print(
            f"{key} {total:>5} {counts.get('success', 0):>7} "
            f"{counts.get('partial', 0):>7} {counts.get('failed', 0):>6} {rate:>11}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
