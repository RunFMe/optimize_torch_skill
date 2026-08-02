#!/usr/bin/env python3
"""Create a small, visualization-oriented crop from a Chrome trace."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from summarize_trace import iter_trace_events


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def overlaps(event: dict[str, Any], start_us: float, end_us: float) -> bool:
    timestamp = finite_number(event.get("ts"))
    if timestamp is None:
        return False
    duration = finite_number(event.get("dur")) or 0.0
    return timestamp <= end_us and timestamp + max(0.0, duration) >= start_us


def matches_filters(
    event: dict[str, Any], keep_names: list[str], keep_categories: list[str]
) -> bool:
    if not keep_names and not keep_categories:
        return True
    name = str(event.get("name", "")).lower()
    category = str(event.get("cat", "")).lower()
    return any(value in name for value in keep_names) or any(
        value in category for value in keep_categories
    )


def clip_complete_event(
    event: dict[str, Any], start_us: float, end_us: float
) -> dict[str, Any]:
    result = dict(event)
    if result.get("ph") != "X":
        return result
    timestamp = finite_number(result.get("ts"))
    duration = finite_number(result.get("dur"))
    if timestamp is None or duration is None:
        return result
    clipped_start = max(timestamp, start_us)
    clipped_end = min(timestamp + max(0.0, duration), end_us)
    result["ts"] = clipped_start
    result["dur"] = max(0.0, clipped_end - clipped_start)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Chrome trace JSON or .json.gz")
    parser.add_argument("output", type=Path, help="output Chrome trace JSON")
    parser.add_argument("--start-us", type=float, required=True)
    parser.add_argument("--duration-us", type=float, required=True)
    parser.add_argument(
        "--keep-name",
        action="append",
        default=[],
        help="keep events whose name contains this value; repeat as needed",
    )
    parser.add_argument(
        "--keep-category",
        action="append",
        default=[],
        help="keep events whose category contains this value; repeat as needed",
    )
    parser.add_argument(
        "--strip-args",
        action="store_true",
        help="drop event argument payloads to keep visualization crops small",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_us <= 0:
        raise SystemExit("--duration-us must be positive")
    end_us = args.start_us + args.duration_us
    keep_names = [value.lower() for value in args.keep_name]
    keep_categories = [value.lower() for value in args.keep_category]
    selected: list[dict[str, Any]] = []
    try:
        for event in iter_trace_events(args.trace):
            if event.get("ph") == "M":
                selected.append(event)
                continue
            if not overlaps(event, args.start_us, end_us):
                continue
            if not matches_filters(event, keep_names, keep_categories):
                continue
            result = clip_complete_event(event, args.start_us, end_us)
            if args.strip_args:
                result.pop("args", None)
            selected.append(result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump({"traceEvents": selected}, output, separators=(",", ":"))
    print(f"Wrote {len(selected):,} events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
