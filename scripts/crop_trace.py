#!/usr/bin/env python3
"""Create a small, visualization-oriented crop from a Chrome trace."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works under `python -I` too

from summarize_trace import iter_trace_events  # noqa: E402


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


def is_enclosing_frame(event: dict[str, Any], start_us: float, end_us: float) -> bool:
    """A Python stack frame that spans the whole window; clipped, it becomes a full-width bar."""
    if str(event.get("cat", "")) != "python_function":
        return False
    timestamp = finite_number(event.get("ts"))
    duration = finite_number(event.get("dur"))
    if timestamp is None or duration is None:
        return False
    return timestamp < start_us and timestamp + duration > end_us


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
        "--drop-category",
        action="append",
        default=[],
        help="drop events whose category contains this value, e.g. python_function",
    )
    parser.add_argument(
        "--keep-enclosing-frames",
        action="store_true",
        help=(
            "keep python_function frames that span the whole window; by default they are "
            "dropped because clipped stacks hide the runtime and GPU rows"
        ),
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
    drop_categories = [value.lower() for value in args.drop_category]
    metadata: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    dropped_frames = 0
    try:
        for event in iter_trace_events(args.trace):
            if event.get("ph") == "M":
                metadata.append(event)
                continue
            if not overlaps(event, args.start_us, end_us):
                continue
            if not matches_filters(event, keep_names, keep_categories):
                continue
            category = str(event.get("cat", "")).lower()
            if any(value in category for value in drop_categories):
                continue
            if not args.keep_enclosing_frames and is_enclosing_frame(
                event, args.start_us, end_us
            ):
                dropped_frames += 1
                continue
            result = clip_complete_event(event, args.start_us, end_us)
            if args.strip_args:
                result.pop("args", None)
            selected.append(result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Metadata first: Kineto writes it at the end, and renderers read lane names from the start.
    with args.output.open("w", encoding="utf-8") as output:
        json.dump({"traceEvents": metadata + selected}, output, separators=(",", ":"))
    print(
        f"Wrote {len(metadata) + len(selected):,} events to {args.output}"
        + (f"; dropped {dropped_frames:,} enclosing Python frames" if dropped_frames else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
