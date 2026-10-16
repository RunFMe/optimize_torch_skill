#!/usr/bin/env python3
"""Summarize PyTorch Chrome traces without loading the whole JSON document."""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, TextIO

GPU_CATEGORY_RE = re.compile(
    r"(?:^|_)(?:kernel|gpu_kernel|gpu_memcpy|gpu_memset)(?:$|_)", re.IGNORECASE
)
COLLECTIVE_RE = re.compile(
    r"nccl|gloo|all[_ -]?(?:gather|reduce|to[_ -]?all)|reduce[_ -]?scatter|broadcast",
    re.IGNORECASE,
)
SYNC_RE = re.compile(
    r"synchroniz|_local_scalar_dense|aten::item|cudaDeviceSynchronize|"
    r"cudaStreamSynchronize|cudaEventSynchronize|cuCtxSynchronize",
    re.IGNORECASE,
)
COPY_RE = re.compile(r"memcpy|memset|copy_|aten::to|aten::_to_copy", re.IGNORECASE)
LAUNCH_RE = re.compile(
    r"(?:cuda|cu)(?:Launch(?:Kernel|CooperativeKernel|KernelEx)|GraphLaunch)", re.IGNORECASE
)


@dataclass
class NameStat:
    count: int = 0
    total_us: float = 0.0
    max_us: float = 0.0

    def add(self, duration_us: float) -> None:
        self.count += 1
        self.total_us += duration_us
        self.max_us = max(self.max_us, duration_us)


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _find_event_array(stream: TextIO, chunk_size: int = 1 << 20) -> tuple[str, int]:
    buffer = ""
    first_nonspace_seen = False
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            raise ValueError(
                "Could not find a top-level event array or traceEvents array"
            )
        buffer += chunk
        if not first_nonspace_seen:
            stripped = buffer.lstrip()
            if stripped:
                first_nonspace_seen = True
                if stripped.startswith("["):
                    return stripped, 1
        match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
        if match:
            return buffer[match.end() :], 0
        if len(buffer) > 4 << 20:
            buffer = buffer[-2 << 20 :]


def iter_trace_events(
    path: Path, chunk_size: int = 1 << 20
) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with open_text(path) as stream:
        buffer, top_level_offset = _find_event_array(stream, chunk_size)
        index = top_level_offset
        eof = False
        while True:
            while index < len(buffer) and buffer[index] in " \t\r\n,":
                index += 1
            if index < len(buffer) and buffer[index] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, index)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("Trace ended before the event array was complete")
                if index:
                    buffer = buffer[index:]
                    index = 0
                chunk = stream.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            index = end
            if isinstance(value, dict):
                yield value
            if index > 8 << 20:
                buffer = buffer[index:]
                index = 0


def event_duration(event: dict[str, Any]) -> float | None:
    duration = event.get("dur")
    if isinstance(duration, (int, float)) and math.isfinite(float(duration)):
        return max(0.0, float(duration))
    return None


def is_gpu_event(event: dict[str, Any]) -> bool:
    category = str(event.get("cat", ""))
    name = str(event.get("name", ""))
    if GPU_CATEGORY_RE.search(category.replace(" ", "_")):
        return True
    lower = category.lower()
    return lower == "kernel" or ("gpu" in lower and bool(COPY_RE.search(name)))


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def format_ms(us: float) -> str:
    return f"{us / 1000.0:.3f} ms"


def print_name_stats(title: str, stats: dict[str, NameStat], top: int) -> None:
    print(f"\n{title}")
    print(f"{'total':>13} {'max':>13} {'count':>9}  name")
    for name, stat in sorted(
        stats.items(), key=lambda item: item[1].total_us, reverse=True
    )[:top]:
        print(
            f"{format_ms(stat.total_us):>13} {format_ms(stat.max_us):>13} "
            f"{stat.count:>9}  {name}"
        )


def summarize(
    path: Path, top: int, min_gap_us: float, stage_pattern: re.Pattern[str] | None
) -> None:
    name_stats: dict[str, NameStat] = defaultdict(NameStat)
    gpu_stats: dict[str, NameStat] = defaultdict(NameStat)
    collective_stats: dict[str, NameStat] = defaultdict(NameStat)
    sync_stats: dict[str, NameStat] = defaultdict(NameStat)
    copy_stats: dict[str, NameStat] = defaultdict(NameStat)
    stage_stats: dict[str, NameStat] = defaultdict(NameStat)
    category_stats: dict[str, NameStat] = defaultdict(NameStat)
    gpu_intervals: list[tuple[float, float]] = []
    launch_intervals: list[tuple[float, float, str]] = []
    begin_stacks: dict[tuple[Any, Any], list[tuple[float, str, str]]] = defaultdict(
        list
    )
    metadata: dict[tuple[Any, Any], str] = {}
    event_count = 0
    complete_count = 0
    min_ts = math.inf
    max_ts = -math.inf

    def consume(event: dict[str, Any], duration_us: float) -> None:
        nonlocal complete_count, min_ts, max_ts
        ts = event.get("ts")
        if not isinstance(ts, (int, float)):
            return
        ts = float(ts)
        name = str(event.get("name", "<unnamed>"))
        category = str(event.get("cat", "<uncategorized>"))
        complete_count += 1
        min_ts = min(min_ts, ts)
        max_ts = max(max_ts, ts + duration_us)
        name_stats[name].add(duration_us)
        category_stats[category].add(duration_us)
        if is_gpu_event(event):
            gpu_stats[name].add(duration_us)
            gpu_intervals.append((ts, ts + duration_us))
        if COLLECTIVE_RE.search(name) or COLLECTIVE_RE.search(category):
            collective_stats[name].add(duration_us)
        if SYNC_RE.search(name):
            sync_stats[name].add(duration_us)
        if COPY_RE.search(name) or COPY_RE.search(category):
            copy_stats[name].add(duration_us)
        if stage_pattern and stage_pattern.search(name):
            stage_stats[name].add(duration_us)
        if LAUNCH_RE.search(name):
            launch_intervals.append((ts, ts + duration_us, name))

    for event in iter_trace_events(path):
        event_count += 1
        phase = event.get("ph")
        if phase == "M":
            args = event.get("args")
            if isinstance(args, dict) and "name" in args:
                metadata[(event.get("pid"), event.get("tid"))] = str(args["name"])
            continue
        duration = event_duration(event)
        if phase == "X" and duration is not None:
            consume(event, duration)
            continue
        key = (event.get("pid"), event.get("tid"))
        if phase == "B" and isinstance(event.get("ts"), (int, float)):
            begin_stacks[key].append(
                (
                    float(event["ts"]),
                    str(event.get("name", "<unnamed>")),
                    str(event.get("cat", "")),
                )
            )
        elif (
            phase == "E"
            and begin_stacks[key]
            and isinstance(event.get("ts"), (int, float))
        ):
            start, name, category = begin_stacks[key].pop()
            synthetic = dict(event, ts=start, name=name, cat=category, ph="X")
            consume(synthetic, max(0.0, float(event["ts"]) - start))

    print(f"Trace: {path}")
    print(f"Events parsed: {event_count:,}; complete/timed events: {complete_count:,}")
    if min_ts != math.inf:
        print(f"Timed event span: {format_ms(max_ts - min_ts)}")

    merged_gpu = merge_intervals(gpu_intervals)
    gpu_busy = sum(end - start for start, end in merged_gpu)
    if merged_gpu:
        gpu_window = merged_gpu[-1][1] - merged_gpu[0][0]
        utilization = 100.0 * gpu_busy / gpu_window if gpu_window else 0.0
        print(
            f"Global GPU busy union: {format_ms(gpu_busy)} over {format_ms(gpu_window)} "
            f"({utilization:.2f}% between first and last GPU event)"
        )
        gaps = [
            (next_start - end, end, next_start)
            for (_, end), (next_start, _) in pairwise(merged_gpu)
            if next_start - end >= min_gap_us
        ]
        print(f"GPU-idle gaps >= {min_gap_us:.0f} us: {len(gaps):,}")
        for gap, start, end in heapq.nlargest(top, gaps):
            print(f"  {format_ms(gap):>13} at ts {start:.3f} -> {end:.3f}")

    launches = sorted(launch_intervals)
    launch_gaps = [
        (next_start - end, end, next_start, next_name)
        for (_, end, _), (next_start, _, next_name) in pairwise(launches)
        if next_start - end >= min_gap_us
    ]
    print(f"Kernel-launch submission events: {len(launches):,}")
    print(f"Launch submission gaps >= {min_gap_us:.0f} us: {len(launch_gaps):,}")
    for gap, start, end, next_name in heapq.nlargest(top, launch_gaps):
        print(
            f"  {format_ms(gap):>13} at ts {start:.3f} -> {end:.3f}; next={next_name}"
        )

    print_name_stats(
        "Top categories by summed duration (overlap may double count)",
        category_stats,
        top,
    )
    print_name_stats("Top GPU events by summed duration", gpu_stats, top)
    print_name_stats("Collective-like events", collective_stats, top)
    print_name_stats("Synchronization/scalar-read events", sync_stats, top)
    print_name_stats("Copy/transfer-like events", copy_stats, top)
    if stage_pattern:
        print_name_stats(
            f"Stage events matching /{stage_pattern.pattern}/", stage_stats, top
        )

    if metadata:
        print("\nNamed lanes")
        for (pid, tid), name in sorted(metadata.items(), key=lambda item: str(item[0]))[
            : 4 * top
        ]:
            print(f"  pid={pid} tid={tid}: {name}")

    print(
        "\nInterpretation warning: summed durations are not critical-path savings. "
        "Confirm overlap, rank arrival, and dependencies in a rendered Perfetto timeline; "
        "use unprofiled runs for speed claims."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Chrome trace JSON or .json.gz")
    parser.add_argument("--top", type=int, default=20, help="rows per summary table")
    parser.add_argument(
        "--min-gap-us",
        type=float,
        default=100.0,
        help="minimum GPU/launch gap to report, in trace microseconds",
    )
    parser.add_argument(
        "--stage-regex",
        default=(
            r"(?:^|/)(?:request|input|load|preprocess|transfer|model|forward|infer|"
            r"iteration|decode|postprocess|serialize|persist|publish|respond)(?:/|$)"
        ),
        help="regex used to select user stage ranges; pass an empty string to disable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top <= 0:
        raise SystemExit("--top must be positive")
    if args.min_gap_us < 0:
        raise SystemExit("--min-gap-us must be non-negative")
    stage_pattern = (
        re.compile(args.stage_regex, re.IGNORECASE) if args.stage_regex else None
    )
    try:
        summarize(args.trace, args.top, args.min_gap_us, stage_pattern)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
