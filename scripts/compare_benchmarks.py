#!/usr/bin/env python3
"""Compare benchmark sample arrays from JSON files or prefixed JSON log lines."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArmStats:
    """Summary statistics for one benchmark arm."""

    label: str
    path: str
    count: int
    minimum: float
    median: float
    mean: float
    p90: float
    maximum: float
    relative_improvement: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Benchmark arm. Repeat for baseline and candidates; first arm is baseline.",
    )
    parser.add_argument(
        "--baseline",
        help="Baseline label. Defaults to the first --arm label.",
    )
    parser.add_argument(
        "--line-prefix",
        help="Read JSON from the last log line beginning with this prefix.",
    )
    parser.add_argument(
        "--samples-path",
        default="samples",
        help="Dot path to the sample array. Use an empty string when the root is an array.",
    )
    parser.add_argument(
        "--metric-path",
        default="",
        help="Dot path inside each sample. Leave empty for numeric sample arrays.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Discard this many leading samples from every arm.",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Treat larger metric values as better. The default treats intervals as lower-is-better.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
    )
    return parser.parse_args()


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Arm must use LABEL=PATH syntax: {value!r}")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise ValueError(f"Arm must use non-empty LABEL=PATH syntax: {value!r}")
    return label, Path(raw_path)


def get_path(value: Any, path: str) -> Any:
    current = value
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise TypeError(f"Cannot traverse {part!r} through {type(current).__name__}")
    return current


def load_json(path: Path, line_prefix: str | None) -> Any:
    text = path.read_text(encoding="utf-8")
    if line_prefix is None:
        return json.loads(text)
    matches = [line for line in text.splitlines() if line.startswith(line_prefix)]
    if not matches:
        raise ValueError(f"No line beginning with {line_prefix!r} in {path}")
    return json.loads(matches[-1][len(line_prefix) :])


def load_samples(
    path: Path,
    *,
    line_prefix: str | None,
    samples_path: str,
    metric_path: str,
    warmup: int,
) -> list[float]:
    root = load_json(path, line_prefix)
    raw_samples = get_path(root, samples_path)
    if not isinstance(raw_samples, list):
        raise TypeError(f"Samples at {samples_path!r} in {path} are not an array")
    values = [float(get_path(sample, metric_path)) for sample in raw_samples]
    if warmup < 0:
        raise ValueError("--warmup must be non-negative")
    values = values[warmup:]
    if not values:
        raise ValueError(f"No timed samples remain for {path} after {warmup} warmups")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite sample in {path}")
    return values


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(
    label: str,
    path: Path,
    values: list[float],
    *,
    baseline_median: float,
    higher_is_better: bool,
) -> ArmStats:
    median = statistics.median(values)
    if baseline_median == 0 or median == 0:
        relative_improvement = 0.0 if median == baseline_median else math.nan
    elif higher_is_better:
        relative_improvement = median / baseline_median - 1
    else:
        relative_improvement = baseline_median / median - 1
    return ArmStats(
        label=label,
        path=str(path),
        count=len(values),
        minimum=min(values),
        median=median,
        mean=statistics.fmean(values),
        p90=percentile(values, 0.9),
        maximum=max(values),
        relative_improvement=relative_improvement,
    )


def markdown(stats: list[ArmStats]) -> str:
    lines = [
        "| Arm | n | min | median | mean | p90 | max | improvement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        improvement = (
            "n/a"
            if math.isnan(item.relative_improvement)
            else f"{item.relative_improvement * 100:+.2f}%"
        )
        lines.append(
            f"| {item.label} | {item.count} | {item.minimum:.6g} | "
            f"{item.median:.6g} | {item.mean:.6g} | {item.p90:.6g} | "
            f"{item.maximum:.6g} | {improvement} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    arms = [parse_arm(value) for value in args.arm]
    labels = [label for label, _ in arms]
    if len(labels) != len(set(labels)):
        raise ValueError("Arm labels must be unique")
    baseline_label = args.baseline or labels[0]
    if baseline_label not in labels:
        raise ValueError(f"Unknown baseline label: {baseline_label!r}")

    samples = {
        label: load_samples(
            path,
            line_prefix=args.line_prefix,
            samples_path=args.samples_path,
            metric_path=args.metric_path,
            warmup=args.warmup,
        )
        for label, path in arms
    }
    baseline_median = statistics.median(samples[baseline_label])
    stats = [
        summarize(
            label,
            path,
            samples[label],
            baseline_median=baseline_median,
            higher_is_better=args.higher_is_better,
        )
        for label, path in arms
    ]
    if args.format == "json":
        print(json.dumps([asdict(item) for item in stats], indent=2, sort_keys=True))
    else:
        print(markdown(stats))


if __name__ == "__main__":
    main()
