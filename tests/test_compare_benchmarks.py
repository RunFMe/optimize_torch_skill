from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from compare_benchmarks import load_samples, markdown, percentile, summarize


class CompareBenchmarksTest(unittest.TestCase):
    def write_file(self, contents: str) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "result.txt"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_loads_numeric_json_array_after_warmup(self) -> None:
        path = self.write_file(json.dumps([10, 9, 8, 7]))

        samples = load_samples(
            path,
            line_prefix=None,
            samples_path="",
            metric_path="",
            warmup=2,
        )

        self.assertEqual(samples, [8.0, 7.0])

    def test_loads_nested_samples_from_last_prefixed_line(self) -> None:
        prefix = "BENCH_RESULT="
        path = self.write_file(
            "noise\n"
            + prefix
            + json.dumps({"measurements": [{"timing": {"seconds": 3}}]})
            + "\nmore noise\n"
            + prefix
            + json.dumps(
                {
                    "measurements": [
                        {"timing": {"seconds": 2}},
                        {"timing": {"seconds": 1}},
                    ]
                }
            )
        )

        samples = load_samples(
            path,
            line_prefix=prefix,
            samples_path="measurements",
            metric_path="timing.seconds",
            warmup=0,
        )

        self.assertEqual(samples, [2.0, 1.0])

    def test_summarizes_lower_is_better_relative_improvement(self) -> None:
        stats = summarize(
            "candidate",
            Path("candidate.json"),
            [8.0, 10.0, 9.0],
            baseline_median=12.0,
            higher_is_better=False,
        )

        self.assertEqual(stats.median, 9.0)
        self.assertAlmostEqual(stats.relative_improvement, 1 / 3)
        self.assertAlmostEqual(stats.p90, percentile([8.0, 10.0, 9.0], 0.9))

    def test_summarizes_higher_is_better_relative_improvement(self) -> None:
        stats = summarize(
            "candidate",
            Path("candidate.json"),
            [120.0],
            baseline_median=100.0,
            higher_is_better=True,
        )

        self.assertAlmostEqual(stats.relative_improvement, 0.2)
        self.assertIn("+20.00%", markdown([stats]))

    def test_zero_baseline_reports_nan_for_different_candidate(self) -> None:
        stats = summarize(
            "candidate",
            Path("candidate.json"),
            [1.0],
            baseline_median=0.0,
            higher_is_better=True,
        )

        self.assertTrue(math.isnan(stats.relative_improvement))


if __name__ == "__main__":
    unittest.main()
