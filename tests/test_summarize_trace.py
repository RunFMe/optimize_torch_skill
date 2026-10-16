from __future__ import annotations

import contextlib
import gzip
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from summarize_trace import LAUNCH_RE, merge_intervals, summarize


def event(name: str, cat: str, ts: float, dur: float, tid: int = 1) -> dict:
    return {"ph": "X", "name": name, "cat": cat, "pid": 1, "tid": tid, "ts": ts, "dur": dur}


class SummarizeTraceTest(unittest.TestCase):
    def write_trace(self, events: list[dict], compressed: bool = False) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        payload = json.dumps({"traceEvents": events}).encode()
        if compressed:
            path = Path(temporary_directory.name) / "trace.json.gz"
            path.write_bytes(gzip.compress(payload))
        else:
            path = Path(temporary_directory.name) / "trace.json"
            path.write_bytes(payload)
        return path

    def run_summary(self, path: Path, min_gap_us: float = 100) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summarize(path, top=10, min_gap_us=min_gap_us, stage_pattern=None)
        return output.getvalue()

    def test_launch_regex_counts_graph_launches(self) -> None:
        for name in ("cudaLaunchKernel", "cuLaunchKernelEx", "cudaLaunchKernelExC", "cudaGraphLaunch", "cuGraphLaunch"):
            self.assertTrue(LAUNCH_RE.search(name), name)
        self.assertFalse(LAUNCH_RE.search("cudaMemcpyAsync"))

    def test_graph_launch_closes_launch_gap(self) -> None:
        events = [
            event("cudaLaunchKernel", "cuda_runtime", 0, 10),
            event("cudaGraphLaunch", "cuda_runtime", 150, 10),
            event("cudaLaunchKernel", "cuda_runtime", 300, 10),
        ]
        summary = self.run_summary(self.write_trace(events))
        self.assertIn("Kernel-launch submission events: 3", summary)
        self.assertIn("Launch submission gaps >= 100 us: 2", summary)
        self.assertIn("next=cudaGraphLaunch", summary)

    def test_gpu_gaps_use_union_across_streams(self) -> None:
        events = [
            event("k1", "kernel", 0, 100, tid=7),
            event("k2", "kernel", 50, 100, tid=8),   # overlaps k1 on another stream
            event("k3", "kernel", 400, 50, tid=7),   # 250 us gap after k2 ends
        ]
        summary = self.run_summary(self.write_trace(events, compressed=True))
        self.assertIn("GPU-idle gaps >= 100 us: 1", summary)
        self.assertIn("at ts 150.000 -> 400.000", summary)

    def test_launch_gap_reports_gpu_idle_inside_it(self) -> None:
        events = [
            event("cudaLaunchKernel", "cuda_runtime", 0, 10),
            event("k1", "kernel", 20, 400, tid=7),       # covers 20-420 of the 10-1000 gap
            event("cudaLaunchKernel", "cuda_runtime", 1000, 10),
        ]
        summary = self.run_summary(self.write_trace(events))
        self.assertIn("(GPU idle inside them: 0.590 ms)", summary)
        self.assertIn("0.990 ms gap, GPU idle 0.590 ms", summary)

    def test_stage_ranges_grouped_by_head_and_category(self) -> None:
        events = [
            event("step[DECODE bs=1 ctx=10]", "user_annotation", 0, 100),
            event("step[DECODE bs=1 ctx=11]", "user_annotation", 200, 100),
            event("step[DECODE bs=1 ctx=10]", "gpu_user_annotation", 10, 100, tid=7),
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summarize(self.write_trace(events), top=10, min_gap_us=100, stage_pattern=re.compile(r"^step\["))
        summary = output.getvalue()
        self.assertRegex(summary, r"0\.200 ms\s+0\.100 ms\s+2  step\[DECODE \[user_annotation\]")
        self.assertRegex(summary, r"0\.100 ms\s+0\.100 ms\s+1  step\[DECODE \[gpu_user_annotation\]")

    def test_collectives_split_by_category_with_gpu_window(self) -> None:
        events = [
            event("gloo:all_reduce", "user_annotation", -1000, 1500, tid=2),  # 500 us inside window
            event("ncclDevKernel_AllReduce", "kernel", 0, 100, tid=7),
            event("k2", "kernel", 900, 100, tid=7),
        ]
        summary = self.run_summary(self.write_trace(events))
        self.assertRegex(summary, r"1\.500 ms\s+0\.500 ms\s+1\.500 ms\s+1  user_annotation: gloo:all_reduce")
        self.assertRegex(summary, r"0\.100 ms\s+0\.100 ms\s+0\.100 ms\s+1  kernel: ncclDevKernel_AllReduce")

    def test_merge_intervals(self) -> None:
        self.assertEqual(merge_intervals([(5, 6), (0, 2), (1, 3)]), [(0, 3), (5, 6)])


if __name__ == "__main__":
    unittest.main()
