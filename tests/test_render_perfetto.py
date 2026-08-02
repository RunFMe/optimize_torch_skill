from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from render_perfetto import (
    discover_accelerator_row_labels,
    discover_default_row_labels,
    discover_host_python_row_labels,
)


class DiscoverAcceleratorRowsTest(unittest.TestCase):
    def write_trace(self, events: list[dict[str, object]]) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        trace = Path(temporary_directory.name) / "trace.json"
        trace.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        return trace

    @staticmethod
    def process_metadata(
        pid: int,
        process_name: str,
        process_label: str,
        thread_name: str | None = None,
    ) -> list[dict[str, object]]:
        events = [
            {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "tid": 0,
                "args": {"name": process_name},
            },
            {
                "ph": "M",
                "name": "process_labels",
                "pid": pid,
                "tid": 0,
                "args": {"labels": process_label},
            },
        ]
        if thread_name is not None:
            events.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": pid,
                    "tid": 7,
                    "args": {"name": thread_name},
                }
            )
        return events

    def test_discovers_single_gpu_process(self) -> None:
        events = self.process_metadata(0, "python", "GPU 0", "stream 7")
        trace = self.write_trace(events)

        self.assertEqual(discover_accelerator_row_labels(trace), ["python 0"])

    def test_discovers_multiple_gpu_processes_in_pid_order(self) -> None:
        events = [
            *self.process_metadata(2, "worker", "CUDA GPU 2", "stream 9"),
            *self.process_metadata(0, "worker", "CUDA GPU 0", "stream 7"),
            *self.process_metadata(3, "worker", "CUDA GPU 3"),
            *self.process_metadata(391, "worker", "CPU", "main thread"),
        ]
        trace = self.write_trace(events)

        self.assertEqual(
            discover_accelerator_row_labels(trace),
            ["worker 0", "worker 2"],
        )

    def test_ignores_cpu_only_trace(self) -> None:
        trace = self.write_trace(self.process_metadata(391, "python", "CPU"))

        self.assertEqual(discover_accelerator_row_labels(trace), [])

    def test_does_not_treat_unrelated_low_pid_as_gpu(self) -> None:
        events = [
            *self.process_metadata(0, "python", "CPU"),
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::add",
                "pid": 0,
                "tid": 0,
                "ts": 1,
                "dur": 1,
            },
        ]
        trace = self.write_trace(events)

        self.assertEqual(discover_accelerator_row_labels(trace), [])

    def test_discovers_host_python_process(self) -> None:
        events = self.process_metadata(
            391,
            "/usr/bin/python3.12",
            "CPU",
            "main thread",
        )
        trace = self.write_trace(events)

        self.assertEqual(
            discover_host_python_row_labels(trace),
            ["/usr/bin/python3.12 391"],
        )

    def test_discovers_multiple_active_host_python_processes(self) -> None:
        events = [
            *self.process_metadata(900, "python", "CPU", "main thread"),
            *self.process_metadata(700, "python3", "host", "worker thread"),
            *self.process_metadata(800, "server", "CPU", "main thread"),
        ]
        trace = self.write_trace(events)

        self.assertEqual(
            discover_host_python_row_labels(trace),
            ["python3 700", "python 900"],
        )

    def test_ignores_inactive_host_python_process(self) -> None:
        trace = self.write_trace(self.process_metadata(391, "python", "CPU"))

        self.assertEqual(discover_host_python_row_labels(trace), [])

    def test_separates_host_python_and_gpu_rows(self) -> None:
        events = [
            *self.process_metadata(391, "python", "CPU", "main thread"),
            *self.process_metadata(0, "python", "GPU 0", "stream 7"),
        ]
        trace = self.write_trace(events)

        self.assertEqual(
            discover_default_row_labels(trace),
            (["python 391"], ["python 0"]),
        )


if __name__ == "__main__":
    unittest.main()
