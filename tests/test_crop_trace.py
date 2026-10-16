from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "crop_trace.py"


def event(name: str, cat: str, ts: float, dur: float) -> dict:
    return {"ph": "X", "name": name, "cat": cat, "pid": 1, "tid": 1, "ts": ts, "dur": dur}


class CropTraceTest(unittest.TestCase):
    def crop(self, events: list[dict], *extra: str) -> list[dict]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        source = Path(directory.name) / "trace.json"
        source.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        output = Path(directory.name) / "crop.json"
        # -I: the script must find summarize_trace without relying on the working directory.
        subprocess.run(
            [sys.executable, "-I", str(SCRIPT), str(source), str(output),
             "--start-us", "100", "--duration-us", "100", *extra],
            check=True, capture_output=True, text=True, cwd=directory.name,
        )
        return json.loads(output.read_text(encoding="utf-8"))["traceEvents"]

    def test_metadata_written_first(self) -> None:
        events = [
            event("aten::mm", "cpu_op", 120, 10),
            {"ph": "M", "name": "process_name", "pid": 1, "tid": 0, "args": {"name": "python"}},
        ]
        self.assertEqual([e["ph"] for e in self.crop(events)], ["M", "X"])

    def test_drops_enclosing_python_frames_by_default(self) -> None:
        events = [
            event("model.py(10): forward", "python_function", 0, 1000),   # spans the window
            event("model.py(20): attn", "python_function", 150, 20),      # inside the window
            event("step[DECODE]", "user_annotation", 0, 1000),             # spans, not a frame
        ]
        names = [e["name"] for e in self.crop(events)]
        self.assertEqual(names, ["model.py(20): attn", "step[DECODE]"])
        kept = [e["name"] for e in self.crop(events, "--keep-enclosing-frames")]
        self.assertIn("model.py(10): forward", kept)

    def test_drop_category(self) -> None:
        events = [event("model.py(20): attn", "python_function", 150, 20), event("k", "kernel", 150, 20)]
        self.assertEqual(
            [e["name"] for e in self.crop(events, "--drop-category", "python_function")], ["k"]
        )


if __name__ == "__main__":
    unittest.main()
