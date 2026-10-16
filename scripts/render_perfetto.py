#!/usr/bin/env python3
"""Render Chrome trace files with the embedded Perfetto web UI."""

from __future__ import annotations

import argparse
import gzip
import http.server
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar
from urllib.parse import quote

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works under `python -I` too

from summarize_trace import iter_trace_events  # noqa: E402

ACCELERATOR_LABEL_MARKERS = (
    "accelerator",
    "cuda",
    "gpu",
    "hip",
    "metal",
    "rocm",
    "xpu",
)
HOST_LABEL_MARKERS = ("cpu", "host")

VIEWER_HTML = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Perfetto trace</title>
<style>html,body,iframe{width:100%;height:100%;margin:0;border:0;overflow:hidden}</style>
</head><body>
<iframe id="perfetto" src="https://ui.perfetto.dev/#!/?mode=embedded&hideSidebar=true"></iframe>
<script>
const iframe=document.getElementById("perfetto");
const traceName=new URLSearchParams(location.search).get("trace");
function ready(){return new Promise(resolve=>{const timer=setInterval(()=>iframe.contentWindow.postMessage("PING","*"),100);function receive(event){if(event.source===iframe.contentWindow&&event.data==="PONG"){clearInterval(timer);window.removeEventListener("message",receive);resolve();}}window.addEventListener("message",receive);});}
(async()=>{await ready();const buffer=await(await fetch("/trace/"+encodeURIComponent(traceName))).arrayBuffer();iframe.contentWindow.postMessage({perfetto:{buffer,title:traceName,fileName:traceName}},"*",[buffer]);})();
</script></body></html>"""


class TraceHandler(http.server.BaseHTTPRequestHandler):
    traces: ClassVar[dict[str, Path]] = {}

    def do_GET(self) -> None:
        if self.path.startswith("/viewer.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(VIEWER_HTML)))
            self.end_headers()
            self.wfile.write(VIEWER_HTML)
            return
        if self.path.startswith("/trace/"):
            from urllib.parse import unquote, urlsplit

            name = unquote(urlsplit(self.path).path.removeprefix("/trace/"))
            trace = self.traces.get(name)
            if trace is None:
                self.send_error(404)
                return
            opener = gzip.open if trace.suffix == ".gz" else open
            with opener(trace, "rb") as source:
                payload = source.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return


def iter_metadata_events(trace: Path) -> Iterator[dict[str, object]]:
    """Yield every metadata event. Kineto writes them at the end of the file, so scan it all."""
    for event in iter_trace_events(trace):
        if event.get("ph") == "M":
            yield event


def discover_default_row_labels(trace: Path) -> tuple[list[str], list[str]]:
    """Return active host and accelerator Perfetto process rows.

    Host rows are CPU-labeled processes named python*; when there are none (SGLang renames its
    processes, e.g. `sglang::scheduler_TP0`), every active CPU-labeled process is used."""
    process_names: dict[object, str] = {}
    process_labels: dict[object, str] = {}
    processes_with_tracks: set[object] = set()

    for event in iter_metadata_events(trace):
        pid = event.get("pid")
        args = event.get("args")
        if pid is None or not isinstance(args, dict):
            continue
        if event.get("name") == "process_name":
            process_name = args.get("name")
            if isinstance(process_name, str) and process_name.strip():
                process_names[pid] = process_name.strip()
        elif event.get("name") == "process_labels":
            process_labels[pid] = " ".join(
                str(value) for value in args.values()
            ).casefold()
        elif event.get("name") == "thread_name":
            processes_with_tracks.add(pid)

    def pid_sort_key(pid: object) -> tuple[int, object]:
        if isinstance(pid, int):
            return (0, pid)
        return (1, str(pid))

    host_python_rows = []
    other_host_rows = []
    accelerator_rows = []
    for pid in sorted(process_names, key=pid_sort_key):
        if pid not in processes_with_tracks:
            continue
        process_name = process_names[pid]
        label_text = process_labels.get(pid, "")
        row_label = f"{process_name} {pid}"
        is_accelerator = any(
            marker in label_text for marker in ACCELERATOR_LABEL_MARKERS
        )
        if is_accelerator:
            accelerator_rows.append(row_label)
        elif any(marker in label_text for marker in HOST_LABEL_MARKERS):
            if Path(process_name).name.casefold().startswith("python"):
                host_python_rows.append(row_label)
            else:
                other_host_rows.append(row_label)

    return host_python_rows or other_host_rows, accelerator_rows


def discover_host_python_row_labels(trace: Path) -> list[str]:
    """Return active host Python process-row labels."""
    host_python_rows, _ = discover_default_row_labels(trace)
    return host_python_rows


def discover_accelerator_row_labels(trace: Path) -> list[str]:
    """Return active accelerator process-row labels."""
    _, accelerator_rows = discover_default_row_labels(trace)
    return accelerator_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--browser", help="Chromium/Chrome executable path")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument(
        "--wait-seconds",
        type=float,
        help="fixed load delay; default scales from 12 to 45 seconds by trace size",
    )
    parser.add_argument(
        "--expand-label",
        action="append",
        default=[],
        help=("additional Perfetto process/rank row label to expand; repeat as needed"),
    )
    parser.add_argument(
        "--no-auto-expand-python",
        action="store_true",
        help="leave host process rows collapsed",
    )
    parser.add_argument(
        "--no-auto-expand-gpu",
        action="store_true",
        help="leave accelerator process rows collapsed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    traces = [trace.resolve() for trace in args.traces]
    for trace in traces:
        if not trace.is_file():
            raise SystemExit(f"trace does not exist: {trace}")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")

    names = {f"{index}-{trace.name}": trace for index, trace in enumerate(traces)}
    TraceHandler.traces = names
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TraceHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    port = server.server_address[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as playwright:
            launch = {
                "headless": True,
                "args": ["--no-sandbox", "--disable-dev-shm-usage"],
            }
            if args.browser:
                launch["executable_path"] = os.path.abspath(args.browser)
            browser = playwright.chromium.launch(**launch)
            for served_name, trace in names.items():
                page = browser.new_page(
                    viewport={"width": args.width, "height": args.height},
                    device_scale_factor=1,
                )
                page.goto(
                    f"http://127.0.0.1:{port}/viewer.html?trace={quote(served_name)}",
                    wait_until="domcontentloaded",
                    timeout=120_000,
                )
                trace_mb = trace.stat().st_size / (1024 * 1024)
                wait_seconds = (
                    args.wait_seconds
                    if args.wait_seconds is not None
                    else min(45.0, max(12.0, 10.0 + trace_mb / 3.0))
                )
                page.wait_for_timeout(wait_seconds * 1000)
                frame = page.frames[-1]
                labels = []
                host_python_rows, accelerator_rows = discover_default_row_labels(trace)
                if not args.no_auto_expand_python:
                    labels.extend(host_python_rows)
                if not args.no_auto_expand_gpu:
                    labels.extend(accelerator_rows)
                labels.extend(args.expand_label)
                expanded = 0
                for label in dict.fromkeys(labels):
                    try:
                        row = frame.get_by_text(label, exact=False).first
                        if row.is_visible(timeout=2_000):
                            row.click(position={"x": 8, "y": 10}, timeout=3_000)
                            page.wait_for_timeout(500)
                            expanded += 1
                        else:
                            print(f"warning: Perfetto row {label!r} not visible")
                    except PlaywrightError:
                        print(f"warning: could not expand Perfetto row {label!r}")
                if not expanded:
                    print(
                        f"warning: no rows expanded in {trace.name}; the image shows "
                        "collapsed processes only. Pass --expand-label with the row names."
                    )
                # The drawer (Trace Doctor import errors, selection details) covers the
                # bottom third; hide it and return to the top rows after expanding.
                try:
                    hide = frame.locator('button[title="Hide panel"]').first
                    if hide.is_visible(timeout=2_000):
                        hide.click(timeout=3_000)
                        page.wait_for_timeout(500)
                except PlaywrightError:
                    print("warning: could not hide the Perfetto bottom panel")
                page.mouse.move(args.width // 2, args.height // 2)
                page.mouse.wheel(0, -100_000)
                page.wait_for_timeout(500)
                output = args.output_dir / f"{trace.stem}.png"
                page.screenshot(path=str(output))
                print(output)
                page.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
