# TraceLens

[TraceLens](https://github.com/AMD-AGI/TraceLens) (AMD, open source) turns a PyTorch profiler trace
into tables: GPU time split into compute, exposed communication, memcpy and idle; kernels by
parent op and stream; ops with shapes and roofline efficiency; per-rank NCCL statistics; and
diffs between two traces. It runs on a CPU host, reads `.json` or `.json.gz` directly, and works
on NVIDIA CUDA traces as well as ROCm traces.

It has no timeline view and reports totals rather than timestamps. Use it for aggregation, and use
[trace-sql.md](trace-sql.md), `crop_trace.py` and the renderer to locate and show a mechanism.

## Table Of Contents

1. Install
2. Capture for TraceLens
3. Reports
4. Roofline
5. TraceLens Agent
6. Cross-checks before trusting a number
7. Python API

## 1. Install

Install into its own environment and pin a commit; the CLIs and sheet names change between releases.

```bash
uv venv .tracelens && uv pip install --python .tracelens \
  "TraceLens[kernel_source] @ git+https://github.com/AMD-AGI/TraceLens.git@994da915a46152bb30a293357a4b604ae6b03338"
export PATH="$PWD/.tracelens/bin:$PATH"
```

An editable install (`pip install -e`) fails with old setuptools; a regular install with current
setuptools works. No GPU is needed. The `kernel_source` extra adds the demangler that
`TraceLens_resolve_kernel_source` uses.

## 2. Capture For TraceLens

Capture requirements depend on which views you need:

| Execution mode | Shapes and roofline | Agent analysis |
|---|---|---|
| Eager | Yes, with `record_shapes=True` | Supported |
| CUDA-graph replay only | Only for kernels outside graphs | Not on one trace; two traces only in reduced `semantic` comparison |
| Graph replay plus an eager-mode trace | Limited | Limited |
| Graph replay plus graph-capture traces | Yes | Supported |

"Agent analysis" refers to the TraceLens Agent (section 5); do not run it on a single graph-replay-only trace.

For CUDA-graph serving (SGLang, vLLM, ATOM, xDiT), the full views need TraceLens's framework
patches, which annotate request packing and trace the graph-capture phase. They live under
`examples/custom_workflows/inference_analysis/`: `sglang_roofline_patches/sglang_<version>/`
(0.5.9 to 0.5.18 at the pinned commit), `sglang_gc_patch/sglang_<version>/` (0.5.18 to 0.5.21),
`vllm_patches/`, `atom_roofline_patches/` and `xdit_patches/`. Apply every patch in the version's
directory with `git apply`. SGLang patches use `python/sglang/...` paths, so apply them from the
SGLang repository root (the directory that contains `python/`); vLLM patches apply from the
parent of the `vllm` package directory.

Patches alone write no capture traces. For SGLang, also:

1. Start the server with `--enable-profile-cuda-graph` (one capture trace per batch size); add
   `--enable-shape-discovery-for-cuda-graph-profile` for wider shape coverage.
2. On 0.5.13 and later, set `SGLANG_GRAPH_BATCH_CAPTURE=1` as well, or no per-batch-size trace is
   written to `capture_traces/`.
3. Pass `shape_discovery=True` and `detailed_annotations=True` in the profile request.

On 0.5.13 and 0.5.14, capture profiling is disabled for the EAGLE/MTP speculative graphs. Pass the
resulting folder to `--capture_folder`. The other frameworks' switches are in the repository's
`docs/how-to/generate-perf-report-pytorch-inference.md`. The bundled Profiling skill and Docker
build scripts target AMD images; on NVIDIA, apply the patches to your own environment.

Without patches, a graph-replay trace still yields the timeline split and the kernel summary, but
most kernel time is attributed to `cudaGraphLaunch` instead of the op that issued it. On one
SGLang trace that was 84% of kernel time.

Keep the diagnostic levels from SKILL.md: a light trace for timing structure, and a separate deep
trace with `record_shapes=True` for TraceLens's op and roofline views. Never compare wall time
between them.

## 3. Reports

Each report writes an Excel workbook (`--output_xlsx_path`) and/or one CSV per sheet
(`--output_csvs_dir`). Read the CSVs with pandas or DuckDB; do not page them through the
conversation.

Per rank, inference (serving or graph workloads):

```bash
TraceLens_generate_perf_report_pytorch_inference \
  --profile_json_path traces/rank0.trace.json.gz --output_csvs_dir tl/rank0 \
  --enable_kernel_summary --short_kernel_study --micro_idle_thresh_us 20 \
  [--capture_folder traces/capture] [--gpu_arch_json_path B300.json] [--enable_pseudo_ops]
```

Per rank, training or eager inference: `TraceLens_generate_perf_report_pytorch` with the same core
flags, plus `--include_call_stack` and `--detect_recompute`.

| Sheet | Answers |
|---|---|
| `gpu_timeline` | Compute, exposed comm, exposed memcpy, micro and macro idle, as ms and % of the GPU span |
| `kernel_summary` | Each kernel by parent op and stream: count, total, mean, min, max, share |
| `ops_summary`, `ops_summary_by_category` | Kernel time per op and per category (GEMM, SDPA, elementwise, ...) |
| `ops_unique_args`, `unified_perf_summary` | Per op and argument set: shapes, dtypes, FLOPs, bytes, TFLOPS/s, TB/s, kernel list |
| `short_kernel_histogram`, `short_kernels_summary` | Launch-bound fragments below `--short_kernel_threshold_us` |
| Per-category perf sheets (`GEMM`, `SDPA_fwd`, `Reduce_fwd`, ...) | Roofline metrics per shape, when shapes exist |

The inference CLI prints a sanity check that the `kernel_launchers` and `unified_perf` tables
account for every GPU event in the tree. Read it; a failed check means those tables do not cover
the trace.

All ranks, collectives:

```bash
TraceLens_generate_multi_rank_collective_report_pytorch \
  --trace_glob 'traces/*.trace.json.gz' --world_size 4 --rank_regex 'TP-(?P<rank>\d+)' \
  --output_csvs_dir tl/collectives --use_multiprocessing
```

Sheets: `nccl_summary_implicit_sync` (per collective type: message size, latency, start and end
skew, algorithm and bus bandwidth), `nccl_implicit_sync` (each collective instance with per-rank
timestamps and waits), `nccl_summary_long` and `nccl_long` (per rank and stream), and
`straggler_summary` (per rank: total wait, times arrived first and last).

Only collectives that carry process-group metadata reach the sync and straggler sheets. NCCL
kernels replayed inside CUDA graphs appear in `nccl_long` as `Unknown_Group` with no collective
name and are left out. On one SGLang trace that was 157 of 218 NCCL kernels per rank, so
`straggler_summary` reported a 17.7 µs wait for a rank that waited about 100 ms. Compare
`num_collectives` in `straggler_summary` with the NCCL kernel count before trusting it, and use
[trace-sql.md](trace-sql.md) section 11 for the rest.

Before and after:

```bash
TraceLens_generate_perf_report_pytorch_inference --profile_json_path after.json.gz \
  --comparison_json_path before.json.gz --output_csvs_dir tl/diff     # adds speedup, delta and diff_stats
TraceLens_compare_perf_reports_pytorch tl/before tl/after --names before after \
  --sheets gpu_timeline ops_summary -o tl/compare.xlsx --output_csvs_dir tl/compare
```

Do not pass `kernel_summary` to the compare CLI for PyTorch reports. It joins on kernel name alone
while the sheet has one row per kernel, parent op and stream, which multiplies rows into a
meaningless cross product. Compare kernels by aggregating each report's `kernel_summary` by name
(or name and stream) in pandas or DuckDB, then joining the two.

Many traces: `TraceLens_trace_index --db campaign.sqlite append --trace-path T --report-dir R`
catalogs reports in SQLite (tables `traces`, `unified_perf_rows`, `op_kernels`, `gemm_perf`, ...)
for questions across a campaign, such as idle percentage per candidate. It stores aggregates, not
events.

Long serving traces: `TraceLens_split_trace TRACE -o out --find-steady-state --num-steps 32`
extracts steady-state iterations. It was written for vLLM; check its output before relying on it
for another framework.

## 4. Roofline

Roofline needs shapes (`record_shapes=True`, or graph-capture traces for graph workloads) and an
arch file with achievable peaks:

```json
{"name": "B300", "mem_bw_gbps": 0, "memory_gb": 0,
 "max_achievable_tflops": {"matrix_fp4": 0, "matrix_fp6": 0, "matrix_fp8": 0, "matrix_int8": 0,
                           "matrix_fp16": 0, "matrix_bf16": 0, "matrix_fp32": 0, "matrix_fp64": 0,
                           "vector_fp16": 0, "vector_bf16": 0, "vector_fp32": 0, "vector_fp64": 0}}
```

A missing key gives that dtype no ceiling, silently. TraceLens ships files only for MI300X, MI325X
and Radeon 8060S. For another GPU, measure one on the target with the bundled suite, which runs on
CUDA and uses `nvidia-smi` for its idle check:

```bash
python -m TraceLens.PerfModel.benchmarking.microbench --device 0 --output B300.json
```

The suite aborts if the GPU is busy (stop the server, or pass `--allow-busy` and treat the result
as a lower bound). It guesses the `name` field from memory size, so a 288 GB GPU is written as
`MI355X`; correct `name` by hand. Pass the file to the CLIs with `--gpu_arch_json_path`.

Performance models exist for standard ops (`aten::mm`, `bmm`, `addmm`, convolution, SDPA, common
elementwise and reductions). Custom extension kernels, such as DeepGEMM, TensorRT-LLM MoE or
FlashInfer attention, have no model unless you add one through `--extension_file`. Check the
`has_perf_model` coverage before relying on the roofline. On one SGLang trace, ops with a model
covered 1% of kernel time; for the rest, follow [kernel-roofline.md](kernel-roofline.md).

## 5. TraceLens Agent

The Agent is a skill, not an MCP server. Its orchestrator runs the reports above once, prepares
per-category data, dispatches subagents in parallel (compute: GEMM, attention, MoE, elementwise,
reduction, normalization, convolution, Triton, generic; system: CPU idle, multi-kernel
communication and memcpy overlap, kernel fusion), validates their outputs, and writes a ranked
`analysis.md`. Findings are grouped as compute kernel, kernel fusion, and system level; each has
Insight, Action and Impact. Compute findings carry an impact score; system findings report impact
as not quantifiable. System analyzers run only when their trigger fires, for example significant
idle time or exposed communication.

Locate and follow it:

```bash
python -c "import TraceLens, pathlib; print(pathlib.Path(TraceLens.__file__).parent / 'Agent/Analysis/skills/analysis-orchestrator/SKILL.md')"
```

Read that `SKILL.md` and its `reference.md` and run them as written. It asks for:

- the trace path(s) and an output directory;
- `standalone` (one trace) or `comparative` (primary versus reference);
- `<platform>`, the name of an arch file;
- the analysis mode: `default` (training and generic eager inference) or `inference` (vLLM,
  SGLang, ATOM); choose `inference` for serving traces, or it uses the training report;
- for `inference`, the execution mode: `eager`, `graph_capture` (needs the capture folder), or
  `graph_replay_only`;
- local or cluster execution.

It finds `<platform>.json` in its installed `Agent/Analysis/utils/arch/` directory, or in the same
subdirectory of each package named in `TL_EXTENSION` (colon-separated importable package names, not
paths). The simplest route is to copy your measured file into the installed `utils/arch/`.

Use it as a source of leads, not as a verdict:

- Do not run it on a single graph-replay-only trace: its orchestrator's `reference.md` says to
  abort standalone `graph_replay_only` runs. Two such traces can be compared, but only in a
  reduced `semantic` mode. Separately, whenever more than 10% of kernel time is attributed to
  `cudaGraphLaunch` rather than to ops (`GRAPH_REPLAY_FRACTION_MAX = 0.10` in
  `utils/deterministic_fallback.py`), it skips the LLM analysis and writes a degraded report with
  only kernel name, time and %E2E. On graph-replay SGLang traces that share is typically far above
  10%. Capture graph-capture traces first (section 2).
- Each finding is a candidate for the ledger only after you locate it in the event data (time,
  rank, stream, launching call) and estimate its exposed time. Impact scores are not Amdahl
  ceilings.
- It analyzes one rank's trace. Cross-rank waits need the collective report or
  [trace-sql.md](trace-sql.md) section 11.

## 6. Cross-Checks Before Trusting A Number

- **Communication classification.** A kernel counts as communication only if its name contains
  `nccl` or `rccl`, or matches the custom pattern `cross_device_reduce`. Framework all-reduce
  kernels (SGLang's `all_reduce_kernel<AllReducePushImpl>`, FlashInfer's
  `oneshotAllreduceFusionKernel`, TensorRT-LLM collectives) are counted as compute. That raises
  `computation_time` and lowers `exposed_comm_time`, and the collective report omits them. On one
  SGLang rank this hid 112 ms of all-reduce, including 96 ms of waiting for other ranks. Sum those
  kernels yourself (`kernel_summary` or trace-sql.md section 11).
- **Collectives inside CUDA graphs.** Even NCCL kernels lose their process group when replayed
  from a graph, so the collective report's skew, wait and straggler sheets cover only the
  collectives issued outside graphs (section 3).
- **Attribution under CUDA graphs.** A parent of `cudaGraphLaunch` says only that the kernel came
  from a graph. Use stream and kernel name, the capture folder, or the correlation query in
  trace-sql.md section 6.
- **Kernel lists per op and shape.** The `kernel_details` column of the unique-args views
  (`ops_unique_args`, `unified_perf_summary`) takes kernel names from the first call in each group
  and pools durations by position. Calls that ran a different kernel with the same kernel count
  are merged under the first call's names, and calls with a different count are skipped with a
  warning. A split dispatch is therefore invisible there; a wide min-to-max range is the only hint.
  Use trace-sql.md section 12.
- **Idle is not a cause.** `macro_idle_time` includes profiler stalls, such as a long
  `cudaGraphLaunch` next to a CUPTI `Activity Buffer Request`. Find the gaps (trace-sql.md section
  3) and look inside them before calling idle time removable.
- **Profiled run.** Every TraceLens number comes from a profiled run. Speed claims still need the
  unprofiled benchmark from SKILL.md section 6.

## 7. Python API

The CLIs cover most needs. For questions they do not answer, load the event tree once and query it
in Python. These entry points were checked against the pinned commit; loading a 535k-event rank
trace takes about 7 s.

```python
from TraceLens import TreePerfAnalyzer, GPUEventAnalyser, TraceDiff, TraceFuse, NcclAnalyser

pa = TreePerfAnalyzer.from_file(path, add_python_func=True)   # also capture_trace_filepath=, enable_pseudo_ops=
pa.get_df_gpu_timeline()                                      # the gpu_timeline sheet
df = pa.get_df_kernel_launchers(include_kernel_details=True, include_call_stack=True)
u = TreePerfAnalyzer.get_df_kernel_launchers_unique_args(df, include_pct=True, group_by_parent_module=True)

evt = pa.tree.get_UID2event(u.iloc[0]["ex_UID"])
pa.tree.traverse_parents_and_print(evt)                  # Python call path that launched it (with_stack=True)
pa.tree.traverse_subtree_and_print(evt, prune_non_gpu=True)   # e.g. every kernel inside one graph replay
GPUEventAnalyser(pa.tree.get_gpu_events(evt)).compute_metrics()   # timeline split for that subtree only

pa.build_nn_module_latency_tree(module_evt)              # GPU time per nn.Module; eager regions only
pa.build_df_perf_metrics(events, include_kernel_details=True)    # FLOPs, bytes, TFLOPS/s; needs shapes

td = TraceDiff(pa_before.tree, pa_after.tree); td.generate_tracediff_report()
td.print_tracediff_report_files(out_dir, prune_non_gpu=True)    # merged tree + diff_stats_df
```

Under CUDA graphs the module view puts nearly all GPU time under "Non-nn.Module GPU Time", and
`build_df_perf_metrics` returns nothing without shapes. Some example notebooks call methods that
no longer exist (for example `TraceDiff.get_corresponding_uid`; use `td.merged_uid_map`); trust
the source over the notebooks.

Tools that live in the repository's `examples/` rather than the package. Fetch only the pinned
commit (a full clone is 1.8 GB):

```bash
git init -q tracelens-src && git -C tracelens-src fetch -q --depth 1 \
  https://github.com/AMD-AGI/TraceLens.git 994da915a46152bb30a293357a4b604ae6b03338 \
  && git -C tracelens-src checkout -q FETCH_HEAD
```

| Tool | Use |
|---|---|
| `examples/custom_workflows/trace_checker.py TRACE --framework pytorch --phase 1` | Pre-flight trace quality: dropped kernels, runtime variability, whether shapes and stacks are present. Run it before trusting any analysis. On live-serving traces its dropped-kernel check can fail on a host-only stretch with no traffic (for example a long prefix before the first request); check for such a stretch before treating it as lost GPU events. |
| `TraceFuse(rank_paths).merge_and_save(out, filter_fn=..., include_pyfunc=False)` | Merge all ranks into one Chrome JSON, optionally filtered (for example to collectives), to render ranks side by side. It needs `args` on every event, so crop without `--strip-args`; it appends `.gz` to the output name (the renderer reads `.gz`) and drops `python_function` events unless `include_pyfunc=True`. Merge two ranks at a time with a `filter_fn`. |
| `examples/custom_workflows/traceMap/main.py --trace1 A --trace2 B --output x.html` | Interactive HTML (Bokeh) of per-kernel latency by launch index plus kernel tables, for two traces side by side. Good for spotting outlier kernels; it has no streams, CPU lanes or wall-clock axis, so it is not a timeline. It loads Bokeh from a CDN. |
| `examples/custom_workflows/roofline_analyzer/main.py report.xlsx cfg.toml --plot-output roofline.png` | Roofline PNG from a report that has shape columns. |
| `TraceLens.PerfModel.perf_model.GEMM.flops_func(M, N, K, bias)`, `.bytes_func(...)`, `SDPA.flops_func(...)` | Speed-of-light FLOPs and bytes from shapes alone, without a trace. |
| `TraceLens_resolve_kernel_source --kernel NAME [--search-path DIR]` | Map a kernel name to its source file, or explain why it has none. |

AMD-only pieces to skip on NVIDIA: the rocprof and pftrace report CLIs, the hipBLASLt tuning UI,
`gemm_dim_eff`'s compute-unit model, and the ROCm Docker images.
