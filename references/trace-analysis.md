# Trace Analysis

Use raw events for measurement and rendered timelines for dependency reasoning. A top-operator table cannot show whether work overlaps, whether a collective waits for peers, or whether the GPU was actually idle.

## Table Of Contents

1. Capture integrity
2. JSON analysis
3. Rendered timeline inspection
4. CPU and launch bubbles
5. Synchronization and transfers
6. Distributed communication
7. Kernel fragmentation and fusion
8. Roofline and ceilings
9. Evidence images

## 1. Capture Integrity

- Capture after warmup, compilation, graph capture, cache population, and model residency stabilization.
- Capture all ranks for distributed jobs and align their wall-clock stage markers.
- Record the profiler options with the artifact. `record_shapes`, stacks, memory profiling, and Python tracing can alter eager and graph paths differently.
- Exclude profiler runs from speed proofs.
- Confirm the trace covers one complete measured invocation and is not truncated during asynchronous accelerator work.
- Preserve the unmodified Chrome JSON. Generate derived summaries and screenshots separately.

## 2. JSON Analysis

Measure at least:

- Coarse `record_function` stage durations.
- GPU busy union, not only summed kernel time.
- Largest global accelerator-idle gaps inside the measured invocation.
- Top kernels by total and maximum duration.
- Kernel launch count and launch submission gaps.
- CUDA synchronization APIs, scalar reads, and synchronous copies.
- H2D/D2H/device copy volume, duration, pinned/pageable behavior, and stream.
- Allocation/free churn and repeated layout/dtype conversions.
- Collective counts, per-rank entry times, exposed duration, and compute overlap.
- Kernel families relevant to the workload, such as GEMM, convolution, attention, reductions, normalization, indexing, elementwise, quantization, and copies.

Treat summed GPU duration as work, not wall time. Concurrent streams make summed durations exceed the critical path.

Where each measurement comes from:

| Measurement | First source | Then |
|---|---|---|
| Stage durations | [trace-sql.md](trace-sql.md) section 8, split by `cat` | `summarize_trace.py --stage-regex`, whose totals add CPU and GPU copies of each range together |
| GPU busy union, idle gaps | `summarize_trace.py` | trace-sql.md section 3; TraceLens `gpu_timeline` for the compute/comm/memcpy/idle split |
| Top kernels | TraceLens `kernel_summary` (by parent op and stream) | trace-sql.md section 7 |
| Launch count and gaps | `summarize_trace.py` | trace-sql.md section 4 |
| Syncs, scalar reads, copies | `summarize_trace.py` | trace-sql.md section 9 |
| Allocation churn | SQL on `aten::empty*`, `cudaMalloc`/`cudaFree` and allocator events; a `profile_memory=True` trace | trace-sql.md section 5 on the window |
| Layout and dtype conversions | TraceLens `ops_summary` (`aten::copy_`, `aten::to`, `aten::contiguous`) | trace-sql.md section 6 to find who calls them |
| Collectives | TraceLens collective report for NCCL | trace-sql.md sections 10 and 11 for host and in-graph collectives |
| Kernel families | TraceLens `ops_summary_by_category` | `kernel_summary` by name when graphs hide the ops |

## 3. Rendered Timeline Inspection

Open the Chrome JSON in Perfetto and inspect:

1. Full invocation: identify stage ordering, accelerator occupancy, host-only tails, and model transitions.
2. Each repeated phase, iteration, or decode step: compare launch cadence, graph replays, collectives, and inter-step gaps.
3. CPU launch thread: determine what delays the next GPU submission.
4. GPU streams: distinguish compute, communication, memcpy, and allocator activity.
5. All ranks: compare entry into the same first collective and later steady collectives.
6. Before/after: use the same zoom, lanes, and time scale.

Inspect the rendered pixels or screenshot directly. Do not rely only on a textual export of Perfetto.

## 4. CPU And Launch Bubbles

A CPU gap is removable only when it delays the next critical GPU work.

- Confirm the GPU is globally idle during the gap.
- Trace backward from the next launch to Python, dispatcher, model loading, queue delivery, or a scalar branch.
- Compare the same interval after warmup; first-use library initialization is not steady state.
- Count repeated small gaps. CUDA Graphs can help a stable, launch-bound loop even when kernel count and GPU compute stay unchanged.
- For iterative samplers or decoders, align several consecutive steps and separate invariant
  per-step scalar reads, state-update work, random generation, and model launches. A bubble
  visible at batch one may disappear after throughput batching.
- Do not claim CUPTI command-buffer waits as application latency without an unprofiled A/B. They may be profiler backpressure.

## 5. Synchronization And Transfers

Investigate:

- `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaEventSynchronize`.
- `aten::item`, `_local_scalar_dense`, Python truth tests on CUDA tensors, and synchronous `.cpu()`/`.to()` calls.
- Device-wide cleanup or allocator synchronization between stable resident models.
- Pageable H2D/D2H transfers and copies that could use pinned buffers plus `non_blocking=True`.

For every synchronization, ask what useful GPU work already overlaps it. A 150 ms CPU wait with the GPU busy and a 50 us next-launch gap has a sub-millisecond latency ceiling.

When removing diagnostic CUDA-dependent branches, preserve error visibility by accumulating flags on
device and reading once outside the measured loop. Do not delete training or correctness diagnostics
for an unproven inference speedup.

## 6. Distributed Communication

Do not equate a collective bar with transfer time.

1. Compare the entry timestamp on every rank.
2. Measure rank-arrival spread.
3. Compare the first collective with later shape-identical collectives after ranks synchronize.
4. Compute payload bandwidth from the actual shard/gather bytes and transfer-only estimate.
5. Run a same-payload, same-topology collective microbenchmark.
6. Measure overlap as the intersection of collective and useful compute intervals.

If the first all-gather is 300 ms but steady gathers are 2 ms, fix rank dispatch or preprocessing skew before tuning NCCL. Adding a barrier merely moves the wait.

For NCCL collectives issued outside CUDA graphs, the TraceLens collective report gives per-rank
start skew, waits, stragglers and bandwidth. Collectives replayed inside graphs lose their process
group and drop out of those sheets, and framework all-reduce kernels are neither reported as
collectives nor counted as communication. Align both across ranks with
[trace-sql.md](trace-sql.md) section 11. A rank
that waits much longer than the others arrives early; look for what delays its peers, often host
work such as request broadcast or scheduling that differs by rank.

Treat communication-library environment changes as topology-specific experiments. Validate startup, every collective order, and long soaks. A hang before model loading can be a node interconnect or fabric problem; test an unchanged known-good runtime on the same node before blaming code.

## 7. Kernel Fragmentation And Fusion

Look for repeated chains around dominant operations such as GEMMs, convolutions, attention, reductions, or gathers:

- residual add, RMSNorm/LayerNorm, modulation, gate, cast, quantize, transpose, and copy;
- expanded or broadcast state materialized across a large dimension;
- Q/K normalization, RoPE, layout conversion, and attention quantization;
- GELU/SILU followed by activation quantization;
- Python tensor-subclass dispatch around invariant FP8 weights.

Prefer fusion that removes intermediate memory traffic. Replacing one isolated norm kernel often loses to framework kernels and leaves all adjacent reads/writes intact.

Use exact target shapes in microbenchmarks. Require the custom kernel to exceed the existing chain enough to survive dispatch overhead and deliver an e2e gain.

## 8. Roofline And Ceilings

For each candidate:

- Compute arithmetic intensity and compare achieved FLOP/s with a device roofline for GEMMs or custom math.
- Compute payload bytes and achieved bandwidth for collectives and copies.
- Use Nsight Compute for representative kernels before replacing vendor kernels already tuned for the target architecture.
- Use an empirical microbenchmark when peak marketing bandwidth is not attainable by the exact topology or payload.
- Calculate the Amdahl ceiling from exposed wall time, not inclusive totals.

Example:

```text
exposed stage = 0.40 s of a 10.0 s invocation
plausible stage reduction = 50%
best realistic e2e gain = 0.20 s = 2%
```

Reject invasive work when its realistic ceiling is below the shipping threshold.

## 9. Evidence Images

For each opportunity, preserve:

- Overview image showing where the region sits on the critical path.
- Zoomed image showing the exact bubble, synchronization, fragmentation, or non-overlapped communication.
- Before/after images at matching scale.
- Caption with trace filename, worker or rank, workload shape, stage, timestamps, and measured exposed time.
- For asynchronous input or output work, an image covering queue fill, steady overlap, backpressure,
  and final drain; a middle steady-state crop alone can hide deferred tails.

Avoid screenshots that show only a selected duration bar without neighboring CPU/GPU dependencies.

Use crops rather than loading multi-gigabyte traces into the browser. Crop by absolute trace
microseconds and retain metadata plus all CPU, GPU, runtime, driver, annotation, and communication
events in the window. Strip large argument payloads only from the derived crop, never the source.

The bundled renderer is intentionally a convenience, not an analysis oracle. Perfetto row order and
labels vary between Torch versions, so inspect every generated image and rerender or expand rows when
the dependency is not legible. Keep the raw source trace, crop command, and screenshot caption together.
