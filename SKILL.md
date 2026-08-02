---
name: optimize-torch-inference
description: Profile and optimize PyTorch inference workloads, models, applications, and services for the user's chosen objective, including latency, throughput, cost, memory, or startup, using controlled benchmarks, profiler traces, and task-quality validation. Use for slow inference, low accelerator utilization, batching or tail-latency problems, CPU launch bubbles, excessive kernels, torch.compile or CUDA Graphs, custom CUDA/Triton/CuTe kernels, precision or quantization experiments, operator/backend tuning, host-device transfers, distributed communication, memory/offload behavior, inference pipelines, multi-candidate optimization campaigns, or requests to prove a speedup without silently reducing required work or output quality.
---

# Optimize Torch Inference

Drive optimization with measured critical-path evidence. Treat traces as diagnostic instruments and unprofiled target-equivalent runs as speed proof.

## Non-Negotiable Rules

1. Freeze the target contract before ranking candidates: primary objective and metric, constraints, workload mix, work budget, output mode, exact hardware and topology, an executable access path, process lifecycle, cache reuse, and quality gate. If the objective, target hardware, or access path is missing and different choices would change the campaign, ask the user one concise question covering them. Never default silently to throughput or the currently visible device. If the contract changes, rerank the ledger.
2. Match the first experiments to the objective. For throughput, measure saturation and batching early; for online latency, preserve the concurrency and tail-latency contract; for startup, measure a clean lifecycle; for memory or cost, optimize the corresponding peak or resource-normalized metric.
3. Use reduced-step or repeated-shape runs only for screening. Never promote a default, publish a speed claim, or reject a fixed-cost candidate solely from a proxy whose stage shares differ materially from production.
4. Inspect both raw Chrome JSON and rendered Perfetto images. Do not approve an opportunity from aggregate tables alone.
5. Capture traces separately from timed runs. Profiler overhead can differ radically between eager, compiled, and CUDA-graph paths.
6. Measure exposed critical-path time. Do not count CPU waits hidden under GPU work, rank rendezvous as transfer time, deferred work outside the timer, or overlapping kernel buckets twice.
7. Keep inputs, shapes, batching, model state, precision, execution algorithm, work budget, and output contracts fixed unless the user explicitly accepts a behavior-changing experiment.
8. Prove each winner in isolation, then remeasure the cumulative candidate. Never add isolated percentages. Refresh the cumulative run after every two accepted changes or sooner when they touch the same stage.
9. Confirm that an optimization is active and beneficial on the target topology. A result from a path disabled under the production launcher does not support the claimed objective.
10. Keep benchmark harnesses, traces, screenshots, generated outputs, and large artifacts outside the implementation diff unless explicitly requested.
11. Maintain the decision ledger during the campaign, including rejected and deferred experiments. Do not reconstruct it from memory at delivery.
12. Use independent review of expensive experiment designs and implemented winners. Fix material findings and rerun proof.

Read [trace-analysis.md](references/trace-analysis.md) before interpreting a trace. Read [optimization-catalog.md](references/optimization-catalog.md) when ranking candidates. Read [benchmark-quality.md](references/benchmark-quality.md) before making a speed or parity claim. For a campaign with multiple hypotheses, compilation, lower precision, custom kernels, distributed execution, or host pipelining, also read [campaign-playbook.md](references/campaign-playbook.md).

## 1. Establish The Contract

Inventory the following from source, runtime configuration, available measurements, and the target machine:

- Workload classes and their target frequency: shapes, batch sizes, sequence lengths, optional paths, and long-tail cases.
- Exact model artifacts and weights, preprocessing/tokenization, adapters, precision, execution algorithm, work budget, backend dispatch, and output settings.
- Target processor or accelerator, device count and capability, interconnect, driver/runtime, Torch build, native extensions, and memory limits.
- Distributed topology and process lifecycle: FSDP, TP, SP, collectives, rank dispatch, model loading, offload, and cache behavior.
- End-to-end boundaries: admission or invocation, input acquisition and deserialization, preprocessing, host-to-device transfer, model phases, communication, device-to-host transfer, postprocessing, serialization or persistence, and result delivery.
- The primary objective and metric, plus constraints on the other dimensions: single-invocation or tail latency, steady-state throughput, queue time, cost, peak memory, and startup.

Obtain the objective from the user, an explicit SLO, or the application contract. If latency versus
throughput, warm versus cold, or model-only versus service-wide optimization remains ambiguous and
would change experiment order, ask before ranking opportunities.

Obtain the exact target hardware and topology from the user or deployment configuration, including
accelerator model/capability, count, memory limit, interconnect, CPU/NUMA/storage constraints, and
runtime environment. Establish how to execute there—local shell, container, remote host, scheduler,
or another user-provided route—and treat access as unverified until a runtime probe succeeds. If no
access path is available, ask the user for one before ranking hardware-dependent experiments. Verify
access with read-only runtime probes and a minimal device operation in the environment where
benchmarks will run. If the target is unreachable, continue only with portable analysis or
explicitly labeled proxy screening; do not claim, reject, default-enable, or tune
architecture-specific work from proxy performance as though it were target proof.

Verify claims against target behavior. Do not optimize an offload, cache miss, fallback, or startup path that representative workloads do not execute.

Create an invariant manifest for baseline and candidate. Diff it before accepting results. Include all quality-sensitive settings, not only application configuration.

Write a short campaign card before the first experiment:

- Primary metric and optimization direction, secondary constraints, and boundary: latency, throughput, cost, memory, or startup; model-only or full pipeline.
- Production workload distribution and work budget, plus the faster screening and confirmation proxies.
- Device topology, launcher, workers/ranks/replicas, hardware access path and verified status, process lifetime, warm/cold cache assumptions, and expected work per fixed-cost execution unit.
- Required outputs, persistence/resume behavior, and the task-quality metrics that can authorize numerical changes.

Use the execution unit that independently pays fixed costs—process, worker, rank, replica, or device—for compile amortization and shape reuse. Do not amortize against global work when each execution unit pays separately.

## 2. Build A Workload-Shaped Benchmark

Run the actual application or service path when feasible. Otherwise reproduce its input pipeline, model lifecycle, distributed launch, preprocessing, execution, and result path.

Use two benchmark tiers:

- **Screen:** cheap, explicitly non-authoritative runs that can reject crashes, large regressions, impossible dispatch, low ceilings, or obvious numerical failures. Record how their stage shares differ from production.
- **Proof:** target-equivalent or defensibly scaled runs used for acceptance, defaults, and published claims.

Proof runs require:

- Identical inputs, seeds, model artifacts, environment, accelerator set, and machine-load conditions.
- Verified access to the claimed accelerator model, capability, device count, memory/resource limits,
  and topology. A different accelerator may be a screening proxy but not target speed proof.
- At least three warmups, or enough to finish compilation, graph capture, caches, allocator settling, and model residency transitions.
- At least five unprofiled timed iterations; use more when noise is comparable to the expected gain.
- Median, minimum, p90, per-stage time, peak allocated/reserved device memory, host memory, and failure count.
- An objective-matched arrival pattern: isolated requests for latency, continuously fed work for throughput, and clean process launches for startup.
- Explicit time boundaries. When using service logs, distinguish completion-in-window from invocations fully contained in the window.
- Sequential or interleaved arms on an otherwise quiescent host. Do not run competing arms concurrently unless CPU, memory, storage, and accelerators are explicitly partitioned.
- For queued or asynchronous work, timing through the final drain and failure propagation. Moving work past the timer is not a speedup.

Instrument coarse stages with `torch.profiler.record_function`, including admission or invocation, input acquisition, preprocessing, H2D, major model phases or repeated iterations, owned collectives, D2H, postprocessing, serialization or persistence, and result delivery. Avoid invasive per-operation instrumentation in proof runs.

Use the bundled comparator when benchmark outputs contain machine-readable sample arrays. It handles
warmup exclusion, prefixed JSON log lines, median/min/mean/p90, and relative improvement:

```bash
uv run python "$SKILL_DIR/scripts/compare_benchmarks.py" \
  --arm baseline=baseline.log --arm candidate=candidate.log \
  --line-prefix 'BENCH_RESULT=' --samples-path measurements \
  --metric-path seconds --warmup 3
```

## 3. Capture Diagnostic Traces

Run at least three complete workload-shaped warmup invocations before entering the profiler, then
capture the next representative invocation on every rank. Profiler-scheduled warmup steps are not a
substitute when they alter lifecycle or dispatch behavior. Export Chrome JSON with
`export_chrome_trace`.

Use two diagnostic levels when needed:

- Light trace: host and applicable accelerator activities, coarse ranges, no stacks or shapes.
- Deep trace: shapes, memory, stacks, or module hierarchy only for a narrow workload. Never compare its wall time with a differently instrumented path.

Run the bundled analyzer:

```bash
uv run python "$SKILL_DIR/scripts/summarize_trace.py" trace.json --top 25 --min-gap-us 100
```

Use the timestamps from the summary and stage ranges to make small, reviewable trace crops:

```bash
uv run python "$SKILL_DIR/scripts/crop_trace.py" trace.json opportunity.json \
  --start-us 123456000 --duration-us 50000 --strip-args
```

Render each crop in Perfetto and inspect the pixels directly. The bundled renderer uses
the embedded Perfetto UI and needs network access plus Chromium:

```bash
uv run --with playwright python "$SKILL_DIR/scripts/render_perfetto.py" \
  opportunity.json --output-dir evidence --browser /path/to/chrome
```

Omit `--browser` when Playwright Chromium is installed. Host Python and accelerator process rows
are expanded automatically so their CPU threads and accelerator streams appear together. Use
`--no-auto-expand-python` or `--no-auto-expand-gpu` only when the corresponding collapsed view is
intentional. Use `--expand-label` for additional worker or rank rows that Perfetto initially
collapses. Produce, at minimum:

- One full critical-path image with host, accelerator, communication, and stage lanes visible.
- One tightly zoomed image for every proposed opportunity, with the time scale and surrounding dependency visible.
- Matched before/after images after implementation.

Screenshots must show why time is removable, not merely that a long bar exists.
When presenting a trace-guided plan, place or link each zoomed screenshot next to the corresponding
opportunity and caption the exposed interval and dependency that makes it actionable.

## 4. Build An Opportunity Ledger

For every candidate, record:

| Field | Required evidence |
|---|---|
| Bottleneck | Raw JSON statistics plus rendered trace location |
| Root cause | Dependency explanation, not only operator name |
| Exposed time | Time on the actual critical path |
| Roofline | FLOPs/bytes/payload, achieved rate, and realistic hardware or empirical ceiling |
| Proposed change | Exact execution behavior to remove or overlap |
| Theoretical maximum | Amdahl-limited end-to-end ceiling |
| Expected gain | Conservative range after overheads |
| Target compatibility | Production topology, launcher, shapes, outputs, and lifecycle where the change is active |
| Hardware evidence | Verified target access, exact device/runtime identity, and any proxy-to-target differences |
| Fixed costs | Compile, autotune, packing, graph capture, warmup, cache misses, and break-even at the execution unit that pays them |
| Risks | Numerical, memory, lifecycle, deadlock, portability, compile, or maintenance |
| Acceptance | Benchmark, parity, soak, and fallback requirements |
| Result | `ship`, `reject`, `defer`, or `retest`, with isolated and cumulative A/B evidence |

Prioritize expected objective value, not inclusive operator time. Estimate `exposed share × plausible removable fraction` before implementation. Agree on an acceptance threshold appropriate to measurement noise, implementation cost, and risk; do not impose a universal percentage. Screen the cheapest disproof first: unsupported dispatch, insufficient ceiling, compile explosion, memory excess, or unacceptable numerical drift.

## 5. Select And Implement One Candidate

Choose the highest-value candidate that has:

- A visible critical-path mechanism.
- A defensible ceiling above the acceptance threshold.
- A target-compatible fallback or kill switch when lifecycle risk is material.
- A validation plan that can detect silent corruption.

Start with an exact-shape microbenchmark for custom kernels and collectives. Then implement the smallest runtime change that tests the hypothesis.

Use three gates for expensive candidates:

1. **Ceiling:** show that production stage share and a realistic stage speedup can clear the threshold.
2. **Mechanism:** prove exact dispatch, numerical behavior, compile cost, and memory on target shapes. A compiler API call is not proof of fusion or backend substitution.
3. **Integration:** run matched A/B/A or randomized proof runs through the real application boundary and target launcher.

For custom CUDA extensions, when applicable:

- Benchmark and validate on every hardware architecture claimed by the optimized path; derive
  architecture guards and tuning from those devices, not from an unverified proxy.
- Use `CUDAGuard` and PyTorch's current CUDA stream.
- Honor allocator stream semantics and avoid implicit legacy-stream launches.
- Validate real, fake/meta, functionalization, aliasing, dtype, layout, device, and graph-capture contracts.
- Compare against a trusted reference on target and edge-case shapes before running the complete model.

For compilation or CUDA Graphs:

- Apply adapters and finalize model state before capture/compile.
- Enumerate supported shapes and inspect graph breaks, recompiles, generated kernels, guards, and pointer stability.
- Validate unload/offload/model-swap behavior. Ordinary `.to(cpu)` changes storage and invalidates captured pointers.
- Require all-rank enablement consensus in distributed services and fail open to a tested eager path.
- Count graphs, recompiles, graph breaks, and cold seconds before timing steady state. Calculate break-even per process and per shape bucket.
- Test dynamic shapes against a representative ordered shape trace. Dynamic compilation can reduce graph count while slowing kernels or retaining guards and recompiles.

For precision and quantization:

- Map which operators actually receive each dtype, including autocast-disabled islands and FP32 accumulators. Do not infer coverage from a top-level autocast setting.
- Benchmark a ladder rather than conflating modes: conservative FP32, TF32-eligible FP32 GEMMs, BF16/FP16 autocast, and narrow FP8 candidates.
- Inventory exact GEMM shapes, frequencies, and epilogues first. For FP8, start with the narrow sites whose end-to-end ceiling and exact-shape benchmarks justify packing and scaling overhead before broad dynamic module replacement.
- Define task-quality metrics and difficult slices before making a numerically different mode a default.

## 6. Prove Performance And Quality

Run matched unprofiled baseline and candidate benchmarks. Randomize or interleave order if thermal or shared-system drift is possible.

Report:

- Absolute and percentage deltas for e2e and affected stages.
- Median, minimum, p90, sample count, and memory.
- Isolated candidate effect and fresh cumulative effect.
- The primary objective metric under its actual operating condition: for example tail latency at the required concurrency, continuously fed throughput, clean-start latency, peak memory, or resource-normalized cost.
- Separate diagnostic trace deltas without using profiled wall time as the speed claim.
- Cold cost, warm cost, break-even in batches or work items per execution unit, graph/recompile counts, and whether caches survive the real lifecycle.
- Full-pipeline time through output persistence and queue drain when the product requires those outputs.

Run contract and task-quality checks appropriate to the output. Audit the final diff and runtime configuration for hidden shortcuts such as less algorithmic work, smaller inputs or batches, shorter sequences, lower precision, changed weights or algorithms, disabled components, relaxed thresholds, reduced output fidelity, or skipped validation and safety work.

Classify the result:

- `Exact-intent`: scheduling, transfer, communication, or mathematically equivalent compute with accepted numerical tolerances.
- `Numerically different`: library/version/kernel-order changes with bounded drift.
- `Behavior-changing`: quantization, lower precision, reduced work budget, distillation, pruning, changed thresholds, or reduced input/output fidelity. Report this contribution separately.

## 7. Review And Iterate

Give an independent reviewer the patch, target configuration, benchmark table, raw traces, rendered evidence, memory data, and quality artifacts. Ask the reviewer to challenge:

- Profiler overhead and incomparable instrumentation.
- Double counting and waits hidden by overlap.
- Rank-arrival skew mislabeled as collective cost.
- Cold-start, cache, workload-mix, hardware, or configuration bias.
- Silent dispatch changes after dependency upgrades.
- Numerical corruption, quality shortcuts, and incomplete optional-path coverage.
- CUDA Graph pointer lifetime, model swaps, FSDP collective order, and deadlocks.
- Unsupported hardware and missing eager fallbacks.
- Proxy-to-production extrapolation, target-topology applicability, timer relocation, undrained asynchronous tails, and compile break-even computed at the wrong process scope.

Use a requested external reviewer, such as Claude from bash, when available. Do not block indefinitely on quota or tool failure; document it and continue with independent review.

For a costly or risky hypothesis, review the experiment design before spending the full GPU run. Send a compact evidence packet with the campaign card, ledger row, commands, raw sample paths, invariant diff, and proposed acceptance threshold. Do not make reviewers rediscover the target contract from chat history.

After review:

1. Fix material findings.
2. Rerun focused correctness tests and the matched benchmark.
3. Capture matched before/after trace images.
4. Reprofile the new cumulative winner and choose the next opportunity from the new trace.

## 8. Stop Deliberately

Stop when any of these holds:

- Two consecutive well-chosen candidates miss the agreed threshold.
- Remaining exposed time is dominated by required compute, communication, or I/O already near its empirical roofline.
- The next gain requires approximate-quality model changes not authorized by the user.
- Expected benefit is smaller than correctness, memory, portability, or maintenance risk.

Before delivery, remove benchmark-only code from the implementation diff if requested, preserve local artifacts, run smoke tests for every touched optional path, and provide a table of shipped, disabled, and rejected experiments with non-additive speedup attribution.

Reconcile the final code, CLI defaults, documentation, benchmark harness, and decision ledger. Explicitly list live opportunities that were deferred rather than disproven. Do not call a campaign complete while a measured winner is unavailable under the target topology or while a promising candidate was screened only with a non-representative proxy.
