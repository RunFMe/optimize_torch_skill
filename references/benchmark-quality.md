# Benchmark And Quality Validation

## Table Of Contents

1. Screening versus proof
2. Matched experiment design
3. Profiler separation
4. Latency and throughput
5. Numerical validation
6. Task behavior and quality
7. Target-environment validation
8. Reporting template

## 1. Screening Versus Proof

Use cheap screens to eliminate impossible dispatch, crashes, large regressions, low ceilings, or
obvious numerical failures. Screens may use fewer iterations, repeated shapes, cached inputs, omitted
output writing, and one to three samples when clearly labeled. They do not authorize defaults or
published speed claims.

Confirm every winner with target-equivalent proof. When a proxy changes the relative cost of model,
host, or fixed startup work, report the proxy stage share, production stage share, and Amdahl-rescaled
ceiling. A reduced-step iterative workload usually exaggerates invariant preprocessing and writing;
a short run exaggerates compilation and autotuning.

## 2. Matched Experiment Design

Hold constant:

- Physical target accelerator and topology.
- Driver and accelerator runtime, clocks/power policy, process or container limits, and competing work.
- Input payloads, preprocessing, invocation options, seeds, model artifacts, adapters, precision, execution algorithm, work budget, and output settings.
- Warmup count and lifecycle state.
- Benchmark orchestration and timing boundaries.

Record exact runtime artifact or container, commit, and dependency versions. A Torch or kernel-library upgrade is part of the candidate and can change both speed and numerical behavior.

Before timing, verify that the benchmark process can execute on the requested hardware and record
the observed device model, architecture/capability, memory, device count, topology, driver/runtime,
and resource limits. Do not substitute the locally visible accelerator without telling the user.
If only proxy hardware is available, label its results as screening and reserve performance
acceptance/rejection for the target unless a validated transfer model exists.

Use at least three warmups and five timed iterations as a floor. Increase samples when the expected gain is under 3%, startup state varies, or host noise is visible. Report all samples or preserve the raw result file.

Run comparison arms sequentially or interleave them on a quiescent host. Do not launch arms
concurrently on shared CPU, memory, storage, or accelerators unless resources are explicitly
partitioned. Cross-arm interference invalidates small deltas and can make the least parallel arm look
artificially slow.

For gains below 3%, prefer ten or more timed samples and A/B/A, B/A/B, or randomized order. Compare
the spread between repeated same-arm runs with the claimed effect.

## 3. Profiler Separation

Never use profiled wall time as the primary speed proof.

Profiler features can penalize eager execution more than CUDA Graph replay because eager exposes every operation while replay hides CPU dispatch. `record_shapes` may retain tensors and amplify the difference.

Use:

- Unprofiled matched runs for latency and throughput.
- Light matched traces for structural before/after evidence.
- Deep traces for diagnosis only.

If profile overhead differs, state it explicitly and discard the profiled ratio.

## 4. Latency And Throughput

Report:

- Median, minimum, p90, mean when useful, and sample count.
- Stage medians measured with the same boundaries.
- Peak allocated and reserved device memory per worker or rank.
- Failures, fallbacks, recompiles, graph recaptures, and cache misses.
- Cold setup time, warm setup time, and break-even batches or work items per fixed-cost execution unit.
- Peak host RSS and shared-memory usage when workers, queues, pinned buffers, or output staging change.

Compute both forms clearly:

```text
latency reduction = 1 - candidate / baseline
speedup ratio = baseline / candidate
throughput increase = baseline_interval / candidate_interval - 1
```

Do not call `1 / isolated latency` a throughput proof when host work overlaps the next invocation. Run a continuously fed workload and measure steady launch/completion intervals after pipeline fill.

For asynchronous pipelines, start the interval before the first steady admission and end it only after
the last steady item is persisted and every queue is drained. Attribute work moved into the next
input gap, background future, callback, or epoch-end hook to the candidate. Measure queue depth in
bytes as well as items and report tail-drain failures.

If work is sharded or replicated, compute reuse and break-even at the process, worker, rank, replica,
or device that independently pays the fixed cost. Global volume is relevant only when the cost and
cache are genuinely shared.

For service-log analysis, use half-open UTC ranges. State whether selection uses completion timestamps or requires each invocation to begin inside the range. Classify workload modes from explicit markers rather than duration when metric tags are incomplete.

## 5. Numerical Validation

Choose tolerances from the accepted precision and downstream sensitivity.

For tensors, inspect:

- Shape, dtype, device, finite values, min/max, mean/std.
- Maximum and mean absolute/relative error.
- Cosine similarity or task-specific metric.
- Repeatability across first and repeated graph replay.
- Multiple target and edge-case shapes.

For attention, validate direct Q/K/V output before whole-model output. Large variance shifts, NaN/inf LSE, or output resembling an input tensor indicate implementation corruption, not expected low-precision drift.

Record actual backend and kernel dispatch. A successful `torch.compile`, SDPA, autocast, or
quantization API call does not prove that the expected fused or lower-precision kernel ran.

## 6. Task Behavior And Quality

Preserve the external contract: input/output schemas, shapes, dtypes, ordering, units, metadata, thresholds, determinism guarantees, safety checks, persistence side effects, errors, and response semantics.

Use task-appropriate review:

- Classification, detection, or regression: golden sets, task metrics, calibration, tolerance bands, and difficult slices.
- Generative text or structured output: exact/schema checks, semantic or judge metrics, factuality, and representative human review.
- Embeddings/retrieval: recall/ranking metrics on representative corpora.
- Images, video, or audio: matched inputs and seeds, blind review, task metrics, artifacts, and temporal or cross-modal consistency where applicable.

Validate the evaluator and comparison harness itself. Misaligned inputs, stale caches, asymmetric preprocessing, nondeterministic seeds, or different lifecycle state can create a false quality difference.

Audit every speed-affecting change after the review. For example, an automatically selected backend may silently change an operand from FP16 to FP8 even when the visible model configuration is unchanged.

## 7. Target-Environment Validation

After controlled proof, validate on the actual target hardware and runtime shape. For deployed services, use a canary; for libraries or batch jobs, use the real launcher and resource limits:

- Confirm the exact runtime artifact or container digest and configuration.
- Reconfirm observed hardware identity and topology; fail the proof rather than silently running on
  a fallback, shared, partitioned, or different-generation device.
- Verify startup, warmup, graph/compile enablement, readiness, and zero restarts.
- Run representative common, optional, large, dynamic-shape, and lifecycle-transition workloads.
- Search logs for failures, eager fallbacks, graph recaptures, out-of-memory errors, communication errors, and quality-sensitive setting changes.
- Compare with a known-good implementation from the same hardware pool and workload mode.
- Keep a kill switch for risky kernel, graph, or distributed changes until target validation proves stable.
- Confirm every accepted optimization remains enabled under the real device count and launcher. A
  path unavailable in the claimed topology must not be included in the result.
- Validate the real output filesystem, final queue drain, crash/resume behavior, and dependent phases
  when persistence is part of the product contract.

When execution hangs, test an unchanged known-good artifact on the same node. This distinguishes application regressions from accelerator fabric, driver, or communication-transport failures.

## 8. Reporting Template

```markdown
### Environment
- Hardware/topology:
- Runtime artifact/commit/dependencies:
- Workload and invariant manifest:
- Process lifecycle and cache assumptions:
- Warmups/timed samples:

### Trace evidence
- Raw trace:
- Overview screenshot:
- Opportunity screenshot:
- Exposed-time and roofline calculation:

### Result
| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| E2E p50 | | | |
| E2E p90 | | | |
| Affected stage | | | |
| Peak reserved/rank | | | |
| Cold setup/process | | | |
| Break-even work/unit | | | |
| Peak host RSS/shared memory | | | |

### Correctness and quality
- Numerical checks:
- Product quality checks:
- Contract/optional-path checks:
- Exact-intent versus approximate-quality changes:

### Review and decision
- Reviewer findings and fixes:
- Fallback/kill switch:
- Rejected experiments:
- Deferred/retest experiments:
- Remaining ceiling and stop rationale:
```
