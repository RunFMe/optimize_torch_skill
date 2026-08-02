# Optimization Campaign Playbook

Use this reference for multi-hypothesis inference campaigns involving latency, throughput, cost,
memory, startup, compilation, lower precision, custom kernels, multiple devices, or full
input/output pipelines.

## Table Of Contents

1. Fast start
2. Campaign card
3. Experiment order
4. Screening and proof
5. Opportunity ledger
6. Compilation lifecycle
7. Precision and quantization
8. Library backends and custom kernels
9. Host pipelines and persistence
10. Default-promotion gate
11. Review packet
12. Final reconciliation

## 1. Fast Start

Do these before editing model code:

1. Freeze the campaign card below.
2. Verify access to the target hardware/runtime and record the observed device identity and topology.
3. Run the real application once and verify outputs.
4. Reproduce the objective's operating point: required concurrency for latency, continuous feed and
   a batch/concurrency sweep for throughput, clean launches for startup, or the constrained resource
   envelope for memory/cost.
5. Capture one light trace at that operating point.
6. Build the opportunity ledger and calculate ceilings.
7. Reject candidates with the cheapest decisive test before writing production code.

Do not start with a fashionable kernel. First determine whether the limiting mechanism is work
formation, compute, launch, memory, communication, lifecycle, or host-pipeline behavior.

## 2. Campaign Card

Keep this in the benchmark output directory and update it only when the user changes the contract.
When it changes, mark earlier rankings as stale.

```yaml
objective: <latency | throughput | cost | memory | startup>
primary_metric: <for example p90_latency_s or requests_per_s>
constraints: <for example minimum throughput, maximum memory, or quality floor>
boundary: <model | pipeline | service>
workload_mix: <representative workload manifest or distribution>
work_budget:
  <algorithm-specific required work>
outputs: <required output and side-effect contract>
quality_gate: <metric and difficult-slice protocol>
hardware:
  accelerator: <target accelerator>
  devices: <target count>
  topology: <deployment or launcher topology>
  access: <local | container | remote host | scheduler | unavailable>
  verified_identity: <observed model, capability, memory, runtime, and topology>
  proxy_differences: <none or explicit differences from target>
lifecycle:
  process_lifetime: <one-shot | long-lived>
  fixed_cost_unit: <process | worker | rank | replica | device>
  expected_work_per_unit: <count or distribution>
  cache_reuse: <cold | warm | mixed>
screen_proxy: <cheap diagnostic workload and disclosed differences>
proof_workload: <objective-matched production-equivalent workload>
```

Record the actual container or environment, commit, weights, driver, Torch build, extension versions,
seeds, precision, and backend dispatch beside the card.

## 3. Experiment Order

Keep the evidence sequence fixed, but rank optimization classes from the measured workload rather
than using one universal implementation order:

1. **Correctness and boundaries:** prove the application path, output contract, timer, and failure
   accounting.
2. **Objective baseline:** measure the primary metric and constraints at the real operating point.
   For throughput, sweep batching/concurrency and padding; for latency, preserve arrival rate and
   concurrency and measure tails; for startup, use clean launches; for memory/cost, hold the relevant
   resource envelope fixed.
3. **Trace and ledger:** measure exposed time, ceilings, fixed costs, and target compatibility.
4. **Cheap disproof:** screen the leading candidates for unsupported dispatch, insufficient ceiling,
   compile explosion, memory excess, lifecycle incompatibility, or unacceptable drift.
5. **Smallest implementation:** implement the highest-value surviving candidate, whether it is work
   formation, redundant-work removal, a runtime/library setting, compilation, a custom kernel,
   precision, host pipelining, or topology.
6. **Proof and iteration:** run objective-matched proof, update the cumulative result, reprofile, and
   select again.

For throughput, test useful batching and continuous feed before replacing small operators. For
latency, topology, admission, batching, or cross-request overlap are valid only within the latency
and throughput constraints. For startup, investigate lifecycle work before steady kernels. Treat
numerically different candidates separately regardless of where evidence ranks them.

Reprofile the cumulative winner after every two accepted changes. Optimizations that attack the same
GEMM, memory chain, or launch bubble are especially non-additive.

## 4. Screening And Proof

### Screening

Screening exists to fail fast. It may use repeated inputs, fewer iterative steps, one device, cached
features, omitted output writing, and one to three timed runs if all differences are disclosed.
Different hardware may screen portable correctness or unsupported dispatch, but its timing cannot
accept or reject target performance unless a validated transfer model exists.

Use a screen only to establish one of these:

- unsupported or incorrect dispatch;
- compile, OOM, NaN, or lifecycle failure;
- a large regression;
- an end-to-end ceiling below the campaign threshold;
- unacceptable direct tensor or trajectory drift.

Do not use a reduced-step screen to reject fixed-cost work such as compilation without rescaling its
break-even to production. Do not use it to accept host-pipeline work whose relative share collapses
at full iteration count.

### Proof

Proof uses the real application boundary and a target-equivalent configuration. Require at least
three warmups and five timed samples. For gains below 3%, prefer at least ten timed samples and an
A/B/A, B/A/B, or randomized interleaving.

Run arms sequentially on a quiescent host. Concurrent arms are valid only with explicit CPU, memory,
storage, and accelerator partitioning. Otherwise cross-arm interference can be larger than the
candidate.

For a proxy whose stage shares differ from production, report both:

```text
proxy stage share
production stage share estimate or measurement
stage-local speedup
production Amdahl ceiling
```

An invariant host cost measured at 5 solver steps can look dominant and become negligible at 200
steps. Conversely, compilation cost measured on four shapes can lose to static specialization over
thousands of repeated bucket hits.

## 5. Opportunity Ledger

Assign a stable ID before the first experiment. Update the row immediately after every run or review.

| Field | Content |
|---|---|
| ID and status | `H07`, `screening`; then `ship`, `reject`, `defer`, or `retest` |
| Contract scope | Workload classes, outputs, devices, launcher, warm/cold lifecycle |
| Hardware access | Target identity, access path, verification evidence, and proxy differences |
| Evidence | Trace/crop paths, command, raw result paths, exact commit |
| Mechanism | Dependency or data-movement explanation |
| Stage share | Exposed production wall-time fraction |
| Ceiling | `stage_share × plausible_removable_fraction` |
| Fixed cost | Compile/autotune/packing/capture seconds and memory |
| Proposal | Smallest behavioral change that tests the mechanism |
| Correctness | Reference, tolerances, edge cases, task metrics |
| Acceptance | Minimum e2e gain, memory limit, failure/fallback rules |
| Result | All samples, isolated delta, cumulative delta, target compatibility |
| Review | Reviewer, findings, fixes, rerun evidence |
| Remaining work | Explicit reason for `defer` or `retest` |

Do not use `reject` for a promising candidate tested only with the wrong implementation or proxy.
Use `defer` or `retest` and state the next decisive experiment.

Useful arithmetic:

```text
latency reduction = 1 - candidate_seconds / baseline_seconds
throughput increase = baseline_interval / candidate_interval - 1
break_even_batches = cold_extra_seconds / warm_seconds_saved_per_batch
break_even_work_items_per_unit = break_even_batches × work_items_per_batch
```

For sharded or replicated work, compare break-even with work handled by the execution unit that pays
the fixed cost, not global campaign volume.

## 6. Compilation Lifecycle

Start with stable, high-share islands. Whole-model compilation is a diagnostic screen, not the
default first implementation when control flow, custom extensions, or Python scalar branches exist.

For every compile candidate:

1. Finalize weights, adapters, packing, and eval mode before compilation.
2. Record cold seconds, graph count, graph breaks, recompiles, guards, and generated kernels.
3. Verify which operations fused. `torch.compile` does not promise to replace a manual
   `QK -> bias/mask -> softmax -> PV` sequence with SDPA or FlashAttention.
4. Replay a representative ordered shape trace for static and dynamic policies.
5. Measure warm improvement and compile break-even per fixed-cost execution unit and shape bucket.
6. Test checkpoint keys, pre-first-use pickling/spawn, DDP/FSDP behavior, fallback, and OOM handling.

Prefer `mode="default"` until a matched experiment proves another mode. `reduce-overhead` and
`max-autotune` can add cold cost or memory without changing steady throughput.

Warm disk cache is not equivalent to a prewarmed process. Verify whether code generation, loading,
autotuning, graph capture, and allocator setup still occur in each new worker.

## 7. Precision And Quantization

Build an operator precision map before benchmarking. Include projections, attention score/value
matmuls, normalizations, reductions, solver state, alignment, and output heads, plus explicit
autocast-disabled regions.

Do not describe BF16 as more precise than TF32. Both have an 8-bit exponent; TF32 has a 10-bit
stored-significand precision while BF16 has 7 bits. BF16 can still be faster or save memory because
it is a 16-bit storage format, while TF32 usually accelerates eligible FP32 GEMMs without shrinking
stored tensors. Actual gain depends on which operators enter each mode.

Use this ladder and measure each increment separately:

1. FP32 with highest matmul precision.
2. FP32 storage with TF32-eligible GEMMs.
3. BF16 or FP16 autocast over a scoped module.
4. Narrow FP8 GEMMs with prepacked weights and calibrated static or delayed scales.
5. Broader quantization only if narrow wins survive integration.

For FP8, inventory exact `M×K×N` shapes, frequency, and surrounding work. Performance varies by
shape and epilogue; do not infer a winning class from dimension labels alone. Broad dynamic wrappers
can spend more time scaling, dispatching tensor subclasses, or compiling than they save. Measure
weight preparation separately and include it in lifecycle economics.

Define paper-, product-, or task-aligned metrics before promoting a numerically different default.
Same-seed coordinate or tensor distance is a guardrail, not a task-quality verdict.

## 8. Library Backends And Custom Kernels

Before writing a kernel:

1. Measure the candidate operator chain's exposed share on the objective-matched target configuration.
2. Calculate the e2e ceiling. A 4% stage made infinitely fast is still at most a 4% win.
3. Test available vendor backends with the exact dtype, additive bias, padding mask, causal mode,
   head dimension, and layout.
4. Inspect the kernel names. Flash, memory-efficient SDPA, math fallback, and a compiled manual path
   are different candidates.
5. Validate left/right/all padding, odd sizes, non-contiguous views, finite outputs, and direct error.
6. Require a microbenchmark margin large enough to survive integration and compilation overhead.

Use three measurements: eager reference, compiled reference, and proposed backend/kernel. A fast
standalone SDPA call is not useful if wrapping it inside the compiled model removes its advantage.

For online-softmax kernels, explicitly handle fully masked rows and finite sentinel masks. A kernel
that passes random dense inputs can still produce NaNs on real padded windows.

## 9. Host Pipelines And Persistence

Measure throughput from admission of the first steady item through completion and final drain of all
steady work. If an asynchronous writer moves work into the next batch's gap or epoch-end hook, include
that time.

Before offloading host work:

- Separate GIL-bound Python serialization from C/C++ work that releases the GIL. A thread may contend
  with the model's Python launch loop rather than overlap it.
- Stage only tensors consumed by the worker. Do not copy an entire prediction tree containing unused
  quadratic embeddings.
- Bound queues by bytes as well as item count and report host RSS and shared-memory usage.
- Propagate failures promptly, drain in `finally`, and shut workers down before process forks or
  dependent phases.
- Keep CUDA-graph output buffers safe: copy dynamic outputs to owned storage before the next replay.
- Use atomic files or an explicit completion sentinel so a killed run cannot leave a directory that
  resume logic mistakes for a finished target.

Benchmark default outputs and large optional outputs separately. Compression of quadratic PAE/PDE or
embeddings can justify asynchronous I/O even when ordinary structure writing cannot.

## 10. Default-Promotion Gate

Make an optimization default-on only when all applicable checks pass:

- positive cumulative proof on the target configuration, not only an isolated microbenchmark;
- measured on verified target hardware for every hardware-specific claim;
- active under the production device count, launcher, optional paths, and output mode;
- acceptable warm/cold economics for expected work per fixed-cost execution unit and shape reuse;
- stable checkpoint/state-dict behavior and pre-first-use serialization/spawn;
- explicit negative kill switch;
- fail-open eager fallback for recognized compile/backend failures;
- OOM re-raised without retrying the same unsafe batch;
- memory, fallback, recompilation, and numerical telemetry visible;
- quality gate complete for numerically different or behavior-changing paths;
- documentation names both the benefit and the unsupported cases.

If the target assumes prewarmed long-lived workers, say so in the CLI help and documentation. Do not
generalize that default to cold one-off use without evidence.

## 11. Review Packet

Send reviewers a bounded packet rather than the full chat history:

- campaign card and invariant diff;
- one opportunity-ledger row;
- exact commands and raw result paths;
- all timed samples, memory, failures, compile counters, and dispatch evidence;
- trace overview and candidate crop;
- patch or prototype;
- correctness and quality artifacts;
- proposed decision and threshold.

Ask for design review before an expensive run when a candidate has numerical, distributed,
compilation, persistence, or custom-kernel risk. Review implemented winners again before default
promotion. Run requested independent reviewers in parallel when tools and capacity allow.

## 12. Final Reconciliation

Before declaring the campaign complete, reconcile:

```text
conversation requirements
campaign card
opportunity ledger
production code
CLI defaults and kill switches
tests
benchmark harnesses and raw outputs
documentation and PR description
```

Deliver four lists:

1. Shipped exact-intent optimizations.
2. Shipped numerically different or behavior-changing optimizations.
3. Rejected experiments with configuration, result, and reason.
4. Deferred/retest opportunities with their remaining ceiling and next decisive experiment.

Do not hide a live opportunity in a `not implemented` paragraph, and do not call a candidate rejected
when only its broad, dynamic, or non-production implementation failed.
