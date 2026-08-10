# Optimization Catalog

Choose from this catalog only after trace evidence identifies the mechanism.

## Table Of Contents

0. Objective and workload formation
1. Dispatch and Python overhead
2. Compilation and CUDA Graphs
3. Attention and custom kernels
4. GEMM, quantization, and block fusion
5. Distributed execution
6. Transfers, memory, and offload
7. Host edge pipelining and I/O
8. Approximate-quality methods
9. Common rejected experiments
10. Model-family and serving checks

## 0. Objective And Workload Formation

Select the objective before replacing operators:

- For online latency, reproduce the target arrival rate, concurrency, queueing policy, and p50/p90/p99
  metric. Do not improve isolated latency by violating the minimum throughput or deployment contract.
- For throughput, test the following work-formation choices before replacing small operators.
- For startup, use clean processes and separate import, model load, weight preparation, compilation,
  warmup, and readiness.
- For cost or memory, hold output quality and service constraints fixed while measuring normalized
  resource use or peak residency.

For throughput objectives:

- Sweep target/request batch, sample parallelism, and sequence bucketing until achieved GEMM rate,
  padding waste, memory, or latency gives a clear knee.
- Measure accelerator busy union and continuous-feed dispatch/completion intervals. A fast isolated
  invocation is not proof of a saturated pipeline.
- Separate model batch size from algorithmic multiplicity such as samples, beams, crops, or diffusion
  trajectories. Enforce the real memory/work budget rather than silently reducing multiplicity.
- Use adaptive caps for quadratic work, context depth, spatial resolution, or other dominant
  dimensions. Report singleton fallbacks and rejected batches.
- Verify batching survives downstream heads, postprocessing, output unpadding, persistence, and
  distributed sharding; model-only batching can silently duplicate or misassign downstream work.

For a throughput objective, optimize small GEMMs only after showing that useful batching cannot make
them large enough to saturate the device.

## 1. Dispatch And Python Overhead

- Capture a static repeated loop with CUDA Graphs when shapes, pointers, control flow, and model state are stable.
- Compile stable model regions when Python dispatch and elementwise fragmentation are exposed but whole-application control flow is dynamic.
- Replace CUDA scalar branches with host-known loop state or precomputed device schedules.
- Cache invariant capability checks and backend dispatch per module, but expect negligible gain unless repeated checks appear in the trace.
- Replace serial manager/proxy rank dispatch with one shared payload plus notifications only when rank entry skew is exposed. Queue fan-out can overlap useful preprocessing, so prove e2e value.
- Fix extensions that launch on legacy stream 0. Launch on PyTorch's current stream and guard the tensor device.

## 2. Compilation And CUDA Graphs

For `torch.compile`:

- Compare supported Torch versions with the complete dependency stack.
- Inspect Dynamo graph breaks, recompiles, guard failures, generated Inductor kernels, and compilation cache behavior.
- Try per-block islands before the entire orchestration layer.
- Test `default`, `reduce-overhead`, and `max-autotune` as evidence warrants.
- Prewarm every target shape and require zero measured-path recompiles in a soak.
- Reject compile paths with thousands of graph breaks, repeated recompilation, or dependency regressions even if one microbenchmark improves.
- Use `TORCH_LOGS="graph_breaks,recompiles,guards,output_code"`, `TORCH_COMPILE_DEBUG=1`,
  `torch._dynamo.explain`, or `tlparse` as supported by the installed Torch version. Inspect the
  generated code and launch timeline; a successful API call is not proof that useful work fused.
- Treat a Torch upgrade as a complete runtime candidate. It can change Inductor, attention dispatch,
  FSDP reshard/storage semantics, allocator behavior, and extension ABI at the same time.
- Compute cold break-even per shape and per worker/rank. Disk cache reuse does not imply that a fresh
  process pays zero code-load, autotune, graph-capture, or allocator cost.
- Replay an ordered production shape trace when comparing static, `dynamic=True`, or automatic
  dynamic promotion. Record graph counts and kernel speed as separate outcomes.

For CUDA Graphs:

- Finalize adapters, quantized weights, and model patching before capture.
- Copy every dynamic tensor input into stable capture buffers before replay.
- Validate eager versus first replay and repeated replay.
- Capture and enable consistently across ranks; mixed graph/eager rank state can deadlock collectives.
- Keep an eager fallback for unsupported shapes, unloads, pointer changes, or insufficient memory.
- Treat graph-private pools and retained model storage as real peak-memory cost.

CUDA Graphs remove launch overhead; they do not inherently reduce transformer FLOPs or kernel count.

## 3. Attention And Custom Kernels

- Benchmark SDPA, FlashAttention, SageAttention, and model-specific kernels on exact dtype/layout/head-dimension/sequence shapes.
- Inspect library version dispatch. An `auto` backend can silently change precision or kernel family after an upgrade.
- Probe Q/K/V directly against a trusted SDPA reference before model integration.
- Check finite values, mean/std, max/mean absolute error, cosine similarity, layout variants, masks, causal behavior, and LSE when consumed.
- Restrict approximate attention to semantically supported calls; keep masked or cross-attention on the reference backend when needed.
- Fuse Q/K norm, RoPE, layout conversion, and quantization only after the main attention kernel is proven.
- Do not assume `torch.compile` rewrites a manual attention sequence into SDPA. Inspect generated
  kernels and compare eager reference, compiled reference, and explicit backend separately.
- Arbitrary dense additive bias, padding masks, FP32 operands, or cross-attention shapes can exclude
  FlashAttention while still allowing a memory-efficient or math backend. Record the exact dispatch.

For CUDA, Triton, or CuTe kernels:

- Start from a repeated memory-bound chain, not a fashionable operator.
- Require access to the claimed target architecture before selecting launch parameters, accepting
  performance, or enabling an architecture-specific default. Keep proxy work clearly diagnostic.
- Fuse residual, reduction, modulation, gating, cast, and compressed-parameter indexing when it removes expanded tensors and round trips to memory.
- Make the kernel graph-capturable and current-stream correct.
- Test odd sizes, strides, non-contiguous layouts, all supported dtypes, devices, and architecture guards.
- Preserve a reference fallback outside supported hardware/shapes.

## 4. GEMM, Quantization, And Block Fusion

- Run an exact-shape dense-kernel roofline search when GEMMs have material exposed time: calculate useful FLOP/s, compare compatible backend families, profile contrasting finalists, and retune the production shape family before writing a one-off kernel.
- Use Nsight Compute before replacing cuBLASLt, CUTLASS, nvJet, or other tuned vendor kernels.
- Prepack stable FP8 weights and scales when runtime patching does not mutate them.
- Fuse scale, bias, activation, quantization, and output conversion into GEMM epilogues when the backend supports it.
- Combine gate/up projections only when adapter semantics and memory layout remain correct.
- Execute unmerged low-rank adapter contributions separately when merging is prohibited.
- Treat residual/RMSNorm/AdaLN/quantization as one optimization pool; their kernel times and intermediates overlap.
- Inventory GEMM `M×K×N`, frequency, and surrounding epilogue before broad FP8 replacement.
  Shape classes can behave differently even at similar FLOP counts. Screen narrow static- or
  delayed-scale FP8 sites before accepting the compile and scaling overhead of dynamic wrappers.

Do not sum quantization, copy, norm, and elementwise buckets to predict savings without reconstructing their dependency chain.

## 5. Distributed Execution

- Reject or redesign a measured winner that is disabled by the target device count or launcher. Test
  model mutation, packing, compilation, and graph capture under spawn before counting its gain.
- Optimize exposed communication, not aggregate NCCL time.
- Prefetch the next FSDP all-gather only after fixing rank-entry skew and proving enough independent compute exists.
- Sweep prefetch depth and retained blocks with peak reserved memory under the target GPU limit.
- Consider keeping the first block replicated or gathered only when its first collective is truly exposed; small gains rarely justify persistent memory and unload complexity.
- Sequence parallelism can regress a launch-bound model by shrinking kernels without reducing launch count. Test SP degree and compile jointly; do not assume additivity.
- Literal compute/communication fusion is usually impossible when a GEMM needs a fully gathered flat-sharded weight. Prefetch, copy-engine movement, or tensor-aligned TP is the practical alternative.
- Use per-shape gather buffers or `all_gather_into_tensor` only when allocation/copy churn is visible.
- Validate collective order, failure propagation, startup, teardown, and optional workload paths in a soak.

## 6. Transfers, Memory, And Offload

- Verify whether the target workload actually offloads parameters or intermediate state before optimizing reload/prefetch.
- Cache inputs, preprocessing results, or features only when target evidence shows repeated values and measured-path misses.
- Use pinned host buffers and asynchronous copies only when transfer is material and safely overlapped.
- Record a CUDA event before releasing or reusing a D2H staging buffer.
- Keep model loading, adapter patching, partial unload, and quantized tensor restoration outside `inference_mode` when framework code mutates parameters; use `no_grad` where appropriate.
- Expect ordinary offload via `.to(cpu)` to change storage and invalidate CUDA Graph pointers.
- Use CUDA virtual-memory remapping only when preserving addresses recovers a proven large graph benefit. Require explicit state, stream synchronization, pointer checks, memory-release proof, and long alternating-model soaks.

## 7. Host Edge Pipelining And I/O

Overlap abundant host work across invocations at the edges:

```text
input acquisition/preparation -> exclusive accelerator phase -> completed D2H -> host postprocess/serialize/deliver
```

- Do not overlap separate accelerator executions when it raises latency, memory, or resource contention beyond the target contract.
- Use bounded queues, per-invocation buffers, failure propagation, and bounded worker pools.
- Prevent data loading, preprocessing, postprocessing, serialization, or storage workers from starving Python launch, communication, or health threads.
- Measure saturated dispatch-to-dispatch intervals after pipeline fill.
- Keep isolated latency and throughput claims separate.
- Include the final drain in the timed boundary. Deferred work that reappears in the next input gap or
  epoch-end hook has moved, not disappeared.
- Distinguish GIL-bound serialization from compression or native I/O that releases the GIL. Threads
  can contend with a Python kernel-launch loop; processes add copies and lifecycle complexity.
- Bound staged output by bytes, propagate failures promptly, and make resume correctness crash-safe
  with atomic artifacts or explicit completion sentinels.

## 8. Approximate-Quality Methods

Keep these separate from exact-intent optimizations:

- Lower activation, value, or accumulator precision.
- Approximate attention.
- Less algorithmic work, such as fewer search candidates, decode iterations, or solver steps.
- Distillation, consistency models, pruning, or smaller models.
- Reduced input size, context length, output size, fidelity, or stricter truncation.

Measure their speed contribution independently. Use representative task metrics, difficult slices, and blind human evaluation when the output is subjective. Unchanged high-level configuration is not proof of semantic equivalence when kernel precision or backend dispatch changed.

## 9. Common Rejected Experiments

Do not assume these help:

- Packed QKV or alternate SP communication without shape-specific proof.
- Higher-priority communication streams when work is launch-bound or already overlapped.
- Broad FSDP prefetch, which can contend with compute and increase memory.
- Whole-model compile with dynamic patching, scalar branches, and model swaps.
- Isolated norm kernels that retain adjacent copies and elementwise launches.
- Collective buffer caching when steady communication is already near empirical roofline.
- Barriers intended to fix rank skew.
- Optimizing defensive fallbacks never observed in the target workload.

Record every rejected experiment, its configuration, result, and reason so later work does not repeat it.

## 10. Model-Family And Serving Checks

Use these only when their mechanism appears in the end-to-end trace:

- Autoregressive models: verify KV-cache reuse, cache layout and growth, paged attention, prefix
  caching, decode batch formation, and stable-shape graph replay separately for prefill and decode.
  Continuous batching can improve throughput while worsening one-invocation latency; measure both.
- Convolutional models: test `channels_last`, cuDNN autotuning for stable shapes, fused
  convolution/norm/activation, and unnecessary NCHW/NHWC conversions. Preserve exact preprocessing.
- Diffusion and iterative solvers: graph or compile the stable denoising loop, eliminate CUDA scalar
  branches using host-known iteration state, and cache only invocation-invariant conditioning. Do not
  reduce steps or alter the scheduler in an exact-intent experiment.
- Recommenders and embedding-heavy models: distinguish memory-bandwidth-bound gathers from network
  sharding, test table placement and workload coalescing, and include cache hit-rate effects.
- Multimodal services: trace modality-specific loading, decoding, tokenization, feature extraction,
  postprocessing, and serialization around the model. Pipeline host edge work across queued invocations
  only after accelerator-owned buffers are safe.
- General serving: inspect batching, bucketing, padding waste, allocator churn, admission control,
  thread oversubscription, serialization, and queue handoff. Use the real workload distribution;
  optimizing one convenient shape can regress the target mix.

Treat quantized KV caches, speculative decoding, early exit, pruning, distillation, and reduced input
fidelity as approximate-quality or behavior-changing unless the product contract explicitly says
otherwise.
