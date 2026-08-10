# Dense-Kernel Roofline Search

Use this workflow when GEMM, convolution, attention, or another compute-dense kernel has material exposed time or performs far below a plausible accelerator ceiling. The goal is to find a better implementation before investing in unrelated fusion or precision changes.

## Table Of Contents

1. Freeze kernel semantics
2. Build the shape ledger
3. Calculate useful throughput and ceilings
4. Search kernel families
5. Diagnose finalists with Nsight Compute
6. Retune the complete shape family
7. Integrate and prove the winner

## 1. Freeze Kernel Semantics

Record every property that can change dispatch, speed, or numerical behavior:

- operand and output dtypes;
- logical shapes, physical strides, transpose state, alignment, and padding;
- scale granularity and layout for quantized operands;
- bias, activation, accumulation mode, output conversion, and other epilogue work;
- sparsity, masks, groups, determinism, and graph-capture requirements;
- allocation versus caller-provided output, weight prepacking, and cache lifecycle.

Reject comparisons that silently move scaling, conversion, allocation, padding, or packing outside the timed boundary. Label a raw-GEMM-only result as a diagnostic ceiling when the production call requires additional work.

## 2. Build The Shape Ledger

Extract exact production shapes and their frequency from traces or runtime instrumentation. Include every shape produced by:

- batch, sequence, image, video, or request-size buckets;
- tensor, sequence, expert, or pipeline parallel degree;
- fused versus split projections;
- optional paths and long-tail requests.

For each shape, record stage share, invocation count, current kernel name, latency distribution, and layout. Start with the highest `exposed time × plausible removable fraction`, not merely the largest FLOP count.

Use repeated exact shapes only for screening. Preserve the real ordered shape mix for integration proof because autotuning, cache reuse, and thermal state can change results.

## 3. Calculate Useful Throughput And Ceilings

For dense GEMM `M×K @ K×N`, calculate useful work as:

```text
useful_flops = 2 × M × N × K
achieved_flops_per_second = useful_flops / unprofiled_kernel_seconds
```

For batched or grouped GEMMs, sum useful FLOPs across groups. Keep useful throughput separate from padded or otherwise executed work.

Use the strongest defensible ceiling:

1. exact-shape empirical throughput from the best compatible library kernel;
2. same-dtype calibration GEMM on the target device;
3. dense hardware peak adjusted for the observed clock when its reference clock is known;
4. published dense peak, clearly labeled as nominal.

Do not compare dense work with a sparse marketing peak. Do not compare profiler-clock throughput with an unadjusted boost-clock peak. Report observed SM and memory clocks, power state, and whether profiling changed them. Treat Nsight's percentage-of-peak metrics as observed-condition diagnostics, not unprofiled speed proof.

Calculate arithmetic intensity from required bytes, then use measured DRAM and cache traffic to test the idealized model. A nominally compute-bound GEMM can become memory-bound because of poor L2 reuse, tile order, scale traffic, padding, or repeated operand loads.

## 4. Search Kernel Families

Benchmark compatible candidates sequentially on an otherwise idle target device:

| Layer | Examples | Purpose |
|---|---|---|
| Current path | Exact production call | Establish dispatch and latency |
| Framework/vendor | `torch.mm`, `torch._scaled_mm`, cuBLASLt, cuDNN, SDPA | Find an optimized maintained baseline |
| Specialized libraries | CUTLASS, nvJet-backed paths, FlashAttention, SageAttention, DeepGEMM, TensorRT-LLM | Test architecture- and datatype-specific kernels |
| Generated kernels | Triton, CuTe DSL, Inductor | Search tiles, schedules, fusion, and layout choices |
| Custom extension | CUDA, CUTLASS, CuTe | Use only when simpler candidates leave a justified ceiling |

For each candidate:

1. Verify the actual kernel name and backend.
2. Run direct tensor correctness before timing.
3. Finish JIT compilation, autotuning, prepacking, and warmup.
4. Collect unprofiled median, minimum, p90, memory, and failures.
5. Record cold cost and break-even per process and shape.

Keep fast-accumulation or lower-precision candidates separate from accurate accumulation. A small tensor error may still require task-quality validation.

## 5. Diagnose Finalists With Nsight Compute

Profile one warmed launch for the current kernel and the fastest compatible candidate. Capture at least:

- duration and observed clocks;
- compute and tensor-pipe throughput;
- DRAM, L2, and L1 throughput plus hit rates;
- registers, shared memory, achieved occupancy, and active warps;
- tile, cluster, grid, stages, scheduler/raster order, and waves.

Interpret the contrast rather than one metric in isolation:

- Low tensor utilization with high DRAM traffic or poor L2 hit rate points to reuse, layout, tile order, multicast, or scale/epilogue traffic.
- Low compute and memory utilization points to launch gaps, insufficient parallel work, dependencies, imbalance, or a poor persistent schedule.
- High tensor utilization at the observed clock means the kernel is near its local compute ceiling; focus on other shapes or surrounding work.
- Low occupancy alone is not a defect. Warp-specialized kernels can be efficient with one resident block when their pipelines stay full.
- A large shape can still underperform when every SM runs a poorly balanced persistent tile.

Do not use replayed profiler duration as the speed claim. Use it only to explain the unprofiled difference.

## 6. Retune The Complete Shape Family

After finding a winning family, sweep all high-frequency and topology-dependent shapes. Tune relevant choices such as:

- tile M/N/K and stage count;
- cluster orientation and multicast operand;
- cooperative, ping-pong, or persistent scheduling;
- raster order, split-K, and work swizzling;
- prepacked layout and fused epilogue.

Require a fallback for untuned shapes. Prefer an architecture/dtype/layout/epilogue/shape-bucket configuration table or existing backend-selection mechanism over model-name checks and exact one-off branches. Preserve benchmark metadata so the table can be regenerated when hardware or dependencies change.

## 7. Integrate And Prove The Winner

Prove the selected kernel in three steps:

1. Exact-shape tensor parity across common and edge shapes.
2. Matched unprofiled application A/B on the target launcher and topology.
3. Task-quality validation when accumulation order, precision, or quantization differs.

Reprofile the cumulative application after integration. Confirm that the expected kernel still dispatches, surrounding quantization and communication did not erase the gain, and every production rank uses the same compatible path.

Record rejected kernels with their exact configuration and reason. Mark unsupported or incompletely tuned candidates `defer` or `retest`, not `reject`.
