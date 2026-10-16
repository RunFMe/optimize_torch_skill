# Trace SQL

Load a trace once into DuckDB, then answer each follow-up question with a query instead of a new
script. On a 159 MB (uncompressed) SGLang rank trace, loading takes about 8 s and a gap query about
30 ms; re-running `summarize_trace.py` costs the full parse every time.

Use this for event-level questions: where and when something happened, what overlapped what, which
rank arrived first. Use TraceLens ([tracelens.md](tracelens.md)) for aggregated tables, and
`crop_trace.py` plus a renderer for images.

## Table Of Contents

1. Load
2. Schema and conventions
3. GPU busy union and idle gaps
4. Kernel-launch gaps
5. What ran in a window
6. Zoom into one kernel
7. Kernels by stream
8. Stages and steps
9. Synchronization and scalar reads
10. CPU-side collectives
11. Cross-rank collective alignment
12. Kernel variants per op and shape
13. Limits

## 1. Load

```bash
uv run --with duckdb python - <<'EOF'
import duckdb, glob, re
con = duckdb.connect("trace.duckdb")
con.execute("SET preserve_insertion_order = false")
files = sorted(glob.glob("traces/*.trace.json.gz"))      # .json works too
for i, path in enumerate(files):
    rank = int(re.search(r"(?:TP|rank)[-_]?(\d+)", path).group(1))   # adapt to the file naming
    con.execute(f"""
      {"CREATE OR REPLACE TABLE ev AS" if i == 0 else "INSERT INTO ev"}
      SELECT {rank} AS rank, e.name, e.cat, e.ph, e.pid->>'$' AS pid, e.tid->>'$' AS tid,
             e.ts::DOUBLE AS ts, e.dur::DOUBLE AS dur, e.args
      FROM (SELECT unnest(traceEvents) AS e FROM read_json(?, maximum_object_size = 1000000000,
            columns = {{traceEvents: 'STRUCT(name VARCHAR, cat VARCHAR, ph VARCHAR, pid JSON,
                        tid JSON, ts JSON, dur DOUBLE, args JSON)[]'}}))""", [path])
print(con.execute("SELECT rank, count(*) FROM ev GROUP BY 1 ORDER BY 1").fetchall())
EOF
```

Load one file per statement. Loading several multi-hundred-MB files in one `read_json` call can
exhaust memory, because each file is a single JSON object. `maximum_object_size` must exceed the
uncompressed file size and is reserved up front, so raise it only for larger files and only as far
as memory allows. Query afterwards from Python
(`duckdb.connect("trace.duckdb").sql(...)`) or the `duckdb` CLI if installed.

## 2. Schema And Conventions

`ev(rank, name, cat, ph, pid, tid, ts, dur, args)`: one row per Chrome-trace event, timestamps
in trace microseconds. Complete events have `ph = 'X'`; metadata (`thread_name`, `process_name`)
has `ph = 'M'` with the label in `args->>'name'`. `pid` and `tid` are text without JSON quotes,
whether the trace wrote them as numbers or strings, so `'7'` matches either form.

Useful categories in Kineto traces: `kernel`, `gpu_memcpy`, `gpu_memset`, `cuda_runtime`,
`cuda_driver`, `cpu_op`, `python_function`, `user_annotation` (CPU `record_function` ranges),
`gpu_user_annotation` (the same ranges projected onto the GPU stream), `ac2g` (flows), and
`overhead` (CUPTI buffer work). For GPU events, `tid` is the stream. `args->>'correlation'` links
a kernel to the runtime call that launched it.

Always parenthesize JSON extraction, as in `(args->>'correlation') = k.corr`. In DuckDB `->>`
binds more loosely than `=` and `AND`, so the unparenthesized form fails with a conversion error.
The same CPU range often appears twice, once as `user_annotation` and once as
`gpu_user_annotation`, and a wait often appears as a Python frame, a `cpu_op` and a runtime call.
Filter by `cat` before summing.

Queries below read their parameters from variables. Set them first:

```sql
SET VARIABLE rank = 0;
SET VARIABLE min_gap_us = 100;
SET VARIABLE t0 = 4777276580000;     -- window start, trace us
SET VARIABLE t1 = 4777276596000;     -- window end, trace us
```

## 3. GPU Busy Union And Idle Gaps

Merge kernel, copy, and memset intervals across all streams, then report gaps where no stream ran.

```sql
WITH k AS (
  SELECT ts AS s, ts + dur AS e FROM ev
  WHERE rank = getvariable('rank') AND ph = 'X' AND cat IN ('kernel', 'gpu_memcpy', 'gpu_memset')),
m AS (SELECT s, e, max(e) OVER (ORDER BY s ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end FROM k),
starts AS (SELECT s, prev_end FROM m WHERE prev_end IS NULL OR s > prev_end)
SELECT round((s - prev_end) / 1000, 3) AS gap_ms, prev_end AS gap_start, s AS gap_end
FROM starts WHERE s - prev_end >= getvariable('min_gap_us')
ORDER BY gap_ms DESC LIMIT 25;
```

```sql
WITH k AS (
  SELECT ts AS s, ts + dur AS e FROM ev
  WHERE rank = getvariable('rank') AND ph = 'X' AND cat IN ('kernel', 'gpu_memcpy', 'gpu_memset')),
m AS (SELECT s, e, max(e) OVER (ORDER BY s ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end FROM k),
grp AS (SELECT s, e, sum(CASE WHEN prev_end IS NULL OR s > prev_end THEN 1 ELSE 0 END) OVER (ORDER BY s) AS g FROM m),
busy AS (SELECT min(s) AS s, max(e) AS e FROM grp GROUP BY g)
SELECT round(sum(e - s) / 1000, 1) AS busy_ms, round((max(e) - min(s)) / 1000, 1) AS window_ms,
       round(100 * sum(e - s) / (max(e) - min(s)), 2) AS busy_pct
FROM busy;
```

A gap is global accelerator idleness only when every stream is idle, which is what the merge
guarantees. Then ask what the CPU was doing during it (section 5).

## 4. Kernel-Launch Gaps

Time between consecutive launch submissions on the host. Count `cudaGraphLaunch` as a launch; a
graph replay submits many kernels at once.

```sql
WITH l AS (
  SELECT ts, ts + dur AS te, name, tid FROM ev
  WHERE rank = getvariable('rank') AND ph = 'X'
    AND regexp_matches(name, '(?i)^(cuda|cu)(Launch(Kernel|CooperativeKernel|KernelEx)|GraphLaunch)')),
n AS (SELECT te AS gap_start, lead(ts) OVER (ORDER BY ts) AS gap_end,
             lead(name) OVER (ORDER BY ts) AS next_call FROM l)
SELECT round((gap_end - gap_start) / 1000, 3) AS gap_ms, gap_start, gap_end, next_call
FROM n WHERE gap_end - gap_start >= getvariable('min_gap_us')
ORDER BY gap_ms DESC LIMIT 25;
```

A launch gap matters only if the GPU goes idle before the next submission. Check it against
section 3 or look inside it with section 5.

## 5. What Ran In A Window

One row per lane, with its covered time and longest events. This is the textual version of a
zoomed Perfetto view.

```sql
WITH w AS (
  SELECT cat, pid, tid, name, dur,
         least(ts + dur, getvariable('t1')) - greatest(ts, getvariable('t0')) AS covered
  FROM ev
  WHERE rank = getvariable('rank') AND ph = 'X' AND cat NOT IN ('Trace', 'python_function')
    AND ts < getvariable('t1') AND ts + dur > getvariable('t0')),
ranked AS (SELECT *, row_number() OVER (PARTITION BY cat, pid, tid ORDER BY dur DESC) AS r FROM w)
SELECT cat, pid, tid, count(*) AS events, round(sum(covered) / 1000, 3) AS covered_ms,
       string_agg(CASE WHEN r <= 3 THEN left(name, 60) || ' (' || round(dur / 1000, 2) || ' ms)' END, ' | ') AS longest
FROM ranked GROUP BY cat, pid, tid ORDER BY covered_ms DESC;
```

Lane labels come from metadata:

```sql
SELECT pid, tid, args->>'name' AS label FROM ev
WHERE rank = getvariable('rank') AND ph = 'M' AND name = 'thread_name' ORDER BY pid, tid;
```

## 6. Zoom Into One Kernel

Find a kernel's launching call through `correlation`, then the host stack open at that moment
(needs `with_stack=True` for Python frames). For a kernel inside a CUDA graph, the launching call is
`cudaGraphLaunch`.

```sql
SET VARIABLE kernel_like = '%all_reduce_kernel%';
WITH k AS (
  SELECT ts, dur, name, tid AS stream, args->>'correlation' AS corr FROM ev
  WHERE rank = getvariable('rank') AND cat = 'kernel' AND name LIKE getvariable('kernel_like')
  ORDER BY dur DESC LIMIT 1),
launch AS (
  SELECT e.ts, e.dur, e.name, e.pid, e.tid FROM ev e, k
  WHERE e.rank = getvariable('rank') AND e.cat IN ('cuda_runtime', 'cuda_driver')
    AND (e.args->>'correlation') = k.corr)
SELECT 'kernel' AS what, k.name, k.ts, round(k.dur / 1000, 3) AS dur_ms, k.stream AS where_ FROM k
UNION ALL
SELECT 'launched by', l.name, l.ts, round(l.dur / 1000, 3), l.pid || '/' || l.tid FROM launch l
UNION ALL
SELECT 'host stack', left(s.name, 100), s.ts, round(s.dur / 1000, 3), s.cat FROM ev s, launch l
WHERE s.rank = getvariable('rank') AND s.pid = l.pid AND s.tid = l.tid
  AND s.cat IN ('python_function', 'cpu_op', 'user_annotation')
  AND s.ts <= l.ts AND s.ts + s.dur >= l.ts + l.dur
ORDER BY what DESC, dur_ms ASC
LIMIT 40;
```

The innermost frames come first under `host stack`. Repeat with `ORDER BY dur DESC LIMIT 1` changed
to pick another instance, for example `ORDER BY ts LIMIT 1 OFFSET 10`.

## 7. Kernels By Stream

```sql
SELECT tid AS stream, count(*) AS kernels, round(sum(dur) / 1000, 2) AS total_ms,
       round(avg(dur), 2) AS mean_us, round(max(dur), 1) AS max_us
FROM ev WHERE rank = getvariable('rank') AND cat = 'kernel'
GROUP BY 1 ORDER BY total_ms DESC LIMIT 10;
```

## 8. Stages And Steps

Durations of `record_function` ranges, and GPU-idle time that falls inside each kind of range.
SGLang names its steps `step[EXTEND ...]`, `step[DECODE ...]`, `step[TARGET_VERIFY ...]`; adapt the
pattern to your framework.

```sql
SELECT regexp_replace(name, '\s.*$', '') AS stage, cat, count(*) AS n,
       round(sum(dur) / 1000, 2) AS total_ms, round(median(dur) / 1000, 3) AS median_ms,
       round(max(dur) / 1000, 3) AS max_ms
FROM ev WHERE rank = getvariable('rank') AND cat IN ('user_annotation', 'gpu_user_annotation')
GROUP BY 1, 2 ORDER BY total_ms DESC LIMIT 20;
```

```sql
WITH k AS (
  SELECT ts AS s, ts + dur AS e FROM ev
  WHERE rank = getvariable('rank') AND ph = 'X' AND cat IN ('kernel', 'gpu_memcpy', 'gpu_memset')),
m AS (SELECT s, max(e) OVER (ORDER BY s ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end FROM k),
gaps AS (SELECT prev_end AS gs, s AS ge FROM m WHERE s - prev_end >= getvariable('min_gap_us')),
steps AS (
  SELECT regexp_extract(name, '^step\[(\w+)', 1) AS kind, ts, ts + dur AS te FROM ev
  WHERE rank = getvariable('rank') AND cat = 'gpu_user_annotation' AND name LIKE 'step[%'),
tagged AS (   -- one row per gap: the innermost step range that contains the gap's start
  SELECT gaps.gs, gaps.ge, steps.kind FROM gaps LEFT JOIN steps ON gaps.gs >= steps.ts AND gaps.gs < steps.te
  QUALIFY row_number() OVER (PARTITION BY gaps.gs ORDER BY steps.te - steps.ts NULLS LAST) = 1)
SELECT coalesce(kind, '(between steps)') AS step_kind, count(*) AS gaps,
       round(sum(ge - gs) / 1000, 2) AS idle_ms
FROM tagged GROUP BY 1 ORDER BY idle_ms DESC;
```

A gap is attributed by its start. Step ranges projected onto several streams can overlap; the
`QUALIFY` keeps each gap once.

## 9. Synchronization And Scalar Reads

```sql
SELECT name, count(*) AS n, round(sum(dur) / 1000, 3) AS total_ms, round(max(dur) / 1000, 3) AS max_ms
FROM ev
WHERE rank = getvariable('rank') AND ph = 'X' AND cat IN ('cuda_runtime', 'cpu_op')
  AND regexp_matches(name, '(?i)synchroniz|_local_scalar_dense|aten::item|cudaMemcpy(Async)?$')
GROUP BY 1 ORDER BY total_ms DESC;
```

The `cat` filter keeps one row per wait; Python frames around the same call are dropped. Drop the
filter to see which Python code issued it.

For each long wait, check what the GPU was doing at the same time (section 5). A wait under busy
GPU work costs little.

## 10. CPU-Side Collectives

Gloo or other host collectives. Split each into the part inside the GPU-active window, because time
outside it is often idle polling for requests.

```sql
WITH win AS (
  SELECT min(ts) AS g0, max(ts + dur) AS g1 FROM ev WHERE rank = getvariable('rank') AND cat = 'kernel')
SELECT name, count(*) AS n, round(sum(dur) / 1000, 1) AS total_ms,
       round(sum(greatest(0, least(ts + dur, g1) - greatest(ts, g0))) / 1000, 1) AS inside_gpu_window_ms
FROM ev, win
WHERE rank = getvariable('rank') AND ph = 'X' AND cat = 'user_annotation'
  AND regexp_matches(name, '^(gloo|nccl):')
GROUP BY 1 ORDER BY total_ms DESC;
```

`gloo:*` ranges are host collectives that block the calling thread. `nccl:*` ranges are only the
host-side enqueue of a GPU collective; its device time is in the kernels (section 11).

## 11. Cross-Rank Collective Alignment

For collectives issued as ordinary kernels, including custom all-reduce kernels inside CUDA graphs
that TraceLens does not classify as communication. Pairs the n-th matching kernel on every rank
and reports how long each rank waited for the last arrival. Check first that every rank has the
same count; if not, narrow the pattern or the window. The ranks' clocks must be comparable, which
holds for one node; check that matched kernels end at nearly the same time.

```sql
SET VARIABLE coll_regex = 'all_reduce_kernel|oneshotAllreduce|ncclDevKernel_AllReduce';
SELECT rank, count(*) AS kernels FROM ev
WHERE cat = 'kernel' AND regexp_matches(name, getvariable('coll_regex')) GROUP BY 1 ORDER BY 1;
```

```sql
WITH c AS (
  SELECT rank, ts, ts + dur AS te,
         row_number() OVER (PARTITION BY rank ORDER BY ts) AS n
  FROM ev WHERE cat = 'kernel' AND regexp_matches(name, getvariable('coll_regex'))),
g AS (SELECT n, max(ts) AS last_arrival, max(te) - min(te) AS end_spread FROM c GROUP BY n),
w AS (SELECT c.rank, c.n, g.last_arrival - c.ts AS wait, g.end_spread FROM c JOIN g USING (n))
SELECT rank, round(sum(wait) / 1000, 1) AS wait_ms, sum(CASE WHEN wait = 0 THEN 1 ELSE 0 END) AS arrived_last,
       count(CASE WHEN wait > 1000 THEN 1 END) AS waits_over_1ms, round(max(end_spread), 1) AS max_end_spread_us
FROM w GROUP BY rank ORDER BY rank;
```

Then list the largest waits to crop and render:

```sql
WITH c AS (
  SELECT rank, ts, ts + dur AS te, row_number() OVER (PARTITION BY rank ORDER BY ts) AS n
  FROM ev WHERE cat = 'kernel' AND regexp_matches(name, getvariable('coll_regex'))),
g AS (SELECT n, max(ts) AS last_arrival, arg_max(rank, ts) AS last_rank FROM c GROUP BY n)
SELECT c.n, c.ts AS start_ts, round((g.last_arrival - c.ts) / 1000, 3) AS wait_ms, g.last_rank
FROM c JOIN g USING (n) WHERE c.rank = getvariable('rank')
ORDER BY wait_ms DESC LIMIT 20;
```

A large `max_end_spread_us` means the pairing is wrong or the clocks disagree. Do not trust the
waits until it is small (a few microseconds on one node).

## 12. Kernel Variants Per Op And Shape

Finds ops whose calls with identical shapes, dtypes and strides run different kernels: a split
dispatch. Each launch is attributed to the latest matching op that is still open on its thread
(an `ASOF` join, which stays fast on large traces where a range join does not). Each call is reduced to the ordered sequence of kernels it launched, and calls are
grouped by that sequence. Because the work is identical, the ratio of median kernel time per call
is the efficiency ratio of the variants, with no roofline needed. Needs `record_shapes=True`, and
ops on the host at launch time, so for CUDA-graph replay use the graph-capture traces
(tracelens.md section 2).

```sql
SET VARIABLE op_regex = '^aten::(mm|addmm|bmm|matmul|linear|_scaled_mm|convolution)$';
WITH k AS (
  SELECT name AS kernel, dur, args->>'correlation' AS corr FROM ev
  WHERE rank = getvariable('rank') AND cat = 'kernel'),
l AS (
  SELECT k.kernel, k.dur, k.corr, r.pid, r.tid, r.ts FROM k JOIN ev r
    ON r.rank = getvariable('rank') AND r.cat IN ('cuda_runtime', 'cuda_driver')
   AND (r.args->>'correlation') = k.corr),
ops AS (
  SELECT name, pid, tid, ts, dur, (args->>'Input Dims') AS dims, (args->>'Input type') AS dtypes,
         (args->>'Input Strides') AS strides FROM ev
  WHERE rank = getvariable('rank') AND cat = 'cpu_op' AND regexp_matches(name, getvariable('op_regex'))),
o AS (   -- the latest matching op that started before each launch and is still open
  SELECT l.kernel, l.dur, l.ts AS launch_ts, op.name AS op, op.ts AS op_ts, op.tid AS op_tid,
         op.dims, op.dtypes, op.strides
  FROM l ASOF JOIN ops op ON l.pid = op.pid AND l.tid = op.tid AND l.ts >= op.ts
  WHERE op.ts + op.dur >= l.ts),
c AS (   -- one row per call: its kernel sequence and summed kernel time
  SELECT op, dims, dtypes, strides, string_agg(kernel, ' + ' ORDER BY launch_ts) AS kernels,
         sum(dur) AS kernel_us
  FROM o GROUP BY op, dims, dtypes, strides, op_ts, op_tid),
v AS (
  SELECT op, dims, dtypes, strides, kernels, count(*) AS calls, median(kernel_us) AS median_us,
         sum(kernel_us) AS total_us FROM c GROUP BY ALL),
s AS (
  SELECT *, count(*) OVER w AS variants, min(median_us) OVER w AS best_median_us,
         sum(calls) OVER w AS group_calls, sum(total_us) OVER w AS group_us FROM v
  WINDOW w AS (PARTITION BY op, dims, dtypes, strides))
SELECT op, left(dims, 60) AS dims, dtypes, left(kernels, 90) AS kernels, calls,
       round(calls / group_calls, 2) AS share_of_calls, round(median_us, 1) AS median_us,
       round(median_us / best_median_us, 2) AS slowdown_vs_best,
       round(calls * (median_us - best_median_us) / 1000, 2) AS excess_ms
FROM s WHERE variants > 1
ORDER BY group_us DESC, median_us
LIMIT 40;
```

`excess_ms` is the kernel time the slower variant spends over the best one; it is removable only
where those kernels are on the critical path. A variant that differs by an extra kernel, such as a
split-K reduction, is still a variant: its median includes that kernel. Profiled kernel durations
are diagnostic; confirm both variants with an unprofiled exact-shape benchmark before forcing one
([kernel-roofline.md](kernel-roofline.md) section 2).

## 13. Limits

- The whole file is parsed as one JSON object, so a trace needs roughly 10x its uncompressed size
  in memory. For multi-GB traces, crop first with `crop_trace.py` or use Perfetto's
  `trace_processor`.
- Flows (`ac2g`) and Python stacks are rows like any other event; there is no prebuilt parent/child
  tree. Reconstruct nesting by interval containment on the same `pid`/`tid`, as in section 6.
- Profiler artifacts stay in the data: CUPTI `overhead` events and the stalls around them look
  like real gaps. Check section 5 before attributing a gap to the application.
