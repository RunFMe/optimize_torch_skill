# Campaign Report

Write the report from the ledger, the raw benchmark outputs and the saved evidence, never from
memory. A reader who did not follow the campaign should be able to check every claim against an
artifact.

This is the final deliverable. It contains the experiment table from SKILL.md section 8 (the
Summary table), the four lists from campaign-playbook.md section 12 (shipped exact-intent, shipped
numerically different or behavior-changing, rejected, deferred), and, inside each opportunity's
**Outcome**, the per-experiment fields from benchmark-quality.md section 8.

## Structure

```markdown
# <Workload> optimization report

## Summary
- Objective and metric, boundary, workload, hardware and topology (from the campaign card).
- Baseline -> final on the primary metric: median, p90, n, and the % change, unprofiled.
- Classification of the final result: exact-intent, numerically different, or behavior-changing.

| # | Opportunity | Source | Tier | Status | Isolated effect | Cumulative after |
|---|---|---|---|---|---|---|

## Opportunities
### O1. <Short name>: <ship | reject | defer | retest>
**Noticed.** What the trace showed, with numbers and location (trace file, rank, stream or thread,
timestamps, stage), the source (required check, TraceLens, exploration), and the exposed time.
**Plan.** The mechanism, the change, its Amdahl ceiling, the expected gain, risks, and the
acceptance threshold agreed before implementation.
**Done.** What changed: files, flags, commits, kill switch. How it was validated: tests, parity,
exact dispatch.
**Outcome.** The benchmark table; the same query or table row before and after; the TraceLens
comparison rows; the before/after images; the quality result; the decision and why.

## Rejected and deferred
Configuration, result and reason for each rejection; remaining ceiling and the next decisive
experiment for each deferral.

## Remaining profile
What the final trace says now: GPU time split, the largest remaining exposed costs, and whether
they sit near an empirical roofline.

## Artifacts
Traces, profiler options, crops, images, raw benchmark logs, and the commands that produced them.
```

## Evidence By Tier

Match the depth of **Noticed** and **Outcome** to the opportunity's tier (SKILL.md section 4):

- **Tier A:** one paragraph each. A benchmark table, the before/after number from the query or
  table that exposed it, and one zoomed crop pair if the mechanism is a timing pattern.
- **Tier B:** add an overview image placing the region on the critical path, matched-scale
  before/after zooms, the exposed-time arithmetic, the TraceLens diff rows, and fixed costs with
  break-even.
- **Tier C:** add microbenchmark and roofline tables, evidence from every rank, quality metrics on
  the difficult slices, fallback proof, the reviewer's findings and how they were resolved, and a
  short explanation of the dependency chain that makes the time removable.

## Images

- Caption every image with trace file, rank, workload shape, stage, window timestamps, and the
  measured exposed time.
- Show before and after at the same zoom, lanes and time scale.
- Show the neighboring CPU, GPU and communication lanes that make the time removable, not only a
  long bar.
- Label stand-in renders that were not produced by Perfetto.

## Claims

- Speed claims come only from unprofiled benchmarks; trace deltas confirm the mechanism.
- Report isolated and cumulative effects separately and never add isolated percentages.
- Report numerically different and behavior-changing contributions separately.
- Keep TraceLens numbers that failed a cross-check (tracelens.md section 6) out of the report, or
  show the corrected value with the reason.
