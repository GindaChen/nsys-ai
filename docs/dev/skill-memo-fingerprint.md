# Skill memo computation identity

This page records the memo-key contract for built-in analysis skills. It is a
developer document: the source of truth remains `Skill.execute`, but the table
below is the review checklist for changing a skill's memo identity.

## Why this exists

The per-connection memo from #295 is deliberately keyed by the skill name and
resolved call parameters. That strict default protects correctness: a
`gpu_idle_gaps(limit=1)` result cannot be returned to a caller that asked for
`limit=5`, and a result for GPU 0 cannot be returned for GPU 1.

One `EvidenceBuilder` pass also has composite callers. The health manifest and
root-cause matcher forward context through the same `Skill.execute` boundary.
Some of that context is not read by the leaf skill, so raw call signatures can
make the same computation look different. The safe abstraction is therefore a
computation fingerprint:

```text
skill identity + input/profile identity + answer-affecting parameters
```

The current implementation still scopes the memo to one connection. The
profile/package identity part of the fingerprint is a future cross-run cache
concern; this page only defines the parameter portion of the fingerprint.

## Static audit for #350

The following trace was collected from a warm `EvidenceBuilder` pass on the
committed `h100_2gpu_1s.sqlite` fixture. It is a call-path inventory, not a
performance claim about large captures.

| Skill | Repeated call shapes | Parameters read by the skill | Parameters deliberately excluded |
|---|---|---|---|
| `sync_cost_analysis` | direct pipeline: `device=0`, trim, injected `overhead_ns`; manifest child: `_skip_device_validation` and no `device` | `device`, `trim_start_ns`, `trim_end_ns` | `overhead_ns`, `communicator_data`; `_skip_device_validation` is consumed before the skill runs |
| `overlap_breakdown` | direct pipeline; manifest child; root-cause call carrying `communicator_data` | `device`, `trim_start_ns`, `trim_end_ns` | `overhead_ns`, `communicator_data`; the nested sync lookup has its own memo key |
| `gpu_idle_gaps` | pipeline `limit=5`; manifest `limit=1`; root-cause default `limit=20` | `device`, trim, `min_gap_ns`, `limit` | `overhead_ns`, `communicator_data` |
| `iteration_timing` | direct pipeline; manifest child with `_skip_device_validation` | `device`, `marker`, `trim_start_ns`, `trim_end_ns` | `overhead_ns`, `communicator_data` |

### Parameter matrix

| Parameter | Where it comes from | Read by these four `execute_fn`s? | Memo treatment |
|---|---|---:|---|
| `device` | `EvidenceBuilder`, manifest, root-cause matcher | Yes | Included |
| `trim_start_ns`, `trim_end_ns` | shared runner and transport trim | Yes | Included |
| `limit` | skill caller, especially `gpu_idle_gaps` | `gpu_idle_gaps` only | Included for `gpu_idle_gaps` |
| `min_gap_ns` | `gpu_idle_gaps` caller/default | `gpu_idle_gaps` | Included |
| `marker` | `iteration_timing` caller/default | `iteration_timing` | Included |
| `overhead_ns` | injected by `Skill.execute` | No | Excluded for these four |
| `communicator_data` | manifest → root-cause handoff | No | Excluded for these four |
| `_skip_device_validation` | manifest child-call policy | No; popped before resolution | Excluded globally by consumption |

The manifest's `sync_cost_analysis` call without `device` is intentionally
not merged with `device=0`: omitted `device` means the sync implementation's
all-device aggregate. This is the counterexample that makes a name-only key
unsafe.

## Implementation rule

`Skill.memo_key_params` is an explicit opt-in allow-list. Skills without it
retain the strict key over every resolved parameter. An allow-list must satisfy
all of the following:

1. Every parameter read by the skill's `execute_fn` is included.
2. Parameters omitted from the list are forwarded context or presentation
   inputs that cannot change returned rows.
3. Defaults are resolved before key construction, so omitted and explicit
   defaults share a computation identity.
4. The skill has tests for both sides: ignored context reuses a result, while
   each answer-affecting parameter still splits the memo.

Do not add a skill to an allow-list because two calls happen to return the same
rows on a small fixture. Read the implementation first, then add the narrowest
safe set and a regression test.

## What this does not claim

This change does not claim that a large profile's warm pass saves a fixed
number of seconds. The original #350 measurement used a 3.5 GB capture that is
not committed here; the repository fixture is suitable for correctness tests,
not for reproducing that wall-clock result. A large-profile benchmark remains
the next verification step before expanding the allow-list or introducing a
cross-process cache.
