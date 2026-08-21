# Surface adapters

CLI, TUI, Web, MCP, and chat are different ways to enter the product. They are not different
analysis implementations. A surface owns transport concerns — parsing input, managing lifecycle,
rendering progress, and emitting its protocol — then delegates profile work to the existing policy,
runner, skill, and session contracts.

The rule is simple:

> A new surface adapts an existing decision point. It does not reimplement the decision point.

This page is the maintainer checklist for adding or reviewing a surface. The detailed ingest rules
remain in [ingest policy](./ingest-policy.md); the detailed skill rules remain in
[skill contract](./skill-contract.md).

## The decision-point map

Use the owner in the right column. If a surface needs behaviour that is not represented there, add
the missing policy to the owner module first and make every surface use it.

| Decision | Owner | Surface entry point | Do not do this in an adapter |
|---|---|---|---|
| Turn an input path into an open profile | `profile.py` | `resolve_profile()`, `resolve_profile_path()`, or read-only `find_ingested_profile()` | Derive `<stem>.sqlite` or `<stem>.parquetdir` yourself |
| Validate a time window | `cli/handlers.py` | `_check_trim_window()` / `_resolve_trim_window()` | Let an empty or out-of-range window become a clean zero result |
| Select and execute analysis | `agent/runner.py` and `skills/registry.py` | `run_diagnose_pack()`, `run_question_evidence()`, `answer_question()`, or `run_skill()` | Call a builtin's private executor or make chat write profile SQL |
| Keep model/tool output bounded | `tool_dispatch.py` and the transport's bounded result helper | `_serialize_skill_result()` for chat; `_cap_rows()` and `MCP_MAX_PAYLOAD_CHARS` for MCP | Serialize an unbounded skill result into a prompt or protocol response |
| Resolve a handoff location | `session_cli.py` and `SessionStore` | `resolve_session_location()` plus the session publisher/reader | Recreate session-id/path rules or write a second session database |
| Mark unavailable evidence | `skills/base.py` and `agent/runner.py` | `abstain()`, `is_abstention_row()`, `is_abstention()` | Turn unavailable data into `[]`, a zero, or a normal Finding |
| Publish a response | The surface adapter | Rich text, SSE events, MCP JSON, or TUI messages | Change the meaning, severity, or provenance of a canonical row |

The same profile may be opened by several surfaces. That is why these functions are seams rather
than convenience helpers: a policy change must reach every entry point, including a surface that is
read-only or runs outside the CLI process.

## The adapter sequence

For a new command, route, tool, or panel, work through the same sequence:

```text
Load input
  → resolve profile through profile.py
  → validate --trim / time scope
  → choose the shared runner or registry skill
  → collect structured rows and typed unavailable states
  → bound and serialize the result for this transport
  → publish the session handoff through SessionStore, when requested
  → render the transport response
```

The surface may stream progress between these steps. Streaming does not make it a second analysis
path: the Web/SSE loop still calls the shared runner, and the TUI still consumes the same structured
session artifacts that the CLI publishes.

### Read-only is still policy-aware

`find_ingested_profile()` exists for a surface that must not convert or write beside a capture. It
returns the already-ingested resolution or `None`; it does not justify reaching into a private helper
or assuming a `.sqlite` sidecar. The MCP server uses this shape so its read-only promise and the CLI's
parquetdir-first policy agree.

If the surface is allowed to prepare a profile, use `resolve_profile()` or
`resolve_profile_path()`. If it is not, use `find_ingested_profile()` and return a typed, actionable
absence. Do not silently fall back to a different backend because the preferred one is missing.

### Trim is a semantic boundary

The CLI receives seconds and converts them to nanoseconds before analysis. The shared trim guard
checks the requested interval against the capture clock, which may not start at zero. Every command
that accepts a window must reject a non-overlapping range with the same out-of-range error rather
than passing an empty interval to a skill.

After the guard, pass `trim_start_ns` and `trim_end_ns` to the runner or skill. Do not validate the
window in the HTTP handler, the TUI widget, and the CLI separately: six commands once drifted this
way, and a capture outside the selected window looked like a healthy profile.

### Bounded output is part of the contract

Different protocols have different budgets, but none may be unbounded:

- The chat dispatcher uses `_serialize_skill_result()` and the shared
  `DEFAULT_MAX_JSON_CHARS` budget. It adds truncation metadata rather than silently dropping the
  tail.
- MCP caps row counts with `max_rows` and rejects a payload above `MCP_MAX_PAYLOAD_CHARS`. Its
  larger budget is intentional because MCP returns a structured contract, not a prompt fragment.
- A Web/SSE endpoint should emit bounded events or a session reference, not inline a complete profile
  tree or every skill row.

The cap belongs at the boundary that hands data to a model or protocol. Do not reduce the SQL result
so early that deterministic analysis changes, and do not add a new uncapped serializer because an
existing one is inconvenient.

## What a surface may own

An adapter may own:

- protocol parsing and validation (`argparse`, HTTP JSON, MCP arguments, or TUI actions);
- lifecycle and cancellation (a worker, SSE connection, or terminal event loop);
- progress and presentation (`text`, `action`, `done`, Rich output, or widgets);
- transport-specific row limits, provided the canonical result remains available through the
  session or bounded response contract;
- compatibility translation when an old caller uses a legacy argument shape.

An adapter may not own:

- a second profile-ingest policy;
- a second keyword/LLM skill selector with a different cap or registry;
- an alternate interpretation of abstention, trim, comparability, or verdict;
- a private session layout or independent answer database;
- direct profile SQL for a question already represented by a registered skill.

The LLM is a planner or synthesizer. It may select registered skills and summarize rows already
queried. It is not a new SQL author hidden inside a transport.

## Historical bypasses and their symptoms

These are the failure patterns reviewers should recognize. Each produced a plausible answer rather
than an obvious crash, which is why looking only for exceptions is insufficient.

| Bypass | What the user saw | Correct owner |
|---|---|---|
| A skill command opened the user's path as SQLite instead of resolving ingest | “No `CUPTI_ACTIVITY_KIND_KERNEL` table” or “file is not a database” while a valid parquetdir existed beside the capture | `resolve_profile()` / `resolve_profile_path()` |
| The MCP server required a `.sqlite` sidecar | MCP refused a capture the CLI could read, or read a different backend when both existed | `find_ingested_profile()` |
| Several commands skipped the shared trim-range guard | An out-of-range window returned `(No kernels found)`, indistinguishable from a capture with no CUDA kernels | `_check_trim_window()` before runner/skill execution |
| The chat `run_skill` path serialized rows directly | A schema skill placed tens of thousands of characters into the model context while its sibling SQL tool had an 8,000-character cap | `_serialize_skill_result()` |

When reviewing a new surface, ask “which owner does this call?” before asking “does this endpoint
return 200?” A 200 response can still contain the wrong backend, the wrong time range, or an
unbounded answer.

## Adding a surface

### 1. Specify the transport contract

Write down the verb, required inputs, output envelope, error codes, and whether the surface may
write. For example, a read-only MCP tool has a profile path, skill name, bounded parameters, and a
JSON result; it does not acquire a session writer lease.

Do not add a new verb only because two surfaces use different words. Prefer the canonical operations:

```text
diagnose | ask | diff | optimize_step
```

The transport can expose a friendly alias, but it should land on one of those workflows and one
session contract.

### 2. Pick the shared execution seam

Use the narrowest existing seam:

- fixed diagnosis → `run_diagnose_pack()`;
- question-driven analysis → `run_question_evidence()` or `answer_question()`;
- one registered analysis → `registry.run_skill(..., raw=True)`;
- pair comparison → the canonical diff command/functions;
- session handoff → `resolve_session_location()` and the existing `SessionStore` publishers.

If none fits, do not copy a neighbouring implementation. Add the missing transport-neutral function,
give it focused tests, and then make the new surface a thin adapter.

### 3. Preserve typed failure states

Pass domain exceptions and abstention markers through the surface's native envelope. A CLI may map a
usage error to exit 2, Web may return its JSON error body, and MCP may return its error object, but the
underlying reason and scope must remain visible. “No data” is not a universal error message: it may
mean no rows, no applicable table, an invalid window, or a refused comparison.

In particular:

- `[]` means a skill ran and found nothing;
- `_abstained: true` means the skill could not answer and must carry a reason;
- a raised execution error means the operation did not finish;
- a gate may turn an inconclusive result into a non-zero assertion outcome, but must still publish
  the inconclusive reason.

### 4. Publish or read the handoff once

If the surface supports `--session` or its equivalent, resolve the same `SessionLocation` and use the
existing `SessionStore` writer/reader. The directory is the handoff between processes; it is not a
second analysis database. Publish complete artifacts atomically, and let a later surface read them
rather than rerunning analysis just to reconstruct the previous answer.

If the surface is read-only, use the session reader and never acquire a writer lease. If a workflow
step is not supported by that surface, say so and print the canonical next command; do not accept a
flag that silently does nothing.

## The command enumeration trap

This repository intentionally has two argparse trees:

```text
nsys_ai.cli.app.main()
  ├─ _build_parser()         public/promoted commands
  └─ _build_legacy_parser()  compatibility and older command families
```

`LEGACY_ROUTED_COMMANDS` decides which tree `main()` chooses. The parser is not the same thing as the
handler table: `baseline`, `evidence`, `skill`, and other command families have nested subcommands,
and one top-level handler can dispatch several actions.

Therefore a coverage check that only walks `_cmd_*` functions in `cli/handlers.py` is incomplete. It
can miss a subparser, advertise a command that the selected parser rejects, or make a command
reachable but absent from top-level help.

Use the same enumeration strategy as `tests/test_cli.py`:

1. collect subcommands from both `_build_parser()` and `_build_legacy_parser()`;
2. account for `LEGACY_ROUTED_COMMANDS` and the routing logic in `cli.app.main()`;
3. walk nested subparsers when a family has subcommands;
4. compare the resulting dispatch set with `--help` and with the command's parser acceptance;
5. test at least one real invocation path for a newly exposed command.

This is why the existing CLI tests assert dispatchable commands are visible and that the legacy
routing set matches the parser that serves it. Extend those tests when adding a new command; do not
replace them with a source-text search.

## Review checklist

- [ ] The surface declares its verb, input, output, write/read-only status, and bounded budget.
- [ ] Profile resolution goes through the owner in `profile.py`.
- [ ] Trim validation is shared and occurs before analysis.
- [ ] Analysis calls the runner or registry, never a private skill executor or ad-hoc SQL.
- [ ] LLM use is limited to registered-skill planning or synthesis of existing evidence.
- [ ] Abstentions, empty results, typed errors, and inconclusive verdicts remain distinguishable.
- [ ] Output is bounded at the transport boundary and includes truncation metadata when applicable.
- [ ] Session mode uses `SessionLocation` / `SessionStore`, or explicitly rejects unsupported writes.
- [ ] Both argparse trees and nested command families are covered if the surface is CLI-facing.
- [ ] The focused surface tests pass, followed by the full suite and a real-profile checkpoint when
      the surface opens or analyzes a profile.

The goal is not to make every surface look identical. The goal is for every surface to tell the same
truth about the same profile, with the session directory carrying the handoff between them.
