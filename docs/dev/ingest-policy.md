# Ingest policy

Every command that reads a profile makes the same decision in the same place. This page describes that
decision, the three functions that expose it, and what goes wrong when a call site skips them — which
has happened, more than once, in ways that were invisible until someone pointed a real capture at the
tool.

## One decision, three entry points

The decision lives in `resolve_profile()` in `profile.py` and returns a `ProfileResolution`:

```python
@dataclass(frozen=True)
class ProfileResolution:
    source_path: str        # what the user typed
    resolved_path: str      # what will actually be opened
    storage_kind: StorageKind          # "nsys-rep" | "parquetdir" | "sqlite"
    backend: Literal["sqlite", "parquetdir"]
    cache_mode: Literal["auto", "direct"]
```

Three functions expose it, and which one you want depends on what you are allowed to do:

| Function | Returns | May run `nsys export` | Use when |
|---|---|---|---|
| `resolve_profile()` | `ProfileResolution` | yes | You need the full decision, including the backend |
| `resolve_profile_path()` | `str` | yes | You only need a path to open |
| `find_ingested_profile()` | `ProfileResolution \| None` | **no** | You must not write or convert |

`find_ingested_profile()` is the read-only counterpart. It applies the same precedence but returns
`None` instead of converting, so a caller that must not have side effects — the MCP server is the one
today — can still follow the policy rather than inventing its own rule.

## The decision

```
input is a directory, or ends .parquetdir / .nsys-cache
    policy=sqlite  -> ExportError: SQLite ingest policy cannot open a parquetdir
    otherwise      -> parquetdir backend, read in place

input ends .nsys-rep
    policy=auto or parquetdir -> convert to <stem>.parquetdir, parquetdir backend
    policy=sqlite             -> convert to <stem>.sqlite, sqlite backend, cache_mode=direct

anything else (treated as SQLite)
    policy=parquetdir -> ExportError: Parquetdir ingest requires a parquetdir directory or a .nsys-rep input.
    policy=sqlite     -> sqlite backend, cache_mode=direct
    policy=auto       -> sqlite backend, cache_mode=auto
```

`policy` comes from `resolve_ingest_policy()`, which reads `NSYS_AI_INGEST` (`auto` by default). An
explicit `backend=` argument overrides the environment for one call; that is how `skill run`
implements `--no-cache`.

## Reuse and staleness

Converting a `.nsys-rep` is the expensive step, so `_resolve_parquetdir_path()` reuses an existing
directory when both hold:

- `<stem>.parquetdir` exists, and its mtime is **not older** than the capture's
- `inspect_local_parquetdir()` accepts it as a complete export

Otherwise it runs `nsys export --type=parquetdir --include-blobs=true`. Note the staleness test is
mtime, not content — copying a capture can invalidate a directory whose contents are still correct.
That is deliberate: the cheap test that never serves stale data is better than the expensive one that
sometimes does.

If `nsys` is not on `PATH`, `ExportToolMissingError` names the manual command rather than failing
generically.

## Where the two environment variables are consumed

`NSYS_AI_INGEST` and `NSYS_AI_CACHE_MODE` are easy to confuse. Their values are documented in
[Environment variables](../user/environment-variables.md); what matters here is where each is read:

- **`NSYS_AI_INGEST`** is consumed by `resolve_ingest_policy()`, which is called from inside
  `resolve_profile()`. It therefore reaches exactly those call sites that go through one of the three
  entry points, and no others.
- **`NSYS_AI_CACHE_MODE`** is consumed by `open_auto_db()` in `parquet_cache.py`, one layer down, and
  is irrelevant once the parquetdir backend has been selected.

The consequence is the reason this page exists: a caller that derives a path itself does not fail
loudly, it just silently ignores `NSYS_AI_INGEST`. That is not hypothetical — see below.

## Call it. Do not reimplement it.

Current call sites:

```
resolve_profile_path()   chat.py · cli/handlers.py · diagnose_command.py · optimize_command.py
                         profile_runner.py · propose_command.py · region_mfu.py · tui_textual.py
resolve_profile()        ai/backend/profile_db_tool.py · cli/handlers.py
find_ingested_profile()  mcp_server.py
```

Two call sites once did not, and both produced failures that looked like something else:

- **`skill run`** passed the user's path straight to the SQLite opener. On a capture whose only
  ingested form was a parquetdir, it reported *"This profile has no CUPTI_ACTIVITY_KIND_KERNEL table …
  the capture did not trace that activity kind"* — describing the capture, while a parquetdir holding
  2.19 million kernel rows sat beside it. On a bare `.nsys-rep` it handed the capture's bytes to
  SQLite and returned *"file is not a database"*.
- **The MCP server** required a `.sqlite` sidecar, so it refused captures the CLI read without
  complaint, and read a different backend than `diagnose` did when both existed.

Both are fixed. They are recorded here because the failure mode is the same in each case and it is not
obvious from the symptom: **bypassing the policy does not produce a missing-file error. It produces a
plausible, wrong statement about the capture.**

If you are adding a code path that opens a profile:

- Call one of the three functions. If none fits, the missing case belongs in `profile.py`, not in your
  module.
- Do not derive `<stem>.sqlite` or `<stem>.parquetdir` yourself. The precedence, the staleness rule and
  the policy override all live in one place so they can change in one place.
- If you cannot afford a conversion, use `find_ingested_profile()` and handle `None`. Do not reach for
  a private helper because the public one might export.
- Report an absent store with a command the user can run. `find_ingested_profile()` returning `None`
  means "nothing ingested yet", not "broken profile".

## What a change here has to preserve

- `NSYS_AI_INGEST` reaches every entry point. A new call site that does not consult
  `resolve_ingest_policy()` makes the variable silently partial, which is worse than not having it.
- The read-only guarantee of `find_ingested_profile()`. It must never convert, and never write beside
  the capture.
- `.sqlite` inputs keep working unchanged. They are the compatibility path, not a deprecated one, and
  committed test fixtures are all `.sqlite`.
- Errors name the input and the way out. `ExportToolMissingError` prints the `nsys export` command;
  the parquetdir/sqlite policy conflicts name the environment variable that resolves them.

## Related

- [What to hand nsys-ai](../user/profile-inputs.md) — the same rules from the user's side
- [Environment variables](../user/environment-variables.md) — `NSYS_AI_INGEST` and `NSYS_AI_CACHE_MODE` values
- [Troubleshooting](../user/troubleshooting.md) — the symptoms a policy bypass produces
- [Support matrix](../support-matrix.md) — export schema versions covered by tests
- [doctor](../doctor.md) — what to run when a capture will not open
