# Session wiring design — DiffLoopState → SessionStore

Design-only gate for issue #268. No production code in this change.

AST-counted on this worktree (`feat/issue-268-session-wiring`):

| type | fields | source |
|---|---:|---|
| `DiffLoopState` | **17** | `loop_state.py:175-192` |
| `SessionState` | **5 total** (`session_id`, `phase`, `before_profile`, `after_profile`, `artifacts`) | `session_store.py:227-234` |
| `Proposal` | **12** | `proposal.py:207-221` |
| `ExpectedImpact` | **2** (`headroom_ms`, `headroom_basis`) | `proposal.py:169-173` |

`kind` is **not** an `ExpectedImpact` dataclass field. It is the constant `"measured_headroom"` emitted by `to_dict` (`proposal.py:189-191`) and required by `from_dict` (`proposal.py:198-202`). An adapter never populates it.

`SessionSnapshot` (`session_store.py:320-325`) is a load view over `SessionState` plus `runspec` / `findings` / `proposal` / `diff`; it is not a sixth `SessionState` field.

Sources read: `loop_state.py`, `session_store.py`, `web.py` (loop handlers), `tree/app.py`, `timeline/app.py`, `proposal.py`, `annotation.py`, `profile_runner.py`, `profile_reference.py`, `fingerprint.py`, `diff_decision.py`, `src/nsys_ai/templates/timeline.js` (packaged path; issue comments call it `templates/timeline.js`), and `gh issue view 268|352 -R GindaChen/nsys-ai --comments`.

Scope constraint (issue #268 comment, recorded decision — not revisited here):

> The CLI drives the loop. The browser and both TUIs **open** a session, render its artifacts, and **record a decision**. `_handle_loop_diagnose`, `_handle_loop_proposal` and `_handle_loop_reprofile` do not execute against a session; in session mode they return a **stated limitation** naming the CLI command that does the job.

---

## Deliverable 1 — DiffLoopState field disposition

Disposition is exactly one of SESSION / DERIVED / DROP / UNRESOLVED. Read sites below are production readers under `src/` that consume the field (or `to_dict` keys the web UI then reads). Writers alone do not earn SESSION.

| field | who reads it (file:line, every site) | disposition | evidence |
|---|---|---|---|
| `before_path` | `loop_state.py:168-169` (`reconcile_h100_loop_paths`); `loop_state.py:197,199` (`to_dict` / label); `loop_state.py:313-329` (`run_diff`); `tree/app.py:152` (ctor seed); `timeline/app.py:178` (ctor seed); `src/nsys_ai/templates/timeline.js:3493-3501` (`before_path` / `before_label`); web handlers return it only via `to_dict` (`web.py:452,465,479,492,508,524,535,543`) | SESSION | Maps to `SessionState.before_profile` (`session_store.py:230`). Types are **not** compatible as-is: loop holds `str` (`loop_state.py:176`); session holds `LocalProfileReference \| None` (`session_store.py:230`, shape `profile_reference.py:19-26`: `path`, `profile_id`, `schema_version`, `product_version`, `kernel_count`). Display path is `before_profile.path`. Construction site is `build_local_profile_reference` (`profile_runner.py:130-255`); see Deliverable 5b. |
| `after_path` | `loop_state.py:170-171,198,200,302-310`; `tree/app.py:152,730`; `timeline/app.py:178,864`; `src/nsys_ai/templates/timeline.js:3329,3494-3510,3724-3733`; web via `to_dict` as above; written by `record_reprofile_artifact` (`loop_state.py:261-265`) and `web.py:491,786` | SESSION | Maps to `SessionState.after_profile` (`session_store.py:231`). Same `str` vs `LocalProfileReference \| None` mismatch. Store writer is `publish_after_profile` (`session_store.py:721-724`), which web/TUI must **not** call under the scope decision (Deliverable 2). |
| `phase` | `loop_state.py:201,246-252`; `tree/app.py:153,691-692,711,754`; `timeline/app.py:179,825-826,845,888`; `src/nsys_ai/templates/timeline.js:3469-3490`; web via `to_dict` | SESSION | Maps to `SessionState.phase` (`session_store.py:229`). Types are compatible: `Phase` (`loop_state.py:18`) and `SessionPhase` (`session_store.py:70`) are the same five-string literal. |
| `selected_scope` | `loop_state.py:202` (`to_dict` only); `loop_state.py:241-242` (`from_dict` validation) | DROP | Repo search under `src/` and `src/nsys_ai/templates/` finds no reader outside `loop_state.py`. `SessionState` has no scope field. Nothing supersedes it beyond deletion. |
| `proposal` | `loop_state.py:203,255-259`; `src/nsys_ai/templates/timeline.js:3329-3330,3517,3694`; web via `to_dict` after `_handle_loop_proposal` (`web.py:478-479`) | DROP | Superseded by the `Proposal` artifact (`proposal.py:207-221`, **twelve** fields AST-counted: `proposal_id`, `source_finding_id`, `source_profile_id`, `summary`, `suggested_actions`, `trace_target`, `expected_impact`, `confidence`, `verification`, `limitations`, `abstained`, `abstention_reason`). Free-text `str` must not be mapped into the store (issue #352 / #268). Session surface after load is `SessionSnapshot.proposal` (`session_store.py:324`), published by CLI via `publish_proposal` (`session_store.py:726-752`). Frontend projection of that artifact is Deliverable 6, not a resurrection of this field. |
| `expected_impact` | `loop_state.py:204,256`; `src/nsys_ai/templates/timeline.js:3522,3687-3702`; web via `to_dict` | DROP | Superseded by `Proposal.expected_impact: ExpectedImpact \| None` (`proposal.py:216`). `ExpectedImpact` has exactly **two** dataclass fields — `headroom_ms`, `headroom_basis` (`proposal.py:172-173`). `kind` is not a field (see header). Free-text `str` is not type-compatible. |
| `decision` | `loop_state.py:205,384`; `src/nsys_ai/templates/timeline.js:3326,3565-3566`; web via `to_dict`; tree/timeline call `set_decision` then read sibling fields (`tree/app.py:748-756`, `timeline/app.py:882-890`) | DERIVED | Not a `SessionState` field. After `publish_decision`, value lives on `SessionSnapshot.diff["decision"]` (`session_store.py:775-829`; shape enforced at `session_store.py:1628-1640`: `status` in `{accepted,rejected}`). Type mismatch: loop / API use `"accept"`/`"reject"` (`loop_state.py:19,364-365`); artifact uses `"accepted"`/`"rejected"`. `publish_decision` normalises both spellings on **input** (`session_store.py:794-798`); only the **read** / projection side needs enum translation for the UI. |
| `decision_reason` | `loop_state.py:206,385`; `src/nsys_ai/templates/timeline.js:3569-3570`; web via `to_dict` | DERIVED | From `diff["decision"]["reason"]` after publication (`diff_decision.py:99-104`; written through `session_store.py:800-806`). |
| `decision_path` | `loop_state.py:207,386`; `src/nsys_ai/templates/timeline.js:3574-3576,3763`; `tree/app.py:756`; `timeline/app.py:890`; web via `to_dict` | DERIVED | Today: filesystem path from `write_diff_decision_json_from_diff_dict` (`loop_state.py:375-386`) — CWD `diff.json` for web, beside the after profile for TUIs (`tree/app.py:723-732`, `timeline/app.py:857-866`). Under SessionStore the artifact is always `<session_dir>/diff.json` (`session_store.py:73-78`, write at `809-811`). Derive as that session path (via `artifacts["diff"].path` relative to the session directory, `session_store.py:214-216` / `_ARTIFACT_PATHS`). |
| `decision_warnings` | `loop_state.py:208,387`; tree/timeline use the **return value** `warnings` from `set_decision` (`tree/app.py:748-758`, `timeline/app.py:882-892`), not a separate JS key | DERIVED | Advisory list from `build_diff_decision_record_from_diff_dict` (`diff_decision.py:84-95,105`). `publish_decision` returns the same class of warnings as its third tuple element (`session_store.py:782,829`) and merges them into `diff["warnings"]` via the builder (`diff_decision.py:93-98`). No separate SessionState field. After reload, only `diff["warnings"]` remains — that list also contains pre-decision warnings, so it is **not** identical to the advisory-only list DiffLoopState stored. |
| `diagnose_ran` | `loop_state.py:209,286`; `src/nsys_ai/templates/timeline.js:3328,3485-3490`; web via `to_dict` | DERIVED | True after `run_diagnose` (`loop_state.py:286`). Derive as `SessionSnapshot.findings is not None` (findings artifact present / loaded at `session_store.py:320-325`). No SessionState bool. |
| `diagnose_findings_count` | `loop_state.py:210,287`; `src/nsys_ai/templates/timeline.js:3562`; web via `to_dict` | DERIVED | Set to `len(ranked)` in `run_diagnose` (`loop_state.py:287`). Derive as `len(snapshot.findings.findings)` from the `EvidenceReport` artifact (`annotation.py:230-251,272-285`). |
| `top_findings` | `loop_state.py:211,288` (`to_dict` / written by `run_diagnose`) | DERIVED | Written as `ranked[:15]` (`loop_state.py:288`) after `_normalize_findings` (`loop_state.py:33-64`). **No** consumer in `src/nsys_ai/templates/timeline.js`, `web.py` handlers beyond blind `to_dict`, `tree/app.py`, or `timeline/app.py`. Derive at read time from `findings.json` (`EvidenceReport.findings`) by applying the same ranking, then `[:15]`. Whether a published EvidenceReport is already opportunity-ranked is UNRESOLVED (Deliverable 4). |
| `diff_summary` | `loop_state.py:212,340,366-377`; `src/nsys_ai/templates/timeline.js:3327,3382,3527-3578,3631-3635,3748`; `tree/app.py:741`; `timeline/app.py:875`; web via `to_dict` | DERIVED | Entire diff payload. Maps to `SessionSnapshot.diff` (`session_store.py:325`), artifact `diff.json`. |
| `comparability_confidence` | `loop_state.py:213,342-344`; `src/nsys_ai/templates/timeline.js:3582-3584`; web via `to_dict` | DERIVED | Copied from diff payload in `run_diff` (`loop_state.py:342-344`). Read from `diff["comparability_confidence"]` (required by session validation at `session_store.py:89,1543-1550`). |
| `verdict` | `loop_state.py:214,341,390`; `src/nsys_ai/templates/timeline.js:3529-3553`; `tree/app.py:713`; `timeline/app.py:847`; web via `to_dict` | DERIVED | From `diff["verdict"]` (`loop_state.py:341`; may be stamped `inconclusive` on decision at `diff_decision.py:96` / `loop_state.py:390`). |
| `last_error` | `loop_state.py:215`; written `web.py:542`; cleared on successful mutations (`loop_state.py:257,265,271,290,346,392`); read `src/nsys_ai/templates/timeline.js:3475` via `to_dict` | DROP | No SessionState / artifact field. HTTP errors already return `{"error": ..., "state": ...}` (`web.py:546`). Durable session restart does not need this string. Sticky survival across a later `GET /api/loop/state` has no store home — UI must treat errors as request-scoped, not as a projected session key (Deliverable 6). |

---

## Deliverable 2 — Web action endpoint → publisher mapping

All five handlers today mutate class-level `_ViewerHandler._loop_state` under `_LOOP_LOCK` (`web.py:27-28,440-535`). Production callers of `SessionStore` remain zero outside tests (repo search: only `session_store.py` defines the class; `propose_command.py` imports `SessionState` only to **reject** session-shaped input).

Scope decision binds the first three endpoints: in **session mode** they do not execute and do not call any `publish_*`. They return a stated limitation naming the CLI command that performs the work. No web propose-path. No free-text proposal field kept alive to satisfy `publish_after_profile`.

### `_handle_loop_diagnose` — `web.py:494-508`

- **Publisher that replaces its loop_state mutation:** **none** in session mode.
- **Finding:** among `publish_runspec` / `publish_findings` / `publish_after_profile` / `publish_proposal` / `publish_diff`, the conceptual match for diagnose output is `publish_findings` (`session_store.py:673-719`), but the endpoint must **not** call it. CLI owns diagnose + publication.
- **Returns today:** `{"state": loop_state.to_dict(), "findings": findings}` (`web.py:508`) where `findings` is the ranked list from `run_diagnose` (`loop_state.py:273-291`), analysis held under `_LOOP_LOCK` (`web.py:495-505`).
- **After change (session mode):** stated limitation naming `nsys-ai evidence build` (CLI verb that builds findings today, `cli/parsers.py:313`). Response must not mutate session state. Browser renders findings from `SessionSnapshot.findings` after a CLI-published session is opened. Note: `evidence build` itself still does not call `SessionStore` today — that CLI publication gap is Deliverable 4 / 5, not a reason to invent a web publisher.

### `_handle_loop_proposal` — `web.py:467-479`

- **Publisher that replaces its loop_state mutation:** **none** in session mode.
- **Finding:** `publish_proposal` (`session_store.py:726-752`) exists, but the endpoint's mutation is free-text `set_proposal` (`web.py:473-478` → `loop_state.py:254-259`). Mapping free text into the store is forbidden by the scope decision and by #352. Building a real `Proposal` needs a selected `Finding` — that is `nsys-ai propose` (`cli/parsers.py:417-438`).
- **Returns today:** `loop_state.to_dict()` (`web.py:479`) with `proposal: str` and `expected_impact: str`.
- **After change (session mode):** stated limitation naming `nsys-ai propose`. Browser renders `SessionSnapshot.proposal` (twelve-field artifact), not a string field.

### `_handle_loop_reprofile` — `web.py:481-492`

- **Publisher that replaces its loop_state mutation:** **none** in session mode.
- **Finding:** conceptual match is `publish_after_profile` (`session_store.py:721-724`), but the endpoint must not call it. Precondition at `_update_state` (`session_store.py:882-890`): phase in `{propose,reprofile}` **and** a non-abstained `Proposal` already present — unreachable from free-text web state, and deliberately not satisfied by inventing a web propose-path.
- **Returns today:** `loop_state.to_dict()` after `record_reprofile_artifact` (`web.py:491-492`, `loop_state.py:261-265`).
- **After change (session mode):** stated limitation naming `nsys-ai profile` for capture/validation (`cli/parsers.py:95`; `profile_command.py` docstring at lines 1-5 states it does not own durable session layout) plus whatever CLI path later calls `publish_after_profile`. Endpoint does not register an after path into the session.

### `_handle_loop_diff` — `web.py:510-524`

- **Publisher that replaces its loop_state mutation:** **none on this endpoint** under the CLI-drives-loop decision.
- **Finding:** conceptual match among the five listed publishers is `publish_diff` (`session_store.py:754-773`). Issue #268 body and the post-#359 comment keep long diff work CLI/optimize-owned. The recorded scope paragraph names stated limitations for diagnose/proposal/reprofile only; it does **not** authorise a web analyze-then-publish path for diff either, because that would reintroduce a second execution path writing the same artifact.
- **Returns today:** `{"state": loop_state.to_dict(), "diff": diff_payload}` (`web.py:524`) from in-process `run_diff` (`loop_state.py:293-347`) under `_LOOP_LOCK`.
- **After change (session mode):** do not run `run_diff` against the session. Browser reads `SessionSnapshot.diff` from an already-published session. Whether the POST returns a stated limitation naming `nsys-ai diff` (`cli/parsers.py:539`) or becomes a no-op reload of the published snapshot is an implementation detail that must not invent a second writer; both keep DiffLoopState out of the path. Prefer a stated limitation for symmetry with the three named endpoints unless a human chooses read-only reload.

### `_handle_loop_decision` — `web.py:526-535`

- **Among the five publishers listed in the mission** (`publish_runspec`, `publish_findings`, `publish_after_profile`, `publish_proposal`, `publish_diff`): **none** matches.
- **Existing API (not invented here):** `SessionWriter.publish_decision` (`session_store.py:775-829`). Preconditions: phase exactly `"diff"` (`:788`); input `accept`/`accepted` and `reject`/`rejected` normalised (`:794-798`).
- **Returns today:** `loop_state.to_dict()` after `set_decision` (`web.py:534-535`), writing CWD `diff.json` via `write_diff_decision_json_from_diff_dict` (`loop_state.py:372-393`).
- **After change:** acquire `SessionStore.writer(session_id)`, call `publish_decision`, return a **projection** of the resulting snapshot (Deliverable 6) — not `DiffLoopState.to_dict()`. Decision fields come from `diff["decision"]`; path from `<session>/diff.json`.

`publish_runspec` has **no** corresponding loop action endpoint among the five.

---

## Deliverable 3 — Where `run_diagnose` / `run_diff` execute

### What they do today

- `DiffLoopState.run_diagnose` (`loop_state.py:273-291`): constructs `EvidenceBuilder`, calls `build()`, ranks findings, **and** mutates loop fields under the caller's lock.
- `DiffLoopState.run_diff` (`loop_state.py:293-347`): opens profiles, runs `diff_profiles` / `to_diff_json`, **and** mutates loop fields.

Web holds `_LOOP_LOCK` for the entire diagnose and diff handlers (`web.py:495-508`, `511-524`). Tree/Timeline call `run_diff` on the UI thread (`tree/app.py:707-713`, `timeline/app.py:841-847`) without a session writer.

### Where the work moves (within the scope decision)

Per `session_store.py:1-4` and issue #268: the store owns persistence only; expensive diagnose / re-profile / diff runs **outside** the state lock; only transitions and publication are serialized.

Per the recorded #268 scope decision and the post-#359 comment: Web/TUI are **read / render / decide**. Long-running diagnose and diff execution remain **CLI-owned**.

Concrete split:

1. **Analysis (no session writer, no state lock):** CLI runs `EvidenceBuilder.build` → `EvidenceReport`, and `diff_profiles` + `to_diff_json` → diff mapping. Same analysis code `run_diagnose` / `run_diff` use today, but not as methods that both execute and own durable UI state.
2. **Publication (serialized, brief):** CLI opens `SessionStore.writer(session_id)` then `publish_findings` / `publish_diff` (and sibling publishers). Web `_handle_loop_decision` is the interactive surface that briefly takes the writer for `publish_decision` only.
3. **Web/TUI session mode:** does not call `run_diagnose` or `run_diff`. No writer lease held across analysis because analysis is not in those processes.

### What holds which lock, and for how long

| lock | where | duration |
|---|---|---|
| `SessionStore.writer()` exclusive **writer** lease | `session_store.py:369-379` (`blocking=False`; conflict → `SessionConflictError`) | Lifetime of the `SessionWriter` context (`__enter__`/`close`, `session_store.py:631-640`). Must cover only publication (`publish_*` / `publish_decision`), never `EvidenceBuilder.build` or `diff_profiles`. |
| Exclusive **state** lock inside each `publish_*` / `publish_decision` | `session_store.py:842-844`, `878-880`, `784-786` (`blocking=True`) | Only load → validate → write artifact/`session.json` → finish journal. Short publication critical section. |
| Today's `_LOOP_LOCK` | `web.py:28,495,511` | Entire diagnose/diff request including analysis — **removed from the session-mode path**. |

`SessionStore.load` also takes the state lock for recovery (`session_store.py:366-367`) but is a read/repair path, not analysis.

---

## Deliverable 4 — Risks

1. **No session_id / session root sourcing on any interactive surface today.** Every endpoint mapping and every TUI open assumes a `session_id` and a `SessionStore(root=...)` that nothing currently provides (Deliverable 5a). Wiring that forgets this cannot load.

2. **CLI still does not publish into sessions.** `profile_command.py:1-5` states it does not own durable session layout; `propose_command.py` rejects session-shaped input; `loop` uses `DiffLoopState` helpers only (issue #268 measured comment). Web/TUI read-render-decide against an empty store until CLI publishers land.

3. **Response-shape break for the web UI.** `src/nsys_ai/templates/timeline.js` reads 16 denormalised `LOOP_STATE` keys (Deliverable 6). `SessionState.to_dict` (`session_store.py:248-261`) emits nothing like that shape. Blind swap loses the stepper/hero rendering.

4. **Stated-limitation UX vs today's free-text / diagnose buttons.** `timeline.js` still posts free text (`:3685-3702`), runs diagnose, and registers after paths. Session mode must surface limitations without silently no-op'ing; otherwise the UI looks broken.

5. **Decision enum mismatch on the read side.** Loop/API/UI use `accept`/`reject` (`loop_state.py:364-365`, `timeline.js:3565,3765`); session diff records `accepted`/`rejected` (`session_store.py:794-799,1639-1640`). Projection must translate or the badge changes meaning.

6. **`publish_after_profile` deadlock if anyone tries to keep web reprofile.** Store requires a non-abstained proposal (`session_store.py:887-890`). Scope decision removes that path deliberately; reintroducing free-text to satisfy the precondition recreates the second source of truth.

7. **TUI `diff.json` location vs session layout.** Tree/Timeline write beside the after profile (`tree/app.py:723-732`, `timeline/app.py:857-866`). SessionStore always writes `<session>/diff.json`. Cross-process handoff tests fail if TUIs keep the old path.

8. **`decision_warnings` round-trip loss.** Advisory-only list is returned by `publish_decision` but not stored as its own artifact; reload only sees full `diff["warnings"]`.

9. **`top_findings` ranking vs persisted findings order.** Loop ranking is `_normalize_findings` (`loop_state.py:33-64`). `publish_findings` stores the `EvidenceReport` as given (`session_store.py:685-718`). If publishers store unranked builder output, deriving `top_findings` as `findings[:15]` silently disagrees with today's ranked slice. I did not verify whether any future CLI publisher ranks before write.

10. **Writer lease held across analysis.** If CLI (or a mistaken web path) acquires `writer()` then runs diagnose/diff before `publish_*`, it violates issue #268 and blocks the conflict model for the whole analysis duration.

11. **Review-session contract still unshipped** (issue #268 post-#359 / post-#363 comments: `mode: loop|review`). Direct before/after review without a proposal still collides with `publish_after_profile`'s proposal requirement. Out of scope for inventing here; it bites anyone trying to open a review-only session under the loop layout.

12. **`_handle_loop_phase` (`web.py:454-465`) is outside the five endpoints** but still mutates phase in memory. Session phases advance via publishers; a free-form `set_phase` has no `publish_*` and desynchronises UI steppers from artifacts if kept in session mode.

13. **Fingerprint cost on naive `str` → `LocalProfileReference` conversion** (Deliverable 5b). Calling `build_local_profile_reference` on every path touch re-opens SQLite and re-runs identity SQL including a full kernel `COUNT(*)`.

14. **CWD-relative default session root.** `SessionStore.__init__` defaults to `.nsys-ai/sessions` resolved from process CWD (`session_store.py:331-332`). A browser server started in a different directory than the CLI that created the session will not see it unless root is passed explicitly.

---

## Deliverable 5 — Gaps the issue body does not cover

### 5a) Session lifecycle — where `session_id` and store `root` come from

| piece | what the store requires | what exists today |
|---|---|---|
| `root` | `SessionStore(root=...)` (`session_store.py:331-332`); default `.nsys-ai/sessions` resolved absolute; locks live at `root.parent / "locks"` (`:333`) | No web/`serve_timeline` parameter (`web.py:727-738`), no tree/timeline constructor argument, no CLI `--session-root` (parsers search shows no `session` flags). Process CWD decides the default. |
| `session_id` | Caller-supplied to `create` / `load` / `writer` (`session_store.py:335-379`); validated by `_SESSION_ID` (`:72`, `:927-930`) | `create` does **not** mint an id. No web query/body field, no TUI prompt, no CLI `--session` flag sources one for open/render/decide. |

Consequence: every Deliverable 2 mapping that says "session mode" is blocked until an explicit open contract exists. Measured absence, not a proposed design:

- `serve_timeline` seeds only `DiffLoopState` from `loop_before` / `loop_after` / H100 preset (`web.py:775-788`).
- Tree/Timeline seed `DiffLoopState(before_path=db_path, after_path=loop_after or "")` (`tree/app.py:152`, `timeline/app.py:178`).
- CLI `loop` passes profile paths into those surfaces (`cli/handlers.py:1398-1414`), never a session id.

What must be decided before implementation (UNRESOLVED product choices, not code):

1. How the user names or discovers `session_id` when opening web/TUI (flag, path to `session.json`, picker over `root`).
2. Whether `root` is always CWD-relative default, an explicit flag, or derived from a session directory path the user passes.
3. Who calls `SessionStore.create` (CLI only, under the scope decision) versus who only `load`s.

### 5b) `str` → `LocalProfileReference`

- Loop fields `before_path` / `after_path` are strings (`loop_state.py:176-177`).
- Session fields `before_profile` / `after_profile` are `LocalProfileReference | None` (`session_store.py:230-231`) carrying path, content-derived `profile_id`, `schema_version`, `product_version`, `kernel_count` (`profile_reference.py:19-26`).
- **Construction site:** `build_local_profile_reference` (`profile_runner.py:130-255`). It opens the `.sqlite` read-only via a descriptor FD (`:194-208`), builds a `Profile`, reads `kernel_count` / schema / product version (`:210-216`), and calls `get_profile_id` (`fingerprint.py:271+`) which hashes capture-time metadata **and** runs `SELECT COUNT(*)` over the resolved kernel table (`fingerprint.py:360-370,431`).
- **Production caller today:** only `LocalProfileRunner` after export validation (`profile_runner.py:534-536`). Web, tree, timeline, and loop handlers never call it. Tests are the other callers.
- **Validation without re-fingerprint:** `validate_local_profile_reference` (`profile_reference.py:164-203`) checks shape + file presence/size; it does **not** recompute `profile_id`.
- **Repeated-call cost:** any path that calls `build_local_profile_reference` on every reprofile POST, every decision, or every GET projection re-opens the DB and re-counts kernels. For large profiles that is multi-second SQLite work proportional to kernel table size, paid on every call. Safe pattern: fingerprint once at CLI publish/`create` time; thereafter pass the stored `LocalProfileReference` and only `validate_local_profile_reference` when the store requires `require_file=True`.

---

## Deliverable 6 — Frontend projection

`src/nsys_ai/templates/timeline.js` reads these **16** `LOOP_STATE` keys (regex inventory of `LOOP_STATE.<name>` matches the issue list exactly):

`after_label`, `after_path`, `before_label`, `before_path`, `comparability_confidence`, `decision`, `decision_path`, `decision_reason`, `diagnose_findings_count`, `diagnose_ran`, `diff_summary`, `expected_impact`, `last_error`, `phase`, `proposal`, `verdict`.

`SessionState.to_dict` (`session_store.py:248-261`) emits `schema_version`, `session_id`, `phase`, `profiles.{before,after}`, `artifacts.{runspec,findings,proposal,diff}` — nothing in the LOOP_STATE shape.

Two keys — `before_label`, `after_label` — are **not** `DiffLoopState` fields. They are computed in `to_dict` (`loop_state.py:199-200`) via `profile_display_name` (`loop_state.py:146-160`). A field-by-field migration of the dataclass misses them.

### Projection key by key (session mode)

Assume input is a loaded `SessionSnapshot` plus the absolute session directory path.

| LOOP_STATE key | projection |
|---|---|
| `phase` | `snapshot.state.phase` |
| `before_path` | `snapshot.state.before_profile.path` if present else `""` |
| `after_path` | `snapshot.state.after_profile.path` if present else `""` |
| `before_label` | `profile_display_name(before_path)` (`loop_state.py:146-160`) — keep the helper; it is not store state |
| `after_label` | `profile_display_name(after_path)` |
| `diagnose_ran` | `snapshot.findings is not None` |
| `diagnose_findings_count` | `len(snapshot.findings.findings)` if findings else `0` |
| `proposal` | **UI must change.** Do not emit free-text. Render `snapshot.proposal` fields (at minimum `summary`, plus structured `expected_impact`, `abstained` / `abstention_reason`). Today's textarea bind at `timeline.js:3517` cannot round-trip a twelve-field artifact. |
| `expected_impact` | **UI must change.** If showing a single string for compatibility during transition, format from `ExpectedImpact.headroom_ms` + `headroom_basis` only — never invent `kind`. Prefer rendering the structured object and retiring the free-text input (`timeline.js:3522,3687-3702`). |
| `diff_summary` | `dict(snapshot.diff)` if present else `null`/`undefined` |
| `verdict` | `snapshot.diff["verdict"]` if diff else `"neutral"` |
| `comparability_confidence` | `snapshot.diff["comparability_confidence"]` if diff else `null` |
| `decision` | translate `snapshot.diff["decision"]["status"]`: `accepted`→`accept`, `rejected`→`reject`; else `null` (UI still speaks accept/reject at `timeline.js:3565,3765`) |
| `decision_reason` | `snapshot.diff["decision"]["reason"]` if decided else `""` |
| `decision_path` | `str(session_dir / "diff.json")` when diff artifact present, else `""` |
| `last_error` | always `""` in the projection. Errors stay on the request (`web.py:546` `error` field / toast). Sticky `LOOP_STATE.last_error` across GET is dropped (Deliverable 1). |

### UI behaviour that must change (not just re-key)

1. **Primary actions for diagnose / save proposal / register after** (`loopSuggestedPhase` / `loopPrimaryLabel` at `timeline.js:3324-3339`, posts at `:3685-3733`) must stop calling the three executing endpoints in session mode and show the stated limitation (CLI command name) instead.
2. **Decision POST** remains live; it must target `publish_decision` and accept the projected response shape.
3. **Stepper `done` logic** that keys off `diagnose_ran` and string `proposal` (`timeline.js:3328-3330,3485-3490`) must key off findings presence and `snapshot.proposal is not None` (and treat abstention explicitly — `proposal.abstained`).

Without that adapter-or-rewrite choice made up front, editing `web.py` alone will ship a session that the existing JS cannot render.

---

## UNRESOLVED register (needs human before code)

1. Exact open contract for `session_id` + store `root` on web and both TUIs (Deliverable 5a).
2. Whether `_handle_loop_diff` in session mode returns a stated limitation naming `nsys-ai diff`, or a read-only reload of `SessionSnapshot.diff` — neither executes analysis; pick one for API stability.
3. Whether persisted `EvidenceReport.findings` order will be opportunity-ranked at CLI `publish_findings` time, for safe `top_findings` derivation.
4. How the proposal/expected_impact controls in `timeline.js` are redesigned to render the twelve-field `Proposal` (minimal: read-only summary + structured impact; full: richer panel).
5. Whether sticky `last_error` across GET must be preserved somehow despite no session field — current disposition is DROP / request-scoped only.
