# Session wiring design — DiffLoopState → SessionStore

Design-only gate for issue #268. No production code in this change.

The design has since landed, so the "measured absence today" readings below are a
record of the state it was designed against, not of current code. Where they name
`loop_after`, that parameter no longer exists: the session store replaced it and
the surfaces now open a session instead.

AST-counted on this worktree (`feat/issue-268-session-wiring`):

| type | fields | source |
|---|---:|---|
| `DiffLoopState` | **17** | `loop_state.py:175-192` |
| `SessionState` | **5 total** (`session_id`, `phase`, `before_profile`, `after_profile`, `artifacts`) | `session_store.py:227-234` |
| `Proposal` | **12** | `proposal.py:207-221` |
| `ExpectedImpact` | **2** (`headroom_ms`, `headroom_basis`) | `proposal.py:169-173` |

`kind` is **not** an `ExpectedImpact` dataclass field. It is the constant `"measured_headroom"` emitted by `to_dict` (`proposal.py:188-191`) and required by `from_dict` (`proposal.py:198-202`). An adapter never populates it.

`SessionSnapshot` (`session_store.py:320-325`) is a load view over `SessionState` plus `runspec` / `findings` / `proposal` / `diff`; it is not a sixth `SessionState` field.

Sources read: `loop_state.py`, `session_store.py`, `web.py` (loop handlers + findings), `tree/app.py`, `timeline/app.py`, `proposal.py`, `annotation.py`, `profile_runner.py`, `profile_reference.py`, `fingerprint.py`, `diff_decision.py`, `src/nsys_ai/templates/timeline.js` (packaged path; issue comments call it `templates/timeline.js`), and `gh issue view 268|352 -R GindaChen/nsys-ai --comments`.

Recorded contracts on #268 (C1–C5 decided; C6 open). Verified against the code where claimed below. Scope that binds Deliverables 2–3:

> The CLI drives the loop. The browser and both TUIs **open** a session, render its artifacts, and **record a decision**. `_handle_loop_diagnose`, `_handle_loop_proposal` and `_handle_loop_reprofile` do not execute against a session; in session mode they return a **stated limitation** naming the CLI command that does the job.

---

## Contract check (C1–C5 hold; deviations called out)

| id | claim | holds? | evidence |
|---|---|---|---|
| C1 | Root defaults to `.nsys-ai/sessions`; id caller-supplied or derived from before content id | Root default **holds**. `--session` / derived-id open path **does not exist yet** | `SessionStore.__init__` default `".nsys-ai/sessions"` (`session_store.py:331`). `create` takes caller `session_id` (`:335`). `cli/parsers.py` contains the string `session` **zero** times. `serve_timeline` has no session parameter (`web.py:727-738`). Tree/Timeline seed `DiffLoopState` from paths (`tree/app.py:152`, `timeline/app.py:178`). |
| C2 | `publish_decision` raises on second decision; UI must disable after first | Store **holds**. UI **does not** yet | Raise at `session_store.py:791-792`. `publish_diff` requires undecided (`:759`, `:1621-1624`). `DiffLoopState.set_decision` overwrites freely (`loop_state.py:349-393`). `loopSetStepUi` keeps accept/reject enabled whenever suggested phase is `accept` (`timeline.js:3415-3416`); `loopSuggestedPhase` returns `accept` whenever `s.decision` **or** `s.diff_summary` (`:3326-3327`), so buttons stay live after a decision. |
| C3 | Overlay loads from `SessionSnapshot.findings` in session mode | Required; **not implemented** | `_findings` at `web.py:189`; GET `/api/findings` at `:218-220`; populated by diagnose (`:507`) or startup `--findings` (`:764`). |
| C4 | TUIs stop writing `diff.json` beside the after profile in session mode | Required; **not implemented** | Write sites `tree/app.py:723-732`, `timeline/app.py:857-866`. Store path always `<session>/diff.json` (`session_store.py:73-78`, write at `:809-811`). |
| C5 | `decision_warnings` DROP; `decision_path` gated on `diff["decision"] is not None` | Applied in Deliverables 1 and 6 | `publish_decision` returns warnings as third tuple element (`session_store.py:829`) but only `diff["warnings"]` is persisted (`diff_decision.py:93-98` via builder). UI announces path whenever key non-empty (`timeline.js:3572-3577`). |

---

## Deliverable 1 — DiffLoopState field disposition

Disposition is exactly one of SESSION / DERIVED / DROP / UNRESOLVED. Read sites are production readers under `src/` that consume the field (or `to_dict` keys the web UI then reads). Writers alone do not earn SESSION. C5 applied: `decision_warnings` is DROP.

| field | who reads it (file:line, every site) | disposition | evidence |
|---|---|---|---|
| `before_path` | `loop_state.py:168-169` (`reconcile_h100_loop_paths`); `loop_state.py:197,199` (`to_dict` / label); `loop_state.py:313-329` (`run_diff`); `tree/app.py:152` (ctor seed); `timeline/app.py:178` (ctor seed); `src/nsys_ai/templates/timeline.js:3493-3501`; web handlers return it only via `to_dict` (`web.py:452,465,479,492,508,524,535,543`) | SESSION | Maps to `SessionState.before_profile` (`session_store.py:230`). Types are **not** compatible as-is: loop holds `str` (`loop_state.py:176`); session holds `LocalProfileReference \| None` (`session_store.py:230`, shape `profile_reference.py:19-26`: `path`, `profile_id`, `schema_version`, `product_version`, `kernel_count`). Display path is `before_profile.path`. Construction site is `build_local_profile_reference` (`profile_runner.py:130`); see Deliverable 5b. Under C1 the content `profile_id` on this reference is also the default session id. |
| `after_path` | `loop_state.py:170-171,198,200,302-310`; `tree/app.py:152,730`; `timeline/app.py:178,864`; `src/nsys_ai/templates/timeline.js:3329,3494-3510,3724-3733`; web via `to_dict`; written by `record_reprofile_artifact` (`loop_state.py:261-265`) and `web.py:491,786` | SESSION | Maps to `SessionState.after_profile` (`session_store.py:231`). Same `str` vs `LocalProfileReference \| None` mismatch. Store writer is `publish_after_profile` (`session_store.py:721-724`), which web/TUI must **not** call under the scope decision (Deliverable 2). |
| `phase` | `loop_state.py:201,246-252`; `tree/app.py:153,691-692,711,754`; `timeline/app.py:179,825-826,845,888`; `src/nsys_ai/templates/timeline.js:3469-3490`; web via `to_dict` | SESSION | Maps to `SessionState.phase` (`session_store.py:229`). Types are compatible: `Phase` (`loop_state.py:18`) and `SessionPhase` (`session_store.py:70`) are the same five-string literal. |
| `selected_scope` | `loop_state.py:202` (`to_dict` only); `loop_state.py:241-242` (`from_dict` validation) | DROP | Repo search under `src/` finds no reader outside `loop_state.py`. `SessionState` has no scope field. Nothing supersedes it beyond deletion. |
| `proposal` | `loop_state.py:203,255-259`; `src/nsys_ai/templates/timeline.js:3329-3330,3517,3694`; web via `to_dict` after `_handle_loop_proposal` (`web.py:478-479`) | DROP | Superseded by the `Proposal` artifact (`proposal.py:207-221`, **twelve** fields AST-counted: `proposal_id`, `source_finding_id`, `source_profile_id`, `summary`, `suggested_actions`, `trace_target`, `expected_impact`, `confidence`, `verification`, `limitations`, `abstained`, `abstention_reason`). Free-text `str` must not be mapped into the store (issue #352 / #268). Session surface after load is `SessionSnapshot.proposal` (`session_store.py:324`), published by CLI via `publish_proposal` (`session_store.py:726-752`). Frontend projection of that artifact is Deliverable 6, not a resurrection of this field. |
| `expected_impact` | `loop_state.py:204,256`; `src/nsys_ai/templates/timeline.js:3522,3687-3702`; web via `to_dict` | DROP | Superseded by `Proposal.expected_impact: ExpectedImpact \| None` (`proposal.py:216`). `ExpectedImpact` has exactly **two** dataclass fields — `headroom_ms`, `headroom_basis` (`proposal.py:172-173`). `kind` is not a field (see header). Free-text `str` is not type-compatible. |
| `decision` | `loop_state.py:205,384`; `src/nsys_ai/templates/timeline.js:3326,3565-3566`; web via `to_dict`; tree/timeline call `set_decision` then read sibling fields (`tree/app.py:748-756`, `timeline/app.py:882-890`) | DERIVED | Not a `SessionState` field. After `publish_decision`, value lives on `SessionSnapshot.diff["decision"]` (`session_store.py:775-829`; shape enforced at `session_store.py:1628-1640`: `status` in `{accepted,rejected}`). Type mismatch: loop / API use `"accept"`/`"reject"` (`loop_state.py:19,364-365`); artifact uses `"accepted"`/`"rejected"`. `publish_decision` normalises both spellings on **input** (`session_store.py:794-798`); only the **read** / projection side needs enum translation for the UI. |
| `decision_reason` | `loop_state.py:206,385`; `src/nsys_ai/templates/timeline.js:3569-3570`; web via `to_dict` | DERIVED | From `diff["decision"]["reason"]` after publication (`diff_decision.py` builder used at `session_store.py:800-806`). |
| `decision_path` | `loop_state.py:207,386`; `src/nsys_ai/templates/timeline.js:3574-3576,3763`; `tree/app.py:756`; `timeline/app.py:890`; web via `to_dict` | DERIVED | Today: filesystem path from `write_diff_decision_json_from_diff_dict` (`loop_state.py:375-386`) — CWD `diff.json` for web, beside the after profile for TUIs (`tree/app.py:723-732`, `timeline/app.py:857-866`). Under SessionStore the artifact is always `<session_dir>/diff.json` (`session_store.py:73-78`, `:809-811`). **C5:** emit the session `diff.json` path only when `snapshot.diff is not None and snapshot.diff.get("decision") is not None`. Gating on artifact presence alone makes `timeline.js:3572-3577` announce a decision that has not been made. |
| `decision_warnings` | `loop_state.py:208,387`; tree/timeline use the **return value** `warnings` from `set_decision` (`tree/app.py:748-758`, `timeline/app.py:882-892`), not a JS key | DROP | **C5 fixed.** `publish_decision` returns advisory warnings as its third tuple element (`session_store.py:829`) but only `diff["warnings"]` survives reload, and that list also holds pre-decision warnings (`diff_decision.py:93-98`). The advisory-only list is not recoverable. Nothing supersedes it as a durable field; callers that need the ephemeral list must use the `publish_decision` return value in-process and must not expect it after restart. |
| `diagnose_ran` | `loop_state.py:209,286`; `src/nsys_ai/templates/timeline.js:3328,3485-3490`; web via `to_dict` | DERIVED | True after `run_diagnose` (`loop_state.py:286`). Derive as `SessionSnapshot.findings is not None` (findings artifact present / loaded at `session_store.py:320-325`). No SessionState bool. |
| `diagnose_findings_count` | `loop_state.py:210,287`; `src/nsys_ai/templates/timeline.js:3562`; web via `to_dict` | DERIVED | Set to `len(ranked)` in `run_diagnose` (`loop_state.py:287`). Derive as `len(snapshot.findings.findings)` from the `EvidenceReport` artifact (`annotation.py:230-251`). |
| `top_findings` | `loop_state.py:211,288` (`to_dict` / written by `run_diagnose`) | DROP | Written as `ranked[:15]` (`loop_state.py:288`) after `_normalize_findings` (`loop_state.py:33-64`). **No** consumer in `src/nsys_ai/templates/timeline.js`, `web.py` handlers beyond blind `to_dict`, `tree/app.py`, or `timeline/app.py`. Superseded by `SessionSnapshot.findings` / GET `/api/findings` (C3) for the overlay. |
| `diff_summary` | `loop_state.py:212,340,366-377`; `src/nsys_ai/templates/timeline.js:3327,3382,3527-3578,3631-3635,3748`; `tree/app.py:741`; `timeline/app.py:875`; web via `to_dict` | DERIVED | Entire diff payload. Maps to `SessionSnapshot.diff` (`session_store.py:325`), artifact `diff.json`. |
| `comparability_confidence` | `loop_state.py:213,342-344`; `src/nsys_ai/templates/timeline.js:3582-3584`; web via `to_dict` | DERIVED | Copied from diff payload in `run_diff` (`loop_state.py:342-344`). Read from `diff["comparability_confidence"]` (required by session validation at `session_store.py:89,1543-1550`). |
| `verdict` | `loop_state.py:214,341,390`; `src/nsys_ai/templates/timeline.js:3529-3553`; `tree/app.py:713`; `timeline/app.py:847`; web via `to_dict` | DERIVED | From `diff["verdict"]` (`loop_state.py:341`; may be stamped `inconclusive` on decision at `diff_decision.py` / `loop_state.py:390`). |
| `last_error` | `loop_state.py:215`; written `web.py:542`; cleared on successful mutations (`loop_state.py:257,265,271,290,346,392`); read `src/nsys_ai/templates/timeline.js:3475` via `to_dict` | DROP | No SessionState / artifact field. HTTP errors already return `{"error": ..., "state": ...}` (`web.py:546`). Durable session restart does not need this string. Sticky survival across a later `GET /api/loop/state` has no store home — UI must treat errors as request-scoped (Deliverable 6). |

No DiffLoopState field is left UNRESOLVED. The only open choice allowed by C6 is the session-mode behaviour of `_handle_loop_diff` (Deliverable 2).

---

## Deliverable 2 — Web action endpoint → publisher mapping

All five handlers today mutate class-level `_ViewerHandler._loop_state` under `_LOOP_LOCK` (`web.py:27-28,440-535`). Production callers of `SessionStore` remain zero outside tests (repo search: only `session_store.py` defines the class; `propose_command.py` imports `SessionState` only to **reject** session-shaped input).

Scope decision binds diagnose / proposal / reprofile: in **session mode** they do not execute and do not call any `publish_*`. They return a stated limitation naming the CLI command that performs the work. No web propose-path. No free-text proposal field kept alive to satisfy `publish_after_profile`.

### `_handle_loop_diagnose` — `web.py:494-508`

- **Publisher that replaces its loop_state mutation:** **none** in session mode.
- **Finding:** among `publish_runspec` / `publish_findings` / `publish_after_profile` / `publish_proposal` / `publish_diff`, the conceptual match for diagnose output is `publish_findings` (`session_store.py:673-719`), but the endpoint must **not** call it. CLI owns diagnose + publication. Store precondition `_validate_findings_provenance` (`:951`) requires `EvidenceReport.profile_id` and `profile_path` to equal the session before profile, and every `finding.selection.profile_id` to match — a CLI publisher problem, not a web one.
- **Returns today:** `{"state": loop_state.to_dict(), "findings": findings}` (`web.py:508`) where `findings` is the ranked list from `run_diagnose` (`loop_state.py:273-291`), analysis held under `_LOOP_LOCK` (`web.py:495-505`).
- **After change (session mode):** stated limitation naming `nsys-ai evidence` (CLI verb that builds findings today, `cli/parsers.py:313`). Response must not mutate session state. Browser renders findings from `SessionSnapshot.findings` after a CLI-published session is opened (see C3 row below).

### `_handle_loop_proposal` — `web.py:467-479`

- **Publisher that replaces its loop_state mutation:** **none** in session mode.
- **Finding:** `publish_proposal` (`session_store.py:726-752`) exists, but the endpoint's mutation is free-text `set_proposal` (`web.py:473-478` → `loop_state.py:254-259`). Mapping free text into the store is forbidden by the scope decision and by #352. Building a real `Proposal` needs a selected `Finding` — that is `nsys-ai propose` (`cli/parsers.py:417-438`).
- **Returns today:** `loop_state.to_dict()` (`web.py:479`) with `proposal: str` and `expected_impact: str`.
- **After change (session mode):** stated limitation naming `nsys-ai propose`. Browser renders `SessionSnapshot.proposal` (twelve-field artifact), not a string field.

### `_handle_loop_reprofile` — `web.py:481-492`

- **Publisher that replaces its loop_state mutation:** **none** in session mode.
- **Finding:** conceptual match is `publish_after_profile` (`session_store.py:721-724`), but the endpoint must not call it. Precondition at `_update_state` (`session_store.py:882-890`): phase in `{propose,reprofile}` **and** a non-abstained `Proposal` already present — unreachable from free-text web state, and deliberately not satisfied by inventing a web propose-path.
- **Returns today:** `loop_state.to_dict()` after `record_reprofile_artifact` (`web.py:491-492`, `loop_state.py:261-265`).
- **After change (session mode):** stated limitation naming `nsys-ai profile` for capture/validation (`cli/parsers.py:95`; `profile_command.py` docstring states it does not own durable session layout) plus whatever CLI path later calls `publish_after_profile`. Endpoint does not register an after path into the session.

### `_handle_loop_diff` — `web.py:510-524`

- **Publisher that replaces its loop_state mutation:** **none on this endpoint** under the CLI-drives-loop decision.
- **Finding:** conceptual match among the five listed publishers is `publish_diff` (`session_store.py:754-773`). Long diff work stays CLI-owned. This endpoint must not analyze-then-publish.
- **Returns today:** `{"state": loop_state.to_dict(), "diff": diff_payload}` (`web.py:524`) from in-process `run_diff` (`loop_state.py:293-347`) under `_LOOP_LOCK`.
- **After change (session mode) — C6, recommend with evidence, not blocking:**

  **Recommendation: read-only reload of `SessionSnapshot.diff` (no publisher, no analysis).**

  Evidence from what the timeline UI does with a diff it did not trigger:

  1. `loopFetchState` (`timeline.js:3435-3443`) loads `GET /api/loop/state` and calls `loopRenderState` with whatever keys arrive — it does not require a prior `POST /api/loop/diff`.
  2. `loopRenderState` (`:3527-3636`) unlocks the decide section, shows the status panel, and paints verdict / step / category deltas solely from `LOOP_STATE.diff_summary`. There is no "I ran this diff" flag.
  3. `loopSuggestedPhase` (`:3326-3327`) returns `'accept'` as soon as `s.diff_summary` is truthy, even when `s.decision` is still empty — so a CLI-published undecided diff advances the stepper to decide without a web-side run.
  4. `loopSetDecision` (`:3747-3748`) only requires `LOOP_STATE.diff_summary`; it does not check that the client triggered the comparison.

  Therefore a projection of `SessionSnapshot.diff` into `diff_summary` (via GET state, or a POST that only reloads the snapshot) is enough for the UI to render and decide. A stated limitation naming `nsys-ai diff` also avoids a second writer, but leaves `loopRunDiff` (`:3723-3744`) as a hard failure on the primary button unless that button is rewritten — reload is the smaller change for a session that already has a published diff.

  Alternative (acceptable under C6): stated limitation naming `nsys-ai diff` (`cli/parsers.py:539`), with GET `/api/loop/state` still projecting the published diff so open-and-render works without POST.

### `_handle_loop_decision` — `web.py:526-535`

- **Among the five publishers listed in the mission** (`publish_runspec`, `publish_findings`, `publish_after_profile`, `publish_proposal`, `publish_diff`): **none** matches.
- **Existing API (not invented here):** `SessionWriter.publish_decision` (`session_store.py:775-829`). Preconditions: phase exactly `"diff"` (`:788`); input `accept`/`accepted` and `reject`/`rejected` normalised (`:794-798`); second decision raises (`:791-792`) — keep that (C2).
- **Returns today:** `loop_state.to_dict()` after `set_decision` (`web.py:534-535`), writing CWD `diff.json` via `write_diff_decision_json_from_diff_dict` (`loop_state.py:372-393`).
- **After change:** acquire `SessionStore.writer(session_id)`, call `publish_decision`, return a **projection** of the resulting snapshot (Deliverable 6) — not `DiffLoopState.to_dict()`. Decision fields come from `diff["decision"]`; `decision_path` only when that key is non-null (C5). UI must then render the decision and **disable** accept/reject (C2) — today `timeline.js:3415-3416` does not.

`publish_runspec` has **no** corresponding loop action endpoint among the five.

### C3 — evidence overlay load path (not a publisher)

| surface | today | session mode |
|---|---|---|
| `GET /api/findings` (`web.py:218-220`) | returns `_ViewerHandler._findings` (`:189`) | return findings from `SessionSnapshot.findings` (`session_store.py:323`), serialised as today (`[f.to_dict() for f in report.findings]`). **Read only — no `publish_*`.** |
| Population today | `_handle_loop_diagnose` (`web.py:507`) or `--findings` at `serve_timeline` (`:764`) | diagnose becomes a stated limitation; without this load path the overlay silently empties |

---

## Deliverable 3 — Where `run_diagnose` / `run_diff` execute

### What they do today

- `DiffLoopState.run_diagnose` (`loop_state.py:273-291`): constructs `EvidenceBuilder`, calls `build()`, ranks findings, **and** mutates loop fields under the caller's lock.
- `DiffLoopState.run_diff` (`loop_state.py:293-347`): opens profiles, runs `diff_profiles` / `to_diff_json`, **and** mutates loop fields.

Web holds `_LOOP_LOCK` for the entire diagnose and diff handlers (`web.py:495-508`, `511-524`). Tree/Timeline call `run_diff` on the UI thread (`tree/app.py:707-713`, `timeline/app.py:841-847`) without a session writer.

### Where the work moves (within the scope decision)

Per `session_store.py` module contract and issue #268: the store owns persistence only; expensive diagnose / re-profile / diff runs **outside** the state lock; only transitions and publication are serialized.

Per the recorded #268 scope decision: Web/TUI are **read / render / decide**. Long-running diagnose and diff execution remain **CLI-owned**.

Concrete split:

1. **Analysis (no session writer, no state lock):** CLI runs `EvidenceBuilder.build` → `EvidenceReport`, and `diff_profiles` + `to_diff_json` → diff mapping. Same analysis code `run_diagnose` / `run_diff` use today, but not as methods that both execute and own durable UI state.
2. **Publication (serialized, brief):** CLI opens `SessionStore.writer(session_id)` then `publish_findings` / `publish_diff` (and sibling publishers). Web `_handle_loop_decision` is the interactive surface that briefly takes the writer for `publish_decision` only. Tree/Timeline in session mode call `publish_decision` the same way and **do not** write beside the after profile (C4).
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

1. **CLI still does not publish into sessions.** `profile_command.py` states it does not own durable session layout; `propose_command.py` rejects session-shaped input; `loop` uses `DiffLoopState` helpers only (issue #268 measured comment). Web/TUI read-render-decide against an empty store until CLI publishers land. Surfaces can open under C1, but there is nothing to render until CLI writes artifacts.

2. **`--session` / derived-id open path is undecided in code.** C1 decides the rule; `cli/parsers.py` still has zero `session` flags, and `serve_timeline` / TUI constructors take no session id (Deliverable 5a). Wiring that forgets an open contract cannot load.

3. **Response-shape break for the web UI.** `src/nsys_ai/templates/timeline.js` reads 16 denormalised `LOOP_STATE` keys (Deliverable 6). `SessionState.to_dict` (`session_store.py:248-261`) emits nothing like that shape. Blind swap loses the stepper/hero rendering.

4. **Stated-limitation UX vs today's free-text / diagnose buttons.** `timeline.js` still posts free text (`:3685-3702`), runs diagnose, and registers after paths. Session mode must surface limitations without silently no-op'ing; otherwise the UI looks broken.

5. **Decision enum mismatch on the read side.** Loop/API/UI use `accept`/`reject` (`loop_state.py:364-365`, `timeline.js:3565,3765`); session diff records `accepted`/`rejected` (`session_store.py:794-799,1639-1640`). Projection must translate or the badge changes meaning.

6. **C2 UI change required on the only interactive write path.** Store rejects a second decision (`session_store.py:791-792`). `timeline.js:3415-3416` / `:3326-3327` keep accept/reject live after a decision exists. Without disabling controls, the second click becomes a 4xx/500 that looks like a bug.

7. **`publish_after_profile` deadlock if anyone tries to keep web reprofile.** Store requires a non-abstained proposal (`session_store.py:887-890`). Scope decision removes that path deliberately; reintroducing free-text to satisfy the precondition recreates the second source of truth.

8. **TUI `diff.json` location vs session layout (C4).** Tree/Timeline write beside the after profile (`tree/app.py:723-732`, `timeline/app.py:857-866`). SessionStore always writes `<session>/diff.json`. Cross-process handoff fails if TUIs keep the old path in session mode.

9. **`decision_path` false positive if gated on artifact presence (C5).** `timeline.js:3572-3577` renders "Decision recorded to …" whenever the key is non-empty. Projecting the path whenever `diff.json` exists announces a decision that has not been made.

10. **`decision_warnings` not recoverable after reload (C5).** Advisory-only list is returned by `publish_decision` but not stored separately; reload only sees full `diff["warnings"]`.

11. **Writer lease held across analysis.** If CLI (or a mistaken web path) acquires `writer()` then runs diagnose/diff before `publish_*`, it violates issue #268 and blocks the conflict model for the whole analysis duration.

12. **Review-session contract still unshipped** (issue #268 post-#359 / post-#363 comments: `mode: loop|review`). Direct before/after review without a proposal still collides with `publish_after_profile`'s proposal requirement. Out of scope for inventing here; it bites anyone trying to open a review-only session under the loop layout.

13. **`_handle_loop_phase` (`web.py:454-465`) is outside the five endpoints** but still mutates phase in memory. Session phases advance via publishers; a free-form `set_phase` has no `publish_*` and desynchronises UI steppers from artifacts if kept in session mode.

14. **Fingerprint cost on naive `str` → `LocalProfileReference` conversion** (Deliverable 5b). Calling `build_local_profile_reference` on every path touch re-opens SQLite and re-runs identity SQL including a full kernel `COUNT(*)`.

15. **CWD-relative default session root (C1).** `SessionStore.__init__` defaults to `.nsys-ai/sessions` resolved from process CWD (`session_store.py:331-332`). A browser server started in a different directory than the CLI that created the session will not see it. C1 explicitly forbids a root flag; processes must share CWD (or an equivalent agreed working directory).

---

## Deliverable 5 — Gaps the issue body does not cover

### 5a) Session lifecycle — where `session_id` and store `root` come from

Under **C1** (decided; not re-opened):

| piece | rule | how a web or TUI process gets it |
|---|---|---|
| `root` | Always `.nsys-ai/sessions` relative to the working directory. **No flag.** | Construct `SessionStore()` / `SessionStore(".nsys-ai/sessions")` (`session_store.py:331`). Same convention as `.git`. Web `serve_timeline` and both TUIs use process CWD. A CLI that created the session must have been run from that same working directory. |
| `session_id` when `--session <id>` is given | Caller-supplied; validated by `_SESSION_ID` (`session_store.py:72`, `:927-930`) | Open contract must accept `--session <id>` (or an equivalent open argument) and pass it to `SessionStore.load` / `writer`. **Today:** `cli/parsers.py` has zero `session` strings — the flag does not exist yet; C1 says it must. |
| `session_id` when no id is given | **Derived** from the before profile's content id that `LocalProfileReference` already carries (`profile_reference.py:23`) | Open with a before profile path → `build_local_profile_reference(before_path)` once (`profile_runner.py:130`) → use `reference.profile_id` as `session_id`. Two processes pointed at the same profile land in the same session without passing anything between them. One session per before-profile is the intended model. |

Measured absence today (not a design choice):

- `serve_timeline` seeds only `DiffLoopState` from `loop_before` / `loop_after` / H100 preset (`web.py:775-788`).
- Tree/Timeline seed `DiffLoopState(before_path=db_path, after_path=loop_after or "")` (`tree/app.py:152`, `timeline/app.py:178`).
- CLI `loop` passes profile paths into those surfaces (`cli/handlers.py`), never a session id.
- `SessionStore.create` takes an id; it does **not** mint one (`session_store.py:335`).

Who calls `create` vs `load` under the scope decision: **CLI creates and publishes**; web/TUI **load** (and `writer` only for `publish_decision`). Web/TUI do not call `create` to invent a parallel session.

### 5b) `str` → `LocalProfileReference`

- Loop fields `before_path` / `after_path` are strings (`loop_state.py:176-177`).
- Session fields `before_profile` / `after_profile` are `LocalProfileReference | None` (`session_store.py:230-231`) carrying path, content-derived `profile_id`, `schema_version`, `product_version`, `kernel_count` (`profile_reference.py:19-26`).
- **Construction site:** `build_local_profile_reference` (`profile_runner.py:130-255`). It opens the `.sqlite` read-only via a descriptor FD (`:194-208`), builds a `Profile`, reads `kernel_count` / schema / product version (`:210-216`), and calls `get_profile_id` (`fingerprint.py:271+`) which hashes capture-time metadata **and** runs `SELECT COUNT(*)` over the resolved kernel table (`fingerprint.py:360-370`).
- **Production caller today:** only `LocalProfileRunner` after export validation (`profile_runner.py:534-536`). Web, tree, timeline, and loop handlers never call it. Tests are the other callers.
- **Validation without re-fingerprint:** `validate_local_profile_reference` (`profile_reference.py:164-203`) checks shape + file presence/size; it does **not** recompute `profile_id`.
- **Repeated-call cost:** any path that calls `build_local_profile_reference` on every reprofile POST, every decision, or every GET projection re-opens the DB and re-counts kernels. For large profiles that is multi-second SQLite work proportional to kernel table size, paid on every call. Safe pattern: fingerprint once at CLI `create` / first open (also yields the C1 default session id); thereafter pass the stored `LocalProfileReference` and only `validate_local_profile_reference` when the store requires `require_file=True`.

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
| `decision_path` | **C5:** `str(session_dir / "diff.json")` **only when** `snapshot.diff is not None and snapshot.diff.get("decision") is not None`; else `""`. Do **not** gate on artifact presence alone. |
| `last_error` | always `""` in the projection. Errors stay on the request (`web.py:546` `error` field / toast). Sticky `LOOP_STATE.last_error` across GET is dropped (Deliverable 1). |

### UI behaviour that must change (not just re-key)

1. **Primary actions for diagnose / save proposal / register after** (`loopSuggestedPhase` / `loopPrimaryLabel` at `timeline.js:3324-3339`, posts at `:3685-3733`) must stop calling the three executing endpoints in session mode and show the stated limitation (CLI command name) instead.
2. **Decision POST** remains live; it must target `publish_decision` and accept the projected response shape. **Once `decision` is non-null, render it and disable accept/reject** (C2) — today buttons stay enabled whenever suggested phase is `accept` (`:3415-3416`).
3. **Stepper `done` logic** that keys off `diagnose_ran` and string `proposal` (`timeline.js:3328-3330,3485-3490`) must key off findings presence and `snapshot.proposal is not None` (and treat abstention explicitly — `proposal.abstained`).
4. **Diff primary button:** under the C6 recommendation, either become a read-only reload or disappear when `diff_summary` is already projected from the session; do not call analyze-then-publish.

Without that adapter-or-rewrite choice made up front, editing `web.py` alone will ship a session that the existing JS cannot render.

---

## UNRESOLVED register

Only C6 remains open (allowed by the mission):

1. **`_handle_loop_diff` in session mode** — stated limitation naming `nsys-ai diff`, or read-only reload of `SessionSnapshot.diff`. **Recommendation above: read-only reload**, with evidence from `timeline.js` treating a preloaded `diff_summary` as first-class. Not blocking.
