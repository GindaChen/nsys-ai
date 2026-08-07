# Session wiring design — DiffLoopState → SessionStore

Design-only gate for issue #268. No production code in this change.

Measured on worktree `feat/issue-268-session-wiring` at `df332ed` parents plus local sources listed below. Every disposition cites `file:line`. Where the code and the issue comments disagree, that disagreement is recorded as a risk or UNRESOLVED, not smoothed over.

Sources read: `src/nsys_ai/loop_state.py`, `session_store.py`, `web.py`, `tree/app.py`, `timeline/app.py`, `proposal.py`, `annotation.py`, `profile_reference.py`, `diff_decision.py`, `templates/timeline.js`, and `gh issue view 268|352 -R GindaChen/nsys-ai --comments`.

Note on field count: issue #268's measured comment says "`DiffLoopState` carries 16 fields". The dataclass at `loop_state.py:175-192` defines **17** fields (listed below). This document dispositions all 17.

`SessionState` (`session_store.py:227-234`) carries five fields plus an artifact map: `session_id`, `phase`, `before_profile`, `after_profile`, `artifacts`.

`SessionSnapshot` (`session_store.py:320-325`) adds loaded artifacts: `runspec`, `findings`, `proposal`, `diff`.

---

## Deliverable 1 — DiffLoopState field disposition

Read sites are production readers (Python under `src/` and the web UI that consumes `to_dict()`). Writers that only assign the field are noted under evidence when they clarify meaning, but disposition follows **who reads**.

| field | who reads it (file:line, every site) | disposition | evidence |
|---|---|---|---|
| `before_path` | `loop_state.py:168-169` (`reconcile_h100_loop_paths`); `loop_state.py:197,199` (`to_dict`); `loop_state.py:313-315,326-329` (`run_diff`); `web.py:452,465,479,492,508,524,535,543` (via `to_dict`); `templates/timeline.js:3493-3502` (`before_path` / `before_label`); `tree/app.py:152` (constructor seed, then later methods use other fields); `timeline/app.py:178` (same) | SESSION | Maps to `SessionState.before_profile` (`session_store.py:230`). Types are **not** compatible as-is: loop holds `str` (`loop_state.py:176`); session holds `LocalProfileReference \| None` (`session_store.py:230`, shape at `profile_reference.py:19-26` with `path`, `profile_id`, `schema_version`, `product_version`, `kernel_count`). Display path becomes `before_profile.path`. Labels (`before_label` in `to_dict`) are not a SessionState field; they are computed today by `profile_display_name` (`loop_state.py:146-160,199`). |
| `after_path` | `loop_state.py:170-171`; `loop_state.py:198,200` (`to_dict`); `loop_state.py:302-310` (`run_diff`); `loop_state.py:262` (written by `record_reprofile_artifact`, read later by `run_diff`/`to_dict`); `templates/timeline.js:3329,3494-3511,3724`; `tree/app.py:730-732`; `timeline/app.py:864-866`; web handlers via `to_dict` as above | SESSION | Maps to `SessionState.after_profile` (`session_store.py:231`). Same type mismatch: `str` vs `LocalProfileReference \| None`. Publication path is `SessionWriter.publish_after_profile` (`session_store.py:721-724`). |
| `phase` | `loop_state.py:201,246-252` (`to_dict` / `set_phase`); `web.py` via `to_dict`; `templates/timeline.js:3469-3490`; `tree/app.py:153,691-692,711,754`; `timeline/app.py:179,825-826,845,888` | SESSION | Maps to `SessionState.phase` (`session_store.py:229`). Types are compatible: `Phase` (`loop_state.py:18`) and `SessionPhase` (`session_store.py:70`) are the same five-string literal. |
| `selected_scope` | `loop_state.py:202` (`to_dict` only); `loop_state.py:241-242` (validation on `from_dict`) | DROP | No reader in `web.py`, `tree/app.py`, `timeline/app.py`, or `templates/timeline.js` (repo search for `selected_scope` under `src/` hits only `loop_state.py`). `SessionState` has no scope field. Nothing supersedes it beyond deletion; it is unused view state. |
| `proposal` | `loop_state.py:203,258-259`; `templates/timeline.js:3329-3330,3517,3694`; web via `to_dict` after `_handle_loop_proposal` (`web.py:478-479`) | DROP | Superseded by the `Proposal` artifact (`proposal.py:207-221`, nine v0 fields). Confirmed by mission text and issue #352 comments: free-text `str` must not be mapped into the store. Session surface is `SessionSnapshot.proposal` (`session_store.py:324`), published by `publish_proposal` (`session_store.py:726-752`). |
| `expected_impact` | `loop_state.py:204,256`; `templates/timeline.js:3522,3687-3702`; web via `to_dict` | DROP | Superseded by `Proposal.expected_impact: ExpectedImpact \| None` (`proposal.py:216`, structured object at `proposal.py:169-203` with `kind`/`headroom_ms`/`headroom_basis`). Free-text `str` in the loop is not type-compatible. |
| `decision` | `loop_state.py:205,384`; `templates/timeline.js:3326,3565-3566`; web via `to_dict`; tree/timeline call `set_decision` then read other fields | DERIVED | Not a `SessionState` field. After `publish_decision`, the value lives on `SessionSnapshot.diff["decision"]` (`session_store.py:775-829`; decision object shape enforced at `session_store.py:1628-1640`: `status` in `{accepted,rejected}`). Type mismatch: loop uses `"accept"`/`"reject"` (`loop_state.py:19,364-365`); diff artifact uses `"accepted"`/`"rejected"`. |
| `decision_reason` | `loop_state.py:206,385`; `templates/timeline.js:3569-3570`; web via `to_dict` | DERIVED | From `diff["decision"]["reason"]` after publication (`diff_decision.py:99-104`; `session_store.py:800-806`). |
| `decision_path` | `loop_state.py:207,386`; `templates/timeline.js:3574-3576,3763`; `tree/app.py:756`; `timeline/app.py:890`; web via `to_dict` | DERIVED | Today: filesystem path returned by `write_diff_decision_json_from_diff_dict` (`loop_state.py:376-386`), often next to the after profile in TUIs (`tree/app.py:723-732`, `timeline/app.py:857-866`) or CWD `diff.json` for web (`loop_state.py:375`). Under SessionStore the artifact path is always `<session_dir>/diff.json` (`session_store.py:_ARTIFACT_PATHS` at `73-78`, write at `809-811`). Derive as session directory + `artifacts["diff"].path` (`session_store.py:214-216`). |
| `decision_warnings` | `loop_state.py:208,387`; `tree/app.py:748-758` (uses the **return value** `warnings` from `set_decision`, not only the field); `timeline/app.py:882-892` (same) | DERIVED | Advisory list from `build_diff_decision_record_from_diff_dict` (`diff_decision.py:84-95,105`). `publish_decision` returns the same class of warnings as its third tuple element (`session_store.py:782,829`) and also merges them into `diff["warnings"]` via the builder (`diff_decision.py:93-98`). There is **no** separate SessionState field. After reload, only `diff["warnings"]` remains on disk — that list also contains pre-decision warnings, so it is not identical to the advisory-only list DiffLoopState stored. |
| `diagnose_ran` | `loop_state.py:209,286`; `templates/timeline.js:3328,3485-3490`; web via `to_dict` | DERIVED | True after `run_diagnose` (`loop_state.py:286`). Derive as `SessionSnapshot.findings is not None` (findings artifact present in `state.artifacts` / loaded at `session_store.py:320-325`). No SessionState bool. |
| `diagnose_findings_count` | `loop_state.py:210,287`; `templates/timeline.js:3562`; web via `to_dict` | DERIVED | Set to `len(ranked)` in `run_diagnose` (`loop_state.py:287`). Derive as `len(snapshot.findings.findings)` from the `findings.json` / `EvidenceReport` artifact (`annotation.py:230-251,272-285`). |
| `top_findings` | `loop_state.py:211,288` (`to_dict` / written by `run_diagnose`) | DERIVED | Written as `ranked[:15]` (`loop_state.py:288`) after `_normalize_findings` (`loop_state.py:33-64`). **No** reader in `templates/timeline.js`, `web.py` handlers (beyond blind `to_dict`), `tree/app.py`, or `timeline/app.py`. Derive at read time from `findings.json` (`EvidenceReport.findings`) by applying the same ranking used at diagnose time, then taking `[:15]`. Whether the stored EvidenceReport order already matches `_normalize_findings` is UNRESOLVED (see Risks) — the artifact key is still `findings`. |
| `diff_summary` | `loop_state.py:212,340,366-377`; `templates/timeline.js:3327,3382,3527-3578,3631-3635,3748`; `tree/app.py:741`; `timeline/app.py:875`; web via `to_dict` | DERIVED | Entire diff payload. Maps to `SessionSnapshot.diff` (`session_store.py:325`), artifact `diff.json`. |
| `comparability_confidence` | `loop_state.py:213,342-344`; `templates/timeline.js:3582-3584`; web via `to_dict` | DERIVED | Copied from diff payload in `run_diff` (`loop_state.py:342-344`). Read from `diff["comparability_confidence"]` (required by session validation at `session_store.py:89,1543-1550`). |
| `verdict` | `loop_state.py:214,341,390`; `templates/timeline.js:3529-3553`; `tree/app.py:713`; `timeline/app.py:847`; web via `to_dict` | DERIVED | From `diff["verdict"]` (`loop_state.py:341`; may be stamped `inconclusive` on decision at `diff_decision.py:96` / `loop_state.py:390`). |
| `last_error` | `loop_state.py:215`; written `web.py:542`; cleared on successful mutations (`loop_state.py:257,265,271,290,346,392`); read `templates/timeline.js:3475` via `to_dict` | DROP | No SessionState / artifact field. HTTP errors already return `{"error": ..., "state": ...}` (`web.py:546`). Durable session restart does not need this string. Sticky survival across a later `GET /api/loop/state` has no store home (see Risks). |

---

## Deliverable 2 — Web action endpoint → publisher mapping

All five handlers today mutate class-level `_ViewerHandler._loop_state` under `_LOOP_LOCK` (`web.py:27-28,494-535`). None import `SessionStore` (production callers of `SessionStore` are still only tests — measured in issue #268 comments; repo search confirms).

### `_handle_loop_diagnose` — `web.py:494-508`

- **Publisher after analysis:** `SessionWriter.publish_findings` (`session_store.py:673-719`), optionally with `before_profile=...`.
- **Not among callers today:** analysis is `loop_state.run_diagnose` **inside** `_LOOP_LOCK` (`web.py:495-505` → `loop_state.py:273-291`).
- **Returns today:** `{"state": loop_state.to_dict(), "findings": findings}` where `findings` is the full ranked list returned by `run_diagnose`, and `state` carries denormalised diagnose fields (`web.py:508`).
- **After change:** `state` comes from a session load/snapshot projection (not `DiffLoopState.to_dict`). `findings` comes from the published `EvidenceReport` / `SessionSnapshot.findings` (`annotation.py:272-285`), not from mutating `top_findings` on loop state. Long `EvidenceBuilder.build` (`evidence_builder.py:104`, invoked from `loop_state.py:282-284`) must finish **before** the writer publishes (see Deliverable 3).

### `_handle_loop_proposal` — `web.py:467-479`

- **Matching publisher among the listed five:** `publish_proposal` (`session_store.py:726-752`) — but only for a validated `Proposal` object.
- **Finding:** the endpoint's mutation is free-text `set_proposal(proposal, expected_impact=...)` (`web.py:473-478` → `loop_state.py:254-259`). That path has **no** matching publisher. Issue #268 comment after PR #359: free-text proposal mutation must be removed rather than mapped. `publish_proposal` requires `Proposal` (`session_store.py:729-730`) and matching `runspec` / findings pointers (`session_store.py:735-744`).
- **Returns today:** `loop_state.to_dict()` (`web.py:479`) with `proposal: str` and `expected_impact: str`.
- **After change:** browser must render `SessionSnapshot.proposal` (`session_store.py:324`, nine fields in `proposal.py:207-221`). The free-text response shape cannot be preserved without inventing a second source of truth.

### `_handle_loop_reprofile` — `web.py:481-492`

- **Publisher:** `SessionWriter.publish_after_profile` (`session_store.py:721-724`).
- **Returns today:** `loop_state.to_dict()` after `record_reprofile_artifact` sets `after_path` + phase `reprofile` (`web.py:491-492`, `loop_state.py:261-265`).
- **After change:** response profile path from `SessionState.after_profile` (`session_store.py:231`). **Constraint measured in store:** `_update_state` requires phase in `{propose,reprofile}` and a **non-abstained** proposal already present (`session_store.py:882-890`). Today's web handler does not check for a proposal (`web.py:481-492`).

### `_handle_loop_diff` — `web.py:510-524`

- **Publisher after analysis:** `SessionWriter.publish_diff` (`session_store.py:754-773`).
- **Returns today:** `{"state": loop_state.to_dict(), "diff": diff_payload}` (`web.py:524`) where `diff_payload` is the return of `run_diff` / `to_diff_json` (`loop_state.py:339-347`).
- **After change:** `diff` from `SessionSnapshot.diff` (same mapping published into `diff.json`). `state` from session snapshot projection. `run_diff` work stays outside the publication lock (Deliverable 3). `publish_diff` requires undecided payload and phase in `{reprofile,diff}` (`session_store.py:759-762,1621-1624`).

### `_handle_loop_decision` — `web.py:526-535`

- **Among the five publishers listed in the mission** (`publish_runspec`, `publish_findings`, `publish_after_profile`, `publish_proposal`, `publish_diff`): **none** matches.
- **Finding / existing API:** `SessionWriter.publish_decision` exists at `session_store.py:775-829` and is the store's decision writer. It was omitted from the mission's "available" list; it is not something invented here.
- **Returns today:** `loop_state.to_dict()` after `set_decision` (`web.py:534-535`), including `decision`, `decision_reason`, `decision_path`, `decision_warnings`, updated `verdict` / phase.
- **After change:** load snapshot after `publish_decision`; derive decision fields from `diff["decision"]` and path from the session `diff.json` location. Warnings are the third return value of `publish_decision` (`session_store.py:829`) at write time; on later GET they are only available via `diff["warnings"]` (see field table).

`publish_runspec` has **no** corresponding loop action endpoint among the five.

---

## Deliverable 3 — Where `run_diagnose` / `run_diff` execute

### What they do today

- `DiffLoopState.run_diagnose` (`loop_state.py:273-291`): constructs `EvidenceBuilder`, calls `build()`, ranks findings, **and** mutates loop fields under the caller's lock.
- `DiffLoopState.run_diff` (`loop_state.py:293-347`): opens profiles, runs `diff_profiles` / `to_diff_json`, **and** mutates loop fields.

Web holds `_LOOP_LOCK` for the entire diagnose and diff handlers (`web.py:495-508`, `511-524`), so analysis and state mutation share one lock. Tree/Timeline call `run_diff` on the UI thread without a session writer (`tree/app.py:707-713`, `timeline/app.py:841-847`).

### Where the work moves

Per `session_store.py:1-4` and issue #268 body: the store owns persistence only; expensive diagnose / re-profile / diff runs **outside** the state lock; only transitions and publication are serialized.

Per issue #268 comment after PR #359 (measured, not inferred): for v0, Web/TUI only need **read / render / decide**; long-running diagnose/diff execution remains **CLI / optimize-owned**. Callers that already run `EvidenceBuilder` include `cli/handlers.py` (e.g. diagnose/evidence paths around the `EvidenceBuilder` imports measured by search) and must then `publish_findings` / `publish_diff`.

Concrete split:

1. **Analysis (no session writer / no state lock):** `EvidenceBuilder.build` → `EvidenceReport`; `diff_profiles` + `to_diff_json` → diff mapping. Same code paths `run_diagnose` / `run_diff` use today, but not as methods that both execute and own durable state.
2. **Publication (serialized):** open `SessionStore.writer(session_id)` then `publish_findings` / `publish_diff` (and sibling publishers).

Whether the web `/api/loop/diagnose` and `/api/loop/diff` endpoints remain thin "analyze then publish" wrappers or are removed in favour of CLI-produced sessions is **UNRESOLVED** by the code (they still execute today) and constrained by the issue comment toward CLI ownership for v0. Either way, analysis must not run while holding the publication locks.

### What holds which lock, and for how long

| lock | where | duration |
|---|---|---|
| `SessionStore.writer()` exclusive **writer** lease | `session_store.py:369-379` (`blocking=False`; conflict → `SessionConflictError`) | Lifetime of the `SessionWriter` context (`__enter__`/`close`, `session_store.py:631-640`). Must not cover `EvidenceBuilder.build` or `diff_profiles`. |
| Exclusive **state** lock inside each `publish_*` / `publish_decision` | `session_store.py:842-844`, `878-880`, `784-786` (`blocking=True`) | Only load → validate → write artifact/`session.json` → finish journal. Short publication critical section. |
| Today's `_LOOP_LOCK` | `web.py:28,495,511` | Entire diagnose/diff request including analysis — **this is what must stop** for those paths. |

`SessionStore.load` also takes the state lock for recovery (`session_store.py:366-367`) but is a read/repair path, not analysis.

---

## Deliverable 4 — Risks

1. **Response-shape break for the web UI.** `templates/timeline.js` reads denormalised keys (`proposal` string, `expected_impact` string, `diagnose_ran`, `diff_summary`, `decision_path`, etc. at lines cited above). `SessionState.to_dict` (`session_store.py:248-261`) and `SessionSnapshot` do not emit that shape. An adapter or UI rewrite is required; a blind swap loses the hero/stepper rendering.

2. **Free-text proposal endpoint cannot call `publish_proposal`.** Removing it is required by issue #268/#352 comments; keeping it creates a second source of truth. The UI still posts free text (`timeline.js:3685-3702`).

3. **Decision enum mismatch.** Loop/API use `accept`/`reject` (`loop_state.py:364-365`, `web.py:532`); session diff records `accepted`/`rejected` (`session_store.py:794-799`). Projection must translate or the badge at `timeline.js:3565` changes meaning.

4. **`publish_after_profile` is stricter than today's reprofile.** Store requires a non-abstained proposal (`session_store.py:887-890`). Web `_handle_loop_reprofile` does not (`web.py:481-492`). H100 preset startup can set `after_path` before any proposal (`web.py:775-787`).

5. **TUI `diff.json` location vs session layout.** Tree/Timeline write beside the after profile (`tree/app.py:723-732`, `timeline/app.py:857-866`). SessionStore always writes `<session>/diff.json`. Cross-process handoff tests in #268 will fail if TUIs keep the old path.

6. **`decision_warnings` round-trip loss.** Advisory-only list is returned by `publish_decision` but not stored as its own artifact; reload only sees full `diff["warnings"]`.

7. **`top_findings` ranking vs persisted findings order.** Loop ranking is `_normalize_findings` (`loop_state.py:33-64`). `publish_findings` stores the `EvidenceReport` as given (`session_store.py:685-718`). If publishers store unranked builder output, deriving `top_findings` as `findings[:15]` silently disagrees with today's UI/state. I did not verify whether CLI diagnose ranks before write.

8. **Writer lease held across analysis.** If an implementation acquires `writer()` then runs diagnose/diff before `publish_*`, it violates issue #268 and blocks concurrent readers'/writers' conflict model for the whole analysis duration.

9. **Issue #268 still lists an explicit review-session contract (`mode: loop|review`) as unshipped.** Direct before/after review without a fabricated proposal collides with `publish_after_profile`'s proposal requirement. Wiring without that contract will force fake proposals or dead ends.

10. **`selected_scope` and sticky `last_error`.** Scope is unused; dropping it is safe for current UI. `last_error` sticky across GET has no session field — confirm UX intentionally becomes request-scoped `error` only.

11. **Zero production `SessionStore` callers.** CLI `profile`/`propose`/`loop` still do not write sessions (issue #268 measured comment). Web/TUI adoption without a CLI publisher leaves nothing durable to open.

12. **`_handle_loop_phase` (`web.py:454-465`) is outside the five endpoints but still mutates phase in memory.** Session phases advance via publishers; a free-form `set_phase` has no `publish_*` and can desynchronise UI steppers from artifacts if kept.

---

## UNRESOLVED register (needs human before code)

1. Do web `/api/loop/diagnose` and `/api/loop/diff` remain execution endpoints that analyze then publish, or become read-only against CLI-produced sessions for v0? (Issue comment says CLI-owned; code still executes in-process.)
2. Exact JSON projection from `SessionSnapshot` → current `LOOP_STATE` keys expected by `timeline.js` (adapter vs UI change).
3. Whether persisted `EvidenceReport.findings` order is already opportunity-ranked when published, for safe `top_findings` derivation.
4. Whether sticky `last_error` across GET must be preserved somehow despite no session field.
