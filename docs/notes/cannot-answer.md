# One shape for "cannot answer"

A consumer of this tool has to tell three states apart: *there is no problem here*, *here is the
problem*, and *I could not look*. The third was said eight different ways, five of them at exit 0, so
in practice it was indistinguishable from the first. That is the failure a verify-first tool can
least afford: an analysis that could not run reads as a clean bill of health.

This note decides the shape. It answers three questions and nothing else — it does not change what
any analysis can answer, only how "cannot" is said. Issue #400 has the observed encodings; the matrix
in `tests/test_cannot_answer.py` drives each of them and is what stops a ninth from appearing.

---

## 1. The payload marker

**Decision.** Any JSON payload that cannot answer carries `_abstained: true` and a non-empty
`reason`. This is the shape `skills.base.abstain` already produces at row level. It becomes canonical
at every level: skill row, CLI JSON envelope, web route body.

**The predicate**, one function, `nsys_ai.cannot_answer.is_cannot_answer(payload)`:

| payload | true when |
|---|---|
| `dict` | `payload.get("_abstained") is True` |
| `list` | non-empty and the first element is such a dict |
| anything else | never |

The list rule is `skills.base.is_abstention` called, not restated: `skills/base.py` stays the single
definition site of the marker string, and `nsys_ai.cannot_answer` holds no copy of it — a guard test
already fails on a second one. Its imports are deferred to call time, so reaching the predicate from
a renderer or a web handler costs nothing at import. Ownership should invert when `skills/base.py` is
next opened — a general contract should not live inside one of its consumers — which is a move to
make alongside #345 rather than in the same change as the decision.

**Which level carries the marker.** Ask the object you are about to read as an answer. If you are
about to index `payload["findings"]`, ask the envelope; if you are about to index `row["total_ms"]`,
ask the row. A payload marks itself when *it* is not an answer, and only then. Partial coverage is
not an abstention: an envelope that produced some findings is an answer, and the analyses that could
not run are recorded beside them.

**Shapes that already exist, and what happens to each:**

- `EvidenceReport.skipped[]` (`{analyzer, skill, reason}`) stays as it is. It is the record of
  *partial* coverage and is deliberately kept out of `findings`, which is right. The envelope itself
  becomes a cannot-answer payload — `_abstained: true` at the top level, `reason` naming the coverage
  that was lost — only when it produced zero findings **and** at least one analysis was skipped, i.e.
  exactly when "no findings" can no longer be read as "no problems".
- `verdict: "inconclusive"` in `diff.json` stays. It is a domain value that says *which* of three
  judgements was reached, and a consumer that knows about diffs should keep reading it. The marker
  goes beside it, for the consumer that does not.
- `proposal.json` carries `abstained` / `abstention_reason`. Those field names stay: the artifact has
  a validated schema with a strict allowed-key check, and renaming a persisted field to gain a
  leading underscore buys nothing. It is the one aliased shape. Whatever envelope reports a proposal
  — a session projection, a web route — sets the marker from those two fields; the predicate itself
  reads one key and only one.
- A bare `error` key is **not** the marker and must not be treated as one. It says the tool broke, not
  that the profile cannot support the question. Where the two were conflated (a missing table
  surfacing as `{"error": ...}`), the fix is to abstain, not to teach the predicate a second key.

---

## 2. The exit code

**Decision.** "Ran cleanly but cannot answer" **stays exit 0**. The payload marker is the only signal.
No new exit code is introduced.

**Why.** An abstention is a correct and complete answer to the question that was asked — "this
profile cannot support that analysis, and here is the reason" — so failing the process would teach
every wrapper to treat the truthful answer as a breakage, and would push callers to suppress the one
output that explains itself. The single caller that genuinely needs a non-zero status is a gate that
asked the tool to *prove* something, and that caller already has one, because there "could not judge"
fails the assertion rather than the command.

**So the exit status means:**

| code | meaning |
|---|---|
| 0 | the command ran and produced its output. The output may be an answer, an empty answer, or a stated abstention — apply the predicate to find out |
| 1 | the command did not run to completion (runtime error) |
| 2 | the command was asked for something it cannot parse or accept (usage) |

**The gate carve-out, stated so it is not read as a contradiction.** `diff --gate` exits non-zero on
`inconclusive` and keeps doing so. The gate is not being asked "what happened", it is being asked to
show that no regression is present, and a comparison that could not be made has not shown that. The
non-zero status there reports a failed *assertion*, not a failed command. Because exit 1 then covers
both "regression found" and "could not judge", the gate's JSON must carry the marker so the two are
distinguishable — that is the payload doing the work the exit code cannot.

**Consequence for a consumer that will not parse JSON.** There is none available, and this is
deliberate: ask for a gate, whose whole purpose is to compress a judgement into a status. Every one of
the eight encodings is a JSON surface or a human-readable one; on the human-readable surfaces an exit
code would not have helped either, because the reader is a person and what they need is the sentence.

---

## 3. The web `limitation` shape

**Decision.** Converge, keeping the existing keys as aliases. `_session_limitation` in `web.py` now
returns:

```json
{"_abstained": true, "reason": "...", "error": "...", "limitation": true, "cli": "nsys-ai ..."}
```

`_abstained` and `reason` are the canonical pair and are what new code branches on. `limitation` and
`error` are retained aliases for browser code already reading them. `cli` is retained because it
carries something the marker does not: the command that *can* do the job. Keeping it is the point of
the shape — a route that declines an action should name the way through, and "abstain with a reason"
is exactly that contract stated in HTTP.

**The HTTP status is unchanged (400) and is not the marker.** The status answers a transport question
— was this request serviceable on this route — and these routes decline an action the server mode does
not perform. The body answers the analysis question. A consumer branches on the body; a status code
was never able to distinguish "your request was malformed" from "your request was fine and I cannot
do it here", which is the same conflation this note removes from the CLI's exit codes.

The two TUIs have a `_session_limitation` of the same name that raises a toast. It is a human surface
with no payload, so the predicate does not apply to it and it is deliberately left alone.

---

## What to emit, for #399, #401 and #402

- **#399 (diff verdict and comparability are computed but never shown).** Keep `verdict` and
  `comparability_confidence` where they are. When `verdict == "inconclusive"`, the JSON payload also
  carries `_abstained: true` and a `reason` — the sanity warnings already collected are the reason
  text, and `diff.py` already has them. Terminal and markdown renderers print the reason instead of,
  not beside, per-kernel deltas presented as findings. Exit stays 0 without `--gate`; with `--gate` it
  stays non-zero, and now the payload says which kind of non-zero it was.
- **#401 (ordinary mistakes produce tracebacks).** A missing required parameter, an unparsable trim, a
  busy port and a non-TTY are **usage**, not abstentions: coded error, exit 2, no marker. Landed. The
  row that still belongs here is the silent one — a command that cannot answer and prints nothing at
  exit 0 must print the reason. In `--format json` that means a marked payload on stdout; in text
  mode, a sentence.
- **#402 (inputs that cannot support a comparison still produce a confident one).** Each new condition
  lowers `comparability_confidence` as #384 established. When the product crosses
  `MIN_COMPARABILITY_CONFIDENCE` the verdict is already `inconclusive`, so the marker follows from
  #399's rule with no extra branch. The self-diff short circuit is the one case that is not an
  abstention: the tool *can* answer, and the answer is "these are the same capture" — a `neutral`
  verdict with a stated note, exit 0, no marker.
- **#345, #283, #281 (the skill layer).** `abstain()` already produces the canonical shape, so these
  need no new decision — they need the guard extended to the paths that currently return `[]`, an
  untyped `error` row, or a raised database error.

## State of the surfaces

Measured on this worktree; each row is one test in `tests/test_cannot_answer.py`.

| surface | today | conforms |
|---|---|---|
| skill row, `abstain()` | `_abstained: true` + `reason`, exit 0 | yes |
| web session route body | marker plus retained aliases, HTTP 400 | yes, changed here |
| `skill run` on a profile missing the table (SQL template) | `{"error": {...}}` on stdout, exit 1 | no — #345 |
| `skill run` on a profile missing the table (`execute_fn`) | `[]`, exit 0 | no — #345 |
| `sync_cost_analysis` with no synchronization table | zero-valued row with an `error` key, exit 0 | no — #345 |
| `skill run` missing a required parameter | coded error, marked JSON on stdout, exit 2 | yes, since #401 |
| `summary` on a profile with no kernel activity | no output at all on either stream, exit 0 | no — #401 |
| `diff` on an incomparable pair | `inconclusive` + confidence 0.0 + warnings, no marker, exit 0 | no — #399 |

A ninth encoding turned up while writing the matrix, which is the argument for having one:
`nccl_payload_breakdown` returns `{"error": ..., "backend_limitation": true}` when the payload column
cannot be read through the active backend, and a second `{"error": ..., "binary_data_rows": 0}` row
when typed payloads were never captured. Both are abstentions wearing an untyped `error` key. They are
the same class as #345's five and belong with them.
