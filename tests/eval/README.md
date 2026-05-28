# Evaluation seed (`tests/eval/`)

Ground truth for **finding correctness**. Each label attaches expected /
forbidden finding assertions to a profile, so changes to skills and the
`EvidenceBuilder` pipeline can be scored — not just run.

This is the **seed** referenced by the roadmap (§4 B / §4 F / §4 H). It exists
because, before it, nothing checked whether `EvidenceBuilder.build()` produced
the *right* findings: skills were gaining `to_findings_fn` callbacks with no
regression net, and the §10 quality metrics (false-positive rate, false-negative
rate, selection overlap) had no data to be computed from.

It is the finding-correctness counterpart to [`tests/trajectories/`](../trajectories/),
which evaluates the *agent's tool-calling* behavior. Same shape (per-case JSON +
a pytest runner + skip-when-profile-absent); different target.

## Layout

```
tests/eval/
  README.md              this file
  labels/<name>.json     one label per profile
tests/test_eval.py       the runner (self-contained; mirrors test_trajectories.py)
```

## Label schema

```jsonc
{
  "id": "eval-mock-001",              // stable test id
  "profile": "tests/fixtures/mock.sqlite",  // path relative to repo root
  "device": 0,
  "description": "...",
  "source": "where the ground truth came from (audit doc, real run, domain analysis)",

  "expect": [                         // findings that SHOULD be produced
    { "category": "idle", "min_severity": "warning", "window_ns": [start, end],
      "required": true }              // "required": gate CI on this; default informational
  ],
  "must_not": [                       // findings that must NOT appear (false positives)
    { "category": "communication" }
  ],
  "known_gaps": [                     // documented-but-unfixed; printed, never gates
    { "observation": "...", "ref": "roadmap §4 B" }
  ]
}
```

Assertion fields (`category`, `min_severity`, `window_ns`) are all optional and
ANDed. `min_severity` accepts either vocabulary (`info|warning|critical` or
`info|low|medium|high|critical`). `window_ns` is a **loose overlap** check
against the finding's `[start_ns, end_ns]`; omit it to ignore location.

### Authoring rule: label, don't snapshot

Labels encode the **intended** result from domain analysis or an audit (e.g.
`docs/audits/l40s-fastvideo-gaps.md`), **not** a snapshot of current
`build()` output. Snapshotting would freeze today's behavior — including known
bugs — as the "correct" answer. Where current output is known-wrong, record it
under `known_gaps` rather than encoding it as `expect`.

## Scoring & CI gating (seed scope)

| Signal | Meaning | CI |
|---|---|---|
| `must_not` matched | a forbidden finding appeared (false positive) | **fails** |
| `expect` with `"required": true` unmet | a calibrated expected finding is missing | **fails** |
| `expect` (default) unmet | an expected finding is missing (false negative) | informational |
| `% categorized`, category histogram | finding-quality metrics | informational |
| `known_gaps` | documented, not yet fixed | printed only |

`must_not` violations and unmet `required` expectations gate; everything else
is printed so regressions are visible without freezing unverified behavior.
Mark an `expect` `"required": true` only once it is a stable, calibrated truth
(e.g. idle gaps on the committed `mock.sqlite`). Promote more expectations to
`required` in later PRs as labels settle.

The matcher/scorer itself is unit-tested in `tests/test_eval.py`
(`TestMatcher` / `TestScore`) so the eval's own logic runs in CI even when every
labeled profile is absent.

## Profiles & CI

Profiles are resolved relative to the repo root; a label whose profile is
**absent is skipped**. Only the committed `tests/fixtures/mock.sqlite` runs in
CI. Large real profiles (e.g. `nano_vllm_qwen3_4b.sqlite`) stay local-only —
keep their `.sqlite` out of the repo and run the eval locally:

```bash
pytest tests/test_eval.py -v        # metrics print via capsys.disabled()
pytest tests/test_eval.py -k mock   # just the CI-reproducible case
```

## Adding a case

1. Drop the profile where the label's `profile` path resolves (committed only if
   small, like `mock.sqlite`).
2. Run `EvidenceBuilder` on it (`nsys-ai analyze <profile> --gpu 0 --format json`)
   to see real output.
3. Write `labels/<name>.json` with `expect` / `must_not` drawn from domain truth
   — not from the raw output. Park known-wrong output under `known_gaps`.
4. `pytest tests/test_eval.py -k <name> -v`.

## Consumers (future work)

- **§4 B** — regression net for the skill-finding upgrade wave.
- **§4 F** — leave-one-skill-out marginal contribution for default-pack governance.
- **§4 H** — verification-agent must-not-claim evaluation.
