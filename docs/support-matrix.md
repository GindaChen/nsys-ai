# Verified Nsight export schemas

This table is the support boundary for the committed real SQLite exports. CI
reads every row, opens the capture, checks the required schema contract, and
runs the canonical diagnose skill pack. A schema is listed here only when the
capture parses and the pack produces at least one usable evidence row.

The matrix guards schemas we actually hold an export for; it does not claim
that an unobserved future Nsight Systems schema is supported. Product versions
are recorded as provenance, while the export schema version is the compatibility
axis.

| Fixture | Export schema | Nsight Systems product | CI coverage |
|---|---|---|---|
| `tests/fixtures/mock.sqlite` | `3.24.14` | `2026.1.1.204` | schema contract + default diagnose pack |
| `tests/fixtures/healthy_1pct.sqlite` | `3.24.14` | `2026.1.1.204` | schema contract + default diagnose pack |
| `tests/fixtures/healthy_judged_1pct.sqlite` | `3.24.14` | `2026.1.1.204` | schema contract + default diagnose pack |
| `tests/fixtures/h100_2gpu_1s.sqlite` | `3.25.0` | `2026.2.1.210` | schema contract + default diagnose pack |
| `tests/fixtures/mfu_2gpu_before.sqlite` | `3.25.0` | `2026.2.1.210` | schema contract + default diagnose pack |
| `tests/fixtures/mfu_2gpu_after.sqlite` | `3.25.0` | `2026.2.1.210` | schema contract + default diagnose pack |

To add a supported schema, commit a real export, add its row here, and let the
matrix test fail until the capture's metadata and analysis output are verified.
