"""Schema contract — the tables/columns the core analysis path requires
(issue #237).

NVIDIA documents that the Nsight SQLite export schema "can and will change".
``NsightSchema.missing_required_columns`` makes the hard requirements explicit
so a future export that drops or renames one fails loudly, by name, in CI and
in ``doctor`` — rather than surfacing as a subtly-wrong number in a report.

Uses the ``minimal_nsys_conn`` fixture (a function-scoped, seeded in-memory
sqlite connection) rather than importing conftest internals, so the synthetic
schema stays a single source of truth and imports work under any pytest rootdir.
"""

import pytest

from nsys_ai.profile import NsightSchema, Profile

# ── The contract passes on real and synthetic exports ───────────────────────


def test_committed_fixture_satisfies_the_contract():
    """The checked-in nsys export (mock.sqlite, schema 3.24.14) must pass — this
    is the CI guard: a fixture update to an incompatible export fails here."""
    with Profile("tests/fixtures/mock.sqlite") as prof:
        assert prof.schema.missing_required_columns() == []
        # Fixture schema version is recorded, not implicit.
        assert prof.schema.schema_version == "3.24.14"


def test_synthetic_fixture_satisfies_the_contract(minimal_nsys_conn):
    assert NsightSchema(minimal_nsys_conn).missing_required_columns() == []


# ── Version detection now has synthetic coverage ────────────────────────────


def test_synthetic_conn_exposes_export_versions(minimal_nsys_conn):
    """conftest seeds META_DATA_EXPORT, so version detection is exercised
    without a binary fixture (it read as None before)."""
    schema = NsightSchema(minimal_nsys_conn)
    assert schema.schema_version == "3.24.14"
    assert schema.version == "2026.1.1.204"


# ── The contract fails loudly, naming exactly what is missing ───────────────


@pytest.mark.parametrize(
    "column",
    ["deviceId", "streamId", "start", "end", "shortName", "demangledName", "correlationId"],
)
def test_missing_kernel_column_is_named(minimal_nsys_conn, column):
    minimal_nsys_conn.execute(
        f"ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL RENAME COLUMN {column} TO {column}_gone"
    )
    missing = NsightSchema(minimal_nsys_conn).missing_required_columns()
    assert f"CUPTI_ACTIVITY_KIND_KERNEL.{column}" in missing


@pytest.mark.parametrize("column", ["id", "value"])
def test_missing_stringids_column_is_named(minimal_nsys_conn, column):
    minimal_nsys_conn.execute(f"ALTER TABLE StringIds RENAME COLUMN {column} TO {column}_gone")
    missing = NsightSchema(minimal_nsys_conn).missing_required_columns()
    assert f"StringIds.{column}" in missing


def test_missing_kernel_table_is_named(minimal_nsys_conn):
    minimal_nsys_conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
    missing = NsightSchema(minimal_nsys_conn).missing_required_columns()
    assert any("kernel activity table" in m for m in missing)


def test_missing_stringids_table_is_named(minimal_nsys_conn):
    minimal_nsys_conn.execute("DROP TABLE StringIds")
    assert "StringIds" in NsightSchema(minimal_nsys_conn).missing_required_columns()


def test_optional_tables_do_not_trip_the_contract(minimal_nsys_conn):
    """A --trace=cuda capture has no NVTX_EVENTS; skills degrade around it, so
    its absence must not be a contract violation."""
    minimal_nsys_conn.execute("DROP TABLE NVTX_EVENTS")
    assert NsightSchema(minimal_nsys_conn).missing_required_columns() == []


def test_column_check_is_case_insensitive(minimal_nsys_conn):
    """Real exports vary column case; the contract must not false-fail on it."""
    minimal_nsys_conn.execute("ALTER TABLE StringIds RENAME COLUMN value TO VALUE")
    assert "StringIds.value" not in NsightSchema(minimal_nsys_conn).missing_required_columns()


# ── doctor surfaces the contract at runtime ─────────────────────────────────


def test_doctor_reports_schema_compatibility_ok():
    from nsys_ai.doctor import _check_profile_health

    with Profile("tests/fixtures/mock.sqlite") as prof:
        section = _check_profile_health(prof)
    check = next(c for c in section.checks if c.name == "Schema compatibility")
    assert check.status == "ok"
    assert "3.24.14" in check.detail


def test_doctor_fails_on_missing_required_column(minimal_nsys_conn):
    from nsys_ai.doctor import _check_profile_health

    # `demangledName` is required by the analysis path but not read during
    # Profile construction (_discover keys off deviceId/streamId/start/end), so
    # the profile still builds and doctor can surface the drift as a clean fail.
    # Columns _discover itself needs would instead raise at construction — a loud
    # failure of a different kind, out of scope for this check.
    minimal_nsys_conn.execute(
        "ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL RENAME COLUMN demangledName TO demangledName_gone"
    )
    prof = Profile._from_conn(minimal_nsys_conn)
    section = _check_profile_health(prof)
    check = next(c for c in section.checks if c.name == "Schema compatibility")
    assert check.status == "fail"
    assert "CUPTI_ACTIVITY_KIND_KERNEL.demangledName" in check.detail
    assert check.hint  # actionable
