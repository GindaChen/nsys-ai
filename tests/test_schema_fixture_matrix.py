"""Verify every published Nsight export schema against a real capture.

The support table is deliberately documentation-owned: keeping the fixture
paths and observed metadata in one visible table makes the support claim easy
to review. This test makes that table load-bearing by checking that it neither
omits a committed SQLite export nor names an uncommitted one.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from nsys_ai.agent.runner import run_diagnose_pack
from nsys_ai.profile import Profile
from nsys_ai.skills.base import is_abstention_row

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
SUPPORT_TABLE = REPO / "docs" / "support-matrix.md"
ROW_RE = re.compile(
    r"^\| `(?P<path>tests/fixtures/[^`]+\.sqlite)` \| `(?P<schema>[^`]+)` "
    r"\| `(?P<product>[^`]+)` \| (?P<coverage>[^|]+) \|$"
)


def _support_rows() -> list[dict[str, str]]:
    rows = []
    in_table = False
    for line in SUPPORT_TABLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Fixture |"):
            in_table = True
            continue
        if not in_table or not line.startswith("|") or line.startswith("|---"):
            continue
        match = ROW_RE.match(line)
        assert match, f"malformed support-matrix row: {line}"
        rows.append(match.groupdict())
    return rows


def _is_git_tracked(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative_path],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def test_support_table_covers_exactly_the_committed_sqlite_exports():
    rows = _support_rows()
    listed = {row["path"] for row in rows}
    committed = {
        str(path.relative_to(REPO))
        for path in FIXTURES.glob("*.sqlite")
    }

    assert len(rows) == len(listed), "support table contains duplicate fixture rows"
    assert listed == committed, (
        "support table drifted from committed exports: "
        f"missing={sorted(committed - listed)}, extra={sorted(listed - committed)}"
    )
    assert all(_is_git_tracked(path) for path in listed), (
        "support table names an export that is not tracked by git"
    )


def test_each_published_schema_parses_and_produces_diagnose_evidence(tmp_path):
    """A fixture that parses but produces no evidence cannot be supported."""
    for row in _support_rows():
        source = REPO / row["path"]
        profile_copy = tmp_path / source.name
        shutil.copy2(source, profile_copy)

        with Profile(profile_copy) as profile:
            assert profile.schema.schema_version == row["schema"], row["path"]
            assert profile.schema.version == row["product"], row["path"]
            assert profile.schema.missing_required_columns() == [], row["path"]

            evidence = run_diagnose_pack(profile.query_conn())
            usable_rows = [
                item
                for rows in evidence.values()
                for item in rows
                if isinstance(item, dict)
                and item
                and not item.get("error")
                and not is_abstention_row(item)
            ]
            assert usable_rows, (
                f"default diagnose pack produced no usable evidence for {row['path']}"
            )
