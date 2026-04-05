"""Root cause pattern store — parse, validate, submit, and list entries.

Supports packaged, user-local, and source-checkout markdown sources:
1. Built-in entries from the package's bundled data (e.g., ``src/nsys_ai/data/book.md``)
2. Community entries from ``docs/root-causes/community/*.md`` (source-checkout / editable install only)
3. User-local entries from ``~/.nsys-ai/root-causes/*.md`` (personal)

Each entry is a markdown file with YAML frontmatter.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RootCauseEntry:
    """A single root cause pattern."""

    name: str
    severity: str = "warning"  # info, warning, critical
    tags: list[str] = field(default_factory=list)
    detection_skill: str = ""
    symptom: str = ""
    mechanism: str = ""  # "Why It Happens"
    detection: str = ""  # "How to Detect"
    fix: str = ""  # "How to Fix"
    example: str = ""  # "Real-World Example"
    source: str = "builtin"  # builtin, community, user
    file_path: str = ""

    def to_summary_row(self) -> dict:
        """Return a dict suitable for tabular display."""
        return {
            "name": self.name,
            "severity": self.severity,
            "source": self.source,
            "tags": ", ".join(self.tags),
            "detection_skill": self.detection_skill,
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.DOTALL)
_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from markdown text."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_yaml = m.group(1)
    body = text[m.end() :]

    # Minimal YAML parser (no PyYAML dependency)
    meta: dict = {}
    for line in raw_yaml.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        # Handle list syntax: [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            items = [x.strip().strip("'\"") for x in val[1:-1].split(",")]
            meta[key] = [x for x in items if x]
        else:
            meta[key] = val.strip("'\"")
    return meta, body


def _extract_sections(body: str) -> dict[str, str]:
    """Extract named sections (## Heading → content) from markdown body."""
    sections: dict[str, str] = {}
    parts = _SECTION_RE.split(body)
    # parts = [preamble, heading1, content1, heading2, content2, ...]
    for i in range(1, len(parts) - 1, 2):
        # Strip trailing whitespace and colons (e.g. "## Symptom:")
        heading = re.sub(r"[:\s]+$", "", parts[i]).strip()
        content = parts[i + 1].strip()
        sections[heading.lower()] = content
    return sections


# Section heading aliases → RootCauseEntry field
_SECTION_MAP = {
    "symptom": "symptom",
    "symptoms": "symptom",
    "why it happens": "mechanism",
    "why": "mechanism",
    "mechanism": "mechanism",
    "how to detect": "detection",
    "detection": "detection",
    "how to fix": "fix",
    "fix": "fix",
    "how to fix it": "fix",
    "real-world example": "example",
    "example": "example",
    "real world example": "example",
}


def parse_entry(text: str, source: str = "user", file_path: str = "") -> RootCauseEntry:
    """Parse a markdown root cause entry into a RootCauseEntry dataclass."""
    meta, body = _parse_frontmatter(text)
    sections = _extract_sections(body)

    entry = RootCauseEntry(
        name=meta.get("name", "Untitled"),
        severity=meta.get("severity", ""),
        tags=meta.get("tags", []) if isinstance(meta.get("tags"), list) else [],
        detection_skill=meta.get("detection_skill", ""),
        source=source,
        file_path=file_path,
    )

    for heading, content in sections.items():
        field_name = _SECTION_MAP.get(heading)
        if field_name:
            setattr(entry, field_name, content)

    return entry


# ---------------------------------------------------------------------------
# Built-in book.md parser
# ---------------------------------------------------------------------------


def _parse_book_md(book_path: str | Path) -> list[RootCauseEntry]:
    """Parse the structured book.md into individual entries.

    book.md uses H2 headings (## N. Title) to separate entries,
    and bold markers (e.g., **Symptom:**) to demarcate sections.
    """
    path = Path(book_path)
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    entries: list[RootCauseEntry] = []

    # Split by H2 headings: ## 1. GPU Bubbles, ## 2. CPU Bottleneck, etc.
    h2_pattern = re.compile(r"^## \d+\.\s+(.+)$", re.MULTILINE)
    parts = h2_pattern.split(text)

    # parts = [preamble, title1, content1, title2, content2, ...]
    for i in range(1, len(parts) - 1, 2):
        title = parts[i].strip()
        content = parts[i + 1].strip()

        # Extract sub-sections (### headings within this entry)
        sub_sections: dict[str, str] = {}
        h3_pattern = re.compile(r"^\*\*(.+?)\*\*(?:\s|:)", re.MULTILINE)
        sub_parts = h3_pattern.split(content)
        for j in range(1, len(sub_parts) - 1, 2):
            sub_heading = sub_parts[j].strip().rstrip(":").lower()
            sub_content = sub_parts[j + 1].strip()
            # Truncate at the next bold heading
            sub_content = sub_content.split("\n**")[0].strip()
            sub_sections[sub_heading] = sub_content

        entry = RootCauseEntry(
            name=title,
            source="builtin",
            file_path=str(path),
        )

        for heading, cont in sub_sections.items():
            field_name = _SECTION_MAP.get(heading)
            if field_name:
                setattr(entry, field_name, cont)

        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Load from directories
# ---------------------------------------------------------------------------


def _load_dir_entries(directory: str | Path, source: str) -> list[RootCauseEntry]:
    """Load all .md files from a directory as root cause entries."""
    d = Path(directory)
    if not d.is_dir():
        return []

    entries = []
    for f in sorted(d.glob("*.md")):
        if f.name.startswith("_") or f.name == "TEMPLATE.md":
            continue
        try:
            text = f.read_text(encoding="utf-8")
            entry = parse_entry(text, source=source, file_path=str(f))
            entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to load {source} root cause from {f}: {e}")
            continue
    return entries


def _default_user_dir() -> Path:
    """Return the default user-local root causes directory."""
    return Path.home() / ".nsys-ai" / "root-causes"


def _find_book_md() -> Path | None:
    """Locate data/book.md relative to the package."""
    # __file__ = src/nsys_ai/root_cause_store.py
    pkg_dir = Path(__file__).resolve().parent  # src/nsys_ai/

    candidates = [
        pkg_dir / "data" / "book.md",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_community_dir() -> Path | None:
    """Locate docs/root-causes/community/ relative to the package."""
    pkg_dir = Path(__file__).resolve().parent  # src/nsys_ai/
    repo_root = pkg_dir.parent.parent  # nsys-tui/

    candidates = [
        repo_root / "docs" / "root-causes" / "community",
        pkg_dir / "docs" / "root-causes" / "community",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def list_entries(root_causes_dir: str | None = None) -> list[RootCauseEntry]:
    """List all root cause entries from all sources.

    Merge order: builtin (book.md) → community → user-local.
    """
    entries: list[RootCauseEntry] = []

    # 1. Builtin
    book = _find_book_md()
    if book:
        entries.extend(_parse_book_md(book))

    # 2. Community
    community = _find_community_dir()
    if community:
        entries.extend(_load_dir_entries(community, source="community"))

    # 3. User-local
    user_dir = Path(root_causes_dir) if root_causes_dir else _default_user_dir()
    entries.extend(_load_dir_entries(user_dir, source="user"))

    return entries


# ---------------------------------------------------------------------------
# Validation + submission
# ---------------------------------------------------------------------------


def validate_entry(entry: RootCauseEntry) -> list[str]:
    """Validate a root cause entry. Returns a list of error messages."""
    errors = []
    if not entry.name or entry.name == "Untitled":
        errors.append("Missing 'name' in frontmatter")
    if not entry.severity:
        errors.append("Missing 'severity' in frontmatter")
    elif entry.severity not in ("info", "warning", "critical"):
        errors.append(f"Invalid severity '{entry.severity}' — must be info/warning/critical")
    if not entry.symptom:
        errors.append("Missing '## Symptom' section")
    if not entry.fix:
        errors.append("Missing '## How to Fix' section")
    return errors


def submit_entry(
    source_path: str | Path,
    dest_dir: str | Path | None = None,
) -> tuple[RootCauseEntry, list[str]]:
    """Validate and copy a root cause markdown file to the destination directory.

    Args:
        source_path: Path to the .md file to submit.
        dest_dir: Destination directory (default: ~/.nsys-ai/root-causes/).

    Returns:
        (entry, errors) — if errors is non-empty, submission was rejected.
    """
    src = Path(source_path)
    if not src.exists():
        return RootCauseEntry(name=""), [f"File not found: {src}"]

    text = src.read_text(encoding="utf-8")
    entry = parse_entry(text, source="user", file_path=str(src))
    errors = validate_entry(entry)

    if errors:
        return entry, errors

    # Write to destination
    dest = Path(dest_dir) if dest_dir else _default_user_dir()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        dest.chmod(0o700)
    except OSError:
        pass

    # Sanitize filename
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", entry.name.lower().replace(" ", "_"))
    dest_file = dest / f"{safe_name}.md"

    if dest_file.exists():
        return entry, [
            f"Submission rejected: destination file already exists: {dest_file}. "
            "Please rename the entry or remove the existing file before submitting."
        ]

    shutil.copy2(str(src), str(dest_file))
    try:
        dest_file.chmod(0o600)
    except OSError:
        pass

    entry.file_path = str(dest_file)
    return entry, []
