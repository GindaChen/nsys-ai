"""Documentation is findable, asserted rather than assumed.

There are two hand-maintained indexes — `docs/README.md` for people reading the repository and
`site/index.html` for people arriving at the project page — and nothing connected them. They had
already drifted: the site listed eight numbered files and none of the project guides.

A page nobody can reach is not written, so both indexes are checked here. The check is deliberately
narrow: it covers `docs/user/` and `docs/dev/`, the two directories whose contents are meant to be
navigable, and says nothing about the rest of `docs/`.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
README = DOCS / "README.md"
SITE = REPO / "site" / "index.html"

#: Directories whose pages must appear in both indexes.
INDEXED_DIRECTORIES = ("user", "dev")


def _indexed_pages() -> list[str]:
    pages = []
    for directory in INDEXED_DIRECTORIES:
        for path in sorted((DOCS / directory).glob("*.md")):
            pages.append(f"{directory}/{path.name}")
    return pages


def test_there_is_something_to_index():
    """Guard the guard: an empty glob would make every assertion below vacuous."""
    pages = _indexed_pages()
    assert pages, f"no pages found under {INDEXED_DIRECTORIES}; the check below would pass on nothing"


def test_every_page_is_listed_in_the_repository_index():
    body = README.read_text(encoding="utf-8")
    missing = [page for page in _indexed_pages() if f"./{page}" not in body]
    assert not missing, (
        "pages missing from docs/README.md: "
        + ", ".join(missing)
        + "\n\nAdd a row to its file index. A page nobody can find is not written."
    )


def test_every_page_is_linked_from_the_site():
    body = SITE.read_text(encoding="utf-8")
    missing = [page for page in _indexed_pages() if f"/docs/{page}" not in body]
    assert not missing, (
        "pages missing from site/index.html: "
        + ", ".join(missing)
        + "\n\nDocumentation is not published to the site — the landing page links GitHub blobs — so a"
        " new page is invisible there until this list is edited."
    )


def test_site_doc_links_point_at_files_that_exist():
    """A link to a renamed or deleted page is worse than no link."""
    body = SITE.read_text(encoding="utf-8")
    linked = set(re.findall(r"blob/main/(docs/[^\"'\s]+\.(?:md|html))", body))
    assert linked, "no documentation links found in site/index.html"
    broken = sorted(path for path in linked if not (REPO / path).is_file())
    assert not broken, "site/index.html links to files that do not exist: " + ", ".join(broken)


def test_repository_index_links_point_at_files_that_exist():
    body = README.read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(\./([^)]+\.(?:md|html))\)", body))
    assert linked, "no documentation links found in docs/README.md"
    broken = sorted(path for path in linked if not (DOCS / path).is_file())
    assert not broken, "docs/README.md links to files that do not exist: " + ", ".join(broken)


def _heading_anchors(path: Path) -> set[str]:
    """GitHub's slug rule, reduced to what these documents actually use."""
    anchors = set()
    for heading in re.findall(r"^#+\s+(.+)$", path.read_text(encoding="utf-8"), re.M):
        slug = re.sub(r"[^a-z0-9\s-]", "", heading.lower()).strip().replace(" ", "-")
        anchors.add(slug)
    return anchors


def test_cross_page_links_resolve():
    """The pages link to each other heavily, so a rename breaks navigation silently.

    Anchors are checked too. A link to a section that has been retitled still renders as a
    working link and lands the reader at the top of the page, which is the failure that is
    hardest to notice by reading the diff.
    """
    broken = []
    for page in _indexed_pages():
        path = DOCS / page
        for _text, target in re.findall(r"\[([^\]]+)\]\((\.[^)]+)\)", path.read_text(encoding="utf-8")):
            file_part, _, anchor = target.partition("#")
            destination = (path.parent / file_part).resolve()
            if not destination.is_file():
                broken.append(f"{page} -> {target} (no such file)")
            elif anchor and anchor not in _heading_anchors(destination):
                broken.append(f"{page} -> {target} (no such heading)")
    assert not broken, "broken links between documentation pages:\n  " + "\n  ".join(broken)
