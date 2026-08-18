"""Keep the CI gating guide aligned with the shipped diff-gate contract."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "ci-diff-gate.md"
INDEX = ROOT / "docs" / "README.md"


def test_ci_guide_documents_the_machine_contract():
    text = GUIDE.read_text(encoding="utf-8")

    for required in (
        "nsys-ai diff --gate",
        "baseline:main",
        "artifacts/diff.json",
        "--exit-on-regression",
        "`0`",
        "`1`",
        "`2`",
        "--accept",
        "--reason",
        "if: always()",
        "--gate-sol REGION:PCT",
        "--theoretical-flops FLOPS",
        "@main",
    ):
        assert required in text


def test_ci_guide_is_linked_from_the_docs_index():
    assert "[ci-diff-gate.md](./ci-diff-gate.md)" in INDEX.read_text(encoding="utf-8")
