"""Basic smoke tests for nsys-ai package."""

import subprocess
import sys


def test_help():
    """CLI --help should exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "nsys-ai" in result.stdout


def test_import():
    """Package should be importable and expose __version__."""
    import nsys_ai

    assert hasattr(nsys_ai, "__version__")
    assert isinstance(nsys_ai.__version__, str)
    assert nsys_ai.__version__  # non-empty


def test_subcommands():
    """Public CLI surface should stay small and web/AI focused."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "--help"], capture_output=True, text=True
    )
    for cmd in [
        "open",
        "web",
        "timeline-web",
        "chat",
        "ask",
        "report",
        "diff",
        "diff-web",
        "export",
        "agent-guide",
    ]:
        assert cmd in result.stdout, f"Missing subcommand: {cmd}"

    # Legacy command names should be hidden from top-level help.
    usage_line = result.stdout.splitlines()[0]
    for hidden in ["info", "summary", "overlap", "skill"]:
        assert hidden not in usage_line

    # 'agent-guide' is public, but 'agent' should be hidden
    assert ",agent," not in usage_line
    assert ",agent}" not in usage_line


def test_chat_subcommand_help():
    """chat subcommand should have --help and accept a profile argument."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "chat", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "profile" in result.stdout


def test_diff_web_subcommand_help():
    """diff-web subcommand should have --help and accept before/after paths."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff-web", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "before" in result.stdout
    assert "after" in result.stdout


def test_diff_subcommand_help():
    """diff subcommand should have --help and accept before/after paths."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "diff", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "before" in result.stdout
    assert "after" in result.stdout


def test_legacy_analyze_still_available():
    """Hidden legacy command should still parse and show help."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "analyze", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--gpu" in result.stdout


def test_agent_guide():
    """agent-guide subcommand should print the system prompt payload."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_ai", "agent-guide"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "nsys-ai Agent Guide" in result.stdout
    assert "Orient" in result.stdout
    assert "Available Skills" in result.stdout
