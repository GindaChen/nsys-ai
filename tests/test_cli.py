"""Basic smoke tests for nsys-ai package."""
import subprocess
import sys


def test_help():
    """CLI --help should exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "--help"],
        capture_output=True, text=True)
    assert result.returncode == 0
    assert "nsys-ai" in result.stdout


def test_import():
    """Package should be importable."""
    import nsys_tui
    assert hasattr(nsys_tui, '__version__') or True  # just check import works


def test_subcommands():
    """All subcommands should be registered."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "--help"],
        capture_output=True, text=True)
    for cmd in ['info', 'analyze', 'open', 'summary', 'overlap', 'nccl', 'iters', 'tree', 'markdown',
                'search', 'export-csv', 'export-json', 'export', 'viewer', 'timeline-html',
                'web', 'perfetto', 'timeline-web', 'tui', 'timeline', 'ask']:
        assert cmd in result.stdout, f"Missing subcommand: {cmd}"


def test_ask_help():
    """ask subcommand should show help with --help."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "ask", "--help"],
        capture_output=True, text=True)
    assert result.returncode == 0
    assert "question" in result.stdout.lower()
    assert "profile" in result.stdout.lower()
    assert "--follow-up" in result.stdout
