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
    """Package should be importable and expose __version__."""
    import nsys_tui
    assert hasattr(nsys_tui, "__version__")
    assert isinstance(nsys_tui.__version__, str)
    assert nsys_tui.__version__  # non-empty


def test_subcommands():
    """All subcommands should be registered."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "--help"],
        capture_output=True, text=True)
    for cmd in ['info', 'analyze', 'open', 'summary', 'overlap', 'nccl', 'iters', 'tree', 'markdown',
                'search', 'export-csv', 'export-json', 'export', 'viewer', 'timeline-html',
                'web', 'perfetto', 'timeline-web', 'tui', 'timeline', 'chat']:
        assert cmd in result.stdout, f"Missing subcommand: {cmd}"


def test_chat_subcommand_help():
    """chat subcommand should have --help and accept a profile argument."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "chat", "--help"],
        capture_output=True, text=True)
    assert result.returncode == 0
    assert "profile" in result.stdout
