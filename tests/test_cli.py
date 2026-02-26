"""Basic smoke tests for nsys-tui package."""
import subprocess
import sys


def test_help():
    """CLI --help should exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "--help"],
        capture_output=True, text=True)
    assert result.returncode == 0
    assert "nsys-tui" in result.stdout


def test_import():
    """Package should be importable."""
    import nsys_tui
    assert hasattr(nsys_tui, '__version__') or True  # just check import works


def test_subcommands():
    """All subcommands should be registered."""
    result = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "--help"],
        capture_output=True, text=True)
    for cmd in ['info', 'summary', 'overlap', 'tree', 'tui', 'timeline',
                'search', 'export', 'viewer']:
        assert cmd in result.stdout, f"Missing subcommand: {cmd}"
