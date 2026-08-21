"""Regression tests for coverage data written by CLI subprocesses."""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import coverage

from nsys_ai.cli.handlers import _cmd_doctor

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cli_subprocess_writes_mergeable_coverage_data(tmp_path):
    """A child launched from a temporary cwd contributes handler lines.

    This is intentionally independent of whether the parent pytest invocation
    uses ``--cov``. It exercises the same environment that conftest installs
    for pytest-cov and catches the two easy regressions: the startup hook is no
    longer importable, or relative ``source`` paths disappear when a CLI test
    changes cwd.
    """
    data_file = tmp_path / ".coverage"
    env = os.environ.copy()
    env["COVERAGE_PROCESS_START"] = str(REPO_ROOT / "pyproject.toml")
    env["COVERAGE_SOURCE"] = str(REPO_ROOT / "src" / "nsys_ai")
    env["COVERAGE_FILE"] = str(data_file)
    source_path = str(REPO_ROOT / "src")
    pythonpath = env.get("PYTHONPATH", "").split(os.pathsep) if env.get("PYTHONPATH") else []
    if source_path not in pythonpath:
        pythonpath.insert(0, source_path)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from argparse import Namespace; "
                "from nsys_ai.cli.handlers import _cmd_doctor; "
                "_cmd_doctor(Namespace(profile=None, deep=False, format='json', "
                "verbose=False, strict=False), None)"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    sidecars = sorted(tmp_path.glob(".coverage.*"))
    assert sidecars, "the CLI subprocess did not write a coverage sidecar"

    combined = coverage.Coverage(data_file=str(data_file), config_file=False)
    combined.combine(data_paths=[str(tmp_path)])
    combined.load()
    handlers = next(
        path for path in combined.get_data().measured_files() if path.endswith("cli/handlers.py")
    )
    covered = set(combined.get_data().lines(handlers))
    start = inspect.getsourcelines(_cmd_doctor)[1]
    body_lines = range(start, start + len(inspect.getsource(_cmd_doctor).splitlines()))
    assert set(body_lines).intersection(covered), (
        "the child imported handlers but did not record the executed doctor path"
    )
