import argparse
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from nsys_ai.profile_command import (
    ProfileCommandError,
    discover_git_provenance,
    normalize_workload,
    parse_environment,
    parse_supported_trace_domains,
    run_profile_command,
    select_trace_domains,
)
from nsys_ai.runspec import RunSpec, RunSpecError

_FAKE_NSYS = r'''#!/usr/bin/env python3
import json
import os
import sqlite3
import sys
from pathlib import Path


def value_after(flag):
    return sys.argv[sys.argv.index(flag) + 1]


if sys.argv[1:] == ['profile', '--help=trace']:
    domains = [item for item in os.environ.get('FAKE_TRACE_DOMAINS', 'cuda,nvtx,nccl').split(',') if item]
    advertised = ', '.join(repr(item) for item in domains + ['none'])
    print(f"""usage: nsys profile

    --unrelated=
       Possible values are 'not-a-domain'.

    -t, --trace=
       Possible values are {advertised}.
       Select the API(s) to trace. If '<api>-annotations' is selected, more text follows.
       Default is 'cuda,nvtx,osrt'.

    --trace-fork-before-exec=
       Possible values are 'true' or 'false'.
    """)
    sys.exit(0)

if sys.argv[1] == 'profile':
    print(json.dumps(sys.argv), flush=True)
    if '--fake-fail' in sys.argv:
        sys.exit(17)
    output = Path(value_after('-o'))
    output.with_suffix('.nsys-rep').write_text('valid')
    sys.exit(0)

if sys.argv[1] == 'export':
    output = Path(value_after('-o'))
    with sqlite3.connect(output) as conn:
        conn.executescript("""
            CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
            INSERT INTO META_DATA_EXPORT VALUES
                ('EXPORT_PRODUCT_VERSION', '2026.2.1.106'),
                ('EXPORT_SCHEMA_VERSION', '3.25.0');
            CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT);
            INSERT INTO TARGET_INFO_GPU VALUES (0, 'Fake GPU');
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES (1, 'fake_kernel');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                deviceId INTEGER, streamId INTEGER, start INTEGER, end INTEGER,
                shortName INTEGER, demangledName INTEGER, correlationId INTEGER
            );
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 7, 10, 20, 1, 1, 1);
        """)
    sys.exit(0)

sys.exit(99)
'''


@pytest.fixture
def fake_nsys(tmp_path):
    executable = tmp_path / "nsys"
    executable.write_text(_FAKE_NSYS)
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def _args(fake_nsys, output, **overrides):
    values = {
        "workload": ["--", "train.py", "--model=public-model"],
        "public_env": [],
        "secret_env": [],
        "env_policy": "inherit",
        "output": str(output),
        "nsys": str(fake_nsys),
        "trace": "auto",
        "dry_run": False,
        "warmup_steps": 0,
        "profile_steps": 1,
        "seed": None,
        "expected_gpu_count": None,
        "expected_rank_count": None,
        "sample": "none",
        "cpuctxsw": "none",
        "capture_range": "none",
        "cuda_memory_usage": False,
        "timeout": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _run_cli(cwd, *argv, env=None):
    cli_env = dict(os.environ)
    cli_env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    if env:
        cli_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "nsys_ai", *argv],
        cwd=cwd,
        env=cli_env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_normalize_workload_strips_one_delimiter_and_preserves_tokens():
    assert normalize_workload(["--", "python", "train.py", "--", "--flag=a b"]) == (
        "python",
        "train.py",
        "--",
        "--flag=a b",
    )
    assert normalize_workload(["train.py", "--lr", "1e-4"]) == (
        sys.executable,
        "train.py",
        "--lr",
        "1e-4",
    )
    assert normalize_workload(["python", "train.py"])[0] == "python"
    with pytest.raises(ProfileCommandError, match="workload"):
        normalize_workload(["--"])


def test_trace_parser_is_limited_to_trace_option_block():
    help_text = """
      --other=
         Possible values are 'wrong'.
      -t, --trace=
         Possible values are 'cuda', 'nvtx', 'nccl' or 'none'.
         Select the API(s) to trace. '<api>-annotations' is explanatory only.
         Default is 'cuda,nvtx'.
      --later=
         Possible values are 'also-wrong'.
    """
    assert parse_supported_trace_domains(help_text) == ("cuda", "nvtx", "nccl")


def test_trace_selection_degrades_auto_but_rejects_explicit_unsupported():
    warnings = []
    assert select_trace_domains("auto", ("cuda",), warning=warnings.append) == ("cuda",)
    assert warnings == [
        "nsys does not support optional nvtx tracing; omitting it",
        "nsys does not support optional nccl tracing; omitting it",
    ]
    with pytest.raises(ProfileCommandError, match="unsupported.*nccl"):
        select_trace_domains("cuda,nccl", ("cuda", "nvtx"))


@pytest.mark.parametrize(
    "help_text",
    [
        "--trace=\nPossible values are 'cuda', 'nvtx'.",
        "--other=\nPossible values are 'cuda'.",
    ],
)
def test_malformed_trace_help_is_rejected(help_text):
    with pytest.raises(ProfileCommandError, match="trace|domain"):
        parse_supported_trace_domains(help_text)


def test_trace_help_without_cuda_is_rejected():
    help_text = """
      -t, --trace=
         Possible values are 'nvtx', 'nccl' or 'none'.
         Select the API(s) to trace.
      --later=
    """
    with pytest.raises(ProfileCommandError, match="does not advertise CUDA"):
        parse_supported_trace_domains(help_text)


@pytest.mark.parametrize(
    ("public", "secrets", "message"),
    [
        (["VISIBLE=one", "VISIBLE=two"], [], "entry 2 duplicates --env entry 1"),
        (["BAD-NAME=value"], [], "entry 1 has an invalid variable name"),
        (["TOKEN=public"], ["TOKEN"], "entry 1 overlaps --env entry 1"),
        (["MISSING_EQUALS"], [], "NAME=VALUE"),
        ([], ["TOKEN", "TOKEN"], "entry 2 duplicates --secret-env entry 1"),
        ([], ["BAD-NAME"], "entry 1 has an invalid variable name"),
    ],
)
def test_environment_declaration_errors_are_rejected(public, secrets, message):
    with pytest.raises(RunSpecError, match=message):
        parse_environment(public, secrets)


def test_git_provenance_records_absolute_root_full_commit_and_relative_cwd(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "training" / "jobs"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "tracked.txt").write_text("tracked")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)

    repository, commit, cwd = discover_git_provenance(nested)

    assert repository == str(repo.resolve())
    assert commit is not None and len(commit) == 40
    assert cwd == "training/jobs"


def test_non_git_provenance_degrades_to_absolute_cwd(tmp_path):
    repository, commit, cwd = discover_git_provenance(tmp_path)
    assert repository is None
    assert commit is None
    assert cwd == str(tmp_path.resolve())


def test_dry_run_is_structurally_redacted_and_does_not_create_output(
    tmp_path, fake_nsys, monkeypatch
):
    monkeypatch.delenv("UNSET_SECRET", raising=False)
    output = tmp_path / "dry-artifacts"
    stdout = io.StringIO()

    def forbidden_runner(*_args):
        raise AssertionError("dry-run must not construct the runner")

    exit_code = run_profile_command(
        _args(
            fake_nsys,
            output,
            dry_run=True,
            workload=["--", "train.py", "--token=workload-value"],
            public_env=["VISIBLE=public-value"],
            secret_env=["UNSET_SECRET"],
        ),
        stdout=stdout,
        stderr=io.StringIO(),
        cwd=tmp_path,
        runner_factory=forbidden_runner,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["workload_token_count"] == 3
    assert payload["public_environment_names"] == ["VISIBLE"]
    assert payload["secret_environment"] == {"UNSET_SECRET": "unresolved"}
    assert "workload-value" not in stdout.getvalue()
    assert "public-value" not in stdout.getvalue()
    assert not output.exists()


def test_fake_nsys_cli_preserves_tokens_python_shorthand_and_artifacts(
    tmp_path, fake_nsys
):
    output = tmp_path / "artifacts"
    result = _run_cli(
        tmp_path,
        "profile",
        "--nsys",
        str(fake_nsys),
        "--output",
        str(output),
        "--warmup-steps",
        "2",
        "--profile-steps",
        "5",
        "--",
        "train.py",
        "--model=x y",
        "--",
        "--tail",
    )

    assert result.returncode == 0, result.stderr
    capture_argv = json.loads((output / "stdout.log").read_text())
    workload_start = capture_argv.index(sys.executable)
    assert capture_argv[workload_start:] == [
        sys.executable,
        "train.py",
        "--model=x y",
        "--",
        "--tail",
    ]
    spec = RunSpec.from_json_bytes((output / "runspec.json").read_bytes())
    assert spec.argv == tuple(capture_argv[workload_start:])
    assert spec.warmup_steps == 2
    assert spec.profile_steps == 5
    assert spec.trace_options.trace == ("cuda", "nccl", "nvtx")
    assert (output / "profile.nsys-rep").is_file()
    assert (output / "profile.sqlite").is_file()
    assert "Profile ID:" in result.stdout
    assert "Kernels: 1" in result.stdout


def test_fake_nsys_cli_records_git_provenance_and_environment(tmp_path, fake_nsys):
    repo = tmp_path / "repo"
    work = repo / "training"
    work.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "tracked.txt").write_text("tracked")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    output = tmp_path / "artifacts"

    result = _run_cli(
        work,
        "profile",
        "--nsys",
        str(fake_nsys),
        "--output",
        str(output),
        "--env",
        "VISIBLE=a=b",
        "--secret-env",
        "CLI_SECRET",
        "python",
        "train.py",
        "--wrapper-looking-token",
        env={"CLI_SECRET": "secret-value"},
    )

    assert result.returncode == 0, result.stderr
    spec = RunSpec.from_json_bytes((output / "runspec.json").read_bytes())
    assert spec.repository == str(repo.resolve())
    assert spec.commit is not None and len(spec.commit) == 40
    assert spec.cwd == "training"
    assert spec.environment.public == {"VISIBLE": "a=b"}
    assert spec.environment.secrets == ("CLI_SECRET",)
    assert spec.argv == ("python", "train.py", "--wrapper-looking-token")
    assert "secret-value" not in (output / "runspec.json").read_text()


def test_auto_trace_omission_warns_and_reaches_persisted_runspec(tmp_path, fake_nsys):
    output = tmp_path / "artifacts"
    result = _run_cli(
        tmp_path,
        "profile",
        "--nsys",
        str(fake_nsys),
        "--output",
        str(output),
        "--",
        "train",
        env={"FAKE_TRACE_DOMAINS": "cuda"},
    )

    assert result.returncode == 0, result.stderr
    assert "optional nvtx tracing; omitting it" in result.stderr
    assert "optional nccl tracing; omitting it" in result.stderr
    spec = RunSpec.from_json_bytes((output / "runspec.json").read_bytes())
    assert spec.trace_options.trace == ("cuda",)


def test_secret_boundary_failure_creates_no_artifacts_and_never_echoes_value(
    tmp_path, fake_nsys
):
    output = tmp_path / "artifacts"
    secret = "highly-sensitive-value"
    result = _run_cli(
        tmp_path,
        "profile",
        "--nsys",
        str(fake_nsys),
        "--output",
        str(output),
        "--secret-env",
        "CLI_SECRET",
        "--",
        "train",
        f"--token={secret}",
        env={"CLI_SECRET": secret},
    )

    assert result.returncode == 1
    assert "secret command-line arguments are unsupported" in result.stderr
    assert secret not in result.stderr
    assert not output.exists()


def test_missing_declared_secret_creates_no_artifacts(
    tmp_path, fake_nsys, monkeypatch
):
    monkeypatch.delenv("DELIBERATELY_UNSET_SECRET", raising=False)
    output = tmp_path / "artifacts"
    result = _run_cli(
        tmp_path,
        "profile",
        "--nsys",
        str(fake_nsys),
        "--output",
        str(output),
        "--secret-env",
        "DELIBERATELY_UNSET_SECRET",
        "--",
        "train",
    )

    assert result.returncode == 1
    assert "declared secret DELIBERATELY_UNSET_SECRET is not set" in result.stderr
    assert not output.exists()


def test_keyboard_interrupt_before_runner_returns_130_without_artifacts(
    tmp_path, fake_nsys, monkeypatch, capsys
):
    import nsys_ai.profile_command as profile_command
    from nsys_ai.cli.handlers import _cmd_profile

    output = tmp_path / "artifacts"

    def interrupt(_selected):
        raise KeyboardInterrupt

    monkeypatch.setattr(profile_command, "resolve_nsys_executable", interrupt)
    with pytest.raises(SystemExit) as exited:
        _cmd_profile(_args(fake_nsys, output), None)

    captured = capsys.readouterr()
    assert exited.value.code == 130
    assert captured.err == "Profile cancelled.\n"
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_keyboard_interrupt_during_lazy_import_returns_130_without_traceback(
    tmp_path, fake_nsys, monkeypatch, capsys
):
    import builtins

    from nsys_ai.cli.handlers import _cmd_profile

    output = tmp_path / "artifacts"
    original_import = builtins.__import__

    def interrupt_profile_command(name, *args, **kwargs):
        if name == "nsys_ai.profile_command":
            raise KeyboardInterrupt
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", interrupt_profile_command)
    with pytest.raises(SystemExit) as exited:
        _cmd_profile(_args(fake_nsys, output), None)

    captured = capsys.readouterr()
    assert exited.value.code == 130
    assert captured.err == "Profile cancelled.\n"
    assert "Traceback" not in captured.err
    assert not output.exists()


@pytest.mark.parametrize(
    ("declarations", "secret_name", "expected_location"),
    [
        (
            ["--env", "DO_NOT_PRINT_THIS_SECRET=one", "--env", "DO_NOT_PRINT_THIS_SECRET=two"],
            "DECLARED_SECRET",
            "--env entry 2 duplicates --env entry 1",
        ),
        (
            ["--env", "BAD-DO_NOT_PRINT_THIS_SECRET=value"],
            "DECLARED_SECRET",
            "--env entry 1 has an invalid variable name",
        ),
        (
            ["--env", "DO_NOT_PRINT_THIS_SECRET=value"],
            "DO_NOT_PRINT_THIS_SECRET",
            "--secret-env entry 1 overlaps --env entry 1",
        ),
    ],
)
def test_environment_declaration_errors_never_echo_secret_values(
    tmp_path, fake_nsys, declarations, secret_name, expected_location
):
    sentinel = "DO_NOT_PRINT_THIS_SECRET"
    output = tmp_path / "artifacts"
    result = _run_cli(
        tmp_path,
        "profile",
        "--nsys",
        str(fake_nsys),
        "--output",
        str(output),
        *declarations,
        "--secret-env",
        secret_name,
        "--",
        "train",
        env={secret_name: sentinel},
    )

    assert result.returncode == 1
    assert expected_location in result.stderr
    assert sentinel not in result.stderr
    assert sentinel not in result.stdout
    assert not output.exists()


@pytest.mark.parametrize("workload", [(), ("--",)])
def test_missing_workload_is_argparse_error_with_no_artifacts(
    tmp_path, fake_nsys, workload
):
    output = tmp_path / "artifacts"
    result = _run_cli(
        tmp_path,
        "profile",
        "--nsys",
        str(fake_nsys),
        "--output",
        str(output),
        *workload,
    )

    assert result.returncode == 2
    assert "a workload command is required" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_typed_runner_failure_returns_one_not_nsys_code(tmp_path, fake_nsys):
    output = tmp_path / "artifacts"
    result = _run_cli(
        tmp_path,
        "profile",
        "--nsys",
        str(fake_nsys),
        "--output",
        str(output),
        "--",
        "train",
        "--fake-fail",
    )

    assert result.returncode == 1
    assert "nsys_failed" in result.stderr
    assert "exit code 17" not in result.stderr
    assert (output / "runspec.json").is_file()


def test_explicit_unsupported_trace_fails_before_artifacts(tmp_path, fake_nsys):
    output = tmp_path / "artifacts"
    result = _run_cli(
        tmp_path,
        "profile",
        "--nsys",
        str(fake_nsys),
        "--output",
        str(output),
        "--trace",
        "cuda,unknown",
        "--",
        "train",
    )

    assert result.returncode == 1
    assert "unsupported trace domain" in result.stderr
    assert not output.exists()


def test_existing_bare_profile_shortcut_remains_viewer_routed():
    from nsys_ai.cli.app import _normalize_default_profile_command

    assert _normalize_default_profile_command(["nsys-ai", "profile.sqlite"]) == [
        "nsys-ai",
        "timeline-web",
        "profile.sqlite",
    ]
