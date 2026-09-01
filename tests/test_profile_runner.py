import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nsys_ai.exceptions import ProfileError
from nsys_ai.profile_runner import (
    LocalProfileRunner,
    RunProgress,
    RunStage,
    RunStatus,
    _assert_capture_does_not_persist_environment,
)
from nsys_ai.runspec import EnvironmentSpec, NsysTraceOptions, RunSpec, RunSpecError

_FAKE_NSYS = r'''#!/usr/bin/env python3
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def value_after(flag):
    index = sys.argv.index(flag)
    return sys.argv[index + 1]


def make_sqlite(path, with_kernel=True):
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
            INSERT INTO META_DATA_EXPORT VALUES
                ('EXPORT_PRODUCT_NAME', 'NVIDIA Nsight Systems'),
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
        """)
        if with_kernel:
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 7, 10, 20, 1, 1, 1)")


if sys.argv[1] == 'profile':
    output = Path(value_after('-o'))
    mode_arg = next((arg for arg in sys.argv if arg.startswith('--fake-mode=')), '')
    mode = mode_arg.partition('=')[2] or 'valid'
    print(json.dumps(sys.argv), flush=True)

    if mode == 'hang':
        child = subprocess.Popen([
            sys.executable,
            '-c',
            'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)',
        ])
        output.with_suffix('.child.pid').write_text(str(child.pid))
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(1)
    if mode == 'race_cancel':
        child = subprocess.Popen([
            sys.executable,
            '-c',
            'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)',
        ])
        output.with_suffix('.child.pid').write_text(str(child.pid))
        time.sleep(0.15)
        sys.exit(17)
    if mode == 'leader_exit_child':
        child = subprocess.Popen([
            sys.executable,
            '-c',
            'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)',
        ])
        output.with_suffix('.child.pid').write_text(str(child.pid))
        time.sleep(0.1)
        sys.exit(17)
    if mode == 'exit_094':
        time.sleep(0.94)
        sys.exit(17)
    if mode == 'exit_070':
        time.sleep(0.70)
        sys.exit(17)
    if mode == 'env':
        ok = (
            os.environ.get('VISIBLE_SETTING') == 'enabled'
            and os.environ.get('RUNNER_SECRET') == 'private-value'
            and os.environ.get('INHERITED_SETTING') == 'inherited'
        )
        if not ok:
            sys.exit(21)
    if mode == 'clean_env':
        ok = (
            os.environ.get('VISIBLE_SETTING') == 'enabled'
            and 'INHERITED_SETTING' not in os.environ
        )
        if not ok:
            sys.exit(22)
    if mode == 'large_logs':
        sys.stdout.write('o' * (2 * 1024 * 1024))
        sys.stderr.write('e' * (2 * 1024 * 1024))
    if mode in {'missing_report', 'nonzero_without_report'}:
        sys.exit(17 if mode == 'nonzero_without_report' else 0)

    report = output.with_suffix('.nsys-rep')
    if mode == 'empty_report':
        report.touch()
    else:
        report.write_text(mode)
    if mode == 'nonzero_with_report':
        sys.exit(19)
    sys.exit(0)

if sys.argv[1] == 'export':
    output = Path(value_after('-o'))
    report = Path(sys.argv[-1])
    mode = report.read_text()
    if mode == 'export_failed':
        print('intentional export failure', file=sys.stderr)
        sys.exit(23)
    if mode == 'invalid':
        output.write_bytes(b'not a sqlite database')
    else:
        make_sqlite(output, with_kernel=mode != 'empty_profile')
    sys.exit(0)

sys.exit(99)
'''


@pytest.fixture
def fake_nsys(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "nsys"
    executable.write_text(_FAKE_NSYS)
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return executable


def _spec(mode="valid", **overrides):
    values = {
        "argv": ("/usr/bin/true", f"--fake-mode={mode}"),
        "environment": EnvironmentSpec(),
    }
    values.update(overrides)
    return RunSpec(**values)


@pytest.fixture(autouse=True)
def sqlite_compat_ingest_policy(monkeypatch):
    """Keep runner artifact tests on the explicit SQLite compatibility path."""
    monkeypatch.setenv("NSYS_AI_INGEST", "sqlite")


def _run(tmp_path, fake_nsys, mode="valid", **spec_overrides):
    runner = LocalProfileRunner(tmp_path / "artifacts", str(fake_nsys))
    return runner.run(_spec(mode, **spec_overrides))


def test_capture_environment_guard_checks_serialized_text_not_metadata_row_count(tmp_path):
    profile = tmp_path / "capture.sqlite"
    with sqlite3.connect(profile) as conn:
        conn.executescript(
            """
            CREATE TABLE TARGET_INFO_SYSTEM_ENV (name TEXT, value TEXT);
            INSERT INTO TARGET_INFO_SYSTEM_ENV VALUES ('CpuCores', '12');
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES (1, 'NSYS_AI_TEST_SECRET=private-value');
            """
        )

    with pytest.raises(ProfileError, match="environment data"):
        _assert_capture_does_not_persist_environment(
            profile, {"NSYS_AI_TEST_SECRET": "private-value"}
        )


def test_capture_environment_guard_allows_machine_metadata(tmp_path):
    profile = tmp_path / "capture.sqlite"
    with sqlite3.connect(profile) as conn:
        conn.executescript(
            """
            CREATE TABLE TARGET_INFO_SYSTEM_ENV (name TEXT, value TEXT);
            INSERT INTO TARGET_INFO_SYSTEM_ENV VALUES
                ('CpuCores', '12'), ('GpuInfo', '{"Gpus": []}');
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES (1, 'fake_kernel');
            """
        )

    _assert_capture_does_not_persist_environment(
        profile, {"NSYS_AI_TEST_SECRET": "private-value"}
    )


def test_descriptor_uri_targets_the_platform_fd_directory(monkeypatch):
    """The guarded read reopens the vetted descriptor by identity, and the fd
    directory that exposes it is Linux-only (``/proc/self/fd``) versus
    ``/dev/fd`` on macOS/BSD. Pin both so the non-Linux branch cannot be
    silently dropped by a refactor that only ever runs on Linux CI.
    """
    import nsys_ai.profile_runner as profile_runner

    monkeypatch.setattr(profile_runner.sys, "platform", "linux")
    assert (
        profile_runner._descriptor_uri(7)
        == "file:/proc/self/fd/7?mode=ro&immutable=1"
    )
    monkeypatch.setattr(profile_runner.sys, "platform", "darwin")
    assert (
        profile_runner._descriptor_uri(7) == "file:/dev/fd/7?mode=ro&immutable=1"
    )


def test_guarded_read_opens_the_capture_on_non_linux(
    tmp_path, fake_nsys, monkeypatch
):
    """Forcing the non-Linux branch must open a real capture, not just format a
    string. ``/dev/fd`` is a symlink to ``/proc/self/fd`` on Linux, so this
    exercises the macOS descriptor path on Linux CI as well as on macOS.
    """
    import nsys_ai.profile_runner as profile_runner

    monkeypatch.setattr(profile_runner.sys, "platform", "darwin")
    result = _run(tmp_path, fake_nsys)

    assert result.status is RunStatus.SUCCEEDED
    assert result.profile is not None
    assert result.profile.kernel_count == 1


def test_success_returns_validated_reference_and_artifacts(tmp_path, fake_nsys):
    stages = []
    runner = LocalProfileRunner(tmp_path / "artifacts", str(fake_nsys))
    result = runner.run(_spec(), lambda update: stages.append(update))

    assert result.status is RunStatus.SUCCEEDED
    assert result.nsys_return_code == 0
    assert result.application_return_code is None
    assert result.profile is not None
    assert result.profile.kernel_count == 1
    assert result.profile.profile_id.startswith("nsys2:sha256:")
    assert result.profile.schema_version == "3.25.0"
    assert result.profile.product_version == "2026.2.1.106"
    assert Path(result.report_path).is_file()
    assert Path(result.sqlite_path).is_file()
    assert RunSpec.from_json_bytes(Path(result.runspec_path).read_bytes()) == _spec()
    assert [update.stage for update in stages] == [
        RunStage.PREPARING,
        RunStage.CAPTURING,
        RunStage.EXPORTING,
        RunStage.VALIDATING,
        RunStage.FINISHED,
    ]
    with pytest.raises(FrozenInstanceError):
        stages[0].stage = RunStage.FINISHED
    with pytest.raises(FrozenInstanceError):
        result.status = RunStatus.NSYS_FAILED

    artifact_dir = Path(result.runspec_path).parent
    assert stat.S_IMODE(artifact_dir.stat().st_mode) == 0o700
    for private_path in (result.runspec_path, result.stdout_path, result.stderr_path):
        assert stat.S_IMODE(Path(private_path).stat().st_mode) == 0o600


def test_runner_reuses_public_profile_reference_factory(
    tmp_path, fake_nsys, monkeypatch
):
    """The runner must not re-implement the guarded read.

    It goes through the shared reader, which pins a descriptor, asserts the
    file did not change under it, and returns the reference and the device
    count from that one open. `build_local_profile_reference` is the thin
    wrapper over the same reader for callers that need only the reference.
    """
    import nsys_ai.profile_runner as profile_runner

    calls = []
    real_reader = profile_runner.read_local_profile_under_guard

    def recording_reader(path, *, resolved_secrets=None):
        calls.append((Path(path), dict(resolved_secrets or {})))
        return real_reader(path, resolved_secrets=resolved_secrets)

    monkeypatch.setattr(
        profile_runner, "read_local_profile_under_guard", recording_reader
    )
    result = _run(tmp_path, fake_nsys)

    assert result.status is RunStatus.SUCCEEDED
    assert calls == [(Path(result.sqlite_path), {})]


def test_runner_keeps_sqlite_result_contract_under_auto_ingest(
    tmp_path, fake_nsys, monkeypatch
):
    monkeypatch.setenv("NSYS_AI_INGEST", "auto")

    result = _run(tmp_path, fake_nsys)

    assert result.status is RunStatus.SUCCEEDED
    assert result.sqlite_path is not None
    assert result.sqlite_path.endswith(".sqlite")
    assert result.profile is not None
    assert result.profile.storage_kind == "sqlite"
    assert result.profile.resolved_path == result.sqlite_path
    assert not Path(result.sqlite_path).with_suffix(".parquetdir").exists()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("missing_report", RunStatus.NSYS_FAILED),
        ("nonzero_without_report", RunStatus.NSYS_FAILED),
        ("nonzero_with_report", RunStatus.APPLICATION_FAILED),
        ("empty_report", RunStatus.INVALID_PROFILE),
        ("export_failed", RunStatus.EXPORT_FAILED),
        ("invalid", RunStatus.INVALID_PROFILE),
        ("empty_profile", RunStatus.INVALID_PROFILE),
    ],
)
def test_capture_export_and_validation_failures_are_distinct(
    tmp_path, fake_nsys, mode, expected
):
    result = _run(tmp_path, fake_nsys, mode)

    assert result.status is expected
    if mode == "nonzero_with_report":
        assert result.nsys_return_code == 19
        assert result.application_return_code is None
        assert "inferred" in result.detail


def test_launch_failure_is_returned_with_file_backed_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    result = LocalProfileRunner(
        tmp_path / "artifacts", "/definitely/missing/nsys"
    ).run(_spec())

    assert result.status is RunStatus.CAPTURE_LAUNCH_FAILED
    assert result.nsys_return_code is None
    assert Path(result.runspec_path).is_file()
    assert Path(result.stdout_path).is_file()
    assert Path(result.stderr_path).is_file()


def test_existing_artifact_directory_is_rejected_without_using_stale_report(
    tmp_path, fake_nsys
):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    stale_report = artifact_dir / "profile.nsys-rep"
    stale_report.write_text("valid")

    result = LocalProfileRunner(artifact_dir, str(fake_nsys)).run(
        _spec("nonzero_with_report")
    )

    assert result.status is RunStatus.ARTIFACT_SETUP_FAILED
    assert result.nsys_return_code is None
    assert result.runspec_path is None
    assert result.stdout_path is None
    assert result.stderr_path is None
    assert result.report_path is None
    assert result.sqlite_path is None
    assert stale_report.read_text() == "valid"


def test_already_cancelled_run_has_no_launch_or_artifact_side_effects(
    tmp_path, fake_nsys
):
    cancellation = threading.Event()
    cancellation.set()
    artifact_dir = tmp_path / "artifacts"

    result = LocalProfileRunner(artifact_dir, str(fake_nsys)).run(
        _spec(), cancellation=cancellation
    )

    assert result.status is RunStatus.CANCELLED
    assert result.nsys_return_code is None
    assert result.runspec_path is None
    assert result.stdout_path is None
    assert result.stderr_path is None
    assert result.report_path is None
    assert result.sqlite_path is None
    assert not artifact_dir.exists()


def test_relative_executable_is_resolved_before_switching_to_spec_cwd(
    tmp_path, monkeypatch
):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    executable = tools_dir / "nsys"
    executable.write_text(_FAKE_NSYS)
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    result = LocalProfileRunner(
        tmp_path / "artifacts", "./tools/nsys"
    ).run(_spec(cwd=str(workload_dir)))

    assert result.status is RunStatus.SUCCEEDED


@pytest.mark.parametrize("bad_location", ["argv", "public", "missing"])
def test_secret_preflight_has_zero_filesystem_side_effects(
    tmp_path, fake_nsys, monkeypatch, bad_location
):
    secret = "private-value"
    if bad_location != "missing":
        monkeypatch.setenv("RUNNER_SECRET", secret)
    argv = (
        ("/usr/bin/true", f"--token={secret}")
        if bad_location == "argv"
        else ("/usr/bin/true", "--fake-mode=valid")
    )
    public = {"ENDPOINT": f"https://example.test/{secret}"} if bad_location == "public" else {}
    spec = RunSpec(
        argv=argv,
        environment=EnvironmentSpec(public=public, secrets=("RUNNER_SECRET",)),
    )
    artifact_dir = tmp_path / "artifacts"

    with pytest.raises(RunSpecError) as exc_info:
        LocalProfileRunner(artifact_dir, str(fake_nsys)).run(spec)

    assert secret not in str(exc_info.value)
    assert not artifact_dir.exists()


@pytest.mark.parametrize(
    ("secret", "overrides", "location"),
    [
        ("private-commit", {"commit": "rev-private-commit"}, "commit"),
        (
            "private-repository",
            {"repository": "/work/private-repository", "cwd": "."},
            "repository",
        ),
        (
            "private-cwd",
            {"repository": "/work/repo", "cwd": "jobs/private-cwd"},
            "cwd",
        ),
        (
            "process-tree",
            {"trace_options": NsysTraceOptions(sample="process-tree")},
            "trace_options.sample",
        ),
    ],
)
def test_secret_preflight_covers_all_persisted_runspec_strings(
    tmp_path, fake_nsys, monkeypatch, secret, overrides, location
):
    monkeypatch.setenv("RUNNER_SECRET", secret)
    spec = _spec(
        environment=EnvironmentSpec(secrets=("RUNNER_SECRET",)),
        **overrides,
    )
    artifact_dir = tmp_path / "artifacts"

    with pytest.raises(RunSpecError, match=location) as exc_info:
        LocalProfileRunner(artifact_dir, str(fake_nsys)).run(spec)

    assert secret not in str(exc_info.value)
    assert not artifact_dir.exists()


def test_invalid_runner_inputs_have_zero_filesystem_side_effects(tmp_path, fake_nsys):
    artifact_dir = tmp_path / "artifacts"
    runner = LocalProfileRunner(artifact_dir, str(fake_nsys))

    with pytest.raises(RunSpecError, match="spec must be"):
        runner.run("not-a-spec")
    with pytest.raises(RunSpecError, match="progress_callback"):
        runner.run(_spec(), progress_callback="not-callable")
    with pytest.raises(RunSpecError, match="cancellation"):
        runner.run(_spec(), cancellation=object())

    assert not artifact_dir.exists()


@pytest.mark.parametrize("executable", [123, "", "bad\x00nsys"])
def test_invalid_nsys_executable_has_zero_filesystem_side_effects(
    tmp_path, executable
):
    artifact_dir = tmp_path / "artifacts"

    with pytest.raises(RunSpecError, match="nsys_executable"):
        LocalProfileRunner(artifact_dir, executable).run(_spec())

    assert not artifact_dir.exists()


@pytest.mark.parametrize(("policy", "mode"), [("inherit", "env"), ("clean", "clean_env")])
def test_environment_policy_public_values_and_declared_secrets(
    tmp_path, fake_nsys, monkeypatch, policy, mode
):
    monkeypatch.setenv("INHERITED_SETTING", "inherited")
    monkeypatch.setenv("RUNNER_SECRET", "private-value")
    environment = EnvironmentSpec(
        policy=policy,
        public={"VISIBLE_SETTING": "enabled"},
        secrets=("RUNNER_SECRET",) if policy == "inherit" else (),
    )

    result = _run(tmp_path, fake_nsys, mode, environment=environment)

    assert result.status is RunStatus.SUCCEEDED
    persisted = Path(result.runspec_path).read_text()
    logs = Path(result.stdout_path).read_text() + Path(result.stderr_path).read_text()
    assert "private-value" not in persisted
    assert "private-value" not in logs


def test_workload_argv_tokens_are_preserved_without_shell_parsing(tmp_path, fake_nsys):
    tokens = ("value with spaces", "$(not-run)", "semi;colon", "--flag=literal")
    spec = RunSpec(argv=("/usr/bin/true", "--fake-mode=valid", *tokens))

    result = LocalProfileRunner(tmp_path / "artifacts", str(fake_nsys)).run(spec)

    assert result.status is RunStatus.SUCCEEDED
    captured_argv = json.loads(Path(result.stdout_path).read_text().splitlines()[0])
    assert captured_argv[-len(tokens) :] == list(tokens)


def test_large_stdout_and_stderr_do_not_deadlock_or_buffer_in_memory(tmp_path, fake_nsys):
    result = _run(tmp_path, fake_nsys, "large_logs")

    assert result.status is RunStatus.SUCCEEDED
    assert Path(result.stdout_path).stat().st_size > 2 * 1024 * 1024
    assert Path(result.stderr_path).stat().st_size == 2 * 1024 * 1024


def test_unexpected_validation_bug_is_not_relabelled_as_invalid_profile(
    tmp_path, fake_nsys, monkeypatch
):
    def fail_identity(*args, **kwargs):
        raise RuntimeError("identity implementation bug")

    monkeypatch.setattr("nsys_ai.profile_runner.get_profile_id", fail_identity)

    with pytest.raises(RuntimeError, match="identity implementation bug"):
        _run(tmp_path, fake_nsys)


def test_progress_callback_exception_does_not_change_completed_result(
    tmp_path, fake_nsys
):
    calls = []

    def broken_observer(update):
        calls.append(update.stage)
        raise RuntimeError("observer failed")

    result = LocalProfileRunner(tmp_path / "artifacts", str(fake_nsys)).run(
        _spec(), progress_callback=broken_observer
    )

    assert result.status is RunStatus.SUCCEEDED
    assert calls[-1] is RunStage.FINISHED


def test_capturing_callback_cancellation_prevents_popen(
    tmp_path, fake_nsys, monkeypatch
):
    cancellation = threading.Event()
    popen_calls = 0

    def cancel_at_capture(update):
        if update.stage is RunStage.CAPTURING:
            cancellation.set()

    def unexpected_popen(*args, **kwargs):
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("Popen must not be called")

    monkeypatch.setattr("nsys_ai.profile_runner.subprocess.Popen", unexpected_popen)
    result = LocalProfileRunner(tmp_path / "artifacts", str(fake_nsys)).run(
        _spec(), progress_callback=cancel_at_capture, cancellation=cancellation
    )

    assert result.status is RunStatus.CANCELLED
    assert popen_calls == 0


def _process_is_running(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


@pytest.mark.parametrize("stop_kind", ["timeout", "cancel"])
def test_timeout_and_cancellation_kill_the_process_tree(
    tmp_path, fake_nsys, monkeypatch, stop_kind
):
    monkeypatch.setattr("nsys_ai.profile_runner._TERMINATION_GRACE_SECONDS", 0.15)
    cancellation = threading.Event()
    timeout = 1 if stop_kind == "timeout" else None
    if stop_kind == "cancel":
        threading.Timer(0.15, cancellation.set).start()
    runner = LocalProfileRunner(tmp_path / "artifacts", str(fake_nsys))

    result = runner.run(
        _spec("hang", timeout_seconds=timeout),
        cancellation=cancellation,
    )

    expected = RunStatus.TIMED_OUT if stop_kind == "timeout" else RunStatus.CANCELLED
    assert result.status is expected
    child_pid = int((tmp_path / "artifacts" / "profile.child.pid").read_text())
    deadline = time.monotonic() + 2
    while _process_is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(child_pid)


def test_due_cancellation_beats_leader_exit_and_kills_surviving_child(
    tmp_path, fake_nsys, monkeypatch
):
    monkeypatch.setattr("nsys_ai.profile_runner._POLL_SECONDS", 0.25)
    monkeypatch.setattr("nsys_ai.profile_runner._TERMINATION_GRACE_SECONDS", 0.1)
    cancellation = threading.Event()
    threading.Timer(0.08, cancellation.set).start()

    result = LocalProfileRunner(tmp_path / "artifacts", str(fake_nsys)).run(
        _spec("race_cancel"), cancellation=cancellation
    )

    assert result.status is RunStatus.CANCELLED
    child_pid = int((tmp_path / "artifacts" / "profile.child.pid").read_text())
    deadline = time.monotonic() + 2
    while _process_is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(child_pid)


def test_leader_exit_always_cleans_surviving_process_group(
    tmp_path, fake_nsys, monkeypatch
):
    monkeypatch.setattr("nsys_ai.profile_runner._TERMINATION_GRACE_SECONDS", 0.1)

    result = _run(tmp_path, fake_nsys, "leader_exit_child")

    assert result.status is RunStatus.NSYS_FAILED
    assert result.nsys_return_code == 17
    child_pid = int((tmp_path / "artifacts" / "profile.child.pid").read_text())
    deadline = time.monotonic() + 2
    while _process_is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(child_pid)


@pytest.mark.parametrize(
    ("mode", "poll_seconds"),
    [("exit_094", 2.0), ("exit_070", 5.0)],
)
@pytest.mark.no_cover
def test_completion_before_deadline_wins_over_coarse_polling(
    tmp_path, fake_nsys, monkeypatch, mode, poll_seconds
):
    # Coverage tracing is deliberately excluded here. This test verifies a
    # sub-second process/deadline race, and tracing both this test and the fake
    # nsys child changes the scheduling it is meant to measure. Keep the
    # correctness check and the coverage report honest instead of making a
    # timing assertion depend on instrumentation overhead.
    monkeypatch.delenv("COVERAGE_PROCESS_START", raising=False)
    monkeypatch.delenv("COVERAGE_SOURCE", raising=False)
    monkeypatch.setattr("nsys_ai.profile_runner._POLL_SECONDS", poll_seconds)

    result = _run(tmp_path, fake_nsys, mode, timeout_seconds=1)

    assert result.status is RunStatus.NSYS_FAILED
    assert result.nsys_return_code == 17


def test_true_timeout_with_coarse_polling_is_still_timed_out(
    tmp_path, fake_nsys, monkeypatch
):
    monkeypatch.setattr("nsys_ai.profile_runner._POLL_SECONDS", 5.0)
    monkeypatch.setattr("nsys_ai.profile_runner._TERMINATION_GRACE_SECONDS", 0.1)

    result = _run(tmp_path, fake_nsys, "hang", timeout_seconds=1)

    assert result.status is RunStatus.TIMED_OUT


def test_cancellation_callable_failure_propagates_after_process_tree_cleanup(
    tmp_path, fake_nsys, monkeypatch
):
    monkeypatch.setattr("nsys_ai.profile_runner._TERMINATION_GRACE_SECONDS", 0.1)
    pid_path = tmp_path / "artifacts" / "profile.child.pid"

    def broken_cancellation():
        if pid_path.exists():
            raise OSError("cancellation source failed")
        return False

    with pytest.raises(OSError, match="cancellation source failed"):
        LocalProfileRunner(tmp_path / "artifacts", str(fake_nsys)).run(
            _spec("hang"), cancellation=broken_cancellation
        )

    child_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2
    while _process_is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(child_pid)


def test_keyboard_interrupt_from_cancellation_cleans_process_tree(
    tmp_path, fake_nsys, monkeypatch
):
    monkeypatch.setattr("nsys_ai.profile_runner._TERMINATION_GRACE_SECONDS", 0.1)
    pid_path = tmp_path / "artifacts" / "profile.child.pid"

    def interrupted_cancellation():
        if pid_path.exists():
            raise KeyboardInterrupt
        return False

    with pytest.raises(KeyboardInterrupt):
        LocalProfileRunner(tmp_path / "artifacts", str(fake_nsys)).run(
            _spec("hang"), cancellation=interrupted_cancellation
        )

    child_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2
    while _process_is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(child_pid)


def test_progress_model_is_frozen():
    progress = RunProgress(RunStage.CAPTURING, 1.0)
    with pytest.raises(FrozenInstanceError):
        progress.elapsed_seconds = 2.0


def test_declared_gpu_count_the_capture_contradicts_fails_validation(tmp_path, fake_nsys):
    """expected_gpu_count was recorded into the RunSpec and never read.

    A one-GPU capture declared as eight used to be published with the
    declaration intact, so every consumer downstream inherited a falsehood the
    artifact asserted about itself.
    """
    result = _run(tmp_path, fake_nsys, expected_gpu_count=8)

    assert result.status is RunStatus.INVALID_PROFILE
    assert result.detail == (
        "declared expected_gpu_count=8 but the capture recorded kernels on 1 GPU(s)"
    )
    # The artifacts still exist, so the mismatch can be inspected.
    assert Path(result.sqlite_path).is_file()


def test_declared_gpu_count_the_capture_meets_succeeds(tmp_path, fake_nsys):
    """The check must not fail a capture that matches its declaration."""
    result = _run(tmp_path, fake_nsys, expected_gpu_count=1)

    assert result.status is RunStatus.SUCCEEDED, result.detail
    assert result.profile is not None


def test_an_unreadable_capture_fails_the_run_rather_than_the_declaration(
    tmp_path, fake_nsys, monkeypatch
):
    """"Could not be checked" must never reach the success path.

    The guarded read either yields a count or raises, so the fail-closed rule
    lives here rather than in a "count is unknown" branch of the comparison,
    which no caller could reach. A capture that cannot be read is invalid
    whether or not a declaration was made.
    """
    from nsys_ai import profile_runner

    def _explode(*_args, **_kwargs):
        raise sqlite3.OperationalError("database disk image is malformed")

    monkeypatch.setattr(profile_runner, "read_local_profile_under_guard", _explode)
    result = _run(tmp_path, fake_nsys, expected_gpu_count=8)

    assert result.status is RunStatus.INVALID_PROFILE
    assert result.detail == "profile validation failed: OperationalError"
    assert result.profile is None


def test_the_gpu_count_is_read_from_the_guarded_handle(tmp_path, fake_nsys):
    """A machine can expose eight GPUs and a run can touch one.

    The count comes back from the same guarded open that builds the reference.
    It used to be a second `sqlite3.connect` by path, made after the pinned
    descriptor was closed -- so the number validation acted on was read from
    whatever sat at that path by then, outside the swap guard, at the cost of a
    second full `GROUP BY deviceId` scan.
    """
    from nsys_ai.profile_runner import read_local_profile_under_guard

    result = _run(tmp_path, fake_nsys)
    assert result.status is RunStatus.SUCCEEDED
    reference, observed = read_local_profile_under_guard(result.sqlite_path)
    assert observed == 1
    assert reference.kernel_count > 0


def test_the_capture_is_opened_once_for_reference_and_gpu_count(tmp_path, fake_nsys, monkeypatch):
    """One guarded open, not two: the second was by path and unguarded."""
    import sqlite3 as _sqlite3

    from nsys_ai import profile_runner

    result = _run(tmp_path, fake_nsys)
    assert result.status is RunStatus.SUCCEEDED

    opened: list[str] = []
    real_connect = _sqlite3.connect

    def _record(target, *args, **kwargs):
        opened.append(str(target))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(profile_runner.sqlite3, "connect", _record)
    profile_runner.read_local_profile_under_guard(result.sqlite_path)

    # The guard reopens the vetted descriptor through the platform's per-process
    # fd directory: /proc/self/fd on Linux, /dev/fd on macOS/BSD. Anything else
    # is a by-path reopen outside the swap guard.
    guarded_prefixes = ("file:/proc/self/fd/", "file:/dev/fd/")
    by_path = [target for target in opened if not target.startswith(guarded_prefixes)]
    assert not by_path, f"the capture was reopened outside the descriptor guard: {by_path}"


def test_teardown_survives_a_group_it_may_not_signal(monkeypatch):
    """A best-effort teardown must not raise out of a finished capture.

    Every ``killpg`` in ``_terminate_process_group`` caught ``ProcessLookupError``
    and nothing else, so an ``EPERM`` propagated out of ``_capture_process`` and
    out of ``run()`` — turning a capture that had already completed into a raised
    ``PermissionError``. Observed on macOS; the caller cannot act on the
    difference between "the group is gone" and "the group is no longer ours",
    and neither is a reason to fail a capture.
    """
    import nsys_ai.profile_runner as profile_runner


    class _FinishedProcess:
        pid = 4242
        waited = False

        def poll(self):
            return 0

        def wait(self, timeout=None):
            type(self).waited = True
            return 0

    def _refuse(_pgid, _sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(profile_runner.os, "killpg", _refuse)

    profile_runner.LocalProfileRunner._terminate_process_group(_FinishedProcess())

    assert _FinishedProcess.waited, "the child was never reaped"


def test_teardown_reports_a_live_tree_it_cannot_stop(monkeypatch):
    """EPERM means the opposite thing while our child is still running.

    The teardown runs from five call sites, and four of them exist to stop the
    tree: timeout, cancellation, the leader-exit sweep, and the BaseException
    handler. If the group refuses us while the child is alive, we have failed to
    do the one thing those callers asked for. Reporting TIMED_OUT or CANCELLED
    over a profiled process that is still running would be a worse answer than
    the failure, so it propagates -- as it did before EPERM was handled at all.
    """
    import nsys_ai.profile_runner as profile_runner

    class _LiveProcess:
        pid = 4244

        def poll(self):
            return None  # still running

        def wait(self, timeout=None):  # pragma: no cover - must not be reached
            raise AssertionError("a live tree must not be reaped as if it stopped")

    def _refuse(_pgid, _sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(profile_runner.os, "killpg", _refuse)

    with pytest.raises(PermissionError):
        profile_runner.LocalProfileRunner._terminate_process_group(_LiveProcess())


def test_teardown_still_stops_at_a_group_that_has_gone(monkeypatch):
    """The pre-existing lookup path keeps its behaviour."""
    import nsys_ai.profile_runner as profile_runner


    class _GoneProcess:
        pid = 4243
        waited = False

        def poll(self):
            return 0

        def wait(self, timeout=None):
            type(self).waited = True
            return 0

    def _gone(_pgid, _sig):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(profile_runner.os, "killpg", _gone)

    profile_runner.LocalProfileRunner._terminate_process_group(_GoneProcess())

    assert _GoneProcess.waited
