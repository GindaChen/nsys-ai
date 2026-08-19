from types import SimpleNamespace

import pytest

from nsys_ai.cli.parsers import _build_legacy_parser, _build_parser
from nsys_ai.exceptions import UsageError
from nsys_ai.profile import select_gpu_device


@pytest.mark.parametrize(
    ("devices", "requested", "expected"),
    [([1, 2, 3], None, 1), ([0, 1], 0, 0), ([7], 7, 7), ([], None, 0)],
)
def test_select_gpu_device_preserves_explicit_zero_and_uses_first_present(
    devices, requested, expected
):
    profile = SimpleNamespace(meta=SimpleNamespace(devices=devices))

    assert select_gpu_device(profile, requested) == expected


def test_select_gpu_device_rejects_absent_explicit_device():
    profile = SimpleNamespace(meta=SimpleNamespace(devices=[1, 2, 3]))

    with pytest.raises(UsageError, match=r"GPU device 0.*available devices: 1, 2, 3"):
        select_gpu_device(profile, 0)


def test_cli_defaults_for_profile_aware_commands_are_unset():
    parser = _build_parser()
    profile = "profile.sqlite"

    diagnose = parser.parse_args(["diagnose", profile])
    evidence = parser.parse_args(["evidence", "build", profile])
    optimize = parser.parse_args(["optimize", "--repo", ".", profile, "--", "true"])
    legacy_parser = _build_legacy_parser()
    analyze = legacy_parser.parse_args(["analyze", profile, "--format", "json"])

    assert diagnose.gpu is None
    assert evidence.gpu is None
    assert analyze.gpu is None
    assert optimize.gpu is None
