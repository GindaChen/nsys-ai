"""A skill that refuses must say so, not crash the formatter it hands the row to.

``Skill.execute`` has three documented outcomes -- no rows, an abstention row, or
data -- and an undocumented fourth: a single ``{"error": ...}`` row returned when
the skill cannot answer for the arguments it was given. Asking for a device the
profile does not have produces one, and it is a good error, carrying
``available_devices`` and a ``hint`` naming the devices that do exist.

That row has none of the data columns. Formatters that index them died on it:
eight of the thirty-seven builtins raised ``KeyError`` rather than printing the
hint written for exactly this case, and ``arithmetic_intensity`` -- the skill
usually cited as the one that handles it well -- was among them. The remaining
twenty-nine were correct by luck, not by contract.

``format_rows`` already renders abstention centrally for the same reason. These
tests pin the error row to the same place.
"""

import pytest

from nsys_ai.skills.base import abstain, is_error_row
from nsys_ai.skills.registry import get_skill, list_skills

#: The row the query layer produces for a device the profile does not have.
DEVICE_ERROR_ROW = {
    "error": "no kernels found",
    "requested_device": 9,
    "available_devices": {"0": 1721, "1": 1721},
    "hint": "Device 9 is not present in this profile. Available devices: 0, 1.",
}


def _skill_names():
    return sorted(list_skills())


def test_there_are_skills_to_check():
    """Guard the guard: an empty registry would make the sweep below vacuous."""
    assert len(_skill_names()) > 10


@pytest.mark.parametrize("name", _skill_names())
def test_every_skill_formats_an_error_row_without_raising(name):
    """The sweep that found the eight. It is the contract, so it covers all of them."""
    text = get_skill(name).format_rows([dict(DEVICE_ERROR_ROW)])

    assert isinstance(text, str) and text.strip()
    assert "no kernels found" in text


@pytest.mark.parametrize("name", ["gpu_idle_gaps", "arithmetic_intensity", "nccl_breakdown"])
def test_the_hint_reaches_the_reader(name):
    """The hint was always written; while the formatters crashed, nobody saw it."""
    text = get_skill(name).format_rows([dict(DEVICE_ERROR_ROW)])

    assert "Device 9 is not present in this profile" in text
    assert "Available devices: 0, 1" in text


def test_an_error_row_without_a_hint_still_renders():
    """``hint`` is optional -- most refusals carry only ``error``."""
    text = get_skill("gpu_idle_gaps").format_rows([{"error": "kernels table is absent"}])

    assert "kernels table is absent" in text


class _Recorder:
    """Stands in for a format_fn so the tests can see whether it was reached."""

    def __init__(self):
        self.called_with = None

    def __call__(self, rows):
        self.called_with = rows
        return "formatted"


def _skill_with(format_fn):
    skill = get_skill("gpu_idle_gaps")
    import dataclasses

    return dataclasses.replace(skill, format_fn=format_fn)


def test_results_carrying_an_error_column_still_reach_the_formatter():
    """The predicate is narrow on purpose.

    A skill's real results may legitimately carry a per-row ``error`` column, and
    collapsing those to one rendered message would throw the results away. Only a
    lone row qualifies.
    """
    rows = [{"error": None, "gap_ns": 1}, {"error": "partial", "gap_ns": 2}]
    recorder = _Recorder()

    assert _skill_with(recorder).format_rows(rows) == "formatted"
    assert recorder.called_with == rows


def test_a_single_data_row_is_not_mistaken_for_an_error():
    """One row with no ``error`` key is data, and goes where data goes."""
    recorder = _Recorder()

    assert _skill_with(recorder).format_rows([{"gap_ns": 1}]) == "formatted"
    assert recorder.called_with == [{"gap_ns": 1}]


def test_abstention_still_wins_over_the_error_branch():
    """Abstention has its own rendering and its own meaning; it is checked first."""
    text = get_skill("gpu_idle_gaps").format_rows(abstain("this profile has no NVTX"))

    assert "not applicable to this profile" in text
    assert "this profile has no NVTX" in text


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([{"error": "x"}], True),
        ([{"error": "x", "hint": "y"}], True),
        ([], False),
        (None, False),
        ([{"gap_ns": 1}], False),
        ([{"error": "x"}, {"error": "y"}], False),
        (abstain("cannot run"), False),
    ],
)
def test_is_error_row_predicate(rows, expected):
    assert is_error_row(rows) is expected


def test_a_skill_that_formats_its_own_error_row_keeps_doing_so():
    """The central rendering is a fallback, not an interception.

    ``code_attribution_candidates`` returns an error row carrying ``limitations``
    and its formatter spells them out. A generic message would be a downgrade, so
    the skill's own output is kept when it mentions the reason.
    """
    row = {
        "error": "No overlapping NVTX ranges found for the selected window",
        "selection": {"start_ns": 0, "end_ns": 1, "device_id": 0},
        "limitations": ["NVTX ranges are a proxy for source location"],
    }

    text = get_skill("code_attribution_candidates").format_rows([row])

    assert "No overlapping NVTX ranges found" in text
    assert "Limitations:" in text
    assert "NVTX ranges are a proxy for source location" in text


def test_tensor_core_usage_keeps_its_own_explanation():
    """The other formatter with a genuinely better message than a generic one."""
    row = {"error": "Tensor Core analysis requires a database connection that exposes ..."}

    text = get_skill("tensor_core_usage").format_rows([row])

    assert "requires a database connection" in text


@pytest.mark.parametrize("name", ["critical_path", "profile_health_manifest"])
def test_a_formatter_that_renders_defaults_over_an_error_does_not_get_to(name):
    """Not raising is not the same as handling it.

    These two render an error row as an empty report -- ``Bound class: None`` over
    a ``0.0ms`` critical path, a manifest with ``?`` for the GPU. That reads as a
    clean bill of health for a question that was never answered, which is worse
    than the crash this change replaced, so the reason wins.
    """
    text = get_skill(name).format_rows([dict(DEVICE_ERROR_ROW)])

    assert "no kernels found" in text
    assert "Device 9 is not present in this profile" in text
    assert "0.0ms" not in text
