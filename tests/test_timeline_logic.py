"""
tests/test_timeline_logic.py — Unit tests for timeline/logic.py.

All pure-function tests: no Textual, no curses, no display required.
Run with: pytest tests/test_timeline_logic.py -v
"""
import pytest

from nsys_ai.timeline.logic import (
    build_stream_kernels,
    center_viewport,
    collect_streams,
    extract_events,
    filter_kernels,
    find_kernel_by_name,
    kernel_at_time,
    kernel_index_at_time,
    nice_tick_interval,
    time_bounds,
    zoom_ns_per_col,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_JSON = [
    {
        "name": "forward",
        "type": "nvtx",
        "duration_ms": 100.0,
        "heat": 0.5,
        "stream": "0",
        "relative_pct": 100,
        "path": "",
        "demangled": "",
        "start_ns": 0,
        "end_ns": 100_000_000,
        "children": [
            {
                "name": "aten::mm",
                "type": "kernel",
                "duration_ms": 30.0,
                "heat": 0.9,
                "stream": "1",
                "relative_pct": 30,
                "path": "forward",
                "demangled": "at::native::matmul",
                "start_ns": 10_000_000,
                "end_ns": 40_000_000,
                "children": [],
            },
            {
                "name": "nccl_allreduce",
                "type": "kernel",
                "duration_ms": 20.0,
                "heat": 0.2,
                "stream": "2",
                "relative_pct": 20,
                "path": "forward",
                "demangled": "",
                "start_ns": 50_000_000,
                "end_ns": 70_000_000,
                "children": [],
            },
        ],
    }
]


@pytest.fixture
def sample_kernels():
    kernels, _ = extract_events(SAMPLE_JSON)
    return kernels


@pytest.fixture
def sample_stream_kernels(sample_kernels):
    streams = collect_streams(sample_kernels)
    return streams, build_stream_kernels(sample_kernels, streams)


# ---------------------------------------------------------------------------
# extract_events
# ---------------------------------------------------------------------------

def test_extract_events_kernels(sample_kernels):
    assert len(sample_kernels) == 2
    names = [k.name for k in sample_kernels]
    assert "aten::mm" in names
    assert "nccl_allreduce" in names


def test_extract_events_sorted_by_start(sample_kernels):
    starts = [k.start_ns for k in sample_kernels]
    assert starts == sorted(starts)


def test_extract_events_nvtx_spans():
    kernels, spans = extract_events(SAMPLE_JSON)
    assert len(spans) == 1
    assert spans[0].name == "forward"


# ---------------------------------------------------------------------------
# collect_streams / build_stream_kernels
# ---------------------------------------------------------------------------

def test_collect_streams(sample_kernels):
    streams = collect_streams(sample_kernels)
    assert "1" in streams
    assert "2" in streams


def test_build_stream_kernels_correct_partitioning(sample_kernels):
    streams = collect_streams(sample_kernels)
    sk = build_stream_kernels(sample_kernels, streams)
    assert all(k.stream == "1" for k in sk.get("1", []))
    assert all(k.stream == "2" for k in sk.get("2", []))


# ---------------------------------------------------------------------------
# filter_kernels
# ---------------------------------------------------------------------------

def test_filter_kernels_by_name(sample_kernels):
    result = filter_kernels(sample_kernels, filter_text="nccl")
    assert len(result) == 1
    assert result[0].name == "nccl_allreduce"


def test_filter_kernels_by_min_dur(sample_kernels):
    result = filter_kernels(sample_kernels, min_dur_us=25_000)  # 25ms threshold
    assert len(result) == 1
    assert result[0].name == "aten::mm"  # only 30ms passes


def test_filter_kernels_demangled(sample_kernels):
    result = filter_kernels(sample_kernels, filter_text="matmul")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# kernel_at_time
# ---------------------------------------------------------------------------

def test_kernel_at_time_inside(sample_kernels):
    # aten::mm is 10ms-40ms on stream 1
    result = kernel_at_time(sample_kernels, 25_000_000)  # 25ms
    assert result is not None
    assert result.name == "aten::mm"


def test_kernel_at_time_nearest(sample_kernels):
    # At 5ms (before any kernel), should return nearest
    result = kernel_at_time(sample_kernels, 5_000_000)
    assert result is not None  # finds closest


def test_kernel_at_time_empty():
    assert kernel_at_time([], 0) is None


# ---------------------------------------------------------------------------
# kernel_index_at_time
# ---------------------------------------------------------------------------

def test_kernel_index_at_time_returns_valid(sample_kernels):
    idx = kernel_index_at_time(sample_kernels, 25_000_000)
    assert 0 <= idx < len(sample_kernels)


def test_kernel_index_at_time_empty():
    assert kernel_index_at_time([], 0) == -1


# ---------------------------------------------------------------------------
# find_kernel_by_name
# ---------------------------------------------------------------------------

def test_find_kernel_by_name_found(sample_stream_kernels):
    streams, sk = sample_stream_kernels
    result = find_kernel_by_name(sk, "aten::mm", 1)
    assert result is not None
    stream, idx = result
    assert sk[stream][idx].name == "aten::mm"


def test_find_kernel_by_name_not_found(sample_stream_kernels):
    _, sk = sample_stream_kernels
    result = find_kernel_by_name(sk, "nonexistent", 1)
    assert result is None


# ---------------------------------------------------------------------------
# Viewport math
# ---------------------------------------------------------------------------

def test_center_viewport():
    # cursor at 500, 10ns/col, 100 cols → viewport_start = 500 - 500 = 0
    vp = center_viewport(500, 10, 100)
    assert vp == 0  # 500 - 10*100//2 = 500-500 = 0


def test_center_viewport_offset():
    vp = center_viewport(1000, 5, 100)
    assert vp == 750  # 1000 - 5*100//2 = 750


def test_nice_tick_interval_reasonable():
    # 1000 cols, 1_000_000 ns/col → total 1e9 ns, expect tick > 1
    interval = nice_tick_interval(1000, 1_000_000)
    assert interval > 0
    # Should not produce more than ~20 ticks
    n_ticks = (1_000_000 * 1000) // interval
    assert n_ticks <= 20


def test_zoom_ns_per_col_in():
    result = zoom_ns_per_col(1000, -1, 1_000_000)
    assert result < 1000


def test_zoom_ns_per_col_out():
    result = zoom_ns_per_col(1000, +1, 1_000_000)
    assert result > 1000


def test_zoom_clamped_to_min():
    result = zoom_ns_per_col(1, -1, 1_000_000)
    assert result >= 1


def test_zoom_clamped_to_time_span():
    result = zoom_ns_per_col(900_000, +1, 1_000_000)
    assert result <= 1_000_000


# ---------------------------------------------------------------------------
# time_bounds
# ---------------------------------------------------------------------------

def test_time_bounds_from_kernels(sample_kernels):
    start, end = time_bounds(sample_kernels, (0, 0))
    assert start == 10_000_000   # aten::mm start
    assert end == 70_000_000     # nccl end


def test_time_bounds_fallback():
    start, end = time_bounds([], (100, 200))
    assert start == 100
    assert end == 200
