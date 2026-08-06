"""Correctness and complexity guards for iteration/runtime correlation."""

import pytest

from nsys_ai.overlap import _correlate_iteration_kernels, detect_iterations
from nsys_ai.profile import Profile


def _kernel(correlation_id):
    return {
        "correlation_id": correlation_id,
        "name": f"kernel_{correlation_id}",
        "start": correlation_id * 100,
        "end": correlation_id * 100 + 10,
    }


def _runtime(correlation_id, start, end):
    return {"correlationId": correlation_id, "start": start, "end": end}


def test_zero_iterations_produce_no_groups():
    assert list(_correlate_iteration_kernels([], [], {})) == []


def test_single_full_capture_iteration_preserves_whole_row_containment():
    kernels = {i: _kernel(i) for i in range(1, 5)}
    rows = [
        _runtime(1, 0, 1),
        _runtime(2, 2, 3),
        _runtime(3, 9, 10),
        _runtime(4, 9, 11),
    ]

    groups = list(
        _correlate_iteration_kernels([{"start": 0, "end": 10}], rows, kernels)
    )

    assert [[kernel["correlation_id"] for kernel in group] for group in groups] == [
        [1, 2, 3]
    ]


def test_multiple_iterations_retain_the_first_row_for_the_next_window():
    kernels = {i: _kernel(i) for i in range(1, 5)}
    rows = [
        _runtime(1, 1, 2),
        _runtime(2, 10, 11),
        _runtime(3, 12, 19),
        _runtime(4, 20, 21),
    ]
    iterations = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]

    groups = list(_correlate_iteration_kernels(iterations, rows, kernels))

    assert [[kernel["correlation_id"] for kernel in group] for group in groups] == [
        [1],
        [2, 3],
    ]


def test_runtime_rows_are_iterated_once_without_rewind():
    class OneShotRows:
        def __init__(self, rows):
            self.rows = rows
            self.iterations = 0
            self.yields = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("runtime rows were restarted")
            for row in self.rows:
                self.yields += 1
                yield row

    rows = OneShotRows([_runtime(i, i * 10, i * 10 + 1) for i in range(1, 5)])
    kernels = {i: _kernel(i) for i in range(1, 5)}
    iterations = [
        {"start": 0, "end": 15},
        {"start": 20, "end": 25},
        {"start": 30, "end": 35},
        {"start": 40, "end": 45},
    ]

    groups = list(_correlate_iteration_kernels(iterations, rows, kernels))

    assert rows.iterations == 1
    assert rows.yields == len(rows.rows)
    assert sum(len(group) for group in groups) == 4


@pytest.mark.parametrize(
    "iterations",
    [
        [{"start": 10, "end": 10}],
        [{"start": 10, "end": 20}, {"start": 19, "end": 30}],
        [{"start": 20, "end": 30}, {"start": 10, "end": 15}],
    ],
)
def test_correlation_rejects_invalid_iteration_windows(iterations):
    with pytest.raises(ValueError):
        list(_correlate_iteration_kernels(iterations, [], {}))


def test_detect_iterations_correlates_multiple_nvtx_windows(minimal_nsys_conn):
    minimal_nsys_conn.execute(
        "INSERT INTO NVTX_EVENTS (globalTid, start, end, text, eventType, rangeId) "
        "VALUES (100, 7500000, 8500000, 'train_step', 59, 2)"
    )
    prof = Profile._from_conn(minimal_nsys_conn)

    rows = detect_iterations(prof, 0, marker="train_step")

    assert [row["kernel_count"] for row in rows] == [4, 1]
    assert all(row["heuristic"] is False for row in rows)


def test_heuristic_windows_follow_cpu_order_across_gpu_stream_reordering(
    minimal_nsys_conn,
):
    minimal_nsys_conn.execute("DELETE FROM NVTX_EVENTS")
    minimal_nsys_conn.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME")
    minimal_nsys_conn.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
    minimal_nsys_conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
        "(globalTid, correlationId, start, end, nameId) VALUES (100, ?, ?, ?, 24)",
        [(1, 100, 110), (2, 300, 310), (3, 200, 210)],
    )
    minimal_nsys_conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "(globalPid, deviceId, streamId, correlationId, start, end, shortName, demangledName) "
        "VALUES (100, 0, ?, ?, ?, ?, 1, 1)",
        [
            (7, 1, 1_000, 2_000),
            (8, 2, 3_000_000, 3_001_000),
            (7, 3, 6_000_000, 6_001_000),
        ],
    )
    prof = Profile._from_conn(minimal_nsys_conn)

    rows = detect_iterations(prof, 0)

    assert [row["gpu_start_ns"] for row in rows] == [1_000, 6_000_000, 3_000_000]
    assert [row["text"] for row in rows] == [
        "heuristic_step_0",
        "heuristic_step_1",
        "heuristic_step_2",
    ]
    assert all(row["heuristic"] is True for row in rows)
