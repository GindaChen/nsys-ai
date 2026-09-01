"""GPU utilization is a fraction of wall time, so it cannot exceed 100%.

It used to. ``compute_ms`` is a sum over kernels, and two kernels resident at the
same instant on different streams each contribute their full duration to it -- so
on any capture where compute and communication overlap, the sum outruns the wall
span it was being divided by. A Megatron capture reported ``Util: 112.5%``, and
the narrative summary repeated the number in prose.

The union was already in the same function: the idle sweep advances a watermark
over sorted kernels and only counts a gap when the next kernel starts after it, so
``span - idle`` is the time at least one kernel was resident. These tests pin the
identity that makes the difference visible -- ``span_ms == busy_ms + idle_ms`` --
because it is what ``compute_ms`` cannot satisfy.
"""

from nsys_ai.summary import gpu_summary


class _Meta:
    def __init__(self):
        self.gpu_info = {}


class _StubProfile:
    """The two attributes ``gpu_summary`` reads, and nothing else."""

    def __init__(self, kernels):
        self._kernels = kernels
        self.meta = _Meta()
        self.path = "stub.sqlite"

    def kernels(self, device, trim=None):
        return self._kernels


def _kernel(name, stream, start_ms, end_ms):
    return {
        "name": name,
        "streamId": stream,
        "start": int(start_ms * 1e6),
        "end": int(end_ms * 1e6),
    }


def test_fully_overlapped_streams_do_not_exceed_one_hundred_percent():
    """Two streams busy for the same 100ms: the device is 100% busy, not 200%."""
    prof = _StubProfile(
        [
            _kernel("gemm", 7, 0, 100),
            _kernel("nccl_allreduce", 15, 0, 100),
        ]
    )

    timing = gpu_summary(prof, device=0)["timing"]

    assert timing["utilization_pct"] == 100.0
    assert timing["busy_ms"] == 100.0
    assert timing["idle_ms"] == 0.0
    # The sum is still reported, still means what it always meant, and is exactly
    # the quantity that would have produced 200%.
    assert timing["compute_ms"] == 200.0


def test_partial_overlap_is_measured_against_the_union_not_the_sum():
    """60ms of union work in a 100ms span is 60%, whatever the per-stream sum says."""
    prof = _StubProfile(
        [
            _kernel("gemm", 7, 0, 50),
            _kernel("nccl_allreduce", 15, 10, 60),
            _kernel("gemm", 7, 90, 100),
        ]
    )

    timing = gpu_summary(prof, device=0)["timing"]

    assert timing["span_ms"] == 100.0
    assert timing["busy_ms"] == 70.0  # [0,60) and [90,100)
    assert timing["idle_ms"] == 30.0  # [60,90)
    assert timing["utilization_pct"] == 70.0
    assert timing["compute_ms"] == 110.0  # 50 + 50 + 10, overlap counted twice


def test_span_reconciles_with_busy_and_idle():
    """The identity a reader checks by eye, on a shape with several overlaps."""
    prof = _StubProfile(
        [
            _kernel("a", 7, 0, 30),
            _kernel("b", 15, 20, 45),
            _kernel("c", 7, 70, 90),
            _kernel("d", 23, 80, 120),
        ]
    )

    timing = gpu_summary(prof, device=0)["timing"]

    assert timing["span_ms"] == timing["busy_ms"] + timing["idle_ms"]
    assert 0.0 <= timing["utilization_pct"] <= 100.0


def test_a_single_stream_is_unaffected():
    """No overlap, so sum and union agree -- the old number was right here."""
    prof = _StubProfile([_kernel("a", 7, 0, 40), _kernel("b", 7, 60, 100)])

    timing = gpu_summary(prof, device=0)["timing"]

    assert timing["compute_ms"] == timing["busy_ms"] == 80.0
    assert timing["utilization_pct"] == 80.0
