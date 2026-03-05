import json
import sqlite3

from nsys_ai.profile import Profile
from nsys_ai.viewer import build_timeline_gpu_data, generate_timeline_data_json


def test_timeline_web_kernel_first_keeps_kernels_outside_nvtx(minimal_nsys_db_path):
    conn = sqlite3.connect(minimal_nsys_db_path)
    conn.execute("INSERT INTO StringIds(id, value) VALUES (?, ?)", (3, "kernel_C"))
    conn.execute(
        """
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
        (globalPid, deviceId, streamId, correlationId, start, end, shortName, demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (100, 0, 9, 3, 4_600_000, 4_800_000, 3, 3, 1, 1, 1, 1, 1, 1),
    )
    conn.commit()
    conn.close()

    with Profile(minimal_nsys_db_path) as prof:
        data = build_timeline_gpu_data(prof, 0, (0, 5_000_000))
        gpu0 = data[0]
        kernel_names = {k["name"] for k in gpu0["kernels"]}

        assert "kernel_C" in kernel_names
        assert len(gpu0["kernels"]) == 3

        k_c = next(k for k in gpu0["kernels"] if k["name"] == "kernel_C")
        assert k_c["path"] == "kernel_C"


def test_timeline_web_trim_uses_overlap_not_containment(minimal_nsys_db_path):
    with Profile(minimal_nsys_db_path) as prof:
        gpu_data = build_timeline_gpu_data(prof, 0, (1_500_000, 1_600_000))
        kernels = gpu_data[0]["kernels"]
        names = [k["name"] for k in kernels]

        # kernel_A spans 1.0ms-2.0ms and must be included by overlap logic.
        assert names == ["kernel_A"]

        payload = json.loads(generate_timeline_data_json(prof, [0], (1_500_000, 1_600_000)))
        assert "gpus" in payload
        assert payload["gpus"][0]["kernels"][0]["name"] == "kernel_A"
