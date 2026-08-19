from __future__ import annotations


def test_explicit_absent_device_has_shared_diagnostic_shape(minimal_nsys_conn):
    from nsys_ai.skills.registry import all_skills

    device_skills = [
        skill
        for skill in all_skills()
        if any(param.name in {"device", "device_id"} for param in skill.params)
    ]

    assert len(device_skills) >= 17
    for skill in device_skills:
        parameter = next(
            param.name for param in skill.params if param.name in {"device", "device_id"}
        )
        rows = skill.execute(minimal_nsys_conn, **{parameter: 99})
        assert len(rows) == 1, skill.name
        row = rows[0]
        assert row["error"] == "no kernels found", skill.name
        assert row["requested_device"] == 99, skill.name
        assert row["available_devices"] == {0: 5}, skill.name
        assert "Try:" in row["hint"], skill.name


def test_root_cause_pivot_reports_the_analysed_device(minimal_nsys_conn):
    minimal_nsys_conn.execute(
        "INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (1, 1, 101, '', 108)"
    )
    minimal_nsys_conn.commit()

    from nsys_ai.skills.registry import get_skill

    rows = get_skill("root_cause_matcher").execute(minimal_nsys_conn, device=1)

    assert rows
    assert all(row["analysed_device"] == 0 for row in rows)
