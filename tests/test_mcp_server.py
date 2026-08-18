"""Contracts for the optional read-only MCP transport."""

import hashlib
import json
import shutil
import sqlite3
import sys
import types
from pathlib import Path


def test_skill_catalog_matches_registry():
    from nsys_ai.mcp_server import _list_skills
    from nsys_ai.skills.registry import all_skills

    payload = _list_skills()

    assert payload["producer"] == "nsys-ai"
    assert [skill["name"] for skill in payload["skills"]] == [
        skill.name for skill in all_skills()
    ]
    assert {"name", "title", "description", "category", "params"} <= set(
        payload["skills"][0]
    )


def test_run_skill_returns_rows_and_canonical_findings(profile_copy):
    from nsys_ai.mcp_server import _run_skill

    profile = profile_copy("h100_2gpu_1s.sqlite")
    result = _run_skill(str(profile), "top_kernels", {"limit": 1}, max_rows=1)

    assert result["producer"] == "nsys-ai"
    assert result["profile"]["path"] == str(profile)
    assert result["rows"]
    assert isinstance(result["findings"], list)
    assert result["abstained"] is False
    json.dumps(result)


def test_run_skill_rejects_undeclared_parameters_before_opening_profile():
    from nsys_ai.mcp_server import _run_skill

    result = _run_skill("does-not-exist.sqlite", "top_kernels", {"kernel_table": "x"})

    assert result["error"]["code"] == "INVALID_PARAMETER"
    assert "kernel_table" in result["error"]["message"]


def test_run_skill_rejects_unsafe_parameter_bounds_before_opening_profile():
    from nsys_ai.mcp_server import MAX_ROWS, _run_skill

    result = _run_skill(
        "does-not-exist.sqlite",
        "top_kernels",
        {"limit": MAX_ROWS + 1},
    )

    assert result["error"]["code"] == "INVALID_PARAMETER"
    assert "limit" in result["error"]["message"]


def test_run_skill_rejects_unsafe_max_rows_before_opening_profile():
    from nsys_ai.mcp_server import MAX_ROWS, _run_skill

    result = _run_skill("does-not-exist.sqlite", "top_kernels", max_rows=MAX_ROWS + 1)

    assert result["error"]["code"] == "INVALID_PARAMETER"
    assert "max_rows" in result["error"]["message"]


def test_run_skill_preserves_read_only_profile_bytes(profile_copy):
    from nsys_ai.mcp_server import _run_skill

    profile = profile_copy("h100_2gpu_1s.sqlite")
    before = hashlib.sha256(Path(profile).read_bytes()).digest()

    result = _run_skill(str(profile), "top_kernels", {"limit": 1}, max_rows=1)

    assert "error" not in result
    assert hashlib.sha256(Path(profile).read_bytes()).digest() == before


def test_nsys_rep_uses_existing_sidecar_without_writing(profile_copy, tmp_path):
    from nsys_ai.mcp_server import _run_skill

    sidecar = tmp_path / "capture.sqlite"
    shutil.copyfile(profile_copy("h100_2gpu_1s.sqlite"), sidecar)
    rep = tmp_path / "capture.nsys-rep"
    rep.write_bytes(b"capture placeholder")
    sidecar.touch()
    before_rep = hashlib.sha256(rep.read_bytes()).digest()
    before_sidecar = hashlib.sha256(sidecar.read_bytes()).digest()

    result = _run_skill(str(rep), "top_kernels", {"limit": 1}, max_rows=1)

    assert "error" not in result
    assert result["profile"]["path"] == str(sidecar)
    assert hashlib.sha256(rep.read_bytes()).digest() == before_rep
    assert hashlib.sha256(sidecar.read_bytes()).digest() == before_sidecar
    assert not (tmp_path / "capture.parquetdir").exists()


def test_nsys_rep_without_current_sidecar_returns_export_error_without_writing(tmp_path):
    from nsys_ai.mcp_server import _run_skill

    rep = tmp_path / "capture.nsys-rep"
    rep.write_bytes(b"capture placeholder")

    result = _run_skill(str(rep), "top_kernels", {"limit": 1})

    assert result["error"]["code"] == "EXPORT_TOOL_MISSING"
    assert not (tmp_path / "capture.sqlite").exists()
    assert not (tmp_path / "capture.parquetdir").exists()


def test_profile_open_errors_are_standard_mcp_payloads(tmp_path):
    from nsys_ai.mcp_server import _diff_payload, _run_skill

    missing = tmp_path / "missing.sqlite"
    skill_result = _run_skill(str(missing), "top_kernels")
    diff_result = _diff_payload(str(missing), str(missing), gpu=0, limit=1)

    assert skill_result["error"]["code"] == "PROFILE_NOT_FOUND"
    assert diff_result["error"]["code"] == "PROFILE_NOT_FOUND"


def test_schema_errors_are_standard_mcp_payloads(tmp_path):
    from nsys_ai.mcp_server import _diff_payload

    before = tmp_path / "before.sqlite"
    after = tmp_path / "after.sqlite"
    for path in (before, after):
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE unrelated (value INTEGER)")

    result = _diff_payload(str(before), str(after), gpu=0, limit=1)

    assert result["error"]["code"] == "SCHEMA_ERROR"


def test_diff_returns_the_persisted_diff_contract(profile_copy):
    from nsys_ai.mcp_server import _diff_payload

    before = profile_copy("mfu_2gpu_before.sqlite")
    after = profile_copy("mfu_2gpu_after.sqlite")
    result = _diff_payload(str(before), str(after), gpu=0, limit=1)

    assert result["schema_version"] == "0.1"
    assert result["producer"] == "nsys-ai"
    assert result["decision"] is None
    assert {"before", "after", "verdict", "comparability_confidence"} <= set(result)
    assert result["before"]["path"] == str(before)
    json.dumps(result)


def test_diff_rejects_an_unbounded_request():
    from nsys_ai.mcp_server import _diff_payload

    result = _diff_payload("before.sqlite", "after.sqlite", limit=51)

    assert result["error"]["code"] == "INVALID_PARAMETER"


def test_mcp_extra_is_lazy_and_has_a_clear_install_message(monkeypatch):
    import builtins

    from nsys_ai.mcp_server import create_server

    original_import = builtins.__import__

    def reject_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("test: mcp extra is absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_mcp)

    try:
        create_server()
    except RuntimeError as exc:
        assert "nsys-ai[mcp]" in str(exc)
    else:
        raise AssertionError("create_server unexpectedly imported the MCP extra")


def test_mcp_registration_exposes_the_three_read_only_tools(monkeypatch):
    from nsys_ai.mcp_server import create_server

    class FakeServer:
        def __init__(self, name):
            self.name = name
            self.tools = {}

        def tool(self, *, name):
            def register(function):
                self.tools[name] = function
                return function

            return register

    servers = []

    def make_server(name):
        server = FakeServer(name)
        servers.append(server)
        return server

    fake_mcp = types.ModuleType("mcp")
    fake_server = types.ModuleType("mcp.server")
    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp.FastMCP = make_server
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)

    server = create_server()

    assert server is servers[0]
    assert set(server.tools) == {"list_skills", "run_skill", "diff_profiles"}
