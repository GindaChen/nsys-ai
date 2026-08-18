"""Contracts for the optional read-only MCP transport."""

import hashlib
import json
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


def test_run_skill_preserves_read_only_profile_bytes(profile_copy):
    from nsys_ai.mcp_server import _run_skill

    profile = profile_copy("h100_2gpu_1s.sqlite")
    before = hashlib.sha256(Path(profile).read_bytes()).digest()

    result = _run_skill(str(profile), "top_kernels", {"limit": 1}, max_rows=1)

    assert "error" not in result
    assert hashlib.sha256(Path(profile).read_bytes()).digest() == before


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
