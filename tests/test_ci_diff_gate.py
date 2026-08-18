"""Contract tests for the thin CI wrappers around the canonical diff gate."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / ".github" / "actions" / "diff-gate" / "action.yml"
WRAPPER = ROOT / "scripts" / "nsys-ai-diff-gate"


def test_github_action_forwards_the_canonical_gate_contract():
    text = ACTION.read_text(encoding="utf-8")

    assert "using: composite" in text
    assert "python -m nsys_ai" in text
    assert "args+=(\"$NSYS_AI_AFTER\" --format json --no-ai --exit-on-regression" in text
    assert "args+=(--gate \"$NSYS_AI_GATE\")" in text
    assert "payload.get(\"verdict\", \"\")" in text


def test_ordinary_ci_wrapper_emits_the_canonical_json_report(tmp_path):
    output = tmp_path / "artifacts" / "diff.json"
    profile = ROOT / "tests" / "fixtures" / "mock.sqlite"
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [str(WRAPPER), str(profile), str(profile), "--output", str(output)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload["verdict"], str)
    assert "--format" not in result.stderr
