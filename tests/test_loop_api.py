from pathlib import Path


def test_web_loop_endpoints_are_registered():
    web_py = Path("src/nsys_ai/web.py").read_text(encoding="utf-8")
    for route in (
        "/api/loop/state",
        "/api/loop/phase",
        "/api/loop/proposal",
        "/api/loop/reprofile",
        "/api/loop/diagnose",
        "/api/loop/diff",
        "/api/loop/decision",
    ):
        assert route in web_py


def test_timeline_template_has_loop_controls():
    html = Path("src/nsys_ai/templates/timeline.html").read_text(encoding="utf-8")
    assert 'id="loopBtn"' in html
    assert 'id="loopSidebar"' in html
    assert 'id="loopPrimaryBtn"' in html
    assert "loopRunPrimary()" in html
    assert 'id="loopDiffStats"' in html
    assert "LOOP_TRIM_NS" in html
