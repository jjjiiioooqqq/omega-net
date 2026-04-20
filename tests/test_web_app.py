import json

from engineering_intelligence.demo.pipeline import run_demo
from engineering_intelligence.web.app import build_dashboard_html


def test_dashboard_html_contains_core_sections() -> None:
    html = build_dashboard_html(run_demo())

    assert "Engineering Intelligence Dashboard" in html
    assert 'id="selected"' in html
    assert 'id="closure"' in html
    assert 'id="physics"' in html
    assert 'id="audit"' in html
    assert 'fetch("/api/demo")' in html


def test_dashboard_embeds_initial_json_payload() -> None:
    result = run_demo()
    html = build_dashboard_html(result)

    payload = json.dumps(result, indent=2)
    assert "let data = JSON.parse(" in html
    assert "selected_design" in payload
    assert "selected_design" in html
