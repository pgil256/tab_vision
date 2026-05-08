"""Unit tests for tabvision.preflight.feedback (rendering)."""

from tabvision.preflight.feedback import render
from tabvision.types import PreflightFinding, PreflightReport


def _report(passed=True, findings=None, suggestions=None) -> PreflightReport:
    return PreflightReport(
        passed=passed,
        findings=findings or [],
        suggested_actions=suggestions or [],
    )


def test_render_pass_with_no_findings():
    out = render(_report(True))
    assert "Preflight: PASS" in out


def test_render_fail_includes_finding_and_action():
    rep = _report(
        passed=False,
        findings=[PreflightFinding("fail", "GUITAR_NOT_DETECTED", "no guitar found")],
        suggestions=["Move the guitar fully into frame."],
    )
    out = render(rep)
    assert "Preflight: FAIL" in out
    assert "GUITAR_NOT_DETECTED" in out
    assert "Move the guitar fully into frame." in out
    assert "FAIL" in out


def test_render_warn_severity():
    rep = _report(
        passed=False,
        findings=[PreflightFinding("warn", "LIGHTING_DIM", "mean luma 30/255")],
    )
    out = render(rep)
    assert "WARN" in out
    assert "LIGHTING_DIM" in out


def test_render_handles_unknown_severity_gracefully():
    rep = _report(
        passed=False,
        findings=[PreflightFinding("info", "GUITAR_VISIBLE", "guitar in 80% of frames")],
    )
    out = render(rep)
    assert "INFO" in out
    assert "GUITAR_VISIBLE" in out
