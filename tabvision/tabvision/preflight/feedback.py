"""Preflight human-readable diagnostics — see SPEC.md §7 Phase 3.

Renders a ``PreflightReport`` as a concise terminal-friendly block.
"""

from __future__ import annotations

from tabvision.types import PreflightReport

SEVERITY_ICON = {"info": "•", "warn": "⚠", "fail": "✗"}


def render(report: PreflightReport) -> str:
    """Render a PreflightReport as a single multi-line string."""
    lines: list[str] = []
    status = "PASS" if report.passed else "FAIL"
    lines.append(f"Preflight: {status}")
    lines.append("")
    if report.findings:
        lines.append("Findings:")
        for f in report.findings:
            icon = SEVERITY_ICON.get(f.severity, "?")
            lines.append(f"  {icon} {f.severity.upper():4s} [{f.code}] {f.message}")
        lines.append("")
    if report.suggested_actions:
        lines.append("Suggested actions:")
        for s in report.suggested_actions:
            lines.append(f"  - {s}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render"]
