"""HTML diagnose report for a single clip."""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from tabvision.types import GuitarConfig, SessionConfig, TabEvent

if TYPE_CHECKING:
    from tabvision.audio.filters import AudioFilterConfig


def write_diagnose_report(
    video_path: str | Path,
    output_path: str | Path | None = None,
    *,
    audio_backend_name: str = "basicpitch",
    lambda_vision: float = 1.0,
    video_stride: int = 3,
    video_enabled: bool = True,
    preflight_enabled: bool = True,
    audio_filters: bool | AudioFilterConfig | None = None,
    cfg: GuitarConfig | None = None,
    session: SessionConfig | None = None,
) -> Path:
    """Run lightweight diagnostics and write a self-contained HTML report.

    The report is deliberately tolerant: if preflight or transcription cannot
    run in the local environment, the HTML still captures the failure and keeps
    the overlay/audio/tab/confidence sections present for debugging.
    """
    cfg = cfg or GuitarConfig()
    session = session or SessionConfig()
    input_path = Path(video_path)
    report_path = Path(output_path) if output_path is not None else _default_report_path(input_path)

    preflight_html = _preflight_section(input_path, preflight_enabled)
    tab_events, pipeline_message = _run_pipeline_for_report(
        input_path,
        audio_backend_name=audio_backend_name,
        lambda_vision=lambda_vision,
        video_stride=video_stride,
        video_enabled=video_enabled,
        audio_filters=audio_filters,
        cfg=cfg,
        session=session,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_html(
            input_path=input_path,
            tab_events=tab_events,
            preflight_html=preflight_html,
            pipeline_message=pipeline_message,
            cfg=cfg,
            session=session,
            video_enabled=video_enabled,
        ),
        encoding="utf-8",
    )
    return report_path


def _default_report_path(input_path: Path) -> Path:
    return input_path.with_suffix(input_path.suffix + ".diagnose.html")


def _preflight_section(input_path: Path, enabled: bool) -> str:
    if not enabled:
        return "<p>Preflight skipped by --no-preflight.</p>"
    try:
        from tabvision.preflight import check, render

        return f"<pre>{html.escape(render(check(input_path)))}</pre>"
    except Exception as exc:  # noqa: BLE001 - diagnose reports degraded environments.
        return f"<p>Preflight unavailable: {html.escape(str(exc))}</p>"


def _run_pipeline_for_report(
    input_path: Path,
    *,
    audio_backend_name: str,
    lambda_vision: float,
    video_stride: int,
    video_enabled: bool,
    audio_filters: bool | AudioFilterConfig | None,
    cfg: GuitarConfig,
    session: SessionConfig,
) -> tuple[list[TabEvent], str]:
    try:
        from tabvision.pipeline import run_pipeline

        events = run_pipeline(
            input_path,
            audio_backend_name=audio_backend_name,
            lambda_vision=lambda_vision,
            video_stride=video_stride,
            video_enabled=video_enabled,
            audio_filters=audio_filters,
            cfg=cfg,
            session=session,
        )
        return list(events), f"Pipeline produced {len(events)} tab events."
    except Exception as exc:  # noqa: BLE001 - a diagnostic report is still useful.
        return [], f"Pipeline unavailable: {exc}"


def _render_html(
    *,
    input_path: Path,
    tab_events: Sequence[TabEvent],
    preflight_html: str,
    pipeline_message: str,
    cfg: GuitarConfig,
    session: SessionConfig,
    video_enabled: bool,
) -> str:
    ascii_tab = _ascii_tab(tab_events, cfg)
    confidence_rows = "\n".join(_confidence_rows(tab_events)) or (
        "<tr><td colspan='6'>No decoded tab events available.</td></tr>"
    )
    video_note = (
        "Video stack enabled; overlay generation is currently represented by "
        "diagnostic placeholders unless downstream overlay assets are available."
        if video_enabled
        else "Video stack disabled by --no-video."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TabVision Diagnose - {html.escape(input_path.name)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }}
    section {{ margin: 0 0 2rem; }}
    pre {{ background: #f4f4f4; padding: 1rem; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
    th {{ background: #eee; }}
  </style>
</head>
<body>
  <h1>TabVision Diagnose</h1>
  <p><strong>Input:</strong> {html.escape(str(input_path))}</p>
  <p><strong>Session:</strong> {session.instrument}, {session.tone}, {session.style}</p>
  <p>{html.escape(pipeline_message)}</p>

  <section id="overlay">
    <h2>Overlay</h2>
    <p>{html.escape(video_note)}</p>
    {preflight_html}
  </section>

  <section id="audio">
    <h2>Audio</h2>
    <p>Audio waveform rendering is a Phase 9 placeholder in this CLI report.
    Transcription diagnostics are summarized by decoded note count and pipeline status.</p>
  </section>

  <section id="tab">
    <h2>Tab</h2>
    <pre>{html.escape(ascii_tab)}</pre>
  </section>

  <section id="confidence">
    <h2>Confidence</h2>
    <table>
      <thead>
        <tr><th>Onset</th><th>Duration</th><th>String</th><th>Fret</th><th>Pitch</th><th>Confidence</th></tr>
      </thead>
      <tbody>
        {confidence_rows}
      </tbody>
    </table>
  </section>
</body>
</html>
"""


def _ascii_tab(events: Sequence[TabEvent], cfg: GuitarConfig) -> str:
    if not events:
        return "No decoded tab events available."
    from tabvision.render.ascii import render

    return render(events, cfg)


def _confidence_rows(events: Sequence[TabEvent]) -> list[str]:
    rows: list[str] = []
    for event in sorted(events, key=lambda e: e.onset_s):
        rows.append(
            "<tr>"
            f"<td>{event.onset_s:.3f}</td>"
            f"<td>{event.duration_s:.3f}</td>"
            f"<td>{event.string_idx}</td>"
            f"<td>{event.fret}</td>"
            f"<td>{event.pitch_midi}</td>"
            f"<td>{event.confidence:.2f}</td>"
            "</tr>"
        )
    return rows


__all__ = ["write_diagnose_report"]
