from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabvision.cli import _build_parser, main


def test_transcribe_format_defaults_to_ascii() -> None:
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "input.mov"])
    assert args.format == "ascii"


def test_transcribe_format_accepts_phase6_formats() -> None:
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "input.mov", "--format", "midi"])
    assert args.format == "midi"


def test_transcribe_json_is_additive_and_disabled_by_default() -> None:
    parser = _build_parser()

    default_args = parser.parse_args(["transcribe", "input.mov"])
    json_args = parser.parse_args(["transcribe", "input.mov", "--json"])

    assert default_args.json_output is False
    assert json_args.json_output is True


def test_transcribe_progress_is_additive_and_disabled_by_default() -> None:
    parser = _build_parser()

    default_args = parser.parse_args(["transcribe", "input.mov"])
    progress_args = parser.parse_args(["transcribe", "input.mov", "--progress"])

    assert default_args.progress is False
    assert progress_args.progress is True


def test_transcribe_progress_reports_stages_on_stderr(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.mov"
    input_path.write_bytes(b"not a real movie; pipeline is injected")
    output_path = tmp_path / "result.tab"

    def fake_run_pipeline(*_args, **kwargs):
        callback = kwargs["progress_callback"]
        assert callback is not None
        for stage in (
            "demux",
            "model_load",
            "audio_inference",
            "video_analysis",
            "decode",
        ):
            callback(stage)
        return []

    monkeypatch.setattr("tabvision.pipeline.run_pipeline", fake_run_pipeline)

    rc = main(
        [
            "transcribe",
            str(input_path),
            "--output",
            str(output_path),
            "--progress",
            "--no-preflight",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "PROGRESS demux 10",
        "PROGRESS model_load 20",
        "PROGRESS audio_inference 35",
        "PROGRESS video_analysis 60",
        "PROGRESS decode 80",
        "PROGRESS render 90",
        "PROGRESS complete 100",
    ]


def test_transcribe_default_has_no_progress_output(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.mov"
    input_path.write_bytes(b"not a real movie; pipeline is injected")
    output_path = tmp_path / "result.tab"

    def fake_run_pipeline(*_args, **kwargs):
        assert kwargs["progress_callback"] is None
        return []

    monkeypatch.setattr("tabvision.pipeline.run_pipeline", fake_run_pipeline)

    rc = main(
        [
            "transcribe",
            str(input_path),
            "--output",
            str(output_path),
            "--no-preflight",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
    assert captured.err == ""


def test_transcribe_json_envelope_reports_output_flags_and_timings(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tabvision.types import TabEvent

    input_path = tmp_path / "input.mov"
    input_path.write_bytes(b"not a real movie; pipeline is injected")
    output_path = tmp_path / "result.tab"
    events = [
        TabEvent(
            onset_s=0.125,
            duration_s=0.25,
            string_idx=0,
            fret=3,
            pitch_midi=43,
            confidence=0.32,
        ),
        TabEvent(
            onset_s=0.5,
            duration_s=0.25,
            string_idx=5,
            fret=0,
            pitch_midi=64,
            confidence=0.95,
        ),
    ]
    monkeypatch.setattr("tabvision.pipeline.run_pipeline", lambda *a, **k: events)

    rc = main(
        [
            "transcribe",
            str(input_path),
            "--output",
            str(output_path),
            "--json",
            "--no-video",
            "--no-preflight",
        ]
    )

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert rc == 0
    assert output_path.exists()
    assert envelope["status"] == "ok"
    assert envelope["output_path"] == str(output_path.resolve())
    assert envelope["low_confidence_flags"] == [
        {
            "type": "low_confidence_note",
            "event_index": 0,
            "onset_s": 0.125,
            "confidence": 0.32,
        }
    ]
    assert set(envelope["timings"]) == {
        "preflight_s",
        "pipeline_s",
        "render_s",
        "total_s",
    }
    assert all(value >= 0.0 for value in envelope["timings"].values())


def test_transcribe_json_requires_output_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.mov"
    input_path.write_bytes(b"not a real movie; pipeline must not run")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_pipeline must not run without --output in JSON mode")

    monkeypatch.setattr("tabvision.pipeline.run_pipeline", fail_if_called)

    rc = main(
        [
            "transcribe",
            str(input_path),
            "--json",
            "--no-video",
            "--no-preflight",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err == "error: --json requires --output so stdout remains valid JSON\n"


def test_transcribe_creates_missing_output_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-o nested/dir/out.tab`` must not crash with a raw FileNotFoundError
    after the (potentially minutes-long) pipeline has already run — mirrors
    the auto-mkdir that ``diagnose`` already does in ``write_diagnose_report``."""
    input_path = tmp_path / "input.mov"
    input_path.write_bytes(b"not a real movie; pipeline is injected")
    output_path = tmp_path / "nested" / "dir" / "out.tab"
    assert not output_path.parent.exists()

    monkeypatch.setattr("tabvision.pipeline.run_pipeline", lambda *a, **k: [])

    rc = main(
        [
            "transcribe",
            str(input_path),
            "-o",
            str(output_path),
            "--no-video",
            "--no-preflight",
        ]
    )

    assert rc == 0
    assert output_path.exists()


def test_diagnose_parser_defaults_to_html_report_next_to_input() -> None:
    parser = _build_parser()
    args = parser.parse_args(["diagnose", "input.mov"])
    assert args.command == "diagnose"
    assert args.input == Path("input.mov")
    assert args.output is None


def test_diagnose_writes_html_report(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "input.mov"
    input_path.write_bytes(b"not a real movie; pipeline is injected")
    output_path = tmp_path / "report.html"

    def fake_write_report(
        video_path: Path,
        output_path_arg: Path,
        *,
        audio_backend_name: str,
        lambda_vision: float,
        video_stride: int,
        video_enabled: bool,
        preflight_enabled: bool,
        audio_filters,
        cfg,
        session,
    ) -> Path:
        assert video_path == input_path
        assert output_path_arg == output_path
        # No --audio-filters flag passed → 'auto' → None (backend default kept).
        assert audio_filters is None
        output_path_arg.write_text(
            "<html><section id='overlay'></section><section id='audio'></section>"
            "<section id='tab'></section><section id='confidence'></section></html>"
        )
        return output_path_arg

    monkeypatch.setattr("tabvision.diagnose.write_diagnose_report", fake_write_report)

    rc = main(
        [
            "diagnose",
            str(input_path),
            "-o",
            str(output_path),
            "--no-video",
            "--no-preflight",
        ]
    )

    assert rc == 0
    html = output_path.read_text()
    assert 'id="overlay"' in html or "id='overlay'" in html
    assert 'id="audio"' in html or "id='audio'" in html
    assert 'id="tab"' in html or "id='tab'" in html
    assert 'id="confidence"' in html or "id='confidence'" in html


def test_waveform_svg_renders_envelope_from_samples() -> None:
    import numpy as np

    from tabvision.diagnose import _waveform_svg

    svg = _waveform_svg(np.sin(np.linspace(0.0, 40.0, 8000)))
    assert svg.startswith("<svg")
    assert "<path" in svg


def test_waveform_svg_is_blank_for_empty_audio() -> None:
    import numpy as np

    from tabvision.diagnose import _waveform_svg

    assert _waveform_svg(np.zeros(0)) == ""


def test_diagnose_uses_supplied_events_and_grades_confidence(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real report path: supplying ``tab_events`` must skip the pipeline
    entirely (no second decode) and colour-band the confidence table."""
    from tabvision.diagnose import write_diagnose_report
    from tabvision.types import TabEvent

    def _must_not_run(*_a, **_k):
        raise AssertionError("run_pipeline must not run when tab_events is supplied")

    monkeypatch.setattr("tabvision.pipeline.run_pipeline", _must_not_run)

    events = [
        TabEvent(onset_s=0.0, duration_s=0.5, string_idx=0, fret=3, pitch_midi=43, confidence=0.32),
        TabEvent(onset_s=0.5, duration_s=0.5, string_idx=5, fret=0, pitch_midi=64, confidence=0.95),
    ]
    output_path = tmp_path / "report.html"

    write_diagnose_report(
        tmp_path / "input.mov",
        output_path,
        video_enabled=False,
        preflight_enabled=False,
        tab_events=events,
    )

    html = output_path.read_text(encoding="utf-8")
    for section in ("overlay", "audio", "tab", "confidence"):
        assert f'id="{section}"' in html
    assert "conf-low" in html and "conf-high" in html
    assert "caller-supplied" in html
