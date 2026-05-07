from __future__ import annotations

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
        cfg,
        session,
    ) -> Path:
        assert video_path == input_path
        assert output_path_arg == output_path
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
