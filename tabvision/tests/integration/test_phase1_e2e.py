"""Phase 1 end-to-end integration test — see SPEC §7 Phase 1 acceptance.

Runs the CLI on a fixture and asserts non-empty ASCII tab output.

Skipped when basic-pitch is not installed (CI's lint+types+smoke job
doesn't install [audio-baseline]).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

basic_pitch = pytest.importorskip(
    "basic_pitch",
    reason="basic-pitch not installed — install with pip install '.[audio-baseline]'",
)
pytest.importorskip("soundfile")

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not on PATH", allow_module_level=True)

FIXTURE = Path(__file__).parent.parent.parent / "data" / "fixtures" / "test_a440.mp4"


@pytest.mark.integration
def test_transcribe_a440_fixture(tmp_path):
    """Phase 1 acceptance: ``tabvision transcribe`` produces non-empty tab."""
    from tabvision.cli import main

    out = tmp_path / "out.tab"
    rc = main(["transcribe", str(FIXTURE), "-o", str(out)])
    assert rc == 0, "CLI should exit 0 on the fixture"
    assert out.exists(), "output file should be written"

    text = out.read_text()
    assert "TabVision ASCII tab" in text
    assert "Tuning:" in text
    # Six tab lines + header + at least one fret position.
    lines = [line for line in text.splitlines() if "|" in line]
    assert len(lines) >= 6, f"expected at least 6 tab-string lines, got {len(lines)}"


@pytest.mark.integration
def test_transcribe_emits_correct_position_for_a440(tmp_path):
    """A440 = MIDI 69. On high-E string (MIDI 64), that's fret 5. Greedy
    audio-only fusion should pick the lowest-fret candidate, which on the
    open instrument is high-E fret 5 (vs B-string fret 10, etc.).
    """
    from tabvision.cli import main

    out = tmp_path / "out.tab"
    main(["transcribe", str(FIXTURE), "-o", str(out)])

    text = out.read_text()
    high_e_line = next(line for line in text.splitlines() if line.startswith("e|"))
    # The fret picked should appear on the high-E line. Without the e-line
    # having the fret number, audio-only fusion is broken.
    assert "5" in high_e_line, f"expected fret 5 on high E, got: {high_e_line!r}"
