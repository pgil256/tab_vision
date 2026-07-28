"""Phase 6 development-row tolerance (2026-07-28).

The 51,130-row assertion was pinned by the banked 2026-07-15 Windows/torch-2.12
run. Regenerating on Linux/torch-2.11 yields 51,126 — a 0.008% drift localised
to four predicted events in the single track ``03_Rock3-148-C_comp``. The
assertion now carries a +/-0.05% tolerance, so these tests pin *both* sides of
it: the known drift must load, and a real corruption must still fail loudly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts.eval.string_assignment_phase0 import DEV_PLAYERS
from scripts.eval.string_assignment_phase6 import (
    DEV_ROW_TOLERANCE,
    EXPECTED_DEV_ROWS,
    _load_note_rows,
)

_FIELDS = ("condition", "evaluation_split", "player", "event_id")


def _write_notes(path: Path, *, rows_per_condition: int) -> None:
    """Minimal note table with ``rows_per_condition`` dev rows per condition."""
    player = sorted(DEV_PLAYERS)[0]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        for condition in ("production_equivalent", "segment-v1"):
            for index in range(rows_per_condition):
                writer.writerow(
                    {
                        "condition": condition,
                        "evaluation_split": "development_oof",
                        "player": player,
                        "event_id": f"{condition}-{index}",
                    }
                )


def test_exact_banked_row_count_loads(tmp_path: Path) -> None:
    path = tmp_path / "notes.csv"
    _write_notes(path, rows_per_condition=EXPECTED_DEV_ROWS)
    production, segment = _load_note_rows(path)
    assert len(production) == EXPECTED_DEV_ROWS
    assert len(segment) == EXPECTED_DEV_ROWS


def test_the_measured_four_row_toolchain_drift_loads(tmp_path: Path) -> None:
    """The actual 2026-07-27 regeneration count must not raise."""
    path = tmp_path / "notes.csv"
    _write_notes(path, rows_per_condition=51_126)
    production, segment = _load_note_rows(path)
    assert len(production) == 51_126
    assert len(segment) == 51_126


@pytest.mark.parametrize("delta", [DEV_ROW_TOLERANCE, -DEV_ROW_TOLERANCE])
def test_tolerance_boundary_is_inclusive(tmp_path: Path, delta: int) -> None:
    path = tmp_path / "notes.csv"
    _write_notes(path, rows_per_condition=EXPECTED_DEV_ROWS + delta)
    production, _ = _load_note_rows(path)
    assert len(production) == EXPECTED_DEV_ROWS + delta


@pytest.mark.parametrize("delta", [DEV_ROW_TOLERANCE + 1, -(DEV_ROW_TOLERANCE + 1)])
def test_drift_beyond_tolerance_still_fails_loudly(tmp_path: Path, delta: int) -> None:
    """A real decode regression must not be absorbed by the tolerance."""
    path = tmp_path / "notes.csv"
    _write_notes(path, rows_per_condition=EXPECTED_DEV_ROWS + delta)
    with pytest.raises(RuntimeError, match="development rows per condition"):
        _load_note_rows(path)


def test_tolerance_stays_within_five_hundredths_of_a_percent() -> None:
    """Guard the tolerance itself — it is a provenance decision, not a knob."""
    assert DEV_ROW_TOLERANCE / EXPECTED_DEV_ROWS <= 0.0005
