"""Tests for the N2 MuScriptor merge-variant harness.

The pilot's conclusions only mean something if (a) the baseline variant is
an exact identity on the ensemble stream — otherwise the paired ΔTab F1 is
measuring the harness, not the merge — and (b) the admission rules do what
their names claim, since the whole design caution in the ROI deep-dive §3.1
is that a naive union trades rescued recall for new false positives.
"""

from __future__ import annotations

from pathlib import Path

from scripts.eval.n2_muscriptor_merge import (
    VARIANTS,
    ClipContext,
    _clip_context,
    _complementarity_summary,
    added_note_yield,
    gold_hits,
    merge_events,
    select_clips,
)
from tabvision.types import AudioEvent, TabEvent

VARIANTS_BY_NAME = {variant.name: variant for variant in VARIANTS}


def _audio(onset: float, pitch: int, *, duration: float = 0.2) -> AudioEvent:
    return AudioEvent(
        onset_s=onset,
        offset_s=onset + duration,
        pitch_midi=pitch,
        velocity=1.0,
        confidence=1.0,
    )


def _gold(onset: float, pitch: int) -> TabEvent:
    return TabEvent(
        onset_s=onset,
        duration_s=0.2,
        string_idx=0,
        fret=0,
        pitch_midi=pitch,
        confidence=1.0,
    )


def _dense_and_sparse() -> list[AudioEvent]:
    """A three-note cluster at ~1.0 s and one isolated note at 5.0 s."""
    return [
        _audio(1.00, 52),
        _audio(1.03, 59),
        _audio(1.06, 64),
        _audio(5.00, 67),
    ]


def test_baseline_variant_is_identity() -> None:
    ensemble = _dense_and_sparse()
    candidates = [_audio(1.02, 55), _audio(5.02, 70)]
    merged, added = merge_events(ensemble, candidates, VARIANTS_BY_NAME["ensemble"])
    assert added == []
    assert merged == sorted(ensemble, key=lambda event: (event.onset_s, event.pitch_midi))


def test_union_drops_pitch_exact_duplicates() -> None:
    ensemble = _dense_and_sparse()
    candidates = [
        _audio(1.02, 52),  # duplicate of the 1.00 s / 52 event (20 ms < 50 ms)
        _audio(1.02, 55),  # new pitch inside the cluster
        _audio(1.30, 52),  # same pitch but 300 ms away — a genuinely new note
    ]
    merged, added = merge_events(ensemble, candidates, VARIANTS_BY_NAME["union"])
    assert len(added) == 2
    assert len(merged) == len(ensemble) + 2
    assert merged == sorted(merged, key=lambda event: (event.onset_s, event.pitch_midi))


def test_cluster_scope_rejects_isolated_additions() -> None:
    ensemble = _dense_and_sparse()
    candidates = [
        _audio(1.02, 55),  # inside the three-note cluster
        _audio(5.02, 70),  # beside the isolated note — not a cluster
        _audio(9.00, 60),  # nowhere near ensemble activity
    ]
    merged, added = merge_events(ensemble, candidates, VARIANTS_BY_NAME["cluster"])
    assert len(added) == 1
    assert [event.pitch_midi for event in merged if event.onset_s > 4.0] == [67]


def test_near80_admits_beside_isolated_onsets() -> None:
    ensemble = _dense_and_sparse()
    candidates = [_audio(5.02, 70), _audio(9.00, 60)]
    _merged, added = merge_events(ensemble, candidates, VARIANTS_BY_NAME["near80"])
    assert len(added) == 1  # the 5.02 s note only; 9.0 s is far from every onset


def test_duration_floor_drops_short_notes() -> None:
    ensemble = _dense_and_sparse()
    candidates = [_audio(1.02, 55, duration=0.02), _audio(1.04, 57, duration=0.20)]
    _merged, union_added = merge_events(ensemble, candidates, VARIANTS_BY_NAME["union"])
    _merged, floored_added = merge_events(ensemble, candidates, VARIANTS_BY_NAME["cluster-dur60"])
    assert len(union_added) == 2
    assert len(floored_added) == 1


def test_added_note_yield_charges_for_false_additions() -> None:
    gold = [_gold(1.0, 60), _gold(2.0, 62), _gold(3.0, 64)]
    ensemble_hits = [True, False, False]  # the 2.0 s and 3.0 s notes were missed
    added = [
        _audio(2.0, 62),  # a real rescue
        _audio(3.0, 71),  # right place, wrong pitch — not a rescue
        _audio(7.0, 64),  # right pitch, nowhere near the missed onset
        _audio(1.0, 60),  # matches a gold note the ensemble already had
    ]
    assert added_note_yield(added, gold, ensemble_hits) == 1


def test_added_note_yield_is_one_to_one() -> None:
    gold = [_gold(2.0, 62)]
    ensemble_hits = [False]
    duplicated = [_audio(2.0, 62), _audio(2.01, 62)]
    assert added_note_yield(duplicated, gold, ensemble_hits) == 1


def test_clip_context_groups_by_cluster_gap() -> None:
    context = _clip_context(_dense_and_sparse())
    assert isinstance(context, ClipContext)
    assert [size for _start, _end, size in context.clusters] == [3, 1]
    assert context.in_dense_cluster(1.02)
    assert not context.in_dense_cluster(5.00)


def test_gold_hits_is_one_to_one() -> None:
    gold = [_gold(1.0, 60), _gold(1.02, 60)]
    # A single prediction can only claim one of the two same-pitch gold notes.
    assert gold_hits(gold, [_audio(1.0, 60)]) == [True, False]
    assert gold_hits(gold, [_audio(1.0, 60), _audio(1.02, 60)]) == [True, True]
    assert gold_hits(gold, [_audio(1.2, 60)]) == [False, False]


def test_select_clips_filters_mode_and_strides(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation"
    annotation.mkdir()
    for player in ("00", "01", "05"):
        for tune in ("A", "B", "C", "D"):
            for mode in ("solo", "comp"):
                (annotation / f"{player}_{tune}_{mode}.jams").write_text("{}", encoding="utf-8")

    solo = select_clips(tmp_path, "solo", 4)
    assert all(track.endswith("_solo") for track in solo)
    assert all(not track.startswith("05") for track in solo)  # 05 is the held-out player
    assert solo == select_clips(tmp_path, "solo", 4)  # deterministic
    # 8 comp ids, count=2 → stride 4: one clip per dev player, same tune.
    assert select_clips(tmp_path, "comp", 2) == ["00_A_comp", "01_A_comp"]
    assert select_clips(tmp_path, "solo", 0) == []


def test_complementarity_summary_splits_by_mode() -> None:
    per_clip = [
        {"mode": "comp", "gold": 100, "ens_wrong": 20, "ms_rescued": 8},
        {"mode": "solo", "gold": 50, "ens_wrong": 10, "ms_rescued": 1},
    ]
    summary = _complementarity_summary(per_clip)
    assert summary["comp"]["complementarity"] == 0.4
    assert summary["comp"]["gate_pass"] is True
    assert summary["solo"]["complementarity"] == 0.1
    assert summary["pooled"]["ensemble_wrong"] == 30
    assert summary["pooled"]["complementarity"] == 0.3
