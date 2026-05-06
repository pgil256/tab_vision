"""Phase 5 acceptance harness — audio+vision vs. audio-only ablation.

Per SPEC §5 and ``docs/plans/2026-05-06-phase5-fusion-design.md`` §6 Step E,
the Phase-5-specific gate is:

    Tab F1 (lambda_vision=1.0) - Tab F1 (lambda_vision=0.0) ≥ 0.08

The absolute Tab F1 ≥ 0.85 bar is currently expected to need Phase 2's
Riley/Edwards audio backbone too — so it's marked ``xfail`` until Phase
2 is wired in. The +8 pp delta is on the hook for Phase 5 alone, since
that's the test for "fusion is doing real work given the current audio".

**Environment caveat (2026-05-06):** the audio backend (basic-pitch +
TF 2.15) requires ``numpy<2`` while MediaPipe (Phase 4) requires
``numpy>=2`` — so a single venv currently can't run both halves of the
pipeline. The test skips when MediaPipe imports fail; once the env is
reconciled (or Phase 2's torch-based audio backbone replaces basic-pitch)
the gate runs unchanged. See ``DECISIONS.md`` if/when this gets fixed.

The gold source is the benchmark index at
``tabvision-server/tests/fixtures/benchmarks/index.json`` — same set the
legacy ``evaluate_transcription.py`` used. Phase 1.5's annotation tool
will eventually fold its labelled clips into the same harness.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Sequence

import pytest

from tabvision.eval.metrics import (
    ChordAccuracyResult,
    TabF1Result,
    chord_instance_accuracy,
    tab_f1,
)
from tabvision.types import TabEvent

PHASE5_TAB_F1_DELTA_GATE = 0.08
"""SPEC §5: audio+vision must beat audio-only by at least this much on Tab F1."""

PHASE5_TAB_F1_ABSOLUTE_GATE = 0.85
"""SPEC §5: target absolute Tab F1. Likely needs Phase 2 SOTA backbone."""

PHASE5_CHORD_ACCURACY_GATE = 0.80
"""SPEC §5: chord-instance accuracy gate."""

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_INDEX = (
    REPO_ROOT
    / "tabvision-server"
    / "tests"
    / "fixtures"
    / "benchmarks"
    / "index.json"
)
EVAL_OUTPUT_DIR = REPO_ROOT / "tabvision-server" / "tools" / "outputs"


@pytest.mark.eval
def test_phase5_audio_plus_vision_beats_audio_only():
    """Run the full pipeline on the eval set under both lambda_vision
    settings; assert audio+vision wins by ≥ 8 pp Tab F1.

    Skips automatically when any heavy dependency (basic-pitch, mediapipe,
    cv2, ffmpeg) is unavailable.
    """
    pytest.importorskip(
        "basic_pitch",
        reason="basic-pitch needed for audio-only ablation; install with "
        "pip install '.[audio-baseline]'",
    )
    pytest.importorskip(
        "mediapipe",
        reason="MediaPipe needed for video evidence; install with "
        "pip install '.[vision]'. NOTE: requires numpy>=2, currently "
        "incompatible with TF 2.15.",
    )
    pytest.importorskip("cv2", reason="opencv-python needed for video frames.")

    benchmarks = _load_benchmarks()
    if not benchmarks:
        pytest.skip("no benchmarks defined in index.json")

    audio_only_scores: list[TabF1Result] = []
    audio_video_scores: list[TabF1Result] = []
    chord_scores: list[ChordAccuracyResult] = []
    rows: list[dict] = []

    for bench in benchmarks:
        video = REPO_ROOT / bench["video_path"]
        gold_path = REPO_ROOT / bench["ground_truth_path"]
        if not video.exists() or not gold_path.exists():
            continue
        gold = _load_gold_tab_events(gold_path)
        if not gold:
            continue

        ao = _run_pipeline(video, lambda_vision=0.0)
        av = _run_pipeline(video, lambda_vision=1.0)

        ao_score = tab_f1(ao, gold)
        av_score = tab_f1(av, gold)
        chord_score = chord_instance_accuracy(av, gold)

        audio_only_scores.append(ao_score)
        audio_video_scores.append(av_score)
        chord_scores.append(chord_score)
        rows.append(
            {
                "id": bench["id"],
                "ao_f1": ao_score.f1,
                "av_f1": av_score.f1,
                "delta": av_score.f1 - ao_score.f1,
                "chord_acc": chord_score.accuracy,
            }
        )

    if not rows:
        pytest.skip("no benchmark videos / ground truth files were available")

    ao_mean = _mean([r.f1 for r in audio_only_scores])
    av_mean = _mean([r.f1 for r in audio_video_scores])
    chord_mean = _mean([r.accuracy for r in chord_scores])
    delta = av_mean - ao_mean

    _write_report(
        rows=rows,
        ao_mean=ao_mean,
        av_mean=av_mean,
        delta=delta,
        chord_mean=chord_mean,
    )

    assert delta >= PHASE5_TAB_F1_DELTA_GATE, (
        f"Phase 5 +{PHASE5_TAB_F1_DELTA_GATE * 100:.0f}pp gate failed: "
        f"audio+vision {av_mean:.3f} - audio-only {ao_mean:.3f} = "
        f"{delta:+.3f}. Per SPEC §5 decision tree, drop lambda_vision and "
        f"investigate vision calibration if equal/worse, or tighten "
        f"hand-span / open-string priors if marginally better."
    )


@pytest.mark.eval
@pytest.mark.xfail(
    reason="absolute Tab F1 ≥ 0.85 likely needs Phase 2 audio SOTA backbone "
    "to also be wired in; track in DECISIONS.md",
    strict=False,
)
def test_phase5_absolute_tab_f1():
    pytest.importorskip("basic_pitch")
    pytest.importorskip("mediapipe")
    pytest.importorskip("cv2")

    benchmarks = _load_benchmarks()
    if not benchmarks:
        pytest.skip("no benchmarks defined in index.json")

    scores: list[TabF1Result] = []
    for bench in benchmarks:
        video = REPO_ROOT / bench["video_path"]
        gold_path = REPO_ROOT / bench["ground_truth_path"]
        if not video.exists() or not gold_path.exists():
            continue
        gold = _load_gold_tab_events(gold_path)
        if not gold:
            continue
        av = _run_pipeline(video, lambda_vision=1.0)
        scores.append(tab_f1(av, gold))

    if not scores:
        pytest.skip("no benchmark videos available")

    mean_f1 = _mean([s.f1 for s in scores])
    assert mean_f1 >= PHASE5_TAB_F1_ABSOLUTE_GATE, (
        f"absolute Tab F1 {mean_f1:.3f} < {PHASE5_TAB_F1_ABSOLUTE_GATE}"
    )


@pytest.mark.eval
def test_phase5_chord_accuracy():
    pytest.importorskip("basic_pitch")
    pytest.importorskip("mediapipe")
    pytest.importorskip("cv2")

    benchmarks = _load_benchmarks()
    if not benchmarks:
        pytest.skip("no benchmarks defined in index.json")

    scores: list[ChordAccuracyResult] = []
    for bench in benchmarks:
        video = REPO_ROOT / bench["video_path"]
        gold_path = REPO_ROOT / bench["ground_truth_path"]
        if not video.exists() or not gold_path.exists():
            continue
        gold = _load_gold_tab_events(gold_path)
        if not gold:
            continue
        av = _run_pipeline(video, lambda_vision=1.0)
        scores.append(chord_instance_accuracy(av, gold))

    if not scores:
        pytest.skip("no benchmark videos available")

    mean_acc = _mean([s.accuracy for s in scores])
    assert mean_acc >= PHASE5_CHORD_ACCURACY_GATE, (
        f"chord accuracy {mean_acc:.3f} < {PHASE5_CHORD_ACCURACY_GATE}"
    )


# ---------- helpers ----------


def _load_benchmarks() -> list[dict]:
    if not BENCHMARK_INDEX.exists():
        return []
    return json.loads(BENCHMARK_INDEX.read_text()).get("benchmarks", [])


def _load_gold_tab_events(path: Path) -> list[TabEvent]:
    """Parse the legacy benchmark ground-truth ``.txt`` format into TabEvents.

    The legacy parser lives in ``tabvision-server/evaluate_transcription.py``;
    this helper imports it lazily to keep the eval module's deps minimal.
    Returns an empty list if the legacy module isn't importable (e.g. when
    the test runs from an environment without the server checked out).
    """
    try:
        import sys

        server_path = REPO_ROOT / "tabvision-server"
        if str(server_path) not in sys.path:
            sys.path.insert(0, str(server_path))
        from evaluate_transcription import parse_ground_truth_tabs
    except Exception:  # noqa: BLE001 — broad: optional dep, want graceful skip
        return []

    text = path.read_text()
    parsed = parse_ground_truth_tabs(text)
    # The legacy parser returns beats; we need seconds. The benchmarks
    # don't carry duration, so this helper currently returns the parsed
    # raw notes without timing. Phase 5 acceptance defers timing
    # alignment to the per-video runner that knows the video duration —
    # see ``_run_pipeline``.
    out: list[TabEvent] = []
    for note in parsed:
        out.append(
            TabEvent(
                onset_s=float(note["beat"]),  # placeholder — runner aligns
                duration_s=0.25,
                # Legacy uses 1=high E, 6=low E; spec uses 0=low E, 5=high E.
                string_idx=6 - int(note["string"]),
                fret=0 if note["fret"] == "X" else int(note["fret"]),
                pitch_midi=0,  # not needed for Tab F1
                confidence=1.0,
            )
        )
    return out


def _run_pipeline(video: Path, *, lambda_vision: float) -> Sequence[TabEvent]:
    """Run audio + video + fusion end-to-end and return TabEvents.

    Stub for now: until the numpy<2 / numpy>=2 environment conflict is
    resolved (or Phase 2's torch-based audio backbone is wired up), this
    raises ``ImportError`` so the surrounding ``importorskip`` calls
    catch it and the test skips with a clear message. Implementation
    will compose ``demux`` → audio backend → guitar/fretboard/hand
    detect → ``fuse(..., lambda_vision=lambda_vision)`` once the env is
    sorted. See the design doc §6 Step E.
    """
    raise ImportError(
        "Phase 5 end-to-end pipeline runner not yet wired — blocked on "
        "numpy<2 vs numpy>=2 env conflict between basic-pitch and "
        "mediapipe. See test docstring."
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_report(
    *,
    rows: list[dict],
    ao_mean: float,
    av_mean: float,
    delta: float,
    chord_mean: float,
) -> None:
    """Emit ``tools/outputs/phase5_eval-YYYY-MM-DD.md`` summary report."""
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out = EVAL_OUTPUT_DIR / f"phase5_eval-{today}.md"
    lines = [
        f"# Phase 5 acceptance — {today}",
        "",
        "Audio-only vs. audio+vision ablation, per SPEC §5.",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean Tab F1 (lambda_vision=0.0) | {ao_mean:.4f} |",
        f"| Mean Tab F1 (lambda_vision=1.0) | {av_mean:.4f} |",
        f"| Delta (audio+vision − audio-only) | {delta:+.4f} |",
        f"| Mean chord-instance accuracy | {chord_mean:.4f} |",
        f"| Phase 5 +{PHASE5_TAB_F1_DELTA_GATE * 100:.0f}pp gate | "
        f"{'PASS' if delta >= PHASE5_TAB_F1_DELTA_GATE else 'FAIL'} |",
        "",
        "## Per-video",
        "",
        "| id | audio-only F1 | audio+vision F1 | delta | chord acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['ao_f1']:.3f} | {r['av_f1']:.3f} | "
            f"{r['delta']:+.3f} | {r['chord_acc']:.3f} |"
        )
    out.write_text("\n".join(lines) + "\n")
