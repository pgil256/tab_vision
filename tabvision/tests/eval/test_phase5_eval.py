"""Phase 5 optional full ablation harness — audio+vision vs. audio-only.

Per SPEC §5 and ``docs/plans/2026-05-06-phase5-fusion-design.md`` §6 Step E,
the Phase-5-specific gate is:

    Tab F1 (lambda_vision=1.0) - Tab F1 (lambda_vision=0.0) ≥ 0.08

The absolute Tab F1 ≥ 0.85 bar likely also needs Phase 7's augmentation
work to clear, so it's marked ``xfail`` for now. The +8 pp delta is on
the hook for Phase 5 alone — that's the test for "fusion is doing real
work given today's audio".

**Audio backend:** uses ``tabvision.audio.backend.make("highres")``
(Phase 2 Riley/Edwards / GAPS via hf-midi-transcription, torch-based,
numpy-2-compatible) — *not* basic-pitch.

**Pipeline:** the full demux → audio + video → fuse chain runs through
:func:`tabvision.pipeline.run_pipeline`. The eval tests skip cleanly
when MediaPipe / cv2 / a YOLO checkpoint aren't available; the gate
runs unchanged once the [vision] extras are installed and the YOLO
weights have been acquired.

The full/home-video ablation is future validation, not a v1 blocker. The
automated v1 gate is the deterministic smoke/public-data report path.

The gold source is the benchmark index at
``tabvision-server/tests/fixtures/benchmarks/index.json`` — same set the
legacy ``evaluate_transcription.py`` used. Optional manual labels can fold
into the same harness later.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest

from tabvision.eval.metrics import (
    chord_instance_accuracy,
    tab_f1,
)
from tabvision.types import (
    DEFAULT_TUNING_MIDI,
    AudioEvent,
    FrameFingering,
    TabEvent,
)

PHASE5_TAB_F1_DELTA_GATE = 0.08
"""SPEC §5: audio+vision must beat audio-only by at least this much on Tab F1."""

PHASE5_TAB_F1_ABSOLUTE_GATE = 0.85
"""SPEC §5: target absolute Tab F1. Likely needs Phase 2 SOTA backbone."""

PHASE5_CHORD_ACCURACY_GATE = 0.80
"""SPEC §5: chord-instance accuracy gate."""

PHASE5_ALIGNMENT_TOLERANCE_S = 0.50
"""Loose pitch-only window used only to find per-clip gold alignment offsets."""

PHASE5_ALIGNMENT_STEP_S = 0.05
"""Offset-search step. Fine enough for later 50 ms strict Tab F1 scoring."""

PHASE5_LAMBDA_SWEEP = (0.0, 0.5, 1.0, 2.0, 5.0)
"""Diagnostic sweep when default ``lambda_vision=1.0`` misses the delta gate."""

LEGACY_MAX_FRET = 24
"""Max fret used by the frozen v0 tab parser when disambiguating 2-digit frets."""

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_INDEX = (
    REPO_ROOT / "tabvision-server" / "tests" / "fixtures" / "benchmarks" / "index.json"
)
EVAL_OUTPUT_DIR = REPO_ROOT / "tabvision-server" / "tools" / "outputs"


@pytest.mark.eval
def test_phase5_audio_plus_vision_beats_audio_only():
    """Run the full pipeline on the eval set under both lambda_vision
    settings; assert audio+vision wins by ≥ 8 pp Tab F1.

    Skips automatically when any heavy dependency (the highres audio
    backend's torch + hf-midi-transcription stack, mediapipe, cv2, ffmpeg)
    is unavailable, *or* when the video-stack-into-pipeline integration
    is still a TODO in ``_run_pipeline``.
    """
    _require_eval_readiness()

    rows = _collect_phase5_rows(lambda_vision=1.0)

    ao_mean = _mean([r["ao_f1"] for r in rows])
    av_mean = _mean([r["av_f1"] for r in rows])
    chord_mean = _mean([r["chord_acc"] for r in rows])
    delta = av_mean - ao_mean
    sweep_rows = []
    if delta < PHASE5_TAB_F1_DELTA_GATE:
        sweep_rows = _run_lambda_sweep()

    _write_report(
        rows=rows,
        ao_mean=ao_mean,
        av_mean=av_mean,
        delta=delta,
        chord_mean=chord_mean,
        sweep_rows=sweep_rows,
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
    _require_eval_readiness()

    rows = _collect_phase5_rows(lambda_vision=1.0)
    mean_f1 = _mean([r["av_f1"] for r in rows])
    assert mean_f1 >= PHASE5_TAB_F1_ABSOLUTE_GATE, (
        f"absolute Tab F1 {mean_f1:.3f} < {PHASE5_TAB_F1_ABSOLUTE_GATE}"
    )


@pytest.mark.eval
def test_phase5_chord_accuracy():
    _require_eval_readiness()

    rows = _collect_phase5_rows(lambda_vision=1.0)
    mean_acc = _mean([r["chord_acc"] for r in rows])
    assert mean_acc >= PHASE5_CHORD_ACCURACY_GATE, (
        f"chord accuracy {mean_acc:.3f} < {PHASE5_CHORD_ACCURACY_GATE}"
    )


# ---------- helpers ----------


@pytest.fixture(autouse=True)
def _phase5_eval_requires_marker(request: pytest.FixtureRequest) -> None:
    markexpr = str(getattr(request.config.option, "markexpr", "") or "")
    if request.node.get_closest_marker("eval") and "eval" not in markexpr:
        pytest.skip(
            "Phase 5 eval is opt-in; run with `pytest -m eval tests/eval/test_phase5_eval.py`."
        )


def _load_benchmarks() -> list[dict]:
    if not BENCHMARK_INDEX.exists():
        return []
    return json.loads(BENCHMARK_INDEX.read_text()).get("benchmarks", [])


def _require_eval_readiness() -> None:
    """Skip only for optional heavy dependencies / model artifacts.

    Benchmark-data problems fail later because those are repo issues, not
    optional local-environment issues.
    """
    pytest.importorskip("torch", reason="highres backend needs torch.")
    pytest.importorskip(
        "hf_midi_transcription",
        reason="highres backend needs hf-midi-transcription.",
    )
    pytest.importorskip("soundfile", reason="highres backend needs soundfile.")
    pytest.importorskip("scipy.signal", reason="highres backend needs scipy.")
    pytest.importorskip("pretty_midi", reason="highres backend needs pretty_midi.")
    pytest.importorskip(
        "mediapipe",
        reason="MediaPipe needed for video evidence; install with pip install '.[vision]'.",
    )
    pytest.importorskip("cv2", reason="opencv-python needed for video frames.")

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH; required by tabvision.demux")
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe not on PATH; required by tabvision.demux")

    from tabvision.video.guitar.yolo_backend import _default_checkpoint_path
    from tabvision.video.hand.mediapipe_backend import _default_model_path

    hand_model = _default_model_path()
    if not hand_model.exists():
        pytest.skip(f"MediaPipe hand model not found at {hand_model}")
    _require_mediapipe_landmarker_loads(hand_model)

    yolo_checkpoint = _default_checkpoint_path()
    if not yolo_checkpoint.exists():
        pytest.skip(f"YOLO-OBB checkpoint not found at {yolo_checkpoint}")


def _require_mediapipe_landmarker_loads(hand_model: Path) -> None:
    """Probe native MediaPipe readiness out-of-process so segfaults become skips."""
    probe = """
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=sys.argv[1])
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
landmarker = vision.HandLandmarker.create_from_options(options)
landmarker.close()
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe, str(hand_model)],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("MediaPipe HandLandmarker readiness probe timed out.")
    if proc.returncode == 0:
        return

    details = (proc.stderr or proc.stdout).strip().splitlines()
    reason = details[-1] if details else f"process exited {proc.returncode}"
    pytest.skip(
        "MediaPipe HandLandmarker readiness probe failed; install compatible "
        f"system OpenGL/GLES runtime libraries. Last error: {reason}"
    )


@cache
def _collect_phase5_rows(*, lambda_vision: float) -> tuple[dict, ...]:
    benchmarks = _load_benchmarks()
    if not benchmarks:
        pytest.fail(f"no benchmarks defined in {BENCHMARK_INDEX}")

    rows: list[dict] = []
    available = 0
    parsed_gold = 0
    runner = _Phase5Runner(lambda_vision=lambda_vision)
    for bench in benchmarks:
        video = REPO_ROOT / bench["video_path"]
        gold_path = REPO_ROOT / bench["ground_truth_path"]
        if not video.exists() or not gold_path.exists():
            continue
        available += 1

        video_duration_s = _video_duration_s(video)
        gold = _load_gold_tab_events(
            gold_path,
            bpm=bench.get("bpm"),
            video_duration_s=video_duration_s,
        )
        if not gold:
            continue
        parsed_gold += 1

        ao, av = runner.run(video)
        aligned_gold, offset_s, alignment_matches = _align_gold_to_audio_only(
            audio_only=ao,
            gold=gold,
            video_duration_s=video_duration_s,
        )

        ao_score = tab_f1(ao, aligned_gold)
        av_score = tab_f1(av, aligned_gold)
        chord_score = chord_instance_accuracy(av, aligned_gold)

        rows.append(
            {
                "id": bench["id"],
                "lambda": lambda_vision,
                "ao_f1": ao_score.f1,
                "av_f1": av_score.f1,
                "delta": av_score.f1 - ao_score.f1,
                "chord_acc": chord_score.accuracy,
                "offset_s": offset_s,
                "alignment_matches": alignment_matches,
                "gold_count": len(aligned_gold),
                "ao_count": len(ao),
                "av_count": len(av),
            }
        )

    if available == 0:
        pytest.fail("benchmark index exists, but no referenced video/ground-truth files exist")
    if parsed_gold == 0:
        pytest.fail("benchmark files exist, but no ground-truth notes parsed")
    if rows and all(r["alignment_matches"] == 0 for r in rows):
        pytest.fail(
            "pitch-only alignment found zero matches on every clip; inspect audio backend "
            "output before trusting strict Tab F1"
        )
    return tuple(rows)


class _Phase5Runner:
    """Eval-only runner that reuses expensive per-clip evidence."""

    def __init__(self, *, lambda_vision: float) -> None:
        from tabvision.types import GuitarConfig, SessionConfig

        self.lambda_vision = lambda_vision
        self.cfg = GuitarConfig()
        self.session = SessionConfig()

    def run(self, video: Path) -> tuple[list[TabEvent], list[TabEvent]]:
        from tabvision.fusion import apply_neck_anchor_priors, fuse

        audio_events = list(_phase5_audio_events(video))
        audio_only = list(
            fuse(
                audio_events,
                [],
                self.cfg,
                self.session,
                lambda_vision=0.0,
            )
        )

        fingerings = []
        av_audio_events = audio_events
        if self.lambda_vision > 0.0:
            fingerings, neck_anchors = _phase5_video_evidence(video)
            if neck_anchors:
                av_audio_events = apply_neck_anchor_priors(
                    audio_events,
                    neck_anchors,
                    self.cfg,
                )

        audio_vision = list(
            fuse(
                av_audio_events,
                fingerings,
                self.cfg,
                self.session,
                lambda_vision=self.lambda_vision,
            )
        )
        return audio_only, audio_vision


@cache
def _phase5_audio_backend():
    from tabvision.pipeline import _make_audio_backend

    return _make_audio_backend("highres")


@cache
def _phase5_video_backends():
    from tabvision.pipeline import (
        _make_fretboard_backend,
        _make_guitar_backend,
        _make_hand_backend,
    )

    return _make_guitar_backend(), _make_fretboard_backend(), _make_hand_backend()


@cache
def _phase5_audio_events(video: Path) -> tuple[AudioEvent, ...]:
    from tabvision.demux import demux
    from tabvision.types import SessionConfig

    demuxed = demux(video)
    return tuple(
        _phase5_audio_backend().transcribe(
            demuxed.wav,
            demuxed.sample_rate,
            SessionConfig(),
        )
    )


@cache
def _phase5_video_evidence(video: Path) -> tuple[tuple[FrameFingering, ...], tuple]:
    from tabvision.demux import demux
    from tabvision.pipeline import _run_video_stack
    from tabvision.types import GuitarConfig

    guitar_backend, fretboard_backend, hand_backend = _phase5_video_backends()
    demuxed = demux(video)
    result = _run_video_stack(
        demuxed.frame_iterator,
        stride=3,
        cfg=GuitarConfig(),
        guitar_backend=guitar_backend,
        fretboard_backend=fretboard_backend,
        hand_backend=hand_backend,
    )
    return tuple(result.fingerings), tuple(result.neck_anchors)


def _load_gold_tab_events(
    path: Path,
    *,
    bpm: float | int | None,
    video_duration_s: float,
) -> list[TabEvent]:
    """Parse legacy tab text and convert beat positions into real seconds."""
    text = path.read_text()
    parsed = _parse_ground_truth_tabs(text)
    return _gold_notes_to_tab_events(parsed, bpm=bpm, video_duration_s=video_duration_s)


def _parse_ground_truth_tabs(text: str) -> list[dict]:
    """Use v0's parser when importable; otherwise mirror its lightweight logic."""
    try:
        server_path = REPO_ROOT / "tabvision-server"
        if str(server_path) not in sys.path:
            sys.path.insert(0, str(server_path))
        from evaluate_transcription import parse_ground_truth_tabs
    except Exception:  # noqa: BLE001 — broad: optional dep, want graceful skip
        return _parse_legacy_tab_text(text)

    return parse_ground_truth_tabs(text)


def _parse_legacy_tab_text(text: str) -> list[dict]:
    """Mirror ``tabvision-server/evaluate_transcription.py`` tab parser."""
    string_map = {"e": 1, "B": 2, "G": 3, "D": 4, "A": 5, "E": 6}
    notes: list[dict] = []

    for line in text.strip().splitlines():
        if "|" not in line:
            continue

        parts = line.split("|")
        if len(parts) < 2:
            continue

        string_id = None
        for char in parts[0].strip():
            if char in string_map:
                string_id = string_map[char]
                break
        if string_id is None:
            continue

        content = "|".join(parts[1:])
        i = 0
        beat_position = 0.0
        while i < len(content):
            char = content[i]
            if char == "|":
                i += 1
            elif char == "-":
                beat_position += 0.25
                i += 1
            elif char.isdigit():
                fret_str = char
                if i + 1 < len(content) and content[i + 1].isdigit():
                    two_digit_fret = int(char + content[i + 1])
                    if two_digit_fret <= LEGACY_MAX_FRET:
                        fret_str = char + content[i + 1]
                        i += 1
                notes.append(
                    {
                        "string": string_id,
                        "fret": int(fret_str),
                        "beat": beat_position,
                    }
                )
                beat_position += 0.25
                i += 1
            elif char in ("X", "x", "/"):
                if char in ("X", "x"):
                    notes.append(
                        {
                            "string": string_id,
                            "fret": "X",
                            "beat": beat_position,
                        }
                    )
                beat_position += 0.25
                i += 1
            else:
                i += 1

    return sorted(notes, key=lambda n: (n["beat"], n["string"]))


def _gold_notes_to_tab_events(
    notes: Sequence[dict],
    *,
    bpm: float | int | None,
    video_duration_s: float,
) -> list[TabEvent]:
    """Convert legacy ``parse_ground_truth_tabs`` dicts to timed TabEvents."""
    pitched = [n for n in notes if n.get("fret") not in ("X", "x")]
    if not pitched:
        return []

    max_beat = max(float(n["beat"]) for n in pitched)
    if bpm is not None and float(bpm) > 0.0:
        beat_to_time = 60.0 / float(bpm)
    else:
        beat_to_time = video_duration_s / max_beat if max_beat > 0.0 else 1.0

    out: list[TabEvent] = []
    for note in pitched:
        fret = int(note["fret"])
        string_idx = 6 - int(note["string"])
        if string_idx < 0 or string_idx >= len(DEFAULT_TUNING_MIDI):
            continue
        out.append(
            TabEvent(
                onset_s=float(note["beat"]) * beat_to_time,
                duration_s=0.25,
                string_idx=string_idx,
                fret=fret,
                pitch_midi=DEFAULT_TUNING_MIDI[string_idx] + fret,
                confidence=1.0,
            )
        )
    return out


def _video_duration_s(video: Path) -> float:
    from tabvision.demux import _probe_metadata

    duration_s, _fps = _probe_metadata(video)
    return duration_s


def _align_gold_to_audio_only(
    *,
    audio_only: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    video_duration_s: float,
) -> tuple[list[TabEvent], float, int]:
    offset_s, matches = _find_best_pitch_offset(
        predicted=audio_only,
        gold=gold,
        video_duration_s=video_duration_s,
        tolerance_s=PHASE5_ALIGNMENT_TOLERANCE_S,
        step_s=PHASE5_ALIGNMENT_STEP_S,
    )
    return [replace(g, onset_s=g.onset_s + offset_s) for g in gold], offset_s, matches


def _find_best_pitch_offset(
    *,
    predicted: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    video_duration_s: float,
    tolerance_s: float,
    step_s: float,
) -> tuple[float, int]:
    """Search positive global offsets using pitch-only matches."""
    if not predicted or not gold:
        return 0.0, 0

    first_gold = min(g.onset_s for g in gold)
    last_gold = max(g.onset_s for g in gold)
    gt_span = max(0.0, last_gold - first_gold)
    pred_max = max((p.onset_s for p in predicted), default=video_duration_s)
    search_duration = max(video_duration_s, pred_max + tolerance_s)
    max_offset = max(0.0, search_duration - gt_span)

    n_steps = int(max_offset / step_s) + 1
    candidate_offsets = [i * step_s for i in range(n_steps)]
    best_offset = 0.0
    best_matches = -1
    best_error = float("inf")

    for offset in candidate_offsets:
        matches, error = _pitch_match_stats(predicted, gold, offset, tolerance_s)
        if matches > best_matches or (matches == best_matches and error < best_error):
            best_matches = matches
            best_offset = offset
            best_error = error

    return best_offset, max(best_matches, 0)


def _count_pitch_matches(
    predicted: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    offset_s: float,
    tolerance_s: float,
) -> int:
    return _pitch_match_stats(predicted, gold, offset_s, tolerance_s)[0]


def _pitch_match_stats(
    predicted: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    offset_s: float,
    tolerance_s: float,
) -> tuple[int, float]:
    gold_used = [False] * len(gold)
    matches = 0
    total_error = 0.0
    for pred in sorted(predicted, key=lambda t: t.onset_s):
        best_j = -1
        best_dt = tolerance_s + 1e-9
        for j, g in enumerate(gold):
            if gold_used[j] or pred.pitch_midi != g.pitch_midi:
                continue
            dt = abs(pred.onset_s - (g.onset_s + offset_s))
            if dt <= tolerance_s and dt < best_dt:
                best_j = j
                best_dt = dt
        if best_j >= 0:
            gold_used[best_j] = True
            matches += 1
            total_error += best_dt
    return matches, total_error if matches else float("inf")


def _run_pipeline(
    video: Path,
    *,
    lambda_vision: float,
    audio_backend_name: str = "highres",
) -> Sequence[TabEvent]:
    """Run audio + video + fusion end-to-end via :func:`tabvision.pipeline.run_pipeline`.

    Returns :class:`TabEvent` sequence directly comparable against
    :func:`_load_gold_tab_events` output.
    """
    from tabvision.pipeline import run_pipeline

    return run_pipeline(
        video,
        audio_backend_name=audio_backend_name,
        lambda_vision=lambda_vision,
        video_enabled=lambda_vision > 0.0,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _run_lambda_sweep() -> list[dict]:
    sweep_rows: list[dict] = []
    for lambda_vision in PHASE5_LAMBDA_SWEEP:
        rows = _collect_phase5_rows(lambda_vision=lambda_vision)
        ao_mean = _mean([r["ao_f1"] for r in rows])
        av_mean = _mean([r["av_f1"] for r in rows])
        sweep_rows.append(
            {
                "lambda": lambda_vision,
                "ao_mean": ao_mean,
                "av_mean": av_mean,
                "delta": av_mean - ao_mean,
                "chord_mean": _mean([r["chord_acc"] for r in rows]),
            }
        )
    return sweep_rows


def _write_report(
    *,
    rows: Sequence[dict],
    ao_mean: float,
    av_mean: float,
    delta: float,
    chord_mean: float,
    sweep_rows: list[dict],
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
        f"| Chord accuracy gate | "
        f"{'PASS' if chord_mean >= PHASE5_CHORD_ACCURACY_GATE else 'FAIL'} |",
        f"| Absolute Tab F1 gate | "
        f"{'PASS' if av_mean >= PHASE5_TAB_F1_ABSOLUTE_GATE else 'DEFER/FAIL'} |",
        "",
        "## Per-video",
        "",
        "| id | audio-only F1 | audio+vision F1 | delta | chord acc | "
        "offset | align matches | gold | ao notes | av notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['ao_f1']:.3f} | {r['av_f1']:.3f} | "
            f"{r['delta']:+.3f} | {r['chord_acc']:.3f} | "
            f"{r['offset_s']:.2f}s | {r['alignment_matches']} | "
            f"{r['gold_count']} | {r['ao_count']} | {r['av_count']} |"
        )

    if sweep_rows:
        lines.extend(
            [
                "",
                "## Diagnostic lambda sweep",
                "",
                "| lambda_vision | audio-only F1 | audio+vision F1 | delta | chord acc |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for r in sweep_rows:
            lines.append(
                f"| {r['lambda']:.1f} | {r['ao_mean']:.3f} | {r['av_mean']:.3f} | "
                f"{r['delta']:+.3f} | {r['chord_mean']:.3f} |"
            )
    out.write_text("\n".join(lines) + "\n")
