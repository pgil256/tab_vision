"""Audio-only GuitarSet eval helpers.

This module deliberately keeps import-time dependencies light. The raw
GuitarSet JAMS files carry string/fret labels, so they are the source of
truth for Tab F1; the derived Basic Pitch TFRecords are useful for
onset/pitch targets but do not retain string/fret.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabvision.errors import BackendError
from tabvision.eval.metrics import TabF1Result, tab_f1
from tabvision.fusion import fuse
from tabvision.fusion.position_prior import (
    PitchPositionPrior,
    apply_pitch_position_prior,
    learn_pitch_position_prior,
)
from tabvision.types import AudioBackend, AudioEvent, GuitarConfig, SessionConfig, TabEvent

DEFAULT_DATA_HOME = Path("~/mir_datasets/guitarset").expanduser()
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[3] / "tabvision-server" / "tools" / "outputs"
)
DEFAULT_VALIDATION_PLAYER = "05"
DEFAULT_ONSET_TOLERANCE_S = 0.05
DEFAULT_POSITION_PRIOR_ALPHA = 1.0
DEFAULT_POSITION_PRIOR_POWER = 2.0


@dataclass(frozen=True)
class EventF1Result:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def total_predicted(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def total_gold(self) -> int:
        return self.true_positives + self.false_negatives


@dataclass(frozen=True)
class AudioOnlyScore:
    onset: EventF1Result
    pitch: EventF1Result
    tab: TabF1Result
    decoded: list[TabEvent]


@dataclass(frozen=True)
class TrackEvalResult:
    track_id: str
    backend: str
    gold_notes: int
    audio_events: int
    decoded_events: int
    onset: EventF1Result
    pitch: EventF1Result
    tab: TabF1Result


@dataclass(frozen=True)
class EvalSummary:
    backend: str
    split: str
    position_prior: str
    n_tracks: int
    total_gold_notes: int
    total_audio_events: int
    mean_onset_f1: float
    mean_pitch_f1: float
    mean_tab_f1: float
    micro_onset: EventF1Result
    micro_pitch: EventF1Result
    micro_tab: TabF1Result


def parse_guitarset_jams(
    jams_path: str | Path,
    cfg: GuitarConfig | None = None,
) -> list[TabEvent]:
    """Parse GuitarSet note_midi annotations into v1 TabEvent gold notes.

    GuitarSet stores one ``note_midi`` annotation per string. Its
    ``data_source`` convention is already low-E to high-E as ``0..5``,
    matching v1 ``string_idx``. Fret is derived from MIDI minus the open
    string pitch so bent/float MIDI labels still land on the nearest fret.
    """
    if cfg is None:
        cfg = GuitarConfig()

    path = Path(jams_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    out: list[TabEvent] = []
    for ann in payload.get("annotations", []):
        if ann.get("namespace") != "note_midi":
            continue

        source = ann.get("annotation_metadata", {}).get("data_source")
        try:
            string_idx = int(source)
        except (TypeError, ValueError):
            continue
        if not 0 <= string_idx < cfg.n_strings:
            continue

        open_pitch = cfg.tuning_midi[string_idx]
        for row in ann.get("data") or []:
            try:
                onset_s = float(row["time"])
                duration_s = float(row["duration"])
                pitch_midi = int(round(float(row["value"])))
            except (KeyError, TypeError, ValueError):
                continue

            fret = pitch_midi - open_pitch
            if fret < cfg.capo or fret > cfg.max_fret:
                continue
            out.append(
                TabEvent(
                    onset_s=onset_s,
                    duration_s=max(0.0, duration_s),
                    string_idx=string_idx,
                    fret=fret,
                    pitch_midi=pitch_midi,
                    confidence=1.0,
                )
            )

    out.sort(key=lambda ev: (ev.onset_s, ev.string_idx, ev.fret))
    return out


def list_guitarset_track_ids(
    data_home: str | Path = DEFAULT_DATA_HOME,
    *,
    split: str = "validation",
    validation_player: str = DEFAULT_VALIDATION_PLAYER,
) -> list[str]:
    """List raw GuitarSet track ids for the requested split.

    The local TFRecord validation split is the held-out player ``05``.
    Reusing that convention avoids TensorFlow as a dependency for this v1
    eval while keeping it aligned with the existing Basic Pitch baseline.
    """
    root = Path(data_home)
    annotation_dir = root / "annotation"
    audio_dir = root / "audio_mono-mic"
    if not annotation_dir.is_dir() or not audio_dir.is_dir():
        return []

    track_ids = sorted(p.stem for p in annotation_dir.glob("*.jams"))
    available = [
        tid for tid in track_ids if (audio_dir / f"{tid}_mic.wav").is_file()
    ]
    if split == "all":
        return available
    if split == "validation":
        return [
            tid
            for tid in available
            if tid.split("_", 1)[0] == validation_player
        ]
    if split == "train":
        return [
            tid
            for tid in available
            if tid.split("_", 1)[0] != validation_player
        ]
    raise ValueError(f"unknown split: {split!r}; expected train, validation, or all")


def _score_event_f1(
    predicted: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    *,
    match_pitch: bool,
    onset_tolerance_s: float = DEFAULT_ONSET_TOLERANCE_S,
) -> EventF1Result:
    pred_sorted = sorted(predicted, key=lambda ev: ev.onset_s)
    gold_sorted = sorted(gold, key=lambda ev: ev.onset_s)
    gold_used = [False] * len(gold_sorted)
    tp = 0
    fp = 0

    for pred in pred_sorted:
        best_j = -1
        best_dt = onset_tolerance_s + 1e-9
        for j, ref in enumerate(gold_sorted):
            if gold_used[j]:
                continue
            if match_pitch and pred.pitch_midi != ref.pitch_midi:
                continue
            dt = abs(pred.onset_s - ref.onset_s)
            if dt <= onset_tolerance_s and dt < best_dt:
                best_j = j
                best_dt = dt
        if best_j >= 0:
            gold_used[best_j] = True
            tp += 1
        else:
            fp += 1

    fn = sum(1 for used in gold_used if not used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EventF1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def score_audio_only(
    audio_events: Sequence[AudioEvent],
    gold: Sequence[TabEvent],
    *,
    cfg: GuitarConfig | None = None,
    session: SessionConfig | None = None,
    onset_tolerance_s: float = DEFAULT_ONSET_TOLERANCE_S,
) -> AudioOnlyScore:
    if cfg is None:
        cfg = GuitarConfig()
    if session is None:
        session = SessionConfig()

    decoded = fuse(audio_events, [], cfg, session, lambda_vision=0.0)
    onset = _score_event_f1(
        decoded,
        gold,
        match_pitch=False,
        onset_tolerance_s=onset_tolerance_s,
    )
    pitch = _score_event_f1(
        decoded,
        gold,
        match_pitch=True,
        onset_tolerance_s=onset_tolerance_s,
    )
    tab = tab_f1(decoded, gold, onset_tolerance_s=onset_tolerance_s)
    return AudioOnlyScore(onset=onset, pitch=pitch, tab=tab, decoded=decoded)


def build_guitarset_position_prior(
    data_home: str | Path = DEFAULT_DATA_HOME,
    *,
    training_split: str = "train",
    validation_player: str = DEFAULT_VALIDATION_PLAYER,
    alpha: float = DEFAULT_POSITION_PRIOR_ALPHA,
    power: float = DEFAULT_POSITION_PRIOR_POWER,
    cfg: GuitarConfig | None = None,
) -> PitchPositionPrior:
    """Learn a pitch-position prior from raw GuitarSet tab annotations."""
    if cfg is None:
        cfg = GuitarConfig()

    examples: list[TabEvent] = []
    for track_id in list_guitarset_track_ids(
        data_home,
        split=training_split,
        validation_player=validation_player,
    ):
        jams_path = Path(data_home) / "annotation" / f"{track_id}.jams"
        examples.extend(parse_guitarset_jams(jams_path, cfg))
    if not examples:
        raise RuntimeError(
            f"no GuitarSet prior-training notes for split={training_split!r} under {data_home}"
        )
    return learn_pitch_position_prior(examples, cfg=cfg, alpha=alpha, power=power)


def load_mono_audio(audio_path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float32, preserving the original sample rate."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency readiness path
        raise RuntimeError("soundfile is required to load GuitarSet WAV files") from exc

    wav, sr = sf.read(str(audio_path), always_2d=False)
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if arr.ndim != 1:
        raise ValueError(f"expected mono/stereo audio, got shape {arr.shape}")
    return arr, int(sr)


def evaluate_track(
    track_id: str,
    backend_name: str,
    *,
    data_home: str | Path = DEFAULT_DATA_HOME,
    cfg: GuitarConfig | None = None,
    session: SessionConfig | None = None,
    position_prior: PitchPositionPrior | None = None,
    backend: AudioBackend | None = None,
) -> TrackEvalResult:
    if cfg is None:
        cfg = GuitarConfig()
    if session is None:
        session = SessionConfig()

    root = Path(data_home)
    audio_path = root / "audio_mono-mic" / f"{track_id}_mic.wav"
    jams_path = root / "annotation" / f"{track_id}.jams"
    if not audio_path.is_file():
        raise FileNotFoundError(f"missing GuitarSet audio: {audio_path}")
    if not jams_path.is_file():
        raise FileNotFoundError(f"missing GuitarSet JAMS: {jams_path}")

    gold = parse_guitarset_jams(jams_path, cfg)
    wav, sr = load_mono_audio(audio_path)

    if backend is None:
        from tabvision.audio.backend import make

        backend = make(backend_name)
    audio_events = list(backend.transcribe(wav, sr, session))
    if position_prior is not None:
        audio_events = apply_pitch_position_prior(audio_events, position_prior)
    scored = score_audio_only(audio_events, gold, cfg=cfg, session=session)
    return TrackEvalResult(
        track_id=track_id,
        backend=backend_name,
        gold_notes=len(gold),
        audio_events=len(audio_events),
        decoded_events=len(scored.decoded),
        onset=scored.onset,
        pitch=scored.pitch,
        tab=scored.tab,
    )


def summarize_results(
    results: Sequence[TrackEvalResult],
    *,
    backend: str,
    split: str,
    position_prior: str = "none",
) -> EvalSummary:
    total_gold = sum(r.gold_notes for r in results)
    total_audio = sum(r.audio_events for r in results)
    n_tracks = len(results)
    mean_onset = _mean(r.onset.f1 for r in results)
    mean_pitch = _mean(r.pitch.f1 for r in results)
    mean_tab = _mean(r.tab.f1 for r in results)
    return EvalSummary(
        backend=backend,
        split=split,
        position_prior=position_prior,
        n_tracks=n_tracks,
        total_gold_notes=total_gold,
        total_audio_events=total_audio,
        mean_onset_f1=mean_onset,
        mean_pitch_f1=mean_pitch,
        mean_tab_f1=mean_tab,
        micro_onset=_sum_event_f1(r.onset for r in results),
        micro_pitch=_sum_event_f1(r.pitch for r in results),
        micro_tab=_sum_tab_f1(r.tab for r in results),
    )


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _sum_event_f1(results: Iterable[EventF1Result]) -> EventF1Result:
    collected = list(results)
    tp = sum(r.true_positives for r in collected)
    fp = sum(r.false_positives for r in collected)
    fn = sum(r.false_negatives for r in collected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EventF1Result(precision, recall, f1, tp, fp, fn)


def _sum_tab_f1(results: Iterable[TabF1Result]) -> TabF1Result:
    collected = list(results)
    tp = sum(r.true_positives for r in collected)
    fp = sum(r.false_positives for r in collected)
    fn = sum(r.false_negatives for r in collected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TabF1Result(precision, recall, f1, tp, fp, fn)


def run_eval(
    *,
    backend_name: str,
    data_home: str | Path = DEFAULT_DATA_HOME,
    split: str = "validation",
    limit: int | None = None,
    validation_player: str = DEFAULT_VALIDATION_PLAYER,
    position_prior_name: str = "none",
    backend_kwargs: Mapping[str, object] | None = None,
) -> tuple[list[TrackEvalResult], EvalSummary]:
    track_ids = list_guitarset_track_ids(
        data_home,
        split=split,
        validation_player=validation_player,
    )
    if limit is not None:
        track_ids = track_ids[:limit]
    if not track_ids:
        raise RuntimeError(f"no GuitarSet tracks found for split={split!r} under {data_home}")

    position_prior: PitchPositionPrior | None = None
    if position_prior_name == "guitarset-train":
        position_prior = build_guitarset_position_prior(
            data_home,
            validation_player=validation_player,
        )
    elif position_prior_name != "none":
        raise ValueError(
            f"unknown position prior: {position_prior_name!r}; expected none or guitarset-train"
        )

    from tabvision.audio.backend import make

    backend = make(backend_name, **(dict(backend_kwargs or {})))
    results: list[TrackEvalResult] = []
    for track_id in track_ids:
        results.append(
            evaluate_track(
                track_id,
                backend_name,
                data_home=data_home,
                position_prior=position_prior,
                backend=backend,
            )
        )
    return results, summarize_results(
        results,
        backend=backend_name,
        split=split,
        position_prior=position_prior_name,
    )


def write_report(
    results: Sequence[TrackEvalResult],
    summary: EvalSummary,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    prior_slug = summary.position_prior.replace("_", "-")
    stem = f"guitarset_audio_eval-{summary.backend}-{summary.split}-{prior_slug}-{today}"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "track_id",
                "backend",
                "gold_notes",
                "audio_events",
                "decoded_events",
                "onset_f1",
                "pitch_f1",
                "tab_f1",
                "tab_tp",
                "tab_fp",
                "tab_fn",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.track_id,
                    r.backend,
                    r.gold_notes,
                    r.audio_events,
                    r.decoded_events,
                    f"{r.onset.f1:.6f}",
                    f"{r.pitch.f1:.6f}",
                    f"{r.tab.f1:.6f}",
                    r.tab.true_positives,
                    r.tab.false_positives,
                    r.tab.false_negatives,
                ]
            )

    lines = [
        f"# GuitarSet Audio Eval ({summary.backend})",
        "",
        f"Split: **{summary.split}**",
        f"Position prior: **{summary.position_prior}**",
        f"Tracks: **{summary.n_tracks}**",
        f"Gold notes: **{summary.total_gold_notes}**",
        f"Audio events: **{summary.total_audio_events}**",
        "",
        "## Aggregate",
        "",
        "| Metric | Mean F1 | Micro P | Micro R | Micro F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
        _metric_row("Onset", summary.mean_onset_f1, summary.micro_onset),
        _metric_row("Pitch", summary.mean_pitch_f1, summary.micro_pitch),
        _metric_row("Tab", summary.mean_tab_f1, summary.micro_tab),
        "",
        "## Per Track",
        "",
        "| Track | Gold | Audio | Decoded | Onset F1 | Pitch F1 | Tab F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| `{r.track_id}` | {r.gold_notes} | {r.audio_events} | "
            f"{r.decoded_events} | {r.onset.f1:.3f} | {r.pitch.f1:.3f} | "
            f"{r.tab.f1:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, csv_path


def _metric_row(name: str, mean_f1: float, result: EventF1Result | TabF1Result) -> str:
    return (
        f"| {name} | {mean_f1:.3f} | {result.precision:.3f} | "
        f"{result.recall:.3f} | {result.f1:.3f} |"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="highres",
        choices=["highres", "highres-fl", "basicpitch"],
    )
    parser.add_argument("--data-home", default=str(DEFAULT_DATA_HOME))
    parser.add_argument("--split", default="validation", choices=["validation", "train", "all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validation-player", default=DEFAULT_VALIDATION_PLAYER)
    parser.add_argument(
        "--device",
        default=None,
        help="optional backend device override, e.g. cuda for GPU runners",
    )
    parser.add_argument(
        "--position-prior",
        default="none",
        choices=["none", "guitarset-train"],
        help="optional pitch-to-string/fret prior attached before audio-only fusion",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    try:
        results, summary = run_eval(
            backend_name=args.backend,
            data_home=args.data_home,
            split=args.split,
            limit=args.limit,
            validation_player=args.validation_player,
            position_prior_name=args.position_prior,
            backend_kwargs={"device": args.device} if args.device else None,
        )
    except (BackendError, FileNotFoundError, RuntimeError) as exc:
        print(f"setup_blocker={exc}", file=sys.stderr)
        return 2
    md_path, csv_path = write_report(results, summary, output_dir=args.output_dir)
    print(f"tracks={summary.n_tracks}")
    print(f"onset_f1={summary.micro_onset.f1:.4f}")
    print(f"pitch_f1={summary.micro_pitch.f1:.4f}")
    print(f"tab_f1={summary.micro_tab.f1:.4f}")
    print(f"report={md_path}")
    print(f"csv={csv_path}")
    return 0


__all__ = [
    "AudioOnlyScore",
    "build_guitarset_position_prior",
    "EvalSummary",
    "EventF1Result",
    "TrackEvalResult",
    "evaluate_track",
    "list_guitarset_track_ids",
    "load_mono_audio",
    "main",
    "parse_guitarset_jams",
    "run_eval",
    "score_audio_only",
    "summarize_results",
    "write_report",
]
