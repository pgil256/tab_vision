"""v1.1 chunk-5 (video half): the REAL CV chain on GAPS, end-to-end.

Closes the loop on the GAPS acceptance corpus by adding the *video* string-axis
lever on top of the chunk-5 audio half. For each clean-12 clip:

    GAPS WAV  --highres-->  AudioEvent[]   (pitch + onset; cached)
    YouTube source video  --xcorr align-->  offset  (video_time = gold + offset)
    frames near each onset  -->  YOLO-OBB neck -> homography
                                 -> MediaPipe hands -> geometric fretting-hand
                                 -> fingertip_to_fret  -->  FrameFingering  (cached)
    fuse(events, fingerings) --> TabEvent[]  -->  Tab F1 vs GAPS gold

Reports three conditions per clip, exactly like the chunk-3 real-chain probe:
``audio-only`` / ``+real-video`` / ``+oracle-video`` (the ceiling). Runs two
audio sources:

  * ``gold``     — gold-pitch events (string/fret stripped). Isolates the
                   *string* axis, so ``+real`` vs ``+oracle`` (=1.0) measures the
                   GAPS video chain's pure string-resolution quality. This is the
                   frame the **0.94** v1.1 single-line video target lives in
                   (chunk-2/3 convention).
  * ``highres``  — the real audio pipeline (in-domain on GAPS). The honest
                   end-to-end number; capped by audio pitch/onset F1 (~0.95 on
                   clean-12), so even oracle video cannot reach 0.94 here.

The expensive layers (highres transcription ~70 s/clip; per-frame CV ~0.5 s) are
cached under ``--cache-dir`` keyed by clip + settings, so fusion/orientation/gate
tuning and report regeneration are cheap.

As of v1.1 chunk-6 (WS0) the per-frame CV cache is the *rich* v2 cache
(:mod:`scripts.eval.gaps_cv_cache`): instead of only the final ``FrameFingering``
it persists the raw intermediates each frame's fingering is built from — the YOLO
``OBBPredictions`` (nut/fret/neck anchors), the fitted ``Homography``, and the
selected fretting ``HandSample``. ``FrameFingering``s are reconstructed from that
cache via ``fingering_from_raw``, so per-clip board calibration, orientation, and
posterior changes (chunk-6 WS1/WS2/WS4) become re-runnable from cache rather than
needing a full MediaPipe/YOLO re-run.

Net-new acquisition + alignment live in ``scripts.acquire.gaps_video``; the
fusion / vision-evidence / scoring helpers are reused verbatim from the v1
package and the chunk-2/3 probes. GAPS is non-commercial offline-eval-only:
media stays local and is never committed or redistributed.

Reproduce::

    cd tabvision
    export TABVISION_DATA_ROOT=~/.tabvision/data
    export PATH=~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin:$PATH
    python -m scripts.acquire.gaps_video --download --clips clean12 \\
        --ffmpeg-location ~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin
    python -m scripts.eval.v1_1_gaps_video_chain_probe \\
        --checkpoint ~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt

This writes the raw auto-report ``v1_1_gaps_video_chain_auto_<date>.md``; the
curated summary (``v1_1_gaps_video_chain_<date>.md``) is hand-authored from it,
matching the chunk-5 report convention.
"""

from __future__ import annotations

import argparse
import datetime
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.acquire.gaps_video import CLEAN_12, AlignmentResult, estimate_offset
from scripts.eval.gaps_cv_cache import (
    CalibrateFn,
    RawFrameCV,
    fingering_from_raw,
    make_board_calibrator,
    make_fret_xs_calibrator,
    needed_frames,
    rawcv_cache_path,
)
from scripts.eval.v1_1_kaggle_oracle_probe import _events_from_gold, _oracle_fingerings
from scripts.eval.v1_1_real_chain_probe import _select_fretting_hand_geometric
from tabvision.demux import _frame_iterator, _probe_metadata
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.error_decomposition import (
    ErrorDecomposition,
    aggregate_decompositions,
    decompose_errors,
)
from tabvision.eval.metrics import tab_f1
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.fusion import fuse
from tabvision.fusion.vision_evidence import (
    ORIENTATION_BY_NAME,
    Orientation,
    choose_orientation,
    combine_fingerings,
    empty_fingering,
    gate_fingering_to_audio,
    orient_fingering,
)
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig, SessionConfig

_TARGET_094 = 0.94


@dataclass
class GateParams:
    vote_frames: int = 3
    vote_window_s: float = 0.06
    min_homography_confidence: float = 0.1
    min_candidate_support: float = 0.02
    min_best_ratio: float = 1.2
    # chunk-3 §5.3 no-regression value: below this clip coverage, the video
    # evidence is dropped and the clip falls back to audio-only. 0.71 holds the
    # no-regression guarantee on GAPS footage (a looser 0.5 leaks corrupting
    # evidence and regresses ~0.05 — see the 2026-06-22 report §3 sweep).
    min_clip_coverage: float = 0.71
    gate: bool = True


@dataclass
class SourceScore:
    """Tab F1 conditions for one clip + one audio source.

    ``real_auto`` is the honest number (orientation auto-selected per clip);
    ``real_best`` is the best over the four fixed orientations — a diagnostic
    *ceiling on orientation selection* that separates "auto picked the wrong
    flip" from "the CV chain can't resolve strings here".
    """

    audio_only: float
    real_auto: float
    real_best: float
    oracle: float
    auto_orient: str
    best_orient: str
    kept_auto: int
    coverage_auto: float
    decomp: ErrorDecomposition  # decomposition of the +real(auto) condition


@dataclass
class ClipResult:
    stem: str
    n_gold: int
    align: AlignmentResult
    scores: dict[str, SourceScore]


# --------------------------------------------------------------------------- #
# Cached expensive layers
# --------------------------------------------------------------------------- #
def _audio_events(
    stem: str,
    wav_path: Path,
    cache_dir: Path,
) -> list[AudioEvent]:
    """highres AudioEvents for a clip (pickle-cached; ~70 s/clip cold)."""
    cache = cache_dir / f"{stem}.audio.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    import soundfile as sf

    from tabvision.audio.highres import HighResBackend

    wav, sr = sf.read(str(wav_path), always_2d=False)
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    events = list(HighResBackend().transcribe(arr, int(sr), SessionConfig()))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(events, fh)
    return events


def _offset(stem: str, wav_path: Path, video_path: Path, cache_dir: Path) -> AlignmentResult:
    """Recovered video<->audio offset for a clip (pickle-cached)."""
    cache = cache_dir / f"{stem}.offset.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    res = estimate_offset(wav_path, video_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(res, fh)
    return res


def _raw_cv_for_frame(
    frame: np.ndarray,
    yolo,  # noqa: ANN001 - YoloOBBBackend
    landmarker,  # noqa: ANN001 - mediapipe HandLandmarker
) -> RawFrameCV | None:
    """Run the CV stack on one frame -> raw intermediates, or None.

    Captures the YOLO ``OBBPredictions`` + fitted ``Homography`` + selected
    fretting ``HandSample`` — everything ``compute_fingering`` consumes — so the
    fingering can be rebuilt downstream without re-running the models (the WS0
    enabler). The None-return conditions exactly mirror the chunk-5 chain (no
    neck/low-confidence homography, no hands, no fretting hand), so the derived
    fingerings reproduce the chunk-5 cache.
    """
    import cv2
    import mediapipe as mp

    from tabvision.video.fretboard.keypoint import predictions_to_homography
    from tabvision.video.hand.mediapipe_backend import _build_hand_sample

    preds = yolo.predict_all(frame)
    homography = predictions_to_homography(preds)
    if homography.confidence <= 0.0:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    h, w = frame.shape[:2]
    hands = [
        _build_hand_sample(lm, hd, frame_width=w, frame_height=h)
        for lm, hd in zip(res.hand_landmarks, res.handedness, strict=False)
    ]
    hand = _select_fretting_hand_geometric(hands, np.linalg.inv(homography.H))
    if hand is None:
        return None
    return RawFrameCV(preds=preds, homography=homography, hand=hand)


def _raw_cv_cache(
    stem: str,
    video_path: Path,
    fps: float,
    needed: set[int],
    yolo,  # noqa: ANN001 - YoloOBBBackend
    landmarker,  # noqa: ANN001
    cache_dir: Path,
    conf: float,
) -> dict[int, RawFrameCV | None]:
    """Raw per-frame CV intermediates for ``needed`` indices (incremental pickle).

    The v2 rich cache (``{stem}.rawcv.c{conf}.pkl``) — see
    :mod:`scripts.eval.gaps_cv_cache`. Decoding is the expensive part, so frames
    are accumulated incrementally and only missing indices are (re)computed.
    """
    cache_path = rawcv_cache_path(cache_dir, stem, conf)
    cache: dict[int, RawFrameCV | None] = {}
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
    missing = sorted(fi for fi in needed if fi not in cache)
    if missing:
        target = set(missing)
        max_fi = missing[-1]
        for fi, (_t, frame) in enumerate(_frame_iterator(video_path, fps)):
            if fi > max_fi:
                break
            if fi in target:
                cache[fi] = _raw_cv_for_frame(frame, yolo, landmarker)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(cache, fh)
    return cache


# --------------------------------------------------------------------------- #
# Fusion-side assembly (cheap; tunable without re-running CV)
# --------------------------------------------------------------------------- #
def _assemble_fingerings(
    events: list[AudioEvent],
    per_onset: dict[int, list[int]],
    raw_cache: dict[int, FrameFingering | None],
    cfg: GuitarConfig,
    *,
    orientation: Orientation | None,
    params: GateParams,
) -> tuple[list[FrameFingering], Orientation, int, float]:
    """One robust FrameFingering per event (orient -> vote -> gate), like chunk-3."""
    raw_by_event: list[list[FrameFingering]] = [
        [raw_cache[fi] for fi in per_onset[i] if raw_cache.get(fi) is not None]
        for i in range(len(events))
    ]
    chosen = orientation
    if chosen is None:
        chosen, _scores = choose_orientation(raw_by_event, events, cfg)

    out: list[FrameFingering] = []
    kept = 0
    for event, raw in zip(events, raw_by_event, strict=False):
        if not raw:
            out.append(empty_fingering(event.onset_s, cfg))
            continue
        oriented = [orient_fingering(f, chosen) for f in raw]
        voted = combine_fingerings(oriented, cfg, t=event.onset_s)
        if params.gate:
            voted = gate_fingering_to_audio(
                event,
                voted,
                cfg,
                min_homography_confidence=params.min_homography_confidence,
                min_candidate_support=params.min_candidate_support,
                min_best_ratio=params.min_best_ratio,
            )
        if voted.homography_confidence > 0.0 and (voted.finger_pos_logits != 0).any():
            kept += 1
        out.append(voted)
    coverage = kept / len(events) if events else 0.0
    if params.gate and coverage < params.min_clip_coverage:
        out = [empty_fingering(event.onset_s, cfg) for event in events]
        kept = 0
    return out, chosen, kept, coverage


def _score_clip(
    stem: str,
    data_root: Path,
    video_cache: Path,
    cfg: GuitarConfig,
    yolo,  # noqa: ANN001 - YoloOBBBackend
    landmarker,  # noqa: ANN001
    cache_dir: Path,
    *,
    conf: float,
    orientation: Orientation | None,
    params: GateParams,
    audio_sources: tuple[str, ...],
    calibrate: CalibrateFn | None = None,
) -> ClipResult | None:
    gaps = data_root / "gaps"
    xml = gaps / "musicxml" / f"{stem}.xml"
    wav = gaps / "audio" / f"{stem}.wav"
    vid = video_cache / f"{stem}.mp4"
    if not (xml.exists() and wav.exists() and vid.exists()):
        print(f"  [skip] {stem}: missing media")
        return None

    gold = parse_gaps(xml)
    if not gold:
        print(f"  [skip] {stem}: empty gold")
        return None

    align = _offset(stem, wav, vid, cache_dir)
    _dur, fps = _probe_metadata(vid)

    # Build per-source events; union the needed frames so CV runs once.
    src_events: dict[str, list[AudioEvent]] = {}
    if "gold" in audio_sources:
        src_events["gold"] = _events_from_gold(gold)
    if "highres" in audio_sources:
        src_events["highres"] = _audio_events(stem, wav, cache_dir)

    all_onsets = sorted({ev.onset_s for evs in src_events.values() for ev in evs})
    needed, _ = needed_frames(
        all_onsets,
        align.offset_s,
        fps,
        window_s=params.vote_window_s,
        max_frames=params.vote_frames,
    )
    rawcv = _raw_cv_cache(stem, vid, fps, needed, yolo, landmarker, cache_dir, conf)
    # Reconstruct the chunk-5 FrameFingerings from the rich cache. This is the
    # WS0 split: the fit/orient/project layer now runs from cached intermediates,
    # so later geometry/posterior changes re-derive ``raw_cache`` without CV.
    raw_cache: dict[int, FrameFingering | None] = {
        fi: fingering_from_raw(rec, cfg, t=fi / fps, calibrate=calibrate)
        for fi, rec in rawcv.items()
    }

    oracle = _oracle_fingerings(gold, cfg)
    scores: dict[str, SourceScore] = {}
    for src, events in src_events.items():
        _, per_onset = needed_frames(
            [e.onset_s for e in events],
            align.offset_s,
            fps,
            window_s=params.vote_window_s,
            max_frames=params.vote_frames,
        )
        audio_only = tab_f1(fuse(events, [], cfg), gold).f1
        oracle_f1 = tab_f1(fuse(events, oracle, cfg), gold).f1

        # Honest condition: orientation auto-selected per clip (unless the user
        # pinned one via --orientation).
        real_auto, auto_ori, kept_auto, cov_auto = _assemble_fingerings(
            events, per_onset, raw_cache, cfg, orientation=orientation, params=params
        )
        auto_tabs = fuse(events, real_auto, cfg)
        f_auto = tab_f1(auto_tabs, gold).f1

        # Diagnostic ceiling: best over the four fixed orientations (cheap —
        # frames are cached; only fusion re-runs).
        f_best, best_name = f_auto, auto_ori.name
        for name, ori in ORIENTATION_BY_NAME.items():
            real_o, _, _, _ = _assemble_fingerings(
                events, per_onset, raw_cache, cfg, orientation=ori, params=params
            )
            f_o = tab_f1(fuse(events, real_o, cfg), gold).f1
            if f_o > f_best:
                f_best, best_name = f_o, name

        scores[src] = SourceScore(
            audio_only=audio_only,
            real_auto=f_auto,
            real_best=f_best,
            oracle=oracle_f1,
            auto_orient=auto_ori.name,
            best_orient=best_name,
            kept_auto=kept_auto,
            coverage_auto=cov_auto,
            decomp=decompose_errors(auto_tabs, gold),
        )

    return ClipResult(stem=stem, n_gold=len(gold), align=align, scores=scores)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt_align_table(results: list[ClipResult]) -> list[str]:
    lines = [
        "| Clip | Gold | offset (s) | xcorr peak ratio | wav dur | vid dur |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        a = r.align
        lines.append(
            f"| {r.stem} | {r.n_gold} | {a.offset_s:+.3f} | {a.peak_ratio:.2f} | "
            f"{a.audio_duration_s:.1f} | {a.video_duration_s:.1f} |"
        )
    return lines


def _fmt_source_table(results: list[ClipResult], src: str) -> list[str]:
    lines = [
        f"#### Audio source: `{src}`",
        "",
        "| Clip | audio-only | +real (auto) | +real (best-orient) | +oracle | "
        "lift (auto) | auto/best orient | kept/cov |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        s = r.scores[src]
        lines.append(
            f"| {r.stem} | {s.audio_only:.4f} | {s.real_auto:.4f} | {s.real_best:.4f} | "
            f"{s.oracle:.4f} | {s.real_auto - s.audio_only:+.4f} | "
            f"{s.auto_orient}/{s.best_orient} | {s.kept_auto}/{s.coverage_auto:.2f} |"
        )
    ao_list = [r.scores[src].audio_only for r in results]
    auto_list = [r.scores[src].real_auto for r in results]
    best_list = [r.scores[src].real_best for r in results]
    oracle_list = [r.scores[src].oracle for r in results]
    ci = bootstrap_ci(auto_list)
    mean_ao = float(np.mean(ao_list))
    mean_auto = float(np.mean(auto_list))
    mean_best = float(np.mean(best_list))
    mean_oracle = float(np.mean(oracle_list))
    lines.append(
        f"| **mean** | **{mean_ao:.4f}** | **{mean_auto:.4f}** | **{mean_best:.4f}** | "
        f"**{mean_oracle:.4f}** | **{mean_auto - mean_ao:+.4f}** | — | — |"
    )
    lines.append("")
    lines.append(
        f"+real (auto-orientation) Tab F1: mean **{mean_auto:.4f}**, "
        f"bootstrap lower-95 **{ci.lower:.4f}** (n={ci.n_observations}). "
        f"Best-fixed-orientation ceiling: mean **{mean_best:.4f}**."
    )
    if src == "gold":
        verdict = (
            "PASS"
            if ci.lower >= _TARGET_094
            else ("meets bar (mean)" if mean_auto >= _TARGET_094 else "below 0.94 bar")
        )
        lines.append(f"vs the 0.94 single-line video target (auto): **{verdict}**.")
    # decomposition (the +real auto condition)
    agg = aggregate_decompositions([r.scores[src].decomp for r in results])
    shares = agg.share_of_loss()
    lines.append("")
    lines.append("Error decomposition (+real, auto-orientation):")
    lines.append("")
    lines.append("| Bucket | Count | Share of loss |")
    lines.append("|---|---:|---:|")
    for col in (
        "correct",
        "wrong_position_same_pitch",
        "pitch_off",
        "timing_only",
        "missed_onset",
        "extra_detection",
    ):
        count = getattr(agg, col)
        if col == "correct":
            lines.append(f"| {col} | {count} | — |")
        else:
            lines.append(f"| {col} | {count} | {shares[col] * 100:.1f}% |")
    lines.append("")
    return lines


def _regressions(
    results: list[ClipResult], src: str, *, tol: float = 1e-9
) -> list[tuple[str, float]]:
    """Clips where +real(auto) drops below audio-only for ``src`` (per-clip).

    The chunk-6 hard invariant: video must stay additive — any clip with weak or
    wrong evidence must fall back to audio-only *exactly*, so ``real_auto`` must
    be ``>= audio_only`` on every clip. ``tol`` absorbs float noise. Returns
    ``(stem, delta)`` for each regressing clip (``delta < 0``).
    """
    out = []
    for r in results:
        s = r.scores[src]
        delta = s.real_auto - s.audio_only
        if delta < -tol:
            out.append((r.stem, delta))
    return out


def _fmt_no_regression(results: list[ClipResult], src: str) -> list[str]:
    regs = _regressions(results, src)
    if not regs:
        return [
            f"No-regression invariant (`{src}`): **HOLDS** (+real(auto) ≥ audio-only on all "
            f"{len(results)} clips)."
        ]
    detail = ", ".join(f"{stem} ({delta:+.4f})" for stem, delta in regs)
    return [
        f"No-regression invariant (`{src}`): **VIOLATED** on {len(regs)}/{len(results)} "
        f"clip(s): {detail}."
    ]


def _write_report(
    results: list[ClipResult],
    output: Path,
    *,
    audio_sources: tuple[str, ...],
    params: GateParams,
) -> None:
    today = datetime.date.today().isoformat()
    lines = [
        "# v1.1 chunk-5 — GAPS video real-chain (video half)",
        "",
        f"**Date:** {today}",
        "**Branch:** `v1.1/oracle-string-resolution`",
        "**Status:** Video half — the chunk-3 confidence-gated CV chain "
        "(YOLO-OBB + MediaPipe + homography + fusion) run over the GAPS clean-12, "
        "with source video acquired via yt-dlp and aligned to the GAPS audio crop "
        "by onset-envelope cross-correlation.",
        "",
        "GAPS is non-commercial, offline-eval-only: source media stays local and is "
        "never committed or redistributed. Clean-12 = the >=80% gold-coverage "
        "standard-tuning test clips from the chunk-5 audio half.",
        "",
        "## 1. Video<->audio crop-offset alignment (net-new)",
        "",
        "GAPS audio is a crop of the YouTube upload, so frame time != gold time. "
        "Per clip, the offset is recovered by cross-correlating onset-strength "
        "envelopes (`video_time = gold_onset + offset`); the peak ratio is the "
        "correlation peak over the best competitor outside a +-5-frame guard band.",
        "",
        *_fmt_align_table(results),
        "",
        f"Offsets are sub-frame (~|offset| < {1.0 / 24:.3f} s, one frame at 24 fps) and "
        "corroborated by the near-equal wav/video durations — the upload is "
        "essentially the GAPS crop.",
        "",
        "## 2. Video-assisted Tab F1",
        "",
        "Conditions per clip: `audio-only` (strings from the playability prior), "
        "`+real (auto)` (CV chain resolves the string; orientation auto-selected — "
        "the honest number), `+real (best-orient)` (best over the 4 fixed "
        "orientations — a diagnostic ceiling on orientation selection), `+oracle` "
        "(gold strings — the absolute ceiling). "
        f"Gate params: vote {params.vote_frames} frames / +-{params.vote_window_s * 1000:.0f} ms, "
        f"min coverage {params.min_clip_coverage:.2f}.",
        "",
    ]
    for src in audio_sources:
        lines.extend(_fmt_source_table(results, src))
        lines.append("")
        lines.extend(_fmt_no_regression(results, src))
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_clips(spec: str) -> tuple[str, ...]:
    if spec == "clean12":
        return CLEAN_12
    return tuple(s.strip() for s in spec.split(",") if s.strip())


def _build_landmarker():  # noqa: ANN202
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model = os.path.expanduser("~/.mediapipe/models/hand_landmarker.task")
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model), num_hands=2
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision" / "data")
    ap.add_argument("--video-cache", type=Path, default=Path.home() / ".tabvision/cache/gaps_video")
    ap.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".tabvision/cache/gaps_video_chain"
    )
    ap.add_argument("--clips", default="clean12", help="'clean12' or comma-separated stems")
    ap.add_argument("--checkpoint", type=Path, default=None, help="YOLO-OBB checkpoint")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--audio-source", choices=["gold", "highres", "both"], default="both")
    ap.add_argument("--orientation", choices=["auto", *ORIENTATION_BY_NAME.keys()], default="auto")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--vote-frames", type=int, default=3)
    ap.add_argument("--vote-window-s", type=float, default=0.06)
    ap.add_argument("--min-homography-conf", type=float, default=0.1)
    ap.add_argument("--min-candidate-support", type=float, default=0.02)
    ap.add_argument("--min-best-ratio", type=float, default=1.2)
    ap.add_argument("--min-clip-coverage", type=float, default=0.71)
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="apply chunk-6 WS1 per-clip nonlinear fret-map calibration (cache-only)",
    )
    ap.add_argument(
        "--calibrate-board",
        action="store_true",
        help="apply chunk-6 WS2 board calibration: nut-axis homography re-fit + fret map",
    )
    ap.add_argument("--limit", type=int, default=None, help="first N clips (smoke)")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("..")
        / "docs"
        / "EVAL_REPORTS"
        / f"v1_1_gaps_video_chain_auto_{datetime.date.today().isoformat()}.md",
    )
    args = ap.parse_args(argv)

    cfg = GuitarConfig()
    params = GateParams(
        vote_frames=args.vote_frames,
        vote_window_s=args.vote_window_s,
        min_homography_confidence=args.min_homography_conf,
        min_candidate_support=args.min_candidate_support,
        min_best_ratio=args.min_best_ratio,
        min_clip_coverage=args.min_clip_coverage,
        gate=not args.no_gate,
    )
    orientation = None if args.orientation == "auto" else ORIENTATION_BY_NAME[args.orientation]
    if args.calibrate_board:
        calibrate: CalibrateFn | None = make_board_calibrator(cfg)
    elif args.calibrate:
        calibrate = make_fret_xs_calibrator(cfg)
    else:
        calibrate = None
    audio_sources: tuple[str, ...] = (
        ("gold", "highres") if args.audio_source == "both" else (args.audio_source,)
    )

    from tabvision.video.guitar.yolo_backend import YoloOBBBackend

    ckpt = args.checkpoint or os.environ.get("TABVISION_GUITAR_YOLO_CHECKPOINT")
    yolo = YoloOBBBackend(checkpoint_path=ckpt, conf=args.conf, device="cpu")
    landmarker = _build_landmarker()

    clips = _resolve_clips(args.clips)
    if args.limit is not None:
        clips = clips[: args.limit]

    _mode = (
        "WS2 board (nut-axis + fret-map)"
        if args.calibrate_board
        else ("WS1 calibrated (nonlinear fret-map)" if args.calibrate else "baseline (uniform)")
    )
    print(f"mode: {_mode}")
    print(f"{'clip':>12} {'gold':>5} {'offset':>7}  src:audio/+real(auto)/+best/+oracle")
    results: list[ClipResult] = []
    for stem in clips:
        res = _score_clip(
            stem,
            args.data_root,
            args.video_cache,
            cfg,
            yolo,
            landmarker,
            args.cache_dir,
            conf=args.conf,
            orientation=orientation,
            params=params,
            audio_sources=audio_sources,
            calibrate=calibrate,
        )
        if res is None:
            continue
        results.append(res)
        summary = "  ".join(
            f"{s}:{res.scores[s].audio_only:.3f}/{res.scores[s].real_auto:.3f}/"
            f"{res.scores[s].real_best:.3f}/{res.scores[s].oracle:.3f}"
            for s in audio_sources
        )
        print(f"{res.stem:>12} {res.n_gold:>5} {res.align.offset_s:>+7.3f}  {summary}")

    if not results:
        print("no clips scored")
        return 1

    print(f"\n{'=' * 60}")
    for src in audio_sources:
        ao = [r.scores[src].audio_only for r in results]
        auto = [r.scores[src].real_auto for r in results]
        best = [r.scores[src].real_best for r in results]
        ci = bootstrap_ci(auto)
        tag = "  <-- vs 0.94" if src == "gold" else ""
        print(
            f"[{src}] audio-only {np.mean(ao):.4f} -> +real(auto) {np.mean(auto):.4f} "
            f"(lower-95 {ci.lower:.4f})  best-orient {np.mean(best):.4f}{tag}"
        )
        regs = _regressions(results, src)
        if regs:
            detail = ", ".join(f"{stem}({delta:+.4f})" for stem, delta in regs)
            print(f"    NO-REGRESSION VIOLATED on {len(regs)}/{len(results)}: {detail}")
        else:
            print(f"    no-regression holds on all {len(results)} clips")

    _write_report(results, args.output, audio_sources=audio_sources, params=params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
