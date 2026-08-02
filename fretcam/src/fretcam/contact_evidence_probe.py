r"""Measure the evidence the FretCam bridge discards.

The shipped `--video-backend fretcam` route ships one thing per accepted
frame: a coarse fret window ``{0} u [N-1, N+4]``, gated on the position
estimator reaching ``locked``/``holding``.  The same detection chain also
computes per-finger :class:`~fretcam.detection.FingerContact` records carrying
a fret *and* a string, and those are dropped on the floor.

This probe quantifies what that costs, on two axes that the 2026-07-24
end-to-end result could not separate:

* **coverage** — how often does each evidence type exist at all, at the
  causal pre-onset instant the bridge reads?
* **strength** — when it exists, how much does it discriminate the gold
  string/fret from the one the audio decoder actually chose?

Strength is reported as a likelihood ratio ``P(hit | gold) / P(hit | audio's
wrong choice)`` over audio-wrong ambiguous notes, and as its log in nats, so it
is directly comparable with the one-nat cap in
:mod:`tabvision.fusion.position_window_prior`.

Every hypothesis is read with the *shipped* causal policy — the latest valid
frame in the 150 ms lookback ending 30 ms before onset — so the comparison
against the current bridge is apples-to-apples.  The probe also counts the
missed-onset population, which a prior-reweighting bridge cannot touch by
construction.

Two stages, so the expensive half is paid once:

    stage 1  run the DetectionChain over a clip and cache every frame's
             contacts plus the estimator state, on the gold clock.
    stage 2  score hypotheses against gold, offline and repeatable.

No inference, download, training, or policy tuning enters the shipped
pipeline.  Development clips (clean-12) only; the source-disjoint split is not
opened.  Reproduce from ``tabvision/`` with the sibling FretCam package
installed in the same environment:

    $env:PYTHONPATH = ((Resolve-Path '../fretcam/src').Path + ';' + (Get-Location).Path)
    .\.venv\Scripts\python -m fretcam.contact_evidence_probe --flip
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import pickle
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from fretcam.detection import DetectionChain
from fretcam.position import EstimatorConfig, PositionEstimator
from fretcam.processing import build_hand_search_hint
from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.fretcam_end_to_end import load_tab_events, production_tab_cache_path
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.position_window_prior import (
    MIN_POSITION_OBSERVATION_CONFIDENCE,
    POSITION_OBSERVATION_LEAD_S,
    POSITION_OBSERVATION_LOOKBACK_S,
    VALID_POSITION_OBSERVATION_STATES,
)
from tabvision.types import GuitarConfig, TabEvent

DATA_ROOT = Path.home() / ".tabvision" / "data"
VIDEO_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video"
OFFSET_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video_chain"
PROD_TAB_CACHE = Path.home() / ".tabvision" / "cache" / "v1_1_second_corpus"
TRACE_CACHE = Path.home() / ".tabvision" / "cache" / "fretcam_contact_trace"

BACKEND_NAME = "highres"
ONSET_TOLERANCE_S = 0.05
TRACE_VERSION = 1
DEFAULT_STRIDE = 3
"""Matches ``tabvision.pipeline.run_pipeline``'s production ``video_stride``."""

ANALYZER_MAX_WIDTH = 640
ANALYZER_MAX_HEIGHT = 480
"""Mirror ``FretCamPositionAnalyzer``'s shipped frame downscale."""


@dataclass
class FrameRecord:
    """One traced frame: what the estimator said and what the fingers did."""

    t_gold: float
    state: str
    est_conf: float
    position: int | None
    contacts: tuple[tuple[int, int, bool, float, float, bool], ...]
    # (string_1based, fret, pressing, pressing_score, quality, barre)


# ---------------------------------------------------------------------------
# stage 1 — trace capture
# ---------------------------------------------------------------------------


def capture_trace(
    stem: str,
    *,
    stride: int,
    refresh: bool,
    max_width: int = ANALYZER_MAX_WIDTH,
    max_height: int = ANALYZER_MAX_HEIGHT,
) -> list[FrameRecord]:
    """Run the detection chain once and cache the per-frame record.

    ``max_width``/``max_height`` default to the analyzer's shipped downscale.
    Raising them tests whether the across-string axis is resolution-limited;
    the values participate in the cache key so arms never collide.
    """
    TRACE_CACHE.mkdir(parents=True, exist_ok=True)
    shipped_size = (max_width, max_height) == (ANALYZER_MAX_WIDTH, ANALYZER_MAX_HEIGHT)
    suffix = "" if shipped_size else f".r{max_width}x{max_height}"
    cache_path = TRACE_CACHE / f"{stem}.v{TRACE_VERSION}.s{stride}{suffix}.pkl"
    if cache_path.is_file() and not refresh:
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    video_path = VIDEO_CACHE / f"{stem}.mp4"
    offset_path = OFFSET_CACHE / f"{stem}.offset.pkl"
    for path in (video_path, offset_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with offset_path.open("rb") as handle:
        offset_s = float(pickle.load(handle).offset_s)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    cfg = GuitarConfig()
    chain = DetectionChain(
        guitar_config=cfg,
        detector_hz=2.0,
        background_detector=False,
        crop_hand=True,
    )
    estimator = PositionEstimator(EstimatorConfig(max_fret=cfg.max_fret))
    records: list[FrameRecord] = []
    started = time.monotonic()
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride:
                frame_index += 1
                continue
            t_video = frame_index / fps
            frame_index += 1

            height, width = frame.shape[:2]
            scale = min(1.0, max_width / max(width, 1), max_height / max(height, 1))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            detection = chain.process_frame(frame, timestamp_s=t_video)

            # Mirror FretCamPositionAnalyzer._position_observation exactly.
            if detection.composite_available or detection.position_fret is not None:
                observed_fret = detection.position_fret
                confidence = detection.observation_confidence
            else:
                observed_fret = detection.index_fret
                confidence = (
                    detection.anchor.confidence if detection.neck_locked else 0.0
                )
            estimate = estimator.update(
                index_fret=observed_fret,
                vision_confidence=confidence,
                timestamp_s=detection.timestamp_s,
            )
            setter = getattr(chain, "set_hand_search_hint", None)
            if setter is not None:
                setter(build_hand_search_hint(detection, estimate))

            records.append(
                FrameRecord(
                    t_gold=float(detection.timestamp_s) - offset_s,
                    state=str(estimate.state),
                    est_conf=float(estimate.confidence),
                    position=(
                        None if estimate.position is None else int(estimate.position)
                    ),
                    contacts=tuple(
                        (
                            int(contact.string),
                            int(contact.fret),
                            bool(contact.pressing),
                            float(contact.pressing_score),
                            float(contact.quality),
                            bool(contact.barre),
                        )
                        for contact in detection.finger_contacts
                        if contact.string is not None and contact.visible
                    ),
                )
            )
    finally:
        chain.close()
        capture.release()

    print(
        f"  [trace] {stem}: {len(records)} frames in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    with cache_path.open("wb") as handle:
        pickle.dump(records, handle)
    return records


# ---------------------------------------------------------------------------
# stage 2 — evidence scoring
# ---------------------------------------------------------------------------


def select_frame(
    records: Sequence[FrameRecord],
    times: Sequence[float],
    onset_s: float,
    *,
    require_accepted: bool,
) -> FrameRecord | None:
    """Latest frame in the shipped causal lookback ending ``onset - lead``."""
    target = onset_s - POSITION_OBSERVATION_LEAD_S
    earliest = target - POSITION_OBSERVATION_LOOKBACK_S
    for index in range(bisect.bisect_right(times, target) - 1, -1, -1):
        record = records[index]
        if record.t_gold < earliest:
            return None
        if not require_accepted:
            return record
        if (
            record.state in VALID_POSITION_OBSERVATION_STATES
            and record.est_conf >= MIN_POSITION_OBSERVATION_CONFIDENCE
        ):
            return record
    return None


def select_window(
    records: Sequence[FrameRecord],
    times: Sequence[float],
    onset_s: float,
) -> list[FrameRecord]:
    """Every frame in the causal lookback, newest first.

    The shipped policy reads one frame. Contacts are silent on most single
    frames, so this variant exists to measure whether the whole window carries
    materially more — at the cost of a looser temporal claim, since a contact
    150 ms before onset need not still be held at onset.
    """
    target = onset_s - POSITION_OBSERVATION_LEAD_S
    earliest = target - POSITION_OBSERVATION_LOOKBACK_S
    window: list[FrameRecord] = []
    for index in range(bisect.bisect_right(times, target) - 1, -1, -1):
        record = records[index]
        if record.t_gold < earliest:
            break
        window.append(record)
    return window


def window_frets(position: int, max_fret: int) -> frozenset[int]:
    """The shipped support set ``{0} u [N-1, N+4]``."""
    return frozenset({0, *range(max(1, position - 1), min(max_fret, position + 4) + 1)})


def contact_sets(
    record: FrameRecord,
    *,
    n_strings: int,
    flip: bool,
    pressing_only: bool,
) -> tuple[frozenset[int], frozenset[tuple[int, int]], frozenset[int]]:
    """Project one frame's contacts into TabVision fret/pair/string sets.

    ``flip`` selects the string-index convention.  FretCam's ``nearest_string``
    returns a one-based index off the canonical board axis; TabVision's
    ``string_idx`` is zero-based from the low E.  Which end of the canonical
    axis is the low E is a geometric fact, not a tunable — the probe reports
    both so the convention is established by measurement rather than assumed.
    """
    frets: set[int] = set()
    pairs: set[tuple[int, int]] = set()
    strings: set[int] = set()
    for string_1based, fret, pressing, _score, _quality, _barre in record.contacts:
        if pressing_only and not pressing:
            continue
        string_idx = (n_strings - string_1based) if flip else (string_1based - 1)
        if not 0 <= string_idx < n_strings:
            continue
        frets.add(int(fret))
        pairs.add((string_idx, int(fret)))
        strings.add(string_idx)
    return frozenset(frets), frozenset(pairs), frozenset(strings)


def match_predictions(
    gold: Sequence[TabEvent],
    predicted: Sequence[TabEvent],
) -> tuple[dict[int, TabEvent], list[TabEvent]]:
    """Greedy onset+pitch match of gold to prediction, at the Tab F1 tolerance."""
    used: set[int] = set()
    matched: dict[int, TabEvent] = {}
    missed: list[TabEvent] = []
    by_pitch: dict[int, list[tuple[float, int]]] = {}
    for index, event in enumerate(predicted):
        entry = (float(event.onset_s), index)
        by_pitch.setdefault(int(event.pitch_midi), []).append(entry)
    for entries in by_pitch.values():
        entries.sort()
    for gold_index, note in enumerate(gold):
        best: tuple[float, int] | None = None
        for onset_s, index in by_pitch.get(int(note.pitch_midi), []):
            if index in used:
                continue
            delta = abs(onset_s - float(note.onset_s))
            if delta <= ONSET_TOLERANCE_S and (best is None or delta < best[0]):
                best = (delta, index)
        if best is None:
            missed.append(note)
        else:
            used.add(best[1])
            matched[gold_index] = predicted[best[1]]
    return matched, missed


def evaluate(
    stems: Sequence[str],
    *,
    stride: int,
    refresh: bool,
    flip: bool,
    pressing_only: bool,
    aggregate: bool = False,
    max_width: int = ANALYZER_MAX_WIDTH,
    max_height: int = ANALYZER_MAX_HEIGHT,
) -> dict[str, Any]:
    """Score every hypothesis over ``stems`` and return deterministic counts."""
    cfg = GuitarConfig()
    totals: Counter[str] = Counter()
    per_clip: list[dict[str, Any]] = []

    for stem in stems:
        records = capture_trace(
            stem,
            stride=stride,
            refresh=refresh,
            max_width=max_width,
            max_height=max_height,
        )
        times = [record.t_gold for record in records]
        gold = parse_gaps(DATA_ROOT / "gaps" / "musicxml" / f"{stem}.xml", cfg)
        predicted = load_tab_events(
            production_tab_cache_path(
                stem, backend_name=BACKEND_NAME, cache_dir=PROD_TAB_CACHE
            )
        )
        matched, _missed = match_predictions(gold, predicted)
        clip: Counter[str] = Counter()

        for gold_index, note in enumerate(gold):
            prediction = matched.get(gold_index)
            if prediction is None:
                clip["missed"] += 1
                frame = select_frame(
                    records, times, float(note.onset_s), require_accepted=False
                )
                if frame is not None:
                    clip["missed_any_frame"] += 1
                    _frets, pairs, _strings = contact_sets(
                        frame,
                        n_strings=cfg.n_strings,
                        flip=flip,
                        pressing_only=pressing_only,
                    )
                    if (int(note.string_idx), int(note.fret)) in pairs:
                        clip["missed_contact_pair_hit"] += 1
                if int(note.fret) == 0:
                    clip["missed_open_string"] += 1
                continue

            clip["matched"] += 1
            if len(candidate_positions(int(note.pitch_midi), cfg)) <= 1:
                continue
            clip["ambiguous"] += 1
            if (int(prediction.string_idx), int(prediction.fret)) == (
                int(note.string_idx),
                int(note.fret),
            ):
                clip["audio_right"] += 1
                # A prior applied to every ambiguous note can also BREAK a note
                # the decoder already had right. Count the exposure: contacts
                # naming some other playable position for this pitch, and not
                # the gold one. Without this the rescue/harm bound is optimistic.
                frame = select_frame(
                    records, times, float(note.onset_s), require_accepted=False
                )
                if frame is None:
                    continue
                _frets, pairs, _strings = contact_sets(
                    frame,
                    n_strings=cfg.n_strings,
                    flip=flip,
                    pressing_only=pressing_only,
                )
                if not pairs:
                    continue
                gold_pair = (int(note.string_idx), int(note.fret))
                rival = {
                    (candidate.string_idx, candidate.fret)
                    for candidate in candidate_positions(int(note.pitch_midi), cfg)
                    if (candidate.string_idx, candidate.fret) != gold_pair
                }
                if pairs & rival and gold_pair not in pairs:
                    clip["AR_breakable"] += 1
                continue
            clip["audio_wrong"] += 1

            gold_pair = (int(note.string_idx), int(note.fret))
            audio_pair = (int(prediction.string_idx), int(prediction.fret))

            accepted = select_frame(
                records, times, float(note.onset_s), require_accepted=True
            )
            if accepted is not None and accepted.position is not None:
                clip["W_cover"] += 1
                supported = window_frets(accepted.position, cfg.max_fret)
                clip["W_gold"] += int(gold_pair[1] in supported)
                clip["W_audio"] += int(audio_pair[1] in supported)

            if aggregate:
                window = select_window(records, times, float(note.onset_s))
            else:
                frame = select_frame(
                    records, times, float(note.onset_s), require_accepted=False
                )
                window = [] if frame is None else [frame]
            frets: set[int] = set()
            pairs: set[tuple[int, int]] = set()
            strings: set[int] = set()
            for record in window:
                record_frets, record_pairs, record_strings = contact_sets(
                    record,
                    n_strings=cfg.n_strings,
                    flip=flip,
                    pressing_only=pressing_only,
                )
                frets |= record_frets
                pairs |= record_pairs
                strings |= record_strings
            if not frets:
                continue
            clip["C_cover"] += 1
            clip["CF_gold"] += int(gold_pair[1] in frets or gold_pair[1] == 0)
            clip["CF_audio"] += int(audio_pair[1] in frets or audio_pair[1] == 0)
            clip["CP_gold"] += int(gold_pair in pairs)
            clip["CP_audio"] += int(audio_pair in pairs)
            clip["CS_gold"] += int(gold_pair[0] in strings)
            clip["CS_audio"] += int(audio_pair[0] in strings)

            # Rescue/harm decomposition for the CP channel. A re-ranker can only
            # help where contacts name the gold position and not the decoder's,
            # and can only hurt in the mirror case. Net bounds the achievable
            # effect far more tightly than the likelihood ratio alone.
            gold_in = gold_pair in pairs
            audio_in = audio_pair in pairs
            if gold_in and not audio_in:
                clip["CP_rescue"] += 1
            elif audio_in and not gold_in:
                clip["CP_harm"] += 1
            elif gold_in and audio_in:
                clip["CP_both"] += 1
            else:
                clip["CP_neither"] += 1

        totals.update(clip)
        per_clip.append({"clip": stem, **clip})
        print(
            f"  {stem}: gold {len(gold)} matched {clip['matched']} "
            f"missed {clip['missed']} ambiguous {clip['ambiguous']} "
            f"audio_wrong {clip['audio_wrong']} "
            f"W_cover {clip['W_cover']} C_cover {clip['C_cover']}",
            flush=True,
        )

    return {
        "config": {
            "stems": list(stems),
            "stride": stride,
            "flip": flip,
            "pressing_only": pressing_only,
            "aggregate": aggregate,
            "max_width": max_width,
            "max_height": max_height,
            "backend": BACKEND_NAME,
        },
        "totals": dict(totals),
        "per_clip": per_clip,
    }


def ratio(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else float("nan")


def format_report(payload: dict[str, Any]) -> str:
    """Render the deterministic counts as the human-readable table."""
    totals = Counter(payload["totals"])
    config = payload["config"]
    lines: list[str] = []
    add = lines.append

    add("=" * 74)
    add(
        "EVIDENCE STRENGTH  (string map: "
        f"{'flipped' if config['flip'] else 'direct'}, contacts: "
        f"{'pressing-only' if config['pressing_only'] else 'all-visible'})"
    )
    add("=" * 74)
    audio_wrong = totals["audio_wrong"]
    add(f"gold notes matched by audio : {totals['matched']}")
    add(f"  ambiguous pitch            : {totals['ambiguous']}")
    add(f"    audio right              : {totals['audio_right']}")
    add(f"    audio WRONG (target)     : {audio_wrong}")
    add("")
    add(
        f"{'hypothesis':<34}{'cover':>8}{'P(gold)':>10}"
        f"{'P(audio)':>10}{'LR':>8}{'nats':>8}"
    )
    add("-" * 74)
    rows = (
        ("W  window {0}u[N-1,N+4]  SHIPPED", "W_cover", "W_gold", "W_audio"),
        ("CF contact frets", "C_cover", "CF_gold", "CF_audio"),
        ("CP contact (string,fret)", "C_cover", "CP_gold", "CP_audio"),
        ("CS contact strings", "C_cover", "CS_gold", "CS_audio"),
    )
    for label, cover_key, gold_key, audio_key in rows:
        cover = totals[cover_key]
        p_gold = ratio(totals[gold_key], cover)
        p_audio = ratio(totals[audio_key], cover)
        if math.isfinite(p_gold) and math.isfinite(p_audio) and p_audio > 0.0:
            likelihood_ratio = p_gold / p_audio
            nats = (
                math.log(likelihood_ratio) if likelihood_ratio > 0.0 else float("-inf")
            )
        else:
            likelihood_ratio = nats = float("nan")
        add(
            f"{label:<34}{ratio(cover, audio_wrong):>7.1%}{p_gold:>10.3f}"
            f"{p_audio:>10.3f}{likelihood_ratio:>8.2f}{nats:>8.2f}"
        )
    add("")
    add("CP RESCUE / HARM  (of the audio-wrong notes contacts cover)")
    add("-" * 74)
    covered = totals["C_cover"]
    for label, key in (
        ("gold named, audio not  -> RESCUE", "CP_rescue"),
        ("audio named, gold not  -> HARM  ", "CP_harm"),
        ("both named             -> no gain", "CP_both"),
        ("neither named          -> silent", "CP_neither"),
    ):
        share = ratio(totals[key], covered)
        add(f"  {label}: {totals[key]:5d}  ({share:6.1%} of covered)")
    net = totals["CP_rescue"] - totals["CP_harm"]
    add(
        f"  NET rescuable          : {net:5d}"
        f"  ({ratio(net, audio_wrong):.1%} of all audio-wrong notes)"
    )
    breakable = totals["AR_breakable"]
    add(
        f"  exposure on audio-RIGHT: {breakable:5d}"
        f"  (contacts name a rival and not the gold, of"
        f" {totals['audio_right']} correct ambiguous notes)"
    )
    add(
        f"  NET after exposure     : {net - breakable:5d}"
        "  (worst case: every exposed note breaks)"
    )
    add("")
    add("MISSED ONSETS (a prior-reweighting bridge cannot touch these)")
    add("-" * 74)
    missed = totals["missed"]
    add(f"  gold notes with no audio detection  : {missed}")
    add(
        f"    with any video frame in window    : {totals['missed_any_frame']}"
        f"  ({ratio(totals['missed_any_frame'], missed):.1%})"
    )
    add(
        f"    video shows the exact (string,fret): {totals['missed_contact_pair_hit']}"
        f"  ({ratio(totals['missed_contact_pair_hit'], missed):.1%})"
    )
    add(
        f"    open string (fretting hand blind) : {totals['missed_open_string']}"
        f"  ({ratio(totals['missed_open_string'], missed):.1%})"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stems", default="clean12", help="'clean12' or a comma list")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--refresh", action="store_true", help="re-run inference")
    parser.add_argument("--flip", action="store_true", help="flipped string map")
    parser.add_argument("--pressing-only", action="store_true")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="union contacts across the whole lookback instead of one frame",
    )
    parser.add_argument("--max-width", type=int, default=ANALYZER_MAX_WIDTH)
    parser.add_argument("--max-height", type=int, default=ANALYZER_MAX_HEIGHT)
    parser.add_argument("--output", default="", help="optional JSON destination")
    args = parser.parse_args(argv)

    if args.stride < 1:
        raise SystemExit("--stride must be a positive integer")
    stems = (
        CLEAN_12
        if args.stems == "clean12"
        else tuple(stem.strip() for stem in args.stems.split(",") if stem.strip())
    )
    payload = evaluate(
        stems,
        stride=args.stride,
        refresh=args.refresh,
        flip=args.flip,
        pressing_only=args.pressing_only,
        aggregate=args.aggregate,
        max_width=args.max_width,
        max_height=args.max_height,
    )
    print("\n" + format_report(payload))
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
