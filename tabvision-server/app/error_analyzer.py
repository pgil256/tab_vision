"""Classify fusion pipeline output errors into named buckets for step-0 analysis.

Given ground truth and detected notes for a single video, emit one ErrorEvent
per classifiable failure. Pure function — no I/O, no subprocess, no pipeline
coupling. Driven by tools/error_analysis.py.

Originally drafted on the feature/audio-finetune branch (commits 6f816b9..552e44c).
This port adds three fixes:
  - auto-align GT timestamps via evaluate_transcription._find_best_time_offset,
    correcting count-in / dead-time offsets that otherwise inflate
    missed_onset and extra_detection;
  - a muted_undetectable bucket so X-fret GT notes (no MIDI) don't pollute
    missed_onset — they cannot produce audio;
  - both detection and ground-truth indices are consumed once per pass, so
    no event is double-counted.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from app.fusion_engine import TabNote  # noqa: F401 — re-export for callers


BUCKETS = (
    'correct',
    'wrong_position_same_pitch',
    'pitch_off',
    'timing_only',
    'missed_onset',
    'muted_undetectable',
    'extra_detection',
    # 'chord_split' deferred — see learned-fusion plan, step 0 follow-up.
)


@dataclass
class ErrorEvent:
    bucket: str
    video_id: str = ''
    # GT side (None when bucket == 'extra_detection').
    gt_string: Optional[int] = None
    gt_fret: Optional[object] = None  # int or 'X'
    gt_time: Optional[float] = None
    gt_midi: Optional[int] = None
    # Detection side (None when bucket == 'missed_onset' / 'muted_undetectable').
    det_string: Optional[int] = None
    det_fret: Optional[int] = None
    det_time: Optional[float] = None
    det_midi: Optional[int] = None

    def to_row(self) -> dict:
        return asdict(self)


def classify_events(detected, ground_truth, time_tolerance, bpm,
                    video_duration, video_id=''):
    if not detected and not ground_truth:
        return []

    gt_with_time = _gt_to_time(ground_truth, bpm, video_duration,
                               detected=detected, time_tolerance=time_tolerance)

    events: list[ErrorEvent] = []
    matched_gt: set[int] = set()
    matched_det: set[int] = set()

    # Pre-pass: GT notes with no MIDI (muted / X-fret) cannot be detected
    # from audio. Classify and remove them from matching so they don't claim
    # real detections in later passes.
    for gi, g in enumerate(gt_with_time):
        if g.get('midi_note') is None:
            matched_gt.add(gi)
            events.append(ErrorEvent(
                bucket='muted_undetectable', video_id=video_id,
                gt_string=g['string'], gt_fret=g['fret'], gt_time=g['time'],
                gt_midi=None,
            ))

    # Pass 1: exact (string + fret + time).
    for di, d in enumerate(detected):
        best_gi, best_dt = None, float('inf')
        for gi, g in enumerate(gt_with_time):
            if gi in matched_gt:
                continue
            dt = abs(d.timestamp - g['time'])
            if dt > time_tolerance:
                continue
            if d.string == g['string'] and d.fret == g['fret'] and dt < best_dt:
                best_gi, best_dt = gi, dt
        if best_gi is not None:
            matched_gt.add(best_gi)
            matched_det.add(di)
            events.append(_pair_event('correct', video_id,
                                      gt_with_time[best_gi], d))

    # Pass 2: same pitch, within time_tolerance, but wrong position.
    for di, d in enumerate(detected):
        if di in matched_det or d.midi_note is None:
            continue
        best_gi, best_dt = None, float('inf')
        for gi, g in enumerate(gt_with_time):
            if gi in matched_gt:
                continue
            dt = abs(d.timestamp - g['time'])
            if dt > time_tolerance:
                continue
            if d.midi_note == g.get('midi_note') and dt < best_dt:
                best_gi, best_dt = gi, dt
        if best_gi is not None:
            matched_gt.add(best_gi)
            matched_det.add(di)
            events.append(_pair_event('wrong_position_same_pitch', video_id,
                                      gt_with_time[best_gi], d))

    # Pass 3: same position outside time_tolerance but within 2x — timing only.
    # Run before pitch_off so a same-position match doesn't get downgraded.
    for di, d in enumerate(detected):
        if di in matched_det:
            continue
        best_gi, best_dt = None, float('inf')
        for gi, g in enumerate(gt_with_time):
            if gi in matched_gt:
                continue
            dt = abs(d.timestamp - g['time'])
            if dt > 2 * time_tolerance:
                continue
            if d.string == g['string'] and d.fret == g['fret'] and dt < best_dt:
                best_gi, best_dt = gi, dt
        if best_gi is not None:
            matched_gt.add(best_gi)
            matched_det.add(di)
            events.append(_pair_event('timing_only', video_id,
                                      gt_with_time[best_gi], d))

    # Pass 4: any unmatched detection within tolerance of any unmatched GT.
    # Audio-side wrong-pitch failure.
    for di, d in enumerate(detected):
        if di in matched_det:
            continue
        best_gi, best_dt = None, float('inf')
        for gi, g in enumerate(gt_with_time):
            if gi in matched_gt:
                continue
            dt = abs(d.timestamp - g['time'])
            if dt <= time_tolerance and dt < best_dt:
                best_gi, best_dt = gi, dt
        if best_gi is not None:
            matched_gt.add(best_gi)
            matched_det.add(di)
            events.append(_pair_event('pitch_off', video_id,
                                      gt_with_time[best_gi], d))

    for gi, g in enumerate(gt_with_time):
        if gi in matched_gt:
            continue
        events.append(ErrorEvent(
            bucket='missed_onset', video_id=video_id,
            gt_string=g['string'], gt_fret=g['fret'], gt_time=g['time'],
            gt_midi=g.get('midi_note'),
        ))
    for di, d in enumerate(detected):
        if di in matched_det:
            continue
        events.append(ErrorEvent(
            bucket='extra_detection', video_id=video_id,
            det_string=d.string, det_fret=d.fret, det_time=d.timestamp,
            det_midi=d.midi_note,
        ))
    return events


def _pair_event(bucket: str, video_id: str, g: dict, d) -> ErrorEvent:
    return ErrorEvent(
        bucket=bucket, video_id=video_id,
        gt_string=g['string'], gt_fret=g['fret'], gt_time=g['time'],
        gt_midi=g.get('midi_note'),
        det_string=d.string, det_fret=d.fret, det_time=d.timestamp,
        det_midi=d.midi_note,
    )


def _gt_to_time(gt: list[dict], bpm, video_duration,
                detected=None, time_tolerance=None) -> list[dict]:
    if not gt:
        return []
    total_beats = max(n['beat'] for n in gt)
    if bpm and bpm > 0:
        beat_to_time = 60.0 / bpm
    else:
        beat_to_time = video_duration / total_beats if total_beats else 1.0
    offset = 0.0
    if bpm and bpm > 0 and detected and time_tolerance:
        from evaluate_transcription import _find_best_time_offset
        offset = _find_best_time_offset(
            detected,
            [n['beat'] for n in gt],
            [n.get('midi_note') for n in gt],
            beat_to_time, video_duration, time_tolerance,
        )
    return [{**n, 'time': n['beat'] * beat_to_time + offset} for n in gt]


def summarize_events(events: list[ErrorEvent]) -> dict[str, int]:
    counts = {b: 0 for b in BUCKETS}
    for e in events:
        counts[e.bucket] = counts.get(e.bucket, 0) + 1
    return counts
