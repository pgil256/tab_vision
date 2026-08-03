"""Opt-in personal position-prior labels harvested from FretCam windows.

Track C priced giving a player their own position prior at **+0.0305
[+0.0183, +0.0430]** aggregate Tab F1 — the largest remaining production
lever (``docs/EVAL_REPORTS/c_prior_adaptation_2026-07-25.md``) — but the
label supply was never built. This module supplies it from the camera.

The mechanism inverts how FretCam has been used so far. As an *evidence*
channel at decode time the camera measured ≈0 three separate ways, because
the decoder's retained paths are the same tab (segment_window_stage1). As a
*labelling* channel its measured profile — position shown on 27% of stable
frames, correct on 100% of them — is exactly right: a labeller needs
precision, not coverage. A confident position window joined with the audio
backend's pitch pins the ``(string, fret)`` whenever exactly one playable
candidate is consistent with the window, and those labels accumulate across
sessions into the counts artifact ``load_pitch_position_prior`` already
loads from a path.

Posture: harvesting a user's own recordings into a **local** personal prior
is the narrow SPEC §1.5 carve-out (2026-08-02). Personal artifacts are
never shipped, never registered as defaults, and never used in eval
corpora or published figures.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from tabvision.fusion.artifact_registry import load_artifact_manifest
from tabvision.types import AudioEvent, GuitarConfig
from tabvision.video.position import PositionWindowObservation

PERSONAL_LABEL_SCHEMA_VERSION = 1

# Track C harvested decoder output at a 0.5 confidence floor (91.6% of notes);
# the same floor is applied to both sides of this join. The 0.25 s gap bound
# is ~7 analyzed frames at the default stride — tight enough that a "locked"
# window still describes the hand that played the onset.
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MAX_GAP_S = 0.25


@dataclass(frozen=True)
class PersonalLabel:
    """One harvested ``(pitch, string, fret)`` example on the media timeline."""

    pitch_midi: int
    string_idx: int
    fret: int
    onset_s: float
    confidence: float
    source: str = "fretcam-window"


def harvest_position_labels(
    audio_events: Sequence[AudioEvent],
    observations: Sequence[PositionWindowObservation],
    cfg: GuitarConfig | None = None,
    *,
    min_audio_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_window_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> list[PersonalLabel]:
    """Join audio pitches against locked position windows into labels.

    A label is emitted only when **exactly one** playable candidate for the
    pitch is consistent with the camera's window — a fretted candidate inside
    ``window_frets``, or an open string, which is playable from any hand
    position. Any ambiguity abstains; the harvest inherits the window
    channel's precision instead of diluting it.

    Only ``state == "locked"`` observations participate: "holding" is a
    display affordance carrying a stale estimate. Harvesting is refused
    under a capo — the counts artifact is capo-0 indexed and a capo session
    would mix coordinate systems (the covariant re-indexing happens at
    *load* time, not label time).
    """
    cfg = cfg or GuitarConfig()
    if cfg.capo != 0:
        raise ValueError("personal-label harvest requires capo 0; the artifact is capo-0 indexed")
    if max_gap_s < 0:
        raise ValueError("max_gap_s must be non-negative")

    usable = sorted(
        (
            obs
            for obs in observations
            if obs.state == "locked" and obs.confidence >= min_window_confidence
        ),
        key=lambda obs: obs.timestamp_s,
    )
    if not usable:
        return []
    timestamps = [obs.timestamp_s for obs in usable]

    labels: list[PersonalLabel] = []
    for event in audio_events:
        if event.confidence < min_audio_confidence:
            continue
        window = _nearest_window(usable, timestamps, event.onset_s, max_gap_s)
        if window is None:
            continue
        window_frets = set(window.window_frets)
        consistent = [
            (string_idx, fret)
            for string_idx, open_pitch in enumerate(cfg.tuning_midi)
            if 0 <= (fret := event.pitch_midi - open_pitch) <= cfg.max_fret
            and (fret == 0 or fret in window_frets)
        ]
        if len(consistent) != 1:
            continue
        string_idx, fret = consistent[0]
        labels.append(
            PersonalLabel(
                pitch_midi=event.pitch_midi,
                string_idx=string_idx,
                fret=fret,
                onset_s=event.onset_s,
                confidence=min(event.confidence, window.confidence),
            )
        )
    return labels


def _nearest_window(
    usable: Sequence[PositionWindowObservation],
    timestamps: Sequence[float],
    onset_s: float,
    max_gap_s: float,
) -> PositionWindowObservation | None:
    import bisect

    index = bisect.bisect_left(timestamps, onset_s)
    best: PositionWindowObservation | None = None
    best_gap = max_gap_s
    for candidate_index in (index - 1, index):
        if 0 <= candidate_index < len(usable):
            gap = abs(timestamps[candidate_index] - onset_s)
            if gap <= best_gap:
                best = usable[candidate_index]
                best_gap = gap
    return best


def append_personal_labels(
    store_path: str | Path,
    labels: Sequence[PersonalLabel],
    *,
    source_media: str,
) -> None:
    """Append labels to a JSONL store, one self-describing object per line."""
    path = Path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    harvested_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            row = {
                "schema_version": PERSONAL_LABEL_SCHEMA_VERSION,
                **asdict(label),
                "media": source_media,
                "harvested_at": harvested_at,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_personal_labels(store_path: str | Path) -> list[PersonalLabel]:
    """Read a JSONL label store back, raising on any malformed row."""
    path = Path(store_path)
    labels: list[PersonalLabel] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON label row") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number}: label row must be an object")
            if row.get("schema_version") != PERSONAL_LABEL_SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number}: unsupported label schema")
            try:
                labels.append(
                    PersonalLabel(
                        pitch_midi=int(row["pitch_midi"]),
                        string_idx=int(row["string_idx"]),
                        fret=int(row["fret"]),
                        onset_s=float(row["onset_s"]),
                        confidence=float(row["confidence"]),
                        source=str(row.get("source", "fretcam-window")),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid label fields") from exc
    return labels


def build_personal_prior_payload(
    labels: Sequence[PersonalLabel],
    *,
    cfg: GuitarConfig | None = None,
    merge_population: str | None = "guitarset-v1",
    min_labels_per_pitch: int = 5,
) -> dict[str, object]:
    """Aggregate labels into a schema-1 counts payload for the prior loader.

    Per-pitch switching, not mixing: a pitch with at least
    ``min_labels_per_pitch`` harvested labels gets *only* its personal
    counts (the loader's ``alpha`` smoothing keeps thin evidence weak); any
    other pitch keeps the population artifact's counts unchanged. This is
    the shape of Track C's ``oracle_player`` arm — your own prior where you
    are known, the population elsewhere — and avoids the unmeasured
    blend-weight hyperparameter entirely. ``min_labels_per_pitch`` is a
    conservatism knob, not a fitted constant; it has never been swept.
    """
    cfg = cfg or GuitarConfig()
    if min_labels_per_pitch < 1:
        raise ValueError("min_labels_per_pitch must be at least 1")

    personal_counts: dict[tuple[int, int, int], int] = {}
    pitch_totals: dict[int, int] = {}
    for label in labels:
        if not (0 <= label.string_idx < cfg.n_strings and 0 <= label.fret <= cfg.max_fret):
            raise ValueError(f"label out of range for the configured guitar: {label}")
        key = (label.pitch_midi, label.string_idx, label.fret)
        personal_counts[key] = personal_counts.get(key, 0) + 1
        pitch_totals[label.pitch_midi] = pitch_totals.get(label.pitch_midi, 0) + 1

    personalized = {pitch for pitch, total in pitch_totals.items() if total >= min_labels_per_pitch}

    alpha = 1.0
    power = 2.0
    population_base: str | None = None
    rows: list[list[int]] = [
        [pitch, string_idx, fret, count]
        for (pitch, string_idx, fret), count in personal_counts.items()
        if pitch in personalized
    ]
    if merge_population and merge_population != "none":
        manifest = load_artifact_manifest(merge_population, expected_kind="position")
        population = json.loads(manifest.artifact_path.read_text(encoding="utf-8"))
        if population.get("schema_version") != 1 or not isinstance(population.get("counts"), list):
            raise ValueError(f"population artifact has an unsupported schema: {merge_population}")
        population_base = merge_population
        alpha = float(population.get("alpha", alpha))
        power = float(population.get("power", power))
        rows.extend(
            [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
            for row in population["counts"]
            if int(row[0]) not in personalized
        )

    rows.sort()
    return {
        "schema_version": 1,
        "kind": "personal-position-prior",
        "alpha": alpha,
        "power": power,
        "population_base": population_base,
        "min_labels_per_pitch": min_labels_per_pitch,
        "label_count": len(labels),
        "personalized_pitches": sorted(personalized),
        "counts": rows,
    }


def write_personal_prior_artifact(path: str | Path, payload: Mapping[str, object]) -> None:
    """Write the artifact JSON the way the loader and the CLI expect it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_MAX_GAP_S",
    "DEFAULT_MIN_CONFIDENCE",
    "PERSONAL_LABEL_SCHEMA_VERSION",
    "PersonalLabel",
    "append_personal_labels",
    "build_personal_prior_payload",
    "harvest_position_labels",
    "read_personal_labels",
    "write_personal_prior_artifact",
]
