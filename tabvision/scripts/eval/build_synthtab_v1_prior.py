"""Build ``synthtab-v1`` position + ``synthtab-seq-v1`` transition priors.

Program S Phase S1a — see
``docs/plans/2026-07-20-nc-second-opinion-and-synthtab-program.md`` and the
S0 audit (``docs/EVAL_REPORTS/s0_synthtab_audit_2026-07-20.md``).

Counts come from the SynthTab symbolic slice
(``all_jams_midi_V2_60000_tracks.zip``, DadaGP-derived, CC-BY-NC-4.0; the
derived count tables inherit that license and are labeled NC in
LICENSES.md). The artifact class and hyperparameters are deliberately
identical to ``guitarset-v1`` / ``gaps-v1``; only the corpus differs.
Tracks are filtered to exact standard tuning (the registered priors'
validated domain) and to a GM-program variant:

- ``acoustic`` — programs 24/25 (nylon/steel acoustic), domain-matched to
  the clean-acoustic route.
- ``all`` — any guitar program 24-31, maximizing scale.

Artifacts land outside the runtime tree (default
``$TABVISION_DATA_ROOT/models/synthtab_priors``): S1a is evaluation-only;
registration is a separate user-gated step.

Timing: JAMS observation times are ticks; each track's tempo annotation
(BPM segments) plus the companion ``string_*.mid`` MThd division (PPQ)
convert ticks to seconds so the 80 ms cluster/anchor rule in
``extract_transitions`` sees real time, exactly as the decode does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import zipfile
from collections import Counter
from pathlib import Path

from tabvision.fusion.transition_prior import extract_transitions
from tabvision.types import TabEvent

STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
ACOUSTIC_PROGRAMS = frozenset({24, 25})
GUITAR_PROGRAMS = frozenset(range(24, 32))
FALLBACK_PPQ = 960
MAX_FRET = 24

# Mirror the accepted guitarset-v1 / gaps-v1 hyperparameters exactly.
POSITION_ALPHA = 1.0
POSITION_POWER = 2.0
SEQUENCE_SCHEME = "delta_fret"
SEQUENCE_ALPHA = 0.5
SEQUENCE_BACKOFF_KAPPA = 8.0


def _read_ppq(archive: zipfile.ZipFile, track_dir: str, names: list[str]) -> int:
    for name in names:
        if not (name.startswith(f"{track_dir}/string_") and name.endswith(".mid")):
            continue
        try:
            header = archive.read(name)[:14]
        except (OSError, zipfile.BadZipFile):
            continue
        if len(header) < 14 or header[:4] != b"MThd":
            continue
        division = struct.unpack(">H", header[12:14])[0]
        if division & 0x8000:  # SMPTE division — unsupported, fall back
            continue
        if division > 0:
            return division
    return FALLBACK_PPQ


class _TempoMap:
    """Piecewise tick->seconds map from JAMS tempo observations."""

    def __init__(self, observations: list[dict], ppq: int) -> None:
        segments: list[tuple[float, float]] = []  # (start_tick, bpm)
        for obs in sorted(observations, key=lambda o: float(o.get("time", 0.0))):
            bpm = float(obs.get("value") or 0.0)
            if bpm > 0:
                segments.append((float(obs.get("time", 0.0)), bpm))
        if not segments or segments[0][0] > 0.0:
            segments.insert(0, (0.0, segments[0][1] if segments else 120.0))
        self._starts = [tick for tick, _ in segments]
        self._bpms = [bpm for _, bpm in segments]
        self._ppq = ppq
        self._cum_seconds = [0.0]
        for index in range(1, len(segments)):
            span = self._starts[index] - self._starts[index - 1]
            rate = 60.0 / (self._bpms[index - 1] * ppq)
            self._cum_seconds.append(self._cum_seconds[-1] + span * rate)

    def seconds(self, tick: float) -> float:
        index = 0
        for candidate in range(len(self._starts)):
            if self._starts[candidate] <= tick:
                index = candidate
            else:
                break
        rate = 60.0 / (self._bpms[index] * self._ppq)
        return self._cum_seconds[index] + (tick - self._starts[index]) * rate


def _track_events(raw: dict, ppq: int) -> list[TabEvent] | None:
    """Parse one SynthTab JAMS dict to TabEvents, or None if out of domain."""
    annotations = raw.get("annotations", [])
    note_tab = [a for a in annotations if a.get("namespace") == "note_tab"]
    if len(note_tab) != 6:
        return None
    tuning: dict[int, int] = {}
    for annotation in note_tab:
        sandbox = annotation.get("sandbox", {})
        index = sandbox.get("string_index")
        open_tuning = sandbox.get("open_tuning")
        if not isinstance(index, int) or not isinstance(open_tuning, int):
            return None
        tuning[index] = open_tuning
    if tuning != STANDARD_TUNING:
        return None

    tempo_obs = [
        obs
        for annotation in annotations
        if annotation.get("namespace") == "tempo"
        for obs in annotation.get("data", [])
    ]
    tempo_map = _TempoMap(tempo_obs, ppq)

    events: list[TabEvent] = []
    for annotation in note_tab:
        index = annotation["sandbox"]["string_index"]
        open_midi = annotation["sandbox"]["open_tuning"]
        string_idx = 6 - index  # SynthTab: 1 = high E; ours: 0 = low E
        for obs in annotation.get("data", []):
            value = obs.get("value") or {}
            fret = value.get("fret")
            if not isinstance(fret, int) or not 0 <= fret <= MAX_FRET:
                continue
            onset = tempo_map.seconds(float(obs.get("time", 0.0)))
            end = tempo_map.seconds(float(obs.get("time", 0.0)) + float(obs.get("duration") or 0.0))
            events.append(
                TabEvent(
                    onset_s=onset,
                    duration_s=max(0.01, end - onset),
                    string_idx=string_idx,
                    fret=fret,
                    pitch_midi=open_midi + fret,
                    confidence=1.0,
                )
            )
    if not events:
        return None
    events.sort(key=lambda e: (e.onset_s, e.string_idx))
    return events


def build_payloads(*, zip_path: Path, variant: str, max_tracks: int) -> tuple[dict, dict, dict]:
    programs = ACOUSTIC_PROGRAMS if variant == "acoustic" else GUITAR_PROGRAMS
    position_counts: Counter[tuple[int, int, int]] = Counter()
    transition_counts: Counter[tuple[int, int, int]] = Counter()
    scanned = eligible = parsed = 0
    program_histogram: Counter[int] = Counter()
    skipped_nonstandard = skipped_program = skipped_empty = 0

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        jams_names = [n for n in names if n.endswith(".jams")]
        members_by_dir: dict[str, list[str]] = {}
        for name in names:
            members_by_dir.setdefault(name.rsplit("/", 1)[0], []).append(name)
        for jams_name in jams_names:
            scanned += 1
            if max_tracks and parsed >= max_tracks:
                break
            try:
                raw = json.loads(archive.read(jams_name))
            except (ValueError, OSError, zipfile.BadZipFile):
                skipped_empty += 1
                continue
            program = raw.get("sandbox", {}).get("instrument")
            if not isinstance(program, int) or program not in programs:
                skipped_program += 1
                continue
            program_histogram[program] += 1
            eligible += 1
            track_dir = jams_name.rsplit("/", 1)[0]
            ppq = _read_ppq(archive, track_dir, members_by_dir.get(track_dir, []))
            events = _track_events(raw, ppq)
            if events is None:
                skipped_nonstandard += 1
                continue
            parsed += 1
            for event in events:
                position_counts[(event.pitch_midi, event.string_idx, event.fret)] += 1
            for sample in extract_transitions(events, singleton_only=True):
                transition_counts[sample] += 1

    if not parsed:
        raise RuntimeError("no SynthTab tracks passed the filters")

    provenance = {
        "zip": str(zip_path),
        "zip_sha256": _sha256(zip_path),
        "variant": variant,
        "gm_programs": sorted(programs),
        "tracks_scanned": scanned,
        "tracks_program_eligible": eligible,
        "tracks_parsed": parsed,
        "skipped_program": skipped_program,
        "skipped_nonstandard_or_empty": skipped_nonstandard + skipped_empty,
        "program_histogram": {str(k): v for k, v in sorted(program_histogram.items())},
        "position_rows": len(position_counts),
        "position_events": sum(position_counts.values()),
        "transition_rows": len(transition_counts),
        "transition_samples": sum(transition_counts.values()),
    }
    source_common = (
        f"counts built from the SynthTab symbolic slice (DadaGP-derived; "
        f"CC-BY-NC-4.0 inherited), standard-tuning tracks, GM programs "
        f"{sorted(programs)}. Evaluation-only until registered."
    )
    position_payload = {
        "schema_version": 1,
        "name": f"synthtab-v1-{variant}",
        "source": f"Pitch-position {source_common}",
        "validation_player": "guitarset-dev-oof",
        "training_tracks": parsed,
        "alpha": POSITION_ALPHA,
        "power": POSITION_POWER,
        "counts": [
            [pitch, string_idx, fret, count]
            for (pitch, string_idx, fret), count in sorted(position_counts.items())
        ],
    }
    sequence_payload = {
        "schema_version": 1,
        "name": f"synthtab-seq-v1-{variant}",
        "source": f"Anchor-to-anchor transition {source_common}",
        "validation_player": "guitarset-dev-oof",
        "training_tracks": parsed,
        "scheme": SEQUENCE_SCHEME,
        "alpha": SEQUENCE_ALPHA,
        "backoff_kappa": SEQUENCE_BACKOFF_KAPPA,
        "singleton_only": True,
        "counts": [
            [dp, ds, prev_fret, count]
            for (dp, ds, prev_fret), count in sorted(transition_counts.items())
        ],
    }
    return position_payload, sequence_payload, provenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=None)
    parser.add_argument("--variant", choices=("acoustic", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-tracks", type=int, default=0)
    args = parser.parse_args()

    data_root = os.environ.get("TABVISION_DATA_ROOT", "")
    zip_path = args.zip_path or (
        Path(data_root) / "datasets" / "synthtab" / "all_jams_midi_V2_60000_tracks.zip"
    )
    output_dir = args.output_dir or (Path(data_root) / "models" / "synthtab_priors")
    output_dir.mkdir(parents=True, exist_ok=True)

    position_payload, sequence_payload, provenance = build_payloads(
        zip_path=zip_path, variant=args.variant, max_tracks=args.max_tracks
    )
    outputs = (
        (output_dir / f"synthtab_v1_{args.variant}.json", position_payload),
        (output_dir / f"synthtab_seq_v1_{args.variant}.json", sequence_payload),
    )
    for path, payload in outputs:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
        print(f"output={path}")
        print(f"sha256={_sha256(path)}")
    manifest_path = output_dir / f"synthtab_v1_{args.variant}.provenance.json"
    manifest_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
