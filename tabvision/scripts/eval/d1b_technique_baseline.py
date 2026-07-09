"""D1-b — expressive-markings (technique) baseline on GuitarSet.

SPEC §1.4 tracks a v1.1 *stretch* of detecting expressive markings (bends,
hammer-ons, pull-offs, slides) at ``>= 0.70`` detection F1. Per §0 rule 7
("flag, don't hallucinate") that number is **UNBASELINED** — no technique-F1
has ever been measured. D1-b is the free, automated first measurement, from
which an honest stretch can be set. This script is that measurement.

Two facts it establishes, both cheaply and without any model download:

1. **What GuitarSet can even baseline.** GuitarSet JAMS carry six annotation
   namespaces: ``note_midi`` (string/fret/pitch), ``pitch_contour`` (dense
   per-string f0), ``beat_position``, ``tempo``, ``chord``, ``key_mode``.
   There are **no discrete technique labels** — no ``bend``/``hammer_on``/
   ``pull_off``/``slide`` events. So technique gold cannot be read directly;
   it must be *derived*. Bends and slides are recoverable as **proxies** from
   ``pitch_contour`` (a bend is a within-note pitch excursion; a slide is a
   legato cross-note glide). **Hammer-ons and pull-offs are articulation, not
   pitch** — indistinguishable from picked notes in ``note_midi`` +
   ``pitch_contour`` without attack/timbre analysis — so GuitarSet cannot
   baseline them at all. They need a technique-labelled corpus (Guitar-TECHS,
   which is electric -> v2 scope).

2. **The operational detector's capability.** The shipping default backend
   (``highres``) constructs every ``AudioEvent`` with an empty ``tags`` tuple
   (``highres.py`` ``_events_from_midi`` — MIDI carries no bend metadata), and
   fusion copies ``tags`` straight through to ``TabEvent.techniques``
   (``viterbi.py``). The only tag-emitting code in the repo is Basic Pitch's
   bend heuristic (``basicpitch.py``), and ``basic-pitch`` is not installed.
   So the operational pipeline emits **zero** technique tags. Technique
   recall is therefore 0 and technique-detection F1 is **0.00** against any
   non-empty gold — a structural (not stochastic) zero. This script scores
   whatever tags the pipeline *does* emit against the derived proxy gold, so
   it doubles as a reusable harness once a real detector is built.

This is a **diagnostic baseline, not a gate**. It does not change any §1.4
acceptance target. Its output is: the measured baseline (0.00) plus the
proxy-technique **support** (how many bend/slide instances even exist to
detect), which together let §15/§1.4 replace the unbaselined 0.70 with an
honest restatement.

The proxy thresholds below are documented and deliberately reported at two
settings, because the derived counts are threshold-sensitive; the *baseline*
(0.00 detector F1) is not.

Reproduce::

    cd tabvision
    export TABVISION_DATA_ROOT=~/.tabvision/data   # or pass --data-home
    python -m scripts.eval.d1b_technique_baseline \
        --output ../docs/EVAL_REPORTS/d1b_technique_baseline_2026-07-09.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- Proxy-derivation thresholds (documented; see module docstring) ----------

# A note is a *bend* proxy if its pitch_contour shows a **sustained net pitch
# shift** between the start and end thirds of the note (after skipping the
# attack transient) of at least this many semitones. "Sustained net shift"
# (not peak excursion) is deliberate: vibrato oscillates around a centre so its
# net first-third-vs-last-third shift is ~0 and it is excluded; the plucked-
# string attack transient is skipped. This *undercounts* bend-and-release
# (where the pitch returns), so it is an honest **floor** on bend prevalence.
# Peak-excursion was tried first and flagged ~30% of notes — not credible; it
# was catching vibrato + attack settling, which is exactly what this rejects.
BEND_SEMITONES_PRIMARY = 1.0  # a clear, held >= 1-semitone (>= 1-fret) bend
BEND_SEMITONES_SUBTLE = 0.5  # inclusive: microbends (also catches slow drift)
# Skip the plucked-string attack transient before measuring the sustained level.
BEND_ATTACK_SKIP_S = 0.05
# Ignore very short notes / sparse contour where a "shift" is just noise.
MIN_BEND_NOTE_DUR_S = 0.12
MIN_BEND_SAMPLES = 6

# A *slide* proxy is a legato transition between two consecutive same-string
# notes: small onset-to-onset gap, a fret jump in this range, and a
# continuously-voiced contour bridging the two (a glide, not a re-pick).
SLIDE_MAX_LEGATO_GAP_S = 0.08
SLIDE_MIN_PITCH_STEP = 1  # semitones (>= 1 fret)
SLIDE_MAX_PITCH_STEP = 7  # semitones (slides beyond ~7 frets are rare)
SLIDE_MIN_VOICED_FRACTION = 0.80

A440 = 440.0


def _hz_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * math.log2(freq / A440)


@dataclass
class _StringContour:
    """Dense f0 samples for one string: parallel time/midi/voiced arrays."""

    times: list[float] = field(default_factory=list)
    midis: list[float] = field(default_factory=list)

    def samples_in(self, t0: float, t1: float) -> list[tuple[float, float]]:
        # times is monotonically increasing; a linear scan is fine at this scale.
        return [(t, m) for t, m in zip(self.times, self.midis, strict=False) if t0 <= t <= t1]

    def voiced_fraction(self, t0: float, t1: float) -> float:
        window = [t for t in self.times if t0 <= t <= t1]
        if not window:
            return 0.0
        # Every retained sample is voiced (unvoiced samples are dropped at parse
        # time), so a nonempty window with samples spanning the interval is
        # "voiced"; measure coverage by expected vs present sample count.
        span = max(1e-6, t1 - t0)
        hop = _median_hop(self.times)
        expected = max(1, round(span / hop)) if hop > 0 else len(window)
        return min(1.0, len(window) / expected)


def _median_hop(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    diffs = sorted(t2 - t1 for t1, t2 in zip(times[:-1], times[1:], strict=False) if t2 > t1)
    return diffs[len(diffs) // 2] if diffs else 0.0


@dataclass
class _Note:
    string_idx: int
    onset_s: float
    duration_s: float
    pitch_midi: int


@dataclass
class TechniqueCensus:
    """Proxy-technique counts for a set of tracks (one threshold setting)."""

    n_tracks: int = 0
    n_notes: int = 0
    n_bend: int = 0
    n_slide: int = 0

    @property
    def bend_pct(self) -> float:
        return 100.0 * self.n_bend / self.n_notes if self.n_notes else 0.0

    @property
    def slide_pct(self) -> float:
        return 100.0 * self.n_slide / self.n_notes if self.n_notes else 0.0

    @property
    def n_derivable(self) -> int:
        return self.n_bend + self.n_slide


def _parse_jams(path: Path, n_strings: int = 6) -> tuple[list[_Note], dict[int, _StringContour]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    notes: list[_Note] = []
    contours: dict[int, _StringContour] = {s: _StringContour() for s in range(n_strings)}

    for ann in payload.get("annotations", []):
        ns = ann.get("namespace")
        source = ann.get("annotation_metadata", {}).get("data_source")
        try:
            string_idx = int(source)
        except (TypeError, ValueError):
            continue
        if not 0 <= string_idx < n_strings:
            continue

        if ns == "note_midi":
            for row in ann.get("data") or []:
                try:
                    onset = float(row["time"])
                    dur = float(row["duration"])
                    pitch = int(round(float(row["value"])))
                except (KeyError, TypeError, ValueError):
                    continue
                notes.append(_Note(string_idx, onset, max(0.0, dur), pitch))

        elif ns == "pitch_contour":
            data = ann.get("data") or {}
            # Columnar layout: {time: [...], value: [{voiced,index,frequency}], ...}
            times = data.get("time") or []
            values = data.get("value") or []
            contour = contours[string_idx]
            for t, v in zip(times, values, strict=False):
                if not isinstance(v, dict) or not v.get("voiced"):
                    continue
                freq = v.get("frequency")
                if not freq or freq <= 0:
                    continue
                contour.times.append(float(t))
                contour.midis.append(_hz_to_midi(float(freq)))

    notes.sort(key=lambda n: (n.string_idx, n.onset_s))
    return notes, contours


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def _is_bend(note: _Note, contour: _StringContour, threshold_st: float) -> bool:
    if note.duration_s < MIN_BEND_NOTE_DUR_S:
        return False
    # Skip the attack transient; measure only the sustained portion of the note.
    start = note.onset_s + BEND_ATTACK_SKIP_S
    samples = contour.samples_in(start, note.onset_s + note.duration_s)
    if len(samples) < MIN_BEND_SAMPLES:
        return False
    midis = [m for _, m in samples]
    third = max(1, len(midis) // 3)
    # Net shift between the note's start third and end third. Vibrato (returns to
    # centre) -> ~0; a held bend -> >= threshold. Median is robust to spikes.
    return abs(_median(midis[-third:]) - _median(midis[:third])) >= threshold_st


def _count_slides(notes_on_string: list[_Note], contour: _StringContour) -> int:
    count = 0
    for a, b in zip(notes_on_string[:-1], notes_on_string[1:], strict=False):
        gap = b.onset_s - (a.onset_s + a.duration_s)
        step = abs(b.pitch_midi - a.pitch_midi)
        if not (0.0 <= gap <= SLIDE_MAX_LEGATO_GAP_S):
            continue
        if not (SLIDE_MIN_PITCH_STEP <= step <= SLIDE_MAX_PITCH_STEP):
            continue
        # Contour must bridge the two notes without an unvoiced gap (a glide).
        bridge_lo = a.onset_s + a.duration_s - 0.02
        bridge_hi = b.onset_s + 0.02
        if contour.voiced_fraction(bridge_lo, bridge_hi) >= SLIDE_MIN_VOICED_FRACTION:
            count += 1
    return count


def census_track(path: Path, bend_threshold_st: float) -> TechniqueCensus:
    notes, contours = _parse_jams(path)
    c = TechniqueCensus(n_tracks=1, n_notes=len(notes))
    by_string: dict[int, list[_Note]] = {}
    for n in notes:
        by_string.setdefault(n.string_idx, []).append(n)
        if _is_bend(n, contours[n.string_idx], bend_threshold_st):
            c.n_bend += 1
    for s, string_notes in by_string.items():
        c.n_slide += _count_slides(string_notes, contours[s])
    return c


def _accumulate(target: TechniqueCensus, other: TechniqueCensus) -> None:
    target.n_tracks += other.n_tracks
    target.n_notes += other.n_notes
    target.n_bend += other.n_bend
    target.n_slide += other.n_slide


def _wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson 95% interval for a proportion — used to show how
    noisy any future technique-F1 would be given the (small) support ``n``."""
    if n <= 0:
        return 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = centre - margin, centre + margin
    return (hi - lo) / 2.0


def _list_jams(data_home: Path, split: str, validation_player: str) -> list[Path]:
    ann_dir = data_home / "guitarset" / "annotation"
    if not ann_dir.is_dir():
        ann_dir = data_home / "annotation"  # allow pointing directly at guitarset/
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"no GuitarSet annotation dir under {data_home}")
    paths = sorted(ann_dir.glob("*.jams"))
    if split == "all":
        return paths
    if split == "validation":
        return [p for p in paths if p.stem.split("_", 1)[0] == validation_player]
    if split == "train":
        return [p for p in paths if p.stem.split("_", 1)[0] != validation_player]
    raise ValueError(f"unknown split {split!r}")


def build_report(
    *,
    data_home: Path,
    splits: tuple[str, ...] = ("all", "validation"),
    validation_player: str = "05",
) -> str:
    # Census per split at both bend thresholds.
    results: dict[str, dict[float, TechniqueCensus]] = {}
    for split in splits:
        paths = _list_jams(data_home, split, validation_player)
        results[split] = {}
        for thr in (BEND_SEMITONES_PRIMARY, BEND_SEMITONES_SUBTLE):
            agg = TechniqueCensus()
            for p in paths:
                _accumulate(agg, census_track(p, thr))
            results[split][thr] = agg

    lines: list[str] = []
    lines.append("# D1-b — Expressive-markings (technique) baseline on GuitarSet")
    lines.append("")
    lines.append(
        "**Diagnostic baseline, not a gate.** First-ever technique-F1 "
        "measurement, to replace the unbaselined `>= 0.70` stretch in "
        "SPEC §1.4 with an honest restatement (§0 rule 7)."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        "- **Operational technique-detection F1 = `0.00`** (structural, "
        "not stochastic). The shipping `highres` backend emits no "
        "technique tags (`highres.py` builds every `AudioEvent` with "
        "empty `tags`; fusion copies `tags`->`TabEvent.techniques` "
        "unchanged). The only tag-emitting path (`basicpitch.py` bend "
        "heuristic) is not installed. Zero detections -> zero recall -> "
        "F1 = 0.00 against any non-empty gold."
    )
    lines.append(
        "- **GuitarSet cannot baseline hammer-ons / pull-offs.** They "
        "are articulation, not pitch, and GuitarSet has no discrete "
        "technique labels — only `pitch_contour`. HO/PO need a "
        "technique-labelled corpus (Guitar-TECHS, electric -> v2)."
    )
    lines.append(
        "- **Bends and slides are derivable as proxies** from "
        "`pitch_contour`; their **support** (below) bounds how precise "
        "any future technique target could be."
    )
    lines.append("")

    for split in splits:
        agg_primary = results[split][BEND_SEMITONES_PRIMARY]
        lines.append(
            f"## Proxy support — split `{split}` "
            f"({agg_primary.n_tracks} tracks, {agg_primary.n_notes} notes)"
        )
        lines.append("")
        lines.append(
            "| Technique | Proxy source | Count | % of notes | "
            "Wilson 95% half-width @ that support |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for thr in (BEND_SEMITONES_PRIMARY, BEND_SEMITONES_SUBTLE):
            c = results[split][thr]
            hw = _wilson_halfwidth(0.70, c.n_bend)  # noise on a hypothetical 0.70 F1
            kind = "clear" if thr >= 1.0 else "incl. microbend"
            lines.append(
                f"| Bend, {kind} (>= {thr} st sustained shift) | "
                f"`pitch_contour` within-note | "
                f"{c.n_bend} | {c.bend_pct:.2f}% | +/- {hw:.3f} |"
            )
        c = results[split][BEND_SEMITONES_PRIMARY]
        hw_s = _wilson_halfwidth(0.70, c.n_slide)
        lines.append(
            f"| Slide (legato glide {SLIDE_MIN_PITCH_STEP}-{SLIDE_MAX_PITCH_STEP} st) "
            f"| `pitch_contour` cross-note | {c.n_slide} | {c.slide_pct:.2f}% | "
            f"+/- {hw_s:.3f} |"
        )
        lines.append("| Hammer-on / pull-off | **not derivable** | n/a | n/a | n/a |")
        lines.append("")

    lines.append("## Interpretation — honest stretch")
    lines.append("")
    val = results.get("validation", results[splits[0]])[BEND_SEMITONES_PRIMARY]
    val_sub = results.get("validation", results[splits[0]])[BEND_SEMITONES_SUBTLE]
    lines.append(
        f"On the canonical validation split (player {validation_player}) there are "
        f"~{val.n_bend} clear-bend + ~{val.n_slide} slide proxies across "
        f"{val.n_notes} notes — enough support for a "
        f"~+/- {_wilson_halfwidth(0.70, val.n_bend):.2f} F1 CI on bends (slides "
        f"thinner, +/- {_wilson_halfwidth(0.70, val.n_slide):.2f}). **Support is "
        f"not the blocker; label quality is.** The 'gold' is a threshold-sensitive "
        f"`pitch_contour` heuristic — the bend count nearly triples "
        f"({val.bend_pct:.1f}% -> {val_sub.bend_pct:.1f}% of notes) between the "
        f"1.0-st and 0.5-st thresholds — not human technique annotation, so "
        f"scoring a detector against it measures agreement-with-a-heuristic, not "
        f"true technique F1. Combined with a **0.00** detector and **unmeasurable** "
        f"hammer-ons/pull-offs, a numeric technique target (the old 0.70) is not "
        f"yet defensible."
    )
    lines.append("")
    lines.append("**Recommended restatement for SPEC §1.4 / §15:**")
    lines.append("")
    lines.append(
        "1. Baseline (2026-07-09): operational technique-detection F1 = "
        "**0.00** — no detector is wired into the default path."
    )
    lines.append(
        "2. GuitarSet baselines **bends & slides only** (proxy, via "
        "`pitch_contour`); **hammer-ons/pull-offs are out until a "
        "technique-labelled corpus is in scope** (Guitar-TECHS -> v2)."
    )
    lines.append(
        "3. First honest milestone = **build any bend/slide detector "
        "and beat 0.00** on this proxy. Defer a numeric F1 target (the "
        "old 0.70) until a detector exists *and* is measured against "
        "**human** technique labels — not this threshold-sensitive "
        "`pitch_contour` heuristic, which measures agreement-with-a-"
        "heuristic rather than true technique F1."
    )
    lines.append("")
    lines.append(
        "_Proxy thresholds are documented in the script; counts are "
        "threshold-sensitive (two settings shown). The 0.00 baseline is "
        "not — it is a structural absence of any detector._"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_home = os.environ.get(
        "TABVISION_DATA_ROOT", str(Path("~/.tabvision/data").expanduser())
    )
    parser.add_argument(
        "--data-home",
        default=default_home,
        help="dir containing guitarset/annotation/*.jams (or $TABVISION_DATA_ROOT)",
    )
    parser.add_argument("--validation-player", default="05")
    parser.add_argument("--output", default=None, help="path to write the markdown report")
    args = parser.parse_args(argv)

    data_home = Path(args.data_home).expanduser()
    try:
        report = build_report(data_home=data_home, validation_player=args.validation_player)
    except (FileNotFoundError, ValueError) as exc:
        print(f"setup_blocker={exc}", file=sys.stderr)
        return 2

    print(report)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nreport={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
