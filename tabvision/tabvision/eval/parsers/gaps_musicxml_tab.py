"""GAPS (Guitar-Aligned Performance Scores) MusicXML-tab annotation parser.

GAPS (ISMIR 2024; arXiv:2408.08653) ships, per piece ``<stem>``, three files
under a shared dataset root::

    <root>/musicxml/<stem>.xml      # score with a TAB part carrying <string>/<fret>
    <root>/midi/<stem>.mid          # high-resolution note-level performance MIDI
    <root>/syncpoints/<stem>.json   # [measure_index, downbeat_seconds[, tempo]] anchors

The ``annotation_path`` passed to :func:`parse` is the **MusicXML** file; the
MIDI and syncpoints siblings are derived by convention. None of the three files
alone yields onset-timed tablature:

- the MusicXML TAB part has ``(string, fret)`` per note but only *score* time;
- the aligned MIDI has exact *performance* onsets (seconds) but no string/fret;
- the syncpoints map score measures to performance seconds (coarse, per-measure).

Gold derivation (validated in the chunk-5 recon, 2026-06-18; repeat unfolding
added in A6, 2026-07-06):

0. **Unfold repeats/voltas** (A6) so the walked measure sequence matches the
   *performance* timeline the syncpoints index (measure 0..N counting repeat
   traversals), not the once-listed written score. Trusted only when the
   unfolded length matches the syncpoint span; otherwise the written order is
   kept (nonstandard volta encodings fall back safely). See
   :func:`_unfold_measures`.
1. Walk the TAB part (the part whose notes carry ``<string>``), honouring
   ``<backup>``/``<forward>``/``<chord>`` so each note gets a correct absolute
   *score* onset in divisions, and reading ``<staff-tuning>`` so scordatura
   (e.g. drop-D) pieces validate. ``our_string_idx = 6 - musicxml_string``
   (MusicXML string 1 = high E; TabVision ``string_idx`` 0 = low E).
2. Warp each note's score onset to approximate seconds via per-measure
   syncpoint interpolation.
3. **Snap to the exact MIDI onset**: for each pitch, monotonically align the
   score notes (warp-time order) to the MIDI notes (time order) maximizing the
   number of matched pairs, using the warp time only as a tiebreaker for where
   the unavoidable gaps (ornaments the written score lists once but the
   performer played more) land. Each matched note takes the MIDI note's exact
   onset and duration; its ``(string, fret)`` comes from the score.

Notes the score lists but the performer did not play (no MIDI counterpart) are
dropped. Before A6 the gold was the *first-traversal-biased* subset (repeat
traversals had no score counterpart, so the performer's replayed notes counted
as false positives against the model); unfolding restores that gold on clips
whose repeat structure is standard.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from tabvision.eval.parsers.registry import register_parser
from tabvision.types import GuitarConfig, TabEvent

FORMAT_NAME = "gaps_musicxml_tab"

_STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Alignment cost model (see module docstring step 3). Gaps cost far more than
# any time-disagreement term so the DP maximises matched pairs (LCS-like) and
# only uses |dt| to place the unavoidable gaps.
_GAP_COST = 5.0

# A6: repeat/volta unfolding. The GAPS syncpoints index the *unfolded*
# performance timeline (measure 0..N where N counts repeat traversals), while
# the written score lists each measure once. We unfold the score to the
# performance order so second-traversal notes get gold (they were dropped
# before — the "first-traversal-biased" artifact the docstring flags). Unfolding
# is only trusted when its length matches the syncpoint span within this
# tolerance (a pickup measure / final-barline marker accounts for ±1); otherwise
# a nonstandard volta encoding is assumed and we fall back to the written order.
_REPEAT_SYNC_TOLERANCE = 3


def _pitch_to_midi(step: str, alter: int, octave: int) -> int:
    return 12 * (octave + 1) + _STEP_TO_SEMITONE[step] + alter


class _ScoreNote:
    __slots__ = ("score_onset", "pitch", "mxml_string", "fret")

    def __init__(self, score_onset: int, pitch: int, mxml_string: int, fret: int) -> None:
        self.score_onset = score_onset
        self.pitch = pitch
        self.mxml_string = mxml_string
        self.fret = fret


def _tab_part(root: ET.Element) -> ET.Element:
    """Return the ``<part>`` whose notes carry ``<string>`` (the TAB staff).

    GAPS MusicXML has two parts: standard notation (P1) and tablature (P2);
    only the latter carries ``<technical><string>/<fret>``.
    """
    for part in root.findall(".//part"):
        if part.find(".//string") is not None:
            return part
    raise ValueError("No MusicXML part carries <string> tab markings")


def read_staff_tuning(annotation_path: str | Path) -> dict[int, int]:
    """Public helper: MusicXML ``<string>`` number (1..6) -> open-string MIDI.

    Used by the eval manifest builder's standard-tuning filter so it can decide
    whether a GAPS clip is playable by the standard-tuning pipeline without
    importing ``pretty_midi`` or parsing the MIDI. Returns an empty dict when
    the score omits staff-tuning.
    """
    return _staff_tuning(_tab_part(ET.parse(annotation_path).getroot()))


def _staff_tuning(part: ET.Element) -> dict[int, int]:
    """Map MusicXML ``<string>`` number (1..6) -> open-string MIDI pitch.

    ``<staff-tuning line="L">`` counts tab lines bottom (1) to top (6); MusicXML
    string number S sits on line ``7 - S`` (string 1 = top line = highest).
    Returns an empty dict if the score omits staff-tuning (standard assumed).
    """
    out: dict[int, int] = {}
    for st in part.findall(".//staff-tuning"):
        line = int(st.get("line", "0"))
        step_el = st.find("tuning-step")
        octave_el = st.find("tuning-octave")
        if step_el is None or octave_el is None or step_el.text is None or octave_el.text is None:
            continue
        alter_el = st.find("tuning-alter")
        alter = int(alter_el.text) if alter_el is not None and alter_el.text else 0
        out[7 - line] = _pitch_to_midi(step_el.text, alter, int(octave_el.text))
    return out


class _RepeatMarks:
    """Per-measure repeat/volta markings read from ``<barline>`` children."""

    __slots__ = ("forward", "backward_times", "ending_start", "ending_stop")

    def __init__(self) -> None:
        self.forward = False
        self.backward_times: int | None = None
        self.ending_start: set[int] | None = None
        self.ending_stop = False


def _measure_marks(measure: ET.Element) -> _RepeatMarks:
    marks = _RepeatMarks()
    for bar in measure.findall("barline"):
        rep = bar.find("repeat")
        if rep is not None:
            direction = rep.get("direction")
            if direction == "forward":
                marks.forward = True
            elif direction == "backward":
                times = rep.get("times")
                marks.backward_times = int(times) if times and times.isdigit() else 2
        end = bar.find("ending")
        if end is not None:
            etype = end.get("type")
            if etype == "start":
                marks.ending_start = {
                    int(x) for x in end.get("number", "").replace(" ", "").split(",") if x.isdigit()
                }
            elif etype in ("stop", "discontinue"):
                marks.ending_stop = True
    return marks


def _has_any_repeat(measures: list[ET.Element]) -> bool:
    return any(m.find(".//repeat") is not None for m in measures)


def _unfold_measures(measures: list[ET.Element]) -> list[int]:
    """Expand simple repeats + 1st/2nd voltas into performance measure order.

    Returns the list of ``measures`` indices in the order the performer plays
    them. Handles forward/backward repeat barlines (with ``times``) and voltas
    whose ``<ending number>`` selects the pass. A backward repeat with no
    matching forward repeat loops from the start of the piece (or the end of the
    previous repeated section). Nested repeats are not modelled; the guard bounds
    pathological input and the caller validates the length before trusting it.
    """
    marks = [_measure_marks(m) for m in measures]
    n = len(measures)
    order: list[int] = []
    played: dict[int, int] = {}
    i = 0
    repeat_start = 0
    cur_pass = 1
    jumped_back = False
    guard = 0
    while i < n and guard < n * 32:
        guard += 1
        mk = marks[i]
        # A forward repeat opens a new section (pass 1) — unless we just looped
        # back into it, in which case the pass counter must be preserved.
        if mk.forward and not jumped_back:
            repeat_start = i
            cur_pass = 1
        jumped_back = False
        # Skip a volta whose ending number does not include the current pass.
        if mk.ending_start is not None and cur_pass not in mk.ending_start:
            j = i
            while j < n and not marks[j].ending_stop:
                j += 1
            i = j + 1
            continue
        order.append(i)
        if mk.backward_times is not None:
            played[i] = played.get(i, 0) + 1
            if played[i] < mk.backward_times:
                cur_pass += 1
                i = repeat_start
                jumped_back = True
                continue
            # Section finished; a later no-forward repeat loops from here.
            repeat_start = i + 1
            cur_pass = 1
        i += 1
    return order


def _walk_tab_notes(measures: list[ET.Element]) -> tuple[list[_ScoreNote], list[int], int]:
    """Return (notes, measure_start_divisions, total_divisions).

    ``measures`` is the (possibly repeat-unfolded) ordered list of ``<measure>``
    elements. Each note's ``score_onset`` is an absolute division offset from the
    start of the piece, so a measure that appears twice (a repeat traversal)
    yields two distinct onset ranges. ``<backup>``/``<forward>`` move the
    in-measure cursor; ``<chord>`` notes share the previous note's onset; grace
    notes consume no time.
    """
    notes: list[_ScoreNote] = []
    measure_starts: list[int] = []
    abs_div = 0
    for measure in measures:
        measure_start = abs_div
        measure_starts.append(measure_start)
        cursor = measure_start
        last_onset = measure_start
        for el in measure:
            if el.tag == "attributes":
                pass  # divisions read implicitly via <duration>; not needed per-note
            elif el.tag == "backup":
                cursor -= _duration(el)
            elif el.tag == "forward":
                cursor += _duration(el)
            elif el.tag == "note":
                is_chord = el.find("chord") is not None
                is_grace = el.find("grace") is not None
                duration = _duration(el)
                onset = last_onset if is_chord else cursor
                if not is_chord:
                    last_onset = cursor
                pitch_el = el.find("pitch")
                string_el = el.find(".//string")
                fret_el = el.find(".//fret")
                if (
                    pitch_el is not None
                    and el.find("rest") is None
                    and string_el is not None
                    and string_el.text is not None
                    and fret_el is not None
                    and fret_el.text is not None
                ):
                    step = pitch_el.find("step")
                    octave = pitch_el.find("octave")
                    if step is not None and step.text and octave is not None and octave.text:
                        alter_el = pitch_el.find("alter")
                        alter = int(alter_el.text) if alter_el is not None and alter_el.text else 0
                        notes.append(
                            _ScoreNote(
                                score_onset=onset,
                                pitch=_pitch_to_midi(step.text, alter, int(octave.text)),
                                mxml_string=int(string_el.text),
                                fret=int(fret_el.text),
                            )
                        )
                if not is_chord and not is_grace:
                    cursor += duration
        abs_div = max(cursor, measure_start)
    return notes, measure_starts, abs_div


def _duration(el: ET.Element) -> int:
    dur = el.find("duration")
    return int(dur.text) if dur is not None and dur.text else 0


def _load_syncpoints(path: Path) -> dict[int, float]:
    """measure-index -> downbeat seconds (first occurrence wins; entries may be
    2- or 3-tuples ``[pos, sec]`` / ``[pos, sec, tempo]``)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    pos_to_sec: dict[int, float] = {}
    for entry in raw:
        pos = int(entry[0])
        if pos not in pos_to_sec:
            pos_to_sec[pos] = float(entry[1])
    return pos_to_sec


def _per_pitch_align(
    score_times: list[tuple[float, int]],
    midi_times: list[tuple[float, int]],
) -> list[tuple[int, int]]:
    """Monotonically align same-pitch score notes to MIDI notes.

    Both inputs are ``(time, payload_index)`` sorted by time. Returns matched
    ``(score_index, midi_index)`` pairs maximizing match count, tiebroken by
    minimal total ``|score_time - midi_time|``.
    """
    n, m = len(score_times), len(midi_times)
    if n == 0 or m == 0:
        return []
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]  # 0=match, 1=skip-score, 2=skip-midi
    for i in range(1, n + 1):
        dp[i][0] = i * _GAP_COST
        back[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j * _GAP_COST
        back[0][j] = 2
    for i in range(1, n + 1):
        si = score_times[i - 1][0]
        row = dp[i]
        prev = dp[i - 1]
        brow = back[i]
        for j in range(1, m + 1):
            match = prev[j - 1] + abs(si - midi_times[j - 1][0])
            skip_score = prev[j] + _GAP_COST
            skip_midi = row[j - 1] + _GAP_COST
            best, choice = match, 0
            if skip_score < best:
                best, choice = skip_score, 1
            if skip_midi < best:
                best, choice = skip_midi, 2
            row[j] = best
            brow[j] = choice
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        choice = back[i][j]
        if choice == 0:
            pairs.append((score_times[i - 1][1], midi_times[j - 1][1]))
            i -= 1
            j -= 1
        elif choice == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def parse(
    annotation_path: str | Path,
    cfg: GuitarConfig | None = None,
) -> list[TabEvent]:
    """Parse a GAPS ``musicxml/<stem>.xml`` into onset-timed gold tab events.

    Derives the ``midi/`` and ``syncpoints/`` siblings from ``annotation_path``.
    Requires ``pretty_midi`` (ships with the ``audio-highres`` extra).
    """
    try:
        import pretty_midi  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - skip path
        raise ImportError(
            "gaps_musicxml_tab parser requires pretty_midi. Install with: "
            "pip install -e 'tabvision[audio-highres]'"
        ) from exc

    if cfg is None:
        cfg = GuitarConfig()

    xml_path = Path(annotation_path)
    stem = xml_path.stem
    root_dir = xml_path.parent.parent
    midi_path = root_dir / "midi" / f"{stem}.mid"
    sync_path = root_dir / "syncpoints" / f"{stem}.json"
    if not midi_path.is_file():
        raise FileNotFoundError(f"GAPS MIDI sibling not found: {midi_path}")
    if not sync_path.is_file():
        raise FileNotFoundError(f"GAPS syncpoints sibling not found: {sync_path}")

    part = _tab_part(ET.parse(xml_path).getroot())
    all_measures = part.findall("measure")
    pos_to_sec = _load_syncpoints(sync_path)
    fallback_sec = max(pos_to_sec.values()) if pos_to_sec else 0.0

    # A6: unfold repeats/voltas to the performance timeline the syncpoints index,
    # so second-traversal notes get gold. Trust the unfold only when its length
    # matches the syncpoint span; otherwise fall back to the written order.
    # ``TABVISION_GAPS_NO_UNFOLD`` forces the pre-A6 behaviour for A/B measurement.
    ordered_measures = all_measures
    if _has_any_repeat(all_measures) and not os.environ.get("TABVISION_GAPS_NO_UNFOLD"):
        unfolded = _unfold_measures(all_measures)
        sync_span = (max(pos_to_sec) + 1) if pos_to_sec else 0
        if sync_span and abs(len(unfolded) - sync_span) <= _REPEAT_SYNC_TOLERANCE:
            ordered_measures = [all_measures[i] for i in unfolded]

    notes, measure_starts, total_div = _walk_tab_notes(ordered_measures)

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    midi_notes = sorted(
        (float(note.start), float(note.end), int(note.pitch))
        for inst in midi.instruments
        for note in inst.notes
    )

    # Map each note's absolute score-onset divisions to approximate seconds by
    # linear interpolation inside its measure between consecutive syncpoints.
    start_to_index = {start: idx for idx, start in enumerate(measure_starts)}

    def warp(note: _ScoreNote) -> float:
        # The note's measure is the largest measure_start <= its onset.
        idx = start_to_index.get(note.score_onset)
        if idx is None:
            # chord/backup onsets land mid-measure; find the enclosing measure.
            idx = _enclosing_measure(measure_starts, note.score_onset)
        d0 = measure_starts[idx]
        d1 = measure_starts[idx + 1] if idx + 1 < len(measure_starts) else total_div
        t0 = pos_to_sec.get(idx, fallback_sec)
        t1 = pos_to_sec.get(idx + 1, fallback_sec)
        if d1 == d0:
            return t0
        return t0 + (note.score_onset - d0) / (d1 - d0) * (t1 - t0)

    score_by_pitch: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for i, note in enumerate(notes):
        score_by_pitch[note.pitch].append((warp(note), i))
    midi_by_pitch: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for j, (start, _end, pitch) in enumerate(midi_notes):
        midi_by_pitch[pitch].append((start, j))

    events: list[TabEvent] = []
    for pitch, score_list in score_by_pitch.items():
        score_list.sort()
        midi_list = sorted(midi_by_pitch.get(pitch, []))
        for score_idx, midi_idx in _per_pitch_align(score_list, midi_list):
            note = notes[score_idx]
            string_idx = 6 - note.mxml_string
            if not (0 <= string_idx < cfg.n_strings) or not (cfg.capo <= note.fret <= cfg.max_fret):
                continue
            start, end, _pitch = midi_notes[midi_idx]
            events.append(
                TabEvent(
                    onset_s=start,
                    duration_s=max(0.0, end - start),
                    string_idx=string_idx,
                    fret=note.fret,
                    pitch_midi=note.pitch,
                    confidence=1.0,
                )
            )

    events.sort(key=lambda event: (event.onset_s, event.string_idx, event.fret))
    return events


def _enclosing_measure(measure_starts: list[int], onset: int) -> int:
    """Index of the measure whose start is the largest <= ``onset``."""
    import bisect  # noqa: PLC0415

    idx = bisect.bisect_right(measure_starts, onset) - 1
    return max(0, idx)


register_parser(FORMAT_NAME, parse)


__all__ = ["FORMAT_NAME", "parse", "read_staff_tuning"]
