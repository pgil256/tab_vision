"""Fusion engine for combining audio and video analysis into tab notes."""
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4
from app.audio_pipeline import DetectedNote, MutedNote, group_notes_into_chords
from app.guitar_mapping import get_candidate_positions, pick_lowest_fret, Position, STANDARD_TUNING, MAX_FRET
from app.video_pipeline import HandObservation, FingerPosition
from app.fretboard_detection import FretboardGeometry, map_finger_to_position
from app.chord_shapes import (
    ChordShapeConfig, PlayingStyle, StyleWeights, STYLE_WEIGHTS,
    find_best_voicing_for_chord, get_voicing_positions,
    score_positions_against_scale, find_matching_voicings,
    score_position_against_voicing, GUITAR_POSITIONS, get_position_for_fret,
)


@dataclass
class TabNote:
    """A note in the guitar tablature."""
    id: str
    timestamp: float        # seconds
    string: int             # 1-6
    fret: int | str         # 0-24 or "X" for muted
    confidence: float       # 0.0-1.0
    confidence_level: str   # "high", "medium", "low"
    midi_note: int          # Original MIDI note for debugging
    # Enhanced attributes
    end_time: Optional[float] = None  # Note end time
    technique: Optional[str] = None   # "normal", "hammer-on", "pull-off", "slide", "bend"
    is_part_of_chord: bool = False
    chord_id: Optional[str] = None    # Groups notes in same chord
    video_matched: bool = False       # Whether video confirmed this position
    audio_confidence: float = 0.0     # Confidence from audio alone
    video_confidence: float = 0.0     # Confidence from video alone
    pitch_bend: float = 0.0          # Pitch bend/vibrato amount from audio


@dataclass
class Chord:
    """A group of simultaneous notes."""
    id: str
    timestamp: float
    notes: list[TabNote]
    confidence: float
    confidence_level: str

    @property
    def duration(self) -> float:
        """Chord duration based on shortest note."""
        if not self.notes or not any(n.end_time for n in self.notes):
            return 0.0
        end_times = [n.end_time for n in self.notes if n.end_time]
        return min(end_times) - self.timestamp if end_times else 0.0


@dataclass
class FusionConfig:
    """Configuration for the fusion engine."""
    # Time tolerance for matching audio to video observations
    video_match_tolerance: float = 0.1  # seconds

    # Confidence adjustments
    video_match_boost: float = 0.3      # Boost when video confirms audio (increased)
    open_string_confidence: float = 0.7   # Confidence for inferred open strings (increased)
    no_match_penalty: float = 0.05      # Penalty when video contradicts audio (reduced)

    # Chord detection
    chord_time_tolerance: float = 0.05  # Max time difference for notes in chord

    # Technique detection thresholds
    hammer_on_max_gap: float = 0.15     # Max gap between notes for hammer-on
    slide_pitch_threshold: int = 2      # Min semitones for slide detection

    # Position selection preferences
    prefer_lower_frets: bool = True
    prefer_same_position: bool = True    # Prefer keeping hand in same position

    # Multi-frame video analysis
    use_multi_frame_video: bool = True
    multi_frame_window: float = 0.1     # seconds around onset

    # Pre-filtering to remove false positive notes
    enable_prefiltering: bool = True
    ghost_note_amplitude_threshold: float = 0.40   # absolute amp floor for ghost detection
    ghost_note_ratio: float = 0.6                  # remove if amp < this * loudest overlapping
    chord_fragment_window: float = 2.0             # seconds to look back for re-detections
    chord_fragment_amplitude_ratio: float = 0.9    # remove if amp < this * original
    max_chord_size: int = 3                        # trim oversized chords by amplitude

    # Muted note detection
    muted_note_min_confidence: float = 0.4           # minimum confidence to include muted note
    muted_note_audio_only_threshold: float = 0.6     # higher threshold without video confirmation
    muted_note_video_boost: float = 0.2              # confidence boost for video-confirmed muted

    # Chord shape and style heuristics
    chord_shape_config: ChordShapeConfig = field(default_factory=ChordShapeConfig)
    playing_style: PlayingStyle = PlayingStyle.DEFAULT


def _create_muted_tab_notes(
    muted_notes: list[MutedNote],
    video_observations: dict | None = None,
    config: FusionConfig | None = None,
) -> list[TabNote]:
    """Convert muted notes into TabNote objects with fret='X'.

    Args:
        muted_notes: Detected muted/percussive notes from audio
        video_observations: Optional video data for confirmation
        config: Fusion configuration

    Returns:
        List of TabNote objects with fret="X"
    """
    if config is None:
        config = FusionConfig()

    tab_notes = []
    for mn in muted_notes:
        # Check video confirmation
        video_confirmed = False
        if video_observations:
            # Look for muting fingers at this timestamp (within tolerance)
            for ts, obs in video_observations.items():
                if abs(ts - mn.timestamp) <= 0.1:
                    if hasattr(obs, 'muting_fingers') and obs.muting_fingers:
                        video_confirmed = True
                        break

        # Apply confidence thresholds
        if mn.confidence < config.muted_note_min_confidence:
            continue
        if not video_confirmed and mn.confidence < config.muted_note_audio_only_threshold:
            continue

        # Calculate final confidence
        final_confidence = mn.confidence
        if video_confirmed:
            final_confidence = min(1.0, mn.confidence + config.muted_note_video_boost)

        tab_notes.append(TabNote(
            id=str(uuid4()),
            timestamp=mn.timestamp,
            string=6,
            fret="X",
            confidence=final_confidence,
            confidence_level="low" if final_confidence < 0.5 else "medium",
            midi_note=0,
            technique="muted",
            video_matched=video_confirmed,
        ))

    return tab_notes


def get_confidence_level(confidence: float) -> str:
    """Map confidence score to level.

    Args:
        confidence: Score from 0.0 to 1.0

    Returns:
        "high" (>0.8), "medium" (0.5-0.8), or "low" (<0.5)
    """
    if confidence > 0.8:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    else:
        return "low"


def _prefilter_notes(
    notes: list[DetectedNote],
    config: FusionConfig
) -> list[DetectedNote]:
    """Remove ghost notes and chord fragment re-detections.

    Ghost notes: low-amplitude notes that overlap temporally with much louder notes
    (sympathetic vibrations, resonance artifacts).

    Chord fragments: a note from a previous chord re-detected as a new event
    because it was still sustaining.

    Args:
        notes: Detected notes
        config: Fusion configuration

    Returns:
        Filtered notes with ghosts and fragments removed
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.start_time)

    # Pass 1: Remove ghost notes (low-amplitude overlapping with loud notes)
    keep_after_ghost = []
    for note in sorted_notes:
        if note.amplitude >= config.ghost_note_amplitude_threshold:
            keep_after_ghost.append(note)
            continue

        # Low amplitude note - check if it overlaps with much louder notes
        max_overlapping_amp = 0.0
        for other in sorted_notes:
            if other is note:
                continue
            # Check temporal overlap
            if other.start_time < note.end_time and other.end_time > note.start_time:
                max_overlapping_amp = max(max_overlapping_amp, other.amplitude)

        if max_overlapping_amp > 0 and note.amplitude < max_overlapping_amp * config.ghost_note_ratio:
            continue  # Ghost note - skip
        keep_after_ghost.append(note)

    # Pass 2: Remove chord fragment re-detections
    # A "chord fragment" is a note that was already playing in a previous chord
    # and gets re-detected. Two strategies:
    # (a) For notes in multi-note chords: check amplitude ratio
    # (b) For isolated single notes: check if the pitch was recently active
    chords = group_notes_into_chords(keep_after_ghost, tolerance=0.05)
    keep = []
    recent_notes: list[DetectedNote] = []  # rolling window of recent notes

    for chord in chords:
        filtered_chord = []

        # Determine which pitches in this chord are "new" vs "continuing"
        new_pitches_in_chord = set()
        continuing_pitches = set()
        for note in chord:
            is_continuing = False
            for prev in recent_notes:
                if prev.midi_note != note.midi_note:
                    continue
                # Same pitch - check if previous note was still ringing
                if prev.end_time >= note.start_time - 0.15:
                    is_continuing = True
                    break
            if is_continuing:
                continuing_pitches.add(note.midi_note)
            else:
                new_pitches_in_chord.add(note.midi_note)

        for note in chord:
            is_fragment = False

            if note.midi_note in continuing_pitches:
                # This pitch was already ringing from a previous chord
                if len(chord) == 1:
                    # Isolated single note - remove if amplitude dropped vs original
                    for prev in recent_notes:
                        if prev.midi_note != note.midi_note:
                            continue
                        if prev.end_time >= note.start_time - 0.15:
                            if note.amplitude < prev.amplitude * 0.9:
                                is_fragment = True
                                break
                elif new_pitches_in_chord:
                    # This chord has some new pitches - keep the continuing ones
                    # only if they have reasonable amplitude (actual re-pluck)
                    for prev in recent_notes:
                        if prev.midi_note != note.midi_note:
                            continue
                        if prev.end_time >= note.start_time - 0.15:
                            if note.amplitude < prev.amplitude * config.chord_fragment_amplitude_ratio:
                                is_fragment = True
                                break
                elif len(chord) == 2:
                    # All chord members are continuing pitches (2-note chord).
                    # Remove notes whose amplitude dropped while the other
                    # member's amplitude increased (re-articulation exciting
                    # a sustaining note).
                    has_reattack = False
                    for other in chord:
                        if other is note:
                            continue
                        for prev in recent_notes:
                            if prev.midi_note != other.midi_note:
                                continue
                            if prev.end_time >= other.start_time - 0.15:
                                if other.amplitude >= prev.amplitude:
                                    has_reattack = True
                                break
                    if has_reattack:
                        for prev in recent_notes:
                            if prev.midi_note != note.midi_note:
                                continue
                            if prev.end_time >= note.start_time - 0.15:
                                if note.amplitude < prev.amplitude * config.chord_fragment_amplitude_ratio:
                                    is_fragment = True
                                break

            if not is_fragment:
                filtered_chord.append(note)

        keep.extend(filtered_chord)
        # Update recent notes (rolling window)
        recent_notes = [n for n in recent_notes
                       if chord[0].start_time - n.start_time < config.chord_fragment_window]
        recent_notes.extend(filtered_chord)

    return keep


def _limit_chord_sizes(
    chords: list[list[DetectedNote]],
    config: FusionConfig
) -> list[list[DetectedNote]]:
    """Trim oversized chords by keeping highest-amplitude notes.

    A real guitar chord event typically has 1-4 simultaneous notes.
    Oversized chord groups (5-6+) usually contain overtone artifacts.

    Args:
        chords: List of chord groups
        config: Fusion configuration

    Returns:
        Chords with size limited to max_chord_size
    """
    result = []
    for chord in chords:
        if len(chord) <= config.max_chord_size:
            result.append(chord)
        else:
            # Keep the highest-amplitude notes
            trimmed = sorted(chord, key=lambda n: n.amplitude, reverse=True)[:config.max_chord_size]
            # Restore time ordering
            trimmed.sort(key=lambda n: n.start_time)
            result.append(trimmed)
    return result


def _postfilter_tab_notes(
    tab_notes: list[TabNote],
    config: FusionConfig
) -> list[TabNote]:
    """Remove post-fusion artifacts: duplicate positions and low-confidence strays.

    Two filters applied in order:
    1. Dedup: same string+fret within 0.3s -> keep highest confidence
    2. Low-confidence isolated: conf < 0.6, not in chord -> remove

    Args:
        tab_notes: Tab notes from fusion
        config: Fusion configuration

    Returns:
        Filtered tab notes
    """
    if not tab_notes:
        return []

    sorted_notes = sorted(tab_notes, key=lambda n: n.timestamp)

    # Pass 1: Dedup same string+fret within 0.3s window
    dedup_window = 0.3
    keep_after_dedup = []
    skip_indices = set()

    for i, note in enumerate(sorted_notes):
        if i in skip_indices:
            continue
        # Look ahead for duplicates
        best = note
        for j in range(i + 1, len(sorted_notes)):
            other = sorted_notes[j]
            if other.timestamp - note.timestamp > dedup_window:
                break
            if other.string == note.string and other.fret == note.fret:
                skip_indices.add(j)
                if other.confidence > best.confidence:
                    best = other
        if best is not note:
            skip_indices.add(i)
        keep_after_dedup.append(best)

    # Pass 2: Remove low-confidence isolated singletons
    # Open strings use a higher threshold since they're prone to sympathetic resonance
    low_conf_threshold = 0.6
    open_string_conf_threshold = 0.7
    result = []
    for note in keep_after_dedup:
        if not note.is_part_of_chord:
            threshold = open_string_conf_threshold if note.fret == 0 else low_conf_threshold
            if note.confidence < threshold:
                continue  # Remove low-confidence isolated note
        result.append(note)

    # Pass 3: Remove isolated open-string resonance
    # Open strings (fret 0) that are not in a chord and whose string is far
    # (2+ strings away) from all neighbors within 1.0s are likely resonance artifacts.
    neighbor_window = 1.0
    min_string_distance = 2  # must be 2+ strings away from ALL neighbors
    after_open = []
    for i, note in enumerate(result):
        if note.fret == 0 and not note.is_part_of_chord:
            # Check if all neighbors within window are on distant strings
            neighbors = [
                other for other in result
                if other is not note
                and abs(other.timestamp - note.timestamp) <= neighbor_window
            ]
            if neighbors:
                min_dist = min(abs(note.string - other.string) for other in neighbors)
                if min_dist >= min_string_distance:
                    continue  # Skip: isolated open string far from all neighbors
        after_open.append(note)

    # Pass 4: Remove sustained-into-chord artifacts
    # When a chord (3+ notes) has exactly ONE note whose string+fret was
    # recently played (within 1.5s) while the other chord members are new,
    # the repeated note is likely a sustain carryover, not a re-articulation.
    # Real re-strums repeat MULTIPLE chord members, not just one.
    sustain_window = 1.5
    after_sustain = []
    for note in after_open:
        if not note.is_part_of_chord or note.chord_id is None:
            after_sustain.append(note)
            continue

        # Get all notes in the same chord
        chord_notes = [n for n in after_open if n.chord_id == note.chord_id]
        if len(chord_notes) < 3:
            after_sustain.append(note)
            continue

        # Count how many chord members are repeats of recent notes
        repeats_in_chord = []
        for cn in chord_notes:
            is_repeat = False
            for prev in after_open:
                if prev.chord_id == cn.chord_id:
                    continue  # Skip same chord
                if (prev.string == cn.string and prev.fret == cn.fret
                        and 0 < cn.timestamp - prev.timestamp <= sustain_window):
                    is_repeat = True
                    break
            if is_repeat:
                repeats_in_chord.append(cn)

        # Only remove if exactly ONE note is a repeat (sustain carryover)
        if (len(repeats_in_chord) == 1
                and repeats_in_chord[0] is note):
            continue  # Skip: lone sustain carryover in a chord of new notes

        after_sustain.append(note)

    return after_sustain


def _estimate_initial_position(
    notes: list[DetectedNote],
    capo_fret: int = 0
) -> Optional[float]:
    """Estimate the initial hand position from the note content.

    Sweeps candidate hand positions (fret centres) from 0 to 15 and picks
    the one where the most notes have at least one candidate within a
    playable 4-fret reach.  Ties are broken by favouring lower positions.

    The fitness score for each candidate position P is the sum of
    ``max(0, 5 - min_distance(note, P))`` across all notes, where
    ``min_distance`` is the minimum |fret - P| over all candidate fret
    positions for that note.  This rewards positions where many notes
    fall close to P and penalises positions where notes must stretch far.

    Args:
        notes: All detected notes
        capo_fret: Capo position

    Returns:
        Estimated hand position fret, or None if fewer than 3 notes
    """
    if len(notes) < 3:
        return None

    # Pre-compute candidate frets for every note.  Notes whose minimum
    # candidate fret exceeds high_only_threshold are "high-only" notes
    # (can only be played in very high positions, usually false pitch
    # detections of overtones).  They're excluded from the sweep so they
    # don't drag the hand position estimate to fret 14+.
    high_only_threshold = 7
    note_candidate_frets: list[list[int]] = []
    for note in notes:
        candidates = get_candidate_positions(note.midi_note, capo_fret)
        if not candidates:
            continue
        frets = [c.fret for c in candidates]
        if min(frets) >= high_only_threshold:
            continue
        note_candidate_frets.append(frets)

    if not note_candidate_frets:
        return None

    reach = 4        # frets reachable from hand position centre
    open_bonus = 3   # extra reward for a note whose best candidate is an open string

    best_pos: Optional[float] = None
    best_score: float = -1.0

    # Sweep integer fret centres; cap at 12 since most guitar playing
    # stays at or below the 12th fret and higher positions are rarely the
    # right answer for ambiguous note sets.
    for hand_pos in range(capo_fret, 13):
        score = 0.0
        for frets in note_candidate_frets:
            # For each candidate fret, compute reward = proximity bonus + open-string bonus
            best_reward = 0.0
            for f in frets:
                proximity = max(0, reach + 1 - abs(f - hand_pos))
                extra = open_bonus if f == 0 else 0
                reward = proximity + extra
                if reward > best_reward:
                    best_reward = reward
            score += best_reward

        # Keep best; ties go to lower position (earlier condition wins)
        if score > best_score:
            best_score = score
            best_pos = float(hand_pos)

    return best_pos


def fuse_audio_only(
    detected_notes: list[DetectedNote],
    capo_fret: int = 0,
    config: Optional[FusionConfig] = None,
    muted_notes: list[MutedNote] | None = None
) -> list[TabNote]:
    """Convert detected audio notes to TabNotes using intelligent position selection.

    Uses ergonomic heuristics including:
    - Prefer lower frets (easier to play)
    - Prefer keeping hand position stable
    - Detect chords and optimize positions together

    Args:
        detected_notes: Notes detected from audio analysis
        capo_fret: Fret where capo is placed (0 = no capo)
        config: Fusion configuration

    Returns:
        List of TabNote objects
    """
    if config is None:
        config = FusionConfig()

    if not detected_notes:
        return []

    # Pre-filter to remove ghost notes and chord fragment re-detections
    if config.enable_prefiltering:
        detected_notes = _prefilter_notes(detected_notes, config)

    # Group notes into chords
    chords = group_notes_into_chords(detected_notes, config.chord_time_tolerance)

    # Limit chord sizes (trim oversized chords by amplitude)
    chords = _limit_chord_sizes(chords, config)

    tab_notes = []
    previous_position = None

    # Estimate initial hand position from note content
    all_notes = [n for chord in chords for n in chord]
    hand_position_fret = _estimate_initial_position(all_notes, capo_fret)

    # Two-pass approach:
    # Pass 1: Process large chords (3+ notes) to establish hand position anchors
    # Pass 2: Process all events using hand position context from nearest anchor
    chord_anchors = {}  # chord_index -> (avg_fret, num_notes)
    for i, chord_notes in enumerate(chords):
        if len(chord_notes) >= 3:  # Only use 3+ note chords as anchors
            chord_candidates = []
            for note in chord_notes:
                candidates = get_candidate_positions(note.midi_note, capo_fret)
                if candidates:
                    chord_candidates.append((note, candidates))
            if len(chord_candidates) >= 3:
                positions = _optimize_chord_positions(
                    chord_candidates, None, config, hand_position_fret=None
                )
                valid = [p for p in positions if p]
                if valid:
                    frets = [p.fret for p in valid if p.fret > 0]
                    if frets:
                        chord_anchors[i] = (sum(frets) / len(frets), len(valid))

    # Propagate anchors: for each chord event, find nearest anchor
    # Weighted by anchor strength (number of notes) and proximity
    def _get_nearest_anchor(chord_idx: int) -> Optional[float]:
        if not chord_anchors:
            return None
        best_anchor = None
        best_score = float('-inf')
        for k, (avg_fret, num_notes) in chord_anchors.items():
            distance = abs(k - chord_idx)
            if distance > 15:
                continue
            # Score: prefer close anchors with many notes
            score = num_notes * 2.0 - distance * 0.5
            if score > best_score:
                best_score = score
                best_anchor = avg_fret
        return best_anchor

    for i, chord_notes in enumerate(chords):
        chord_id = str(uuid4()) if len(chord_notes) > 1 else None

        # Get all candidates for each note in the chord
        chord_candidates = []
        for note in chord_notes:
            candidates = get_candidate_positions(note.midi_note, capo_fret)
            if candidates:
                chord_candidates.append((note, candidates))

        if not chord_candidates:
            continue

        # Use anchor-based hand position if available
        anchor = _get_nearest_anchor(i)
        effective_hand_pos = hand_position_fret
        if anchor is not None:
            if effective_hand_pos is None:
                effective_hand_pos = anchor
            else:
                # Blend: anchors are strong evidence
                effective_hand_pos = effective_hand_pos * 0.3 + anchor * 0.7

        # If single note, use simple selection
        if len(chord_candidates) == 1:
            note, candidates = chord_candidates[0]
            position = _select_best_position(
                candidates, previous_position, config,
                hand_position_fret=effective_hand_pos
            )
            if position:
                tab_note = _create_tab_note(
                    note, position, chord_id, audio_confidence=note.confidence
                )
                tab_notes.append(tab_note)
                previous_position = position
                # Update hand position with smoothing
                if hand_position_fret is None:
                    hand_position_fret = float(position.fret)
                else:
                    hand_position_fret = hand_position_fret * 0.7 + position.fret * 0.3
        else:
            # Multiple notes - optimize as chord
            chord_hand_pos = effective_hand_pos
            selected_positions = _optimize_chord_positions(
                chord_candidates, previous_position, config,
                hand_position_fret=chord_hand_pos
            )
            for (note, _), position in zip(chord_candidates, selected_positions):
                if position:
                    tab_note = _create_tab_note(
                        note, position, chord_id, audio_confidence=note.confidence
                    )
                    tab_notes.append(tab_note)

            # Update previous position and hand position from chord center
            if selected_positions:
                valid_positions = [p for p in selected_positions if p]
                if valid_positions:
                    avg_fret = sum(p.fret for p in valid_positions) / len(valid_positions)
                    previous_position = min(
                        valid_positions,
                        key=lambda p: abs(p.fret - avg_fret)
                    )
                    # Chords update hand position more aggressively
                    if hand_position_fret is None:
                        hand_position_fret = avg_fret
                    else:
                        hand_position_fret = hand_position_fret * 0.4 + avg_fret * 0.6

    # Post-processing: correct slide/legato positions
    # When consecutive notes are close in time and pitch, put them on same string
    tab_notes = _correct_slide_positions(tab_notes, capo_fret)

    # Post-processing: correct melodic segment string assignments
    # Ensures scale passages stay on one string rather than scattering across strings
    tab_notes = _correct_melodic_segments(tab_notes, capo_fret, config)

    # Post-filter: remove duplicate positions and low-confidence strays
    tab_notes = _postfilter_tab_notes(tab_notes, config)

    # Detect techniques (hammer-ons, pull-offs, slides)
    tab_notes = _detect_techniques(tab_notes, config)

    # Add muted notes as "X" fret TabNotes
    if muted_notes:
        muted_tab = _create_muted_tab_notes(muted_notes, config=config)
        tab_notes.extend(muted_tab)
        tab_notes.sort(key=lambda n: n.timestamp)

    return tab_notes


def _select_best_position(
    candidates: list[Position],
    previous_position: Optional[Position],
    config: FusionConfig,
    hand_position_fret: Optional[float] = None
) -> Optional[Position]:
    """Select the best position from candidates.

    Uses hand position tracking for continuity. When no hand position context
    is available, prefers lower frets. When hand position is established,
    strongly prefers staying near it. Also considers scale patterns and
    positional awareness when enabled.

    Args:
        candidates: Valid positions for the note
        previous_position: Previous note position (for continuity)
        config: Fusion configuration
        hand_position_fret: Estimated center fret of current hand position

    Returns:
        Best position, or None if no candidates
    """
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    style_weights = STYLE_WEIGHTS.get(
        config.playing_style, STYLE_WEIGHTS[PlayingStyle.DEFAULT]
    )

    # Score each candidate
    def score_position(pos: Position) -> float:
        score = 0.0

        # Prefer lower frets (weight from style)
        if config.prefer_lower_frets:
            score -= pos.fret * style_weights.lower_fret_weight

        # Open string bonus in low position: open strings are ergonomically
        # easy and help the algorithm cross strings correctly in scales
        if pos.fret == 0:
            if hand_position_fret is None or hand_position_fret <= 4:
                score += 0.3

        # Strong preference for staying near hand position
        if config.prefer_same_position and hand_position_fret is not None:
            fret_distance = abs(pos.fret - hand_position_fret)
            score -= fret_distance * style_weights.position_stay_weight
            # Extra stretch penalty beyond a 4-fret span from hand position
            if fret_distance > 4 and pos.fret > 0:
                score -= (fret_distance - 4) * 0.3

        # Prefer staying near previous position
        if config.prefer_same_position and previous_position:
            fret_distance = abs(pos.fret - previous_position.fret)
            string_distance = abs(pos.string - previous_position.string)
            score -= fret_distance * 0.15
            score -= string_distance * 0.05

        # Slight preference for middle strings (2-5)
        if 2 <= pos.string <= 5:
            score += 0.03

        # Positional pattern awareness: bonus for frets within the current
        # guitar position's expected range
        if (config.chord_shape_config.position_awareness_enabled
                and hand_position_fret is not None):
            guitar_pos = get_position_for_fret(round(hand_position_fret))
            if guitar_pos and guitar_pos.contains_fret(pos.fret):
                score += 0.1
            elif pos.fret > 0:
                # Penalty for notes outside the position range
                score -= 0.05

        # Scale pattern bonus: if this note fits within a recognized scale
        # pattern at the current position
        if (config.chord_shape_config.scale_pattern_enabled
                and hand_position_fret is not None):
            scale_score = score_positions_against_scale(
                [pos], round(hand_position_fret), config.playing_style
            )
            score += scale_score * 0.1

        return score

    return max(candidates, key=score_position)


def _optimize_chord_positions(
    chord_candidates: list[tuple[DetectedNote, list[Position]]],
    previous_position: Optional[Position],
    config: FusionConfig,
    hand_position_fret: Optional[float] = None
) -> list[Optional[Position]]:
    """Optimize positions for a chord (multiple simultaneous notes).

    Uses a two-phase approach:
    1. Find the best fret region that can accommodate all chord notes
    2. Assign notes to specific strings within that region

    Ensures:
    - No two notes on same string
    - Positions are ergonomically playable (within 4-fret span)
    - Minimizes hand span

    Args:
        chord_candidates: List of (note, candidates) tuples
        previous_position: Previous position for continuity
        config: Fusion configuration
        hand_position_fret: Estimated hand position center fret

    Returns:
        List of selected positions (same order as input)
    """
    if not chord_candidates:
        return []

    # --- Context-aware chord shape recognition ---
    # If the chord MIDI notes match a known voicing, try using that voicing's
    # positions directly. This handles cases like "G major chord should use
    # fret 3 on B string, not fret 8 on G string".
    shape_config = config.chord_shape_config
    voicing_assignments = None

    if shape_config.enabled and len(chord_candidates) >= shape_config.min_chord_notes:
        chord_midi = [note.midi_note for note, _ in chord_candidates]
        best_voicing = find_best_voicing_for_chord(
            chord_midi, hand_position_fret, config.playing_style
        )
        if best_voicing is not None:
            voicing_pos_map = get_voicing_positions(best_voicing, chord_midi)
            if len(voicing_pos_map) >= len(chord_candidates) * 0.7:
                # Good coverage — try using voicing positions
                v_assignments = {}
                v_used_strings = set()
                v_matched = 0
                for idx, (note, _candidates) in enumerate(chord_candidates):
                    pos = voicing_pos_map.get(note.midi_note)
                    if pos and pos.string not in v_used_strings:
                        v_assignments[idx] = pos
                        v_used_strings.add(pos.string)
                        v_matched += 1
                    else:
                        v_assignments[idx] = None

                if v_matched >= len(chord_candidates) * 0.7:
                    voicing_assignments = v_assignments

    # Phase 1: Find best fret region by scoring candidate combinations
    # For each note, get all candidate frets and find the region that
    # satisfies the most notes with minimum span
    best_region_score = float('-inf')
    best_assignments = None

    # Collect all possible fret values across all candidates
    all_frets = set()
    for _, candidates in chord_candidates:
        for c in candidates:
            all_frets.add(c.fret)

    # Try each fret as center of a 5-fret window (typical hand span)
    for center_fret in all_frets:
        fret_min = max(0, center_fret - 2)
        fret_max = center_fret + 3

        # Score this region
        region_score = 0.0
        used_strings = set()
        assignments = {}

        # Sort by most constrained (fewest candidates in region)
        indexed = list(enumerate(chord_candidates))
        indexed.sort(key=lambda x: sum(
            1 for c in x[1][1]
            if fret_min <= c.fret <= fret_max and c.string not in used_strings
        ))

        for orig_idx, (note, candidates) in indexed:
            # Filter to candidates in this fret region, on unused strings
            available = [
                c for c in candidates
                if fret_min <= c.fret <= fret_max and c.string not in used_strings
            ]
            # Also include open strings (fret 0) which are always reachable
            if fret_min > 0:
                available.extend([
                    c for c in candidates
                    if c.fret == 0 and c.string not in used_strings
                    and c not in available
                ])

            if available:
                # Pick the one closest to center
                pos = min(available, key=lambda c: abs(c.fret - center_fret))
                used_strings.add(pos.string)
                assignments[orig_idx] = pos
                region_score += note.confidence
            else:
                assignments[orig_idx] = None

        # Score the region
        if hand_position_fret is not None:
            region_score -= abs(center_fret - hand_position_fret) * 0.3
        elif config.prefer_lower_frets:
            region_score -= center_fret * 0.02

        # Prefer compact string arrangements (notes on adjacent strings)
        assigned_positions = [p for p in assignments.values() if p]
        if len(assigned_positions) >= 2:
            assigned_strings = [p.string for p in assigned_positions]
            string_span = max(assigned_strings) - min(assigned_strings)
            region_score -= string_span * 0.1

        # Chord voicing match bonus: if these positions closely match a known
        # voicing, boost the score
        if shape_config.enabled and assigned_positions:
            chord_midi = [note.midi_note for note, _ in chord_candidates]
            voicing_matches = find_matching_voicings(chord_midi, config.playing_style)
            if voicing_matches:
                best_voicing, v_score = voicing_matches[0]
                pos_match = score_position_against_voicing(
                    assigned_positions, best_voicing
                )
                style_weights = STYLE_WEIGHTS.get(
                    config.playing_style, STYLE_WEIGHTS[PlayingStyle.DEFAULT]
                )
                region_score += pos_match * style_weights.voicing_match_bonus

        if region_score > best_region_score:
            best_region_score = region_score
            best_assignments = assignments

    # If we have a strong voicing match, compare it against the region-based
    # result and use whichever is better — but only if the voicing is
    # consistent with the current hand position.  If the voicing positions
    # are far from hand_position_fret (> 5 frets), the region-based result
    # is more trustworthy and the voicing override is skipped.
    if voicing_assignments is not None and best_assignments is not None:
        v_count = sum(1 for p in voicing_assignments.values() if p is not None)
        r_count = sum(1 for p in best_assignments.values() if p is not None)

        voicing_ok = True
        if hand_position_fret is not None:
            v_frets = [
                p.fret for p in voicing_assignments.values()
                if p is not None and p.fret > 0
            ]
            if v_frets:
                v_avg = sum(v_frets) / len(v_frets)
                if abs(v_avg - hand_position_fret) > 3:
                    voicing_ok = False  # voicing is far from hand position

        # Prefer voicing if it covers at least as many notes and is consistent
        if voicing_ok and v_count >= r_count:
            best_assignments = voicing_assignments

    if best_assignments is None:
        # Fallback: greedy with simple selection
        used_strings = set()
        best_assignments = {}
        indexed_candidates = list(enumerate(chord_candidates))
        indexed_candidates.sort(key=lambda x: len(x[1][1]))
        for orig_idx, (note, candidates) in indexed_candidates:
            available = [c for c in candidates if c.string not in used_strings]
            if available:
                position = _select_best_position(
                    available, previous_position, config,
                    hand_position_fret=hand_position_fret
                )
                if position:
                    used_strings.add(position.string)
                    best_assignments[orig_idx] = position
            else:
                best_assignments[orig_idx] = None

    # Return in original order
    return [best_assignments.get(i) for i in range(len(chord_candidates))]


def _create_tab_note(
    note: DetectedNote,
    position: Position,
    chord_id: Optional[str],
    audio_confidence: float = 0.0,
    video_confidence: float = 0.0,
    video_matched: bool = False
) -> TabNote:
    """Create a TabNote from a detected note and position."""
    # Combined confidence
    if video_matched:
        confidence = min(1.0, audio_confidence + video_confidence * 0.3)
    else:
        confidence = audio_confidence

    return TabNote(
        id=str(uuid4()),
        timestamp=note.start_time,
        string=position.string,
        fret=position.fret,
        confidence=confidence,
        confidence_level=get_confidence_level(confidence),
        midi_note=note.midi_note,
        end_time=note.end_time,
        is_part_of_chord=chord_id is not None,
        chord_id=chord_id,
        video_matched=video_matched,
        audio_confidence=audio_confidence,
        video_confidence=video_confidence,
        pitch_bend=note.pitch_bend,
    )


def _correct_slide_positions(
    tab_notes: list[TabNote],
    capo_fret: int = 0
) -> list[TabNote]:
    """Correct positions for consecutive notes that form melodic runs.

    Applies two corrections:
    1. Adjacent notes on different strings with close pitches → same string
    2. Ascending runs where earlier notes are on one string → keep on same string

    Args:
        tab_notes: List of tab notes
        capo_fret: Capo position

    Returns:
        Tab notes with corrected positions
    """
    if len(tab_notes) < 2:
        return tab_notes

    sorted_notes = sorted(tab_notes, key=lambda n: n.timestamp)

    # Pass 1: Fix pairs on different strings with close pitches
    for i in range(1, len(sorted_notes)):
        prev = sorted_notes[i - 1]
        curr = sorted_notes[i]

        time_gap = curr.timestamp - prev.timestamp
        if time_gap > 0.5:
            continue

        midi_diff = abs(curr.midi_note - prev.midi_note)
        if midi_diff < 1 or midi_diff > 2:
            continue

        if prev.string == curr.string:
            continue

        prev_candidates = get_candidate_positions(prev.midi_note, capo_fret)
        curr_candidates = get_candidate_positions(curr.midi_note, capo_fret)

        prev_by_string = {p.string: p for p in prev_candidates}
        curr_by_string = {p.string: p for p in curr_candidates}
        common_strings = set(prev_by_string.keys()) & set(curr_by_string.keys())

        if common_strings:
            # Break ties by preferring the string already assigned to prev or curr
            # This avoids gratuitously moving notes to a different string
            def _string_score(s):
                fret_dist = abs(prev_by_string[s].fret - curr_by_string[s].fret)
                # Tiebreaker: prefer string matching prev, then curr
                existing_bonus = 0
                if s == prev.string:
                    existing_bonus = -2  # strong preference
                elif s == curr.string:
                    existing_bonus = -1  # mild preference
                return (fret_dist, existing_bonus, s)

            best_string = min(common_strings, key=_string_score)
            prev_pos = prev_by_string[best_string]
            curr_pos = curr_by_string[best_string]

            # Guard: don't merge if a note at a low fret (0-4) would be pushed
            # to a much higher fret. This indicates a string crossing, not a slide.
            prev_fret_val = prev.fret if isinstance(prev.fret, int) else 0
            curr_fret_val = curr.fret if isinstance(curr.fret, int) else 0
            prev_fret_jump = abs(prev_pos.fret - prev_fret_val)
            curr_fret_jump = abs(curr_pos.fret - curr_fret_val)
            if (prev_fret_val <= 4 and prev_fret_jump > 3) or \
               (curr_fret_val <= 4 and curr_fret_jump > 3):
                continue  # Skip: string crossing, not a slide

            prev.string = prev_pos.string
            prev.fret = prev_pos.fret
            curr.string = curr_pos.string
            curr.fret = curr_pos.fret

    # Pass 2: Fix ascending/descending runs across string boundaries
    # Look for sequences where notes move to a different string but could
    # stay on the same string with ascending frets (e.g., slide up the neck)
    for i in range(2, len(sorted_notes)):
        prev2 = sorted_notes[i - 2]
        prev1 = sorted_notes[i - 1]
        curr = sorted_notes[i]

        # Check for sequential timing (within 1s each pair)
        if curr.timestamp - prev1.timestamp > 1.0:
            continue
        if prev1.timestamp - prev2.timestamp > 1.0:
            continue

        # Check if prev2 and prev1 are on the same string but curr jumped
        if prev2.string == prev1.string and curr.string != prev1.string:
            # Can curr be on the same string as prev1?
            target_string = prev1.string
            curr_candidates = get_candidate_positions(curr.midi_note, capo_fret)
            curr_by_string = {p.string: p for p in curr_candidates}

            if target_string in curr_by_string:
                new_pos = curr_by_string[target_string]
                # Only reassign if the fret follows the direction of the run
                prev_fret = prev1.fret
                if isinstance(prev_fret, int) and isinstance(new_pos.fret, int):
                    # Guard: don't move low-fret notes to high frets
                    curr_fret = curr.fret if isinstance(curr.fret, int) else 0
                    if curr_fret <= 4 and abs(new_pos.fret - curr_fret) > 3:
                        continue
                    # Check ascending run (fret increases)
                    if prev1.fret > prev2.fret and new_pos.fret > prev1.fret:
                        curr.string = new_pos.string
                        curr.fret = new_pos.fret
                    # Check descending run (fret decreases)
                    elif prev1.fret < prev2.fret and new_pos.fret < prev1.fret:
                        curr.string = new_pos.string
                        curr.fret = new_pos.fret

    # Pass 3: Re-run Pass 1 to catch cases where Pass 2 created new
    # same-string pairs that now qualify for semitone correction
    for i in range(1, len(sorted_notes)):
        prev = sorted_notes[i - 1]
        curr = sorted_notes[i]

        time_gap = curr.timestamp - prev.timestamp
        if time_gap > 0.5:
            continue

        midi_diff = abs(curr.midi_note - prev.midi_note)
        if midi_diff < 1 or midi_diff > 2:
            continue

        if prev.string == curr.string:
            continue

        prev_candidates = get_candidate_positions(prev.midi_note, capo_fret)
        curr_candidates = get_candidate_positions(curr.midi_note, capo_fret)

        prev_by_string = {p.string: p for p in prev_candidates}
        curr_by_string = {p.string: p for p in curr_candidates}
        common_strings = set(prev_by_string.keys()) & set(curr_by_string.keys())

        if common_strings:
            def _string_score(s):
                fret_dist = abs(prev_by_string[s].fret - curr_by_string[s].fret)
                existing_bonus = 0
                if s == prev.string:
                    existing_bonus = -2
                elif s == curr.string:
                    existing_bonus = -1
                return (fret_dist, existing_bonus, s)

            best_string = min(common_strings, key=_string_score)
            prev_pos = prev_by_string[best_string]
            curr_pos = curr_by_string[best_string]

            # Guard: don't merge if a low-fret note would be pushed high
            prev_fret_val = prev.fret if isinstance(prev.fret, int) else 0
            curr_fret_val = curr.fret if isinstance(curr.fret, int) else 0
            prev_fret_jump = abs(prev_pos.fret - prev_fret_val)
            curr_fret_jump = abs(curr_pos.fret - curr_fret_val)
            if (prev_fret_val <= 4 and prev_fret_jump > 3) or \
               (curr_fret_val <= 4 and curr_fret_jump > 3):
                continue

            prev.string = prev_pos.string
            prev.fret = prev_pos.fret
            curr.string = curr_pos.string
            curr.fret = curr_pos.fret

    # Pass 4: Re-assign consecutive single notes when a recent chord suggests a different string
    # When two consecutive non-chord notes are on the same string, and both could be
    # on a string that was active in a recent chord, prefer the chord string.
    # This catches slides following a bass line (e.g., A string open → A string fret 8).
    for i in range(1, len(sorted_notes)):
        curr = sorted_notes[i]
        prev = sorted_notes[i - 1]

        # Both must be non-chord, same string, close in time
        if curr.is_part_of_chord or prev.is_part_of_chord:
            continue
        if curr.string != prev.string:
            continue
        if curr.timestamp - prev.timestamp > 0.5:
            continue

        # Look for a recent chord (within 1.5s) with a member on a different string
        # that both notes could be placed on
        prev_candidates = get_candidate_positions(prev.midi_note, capo_fret)
        curr_candidates = get_candidate_positions(curr.midi_note, capo_fret)
        prev_by_string = {p.string: p for p in prev_candidates}
        curr_by_string = {p.string: p for p in curr_candidates}
        common_alt_strings = (set(prev_by_string.keys()) & set(curr_by_string.keys())) - {curr.string}

        if not common_alt_strings:
            continue

        # Check recent chord members
        best_alt_string = None
        for j in range(i - 2, max(-1, i - 8), -1):
            earlier = sorted_notes[j]
            if prev.timestamp - earlier.timestamp > 1.5:
                break
            if earlier.is_part_of_chord and earlier.string in common_alt_strings:
                best_alt_string = earlier.string
                break

        if best_alt_string is not None:
            prev_pos = prev_by_string[best_alt_string]
            curr_pos = curr_by_string[best_alt_string]
            prev.string = prev_pos.string
            prev.fret = prev_pos.fret
            curr.string = curr_pos.string
            curr.fret = curr_pos.fret

    return sorted_notes


def _correct_melodic_segments(
    tab_notes: list[TabNote],
    capo_fret: int = 0,
    config: Optional[FusionConfig] = None
) -> list[TabNote]:
    """Correct string assignments for melodic segments.

    Identifies segments of consecutive non-chord notes and determines
    the optimal single string for the segment. Melodic passages (scales,
    runs) typically stay on one string — this corrects cases where the
    position selector scattered notes across multiple strings.

    Also uses scale box pattern awareness to prefer strings/frets that
    align with recognized scale patterns.

    Only applies when:
    - Segment has 3+ notes
    - A single string can accommodate 90%+ of notes
    - Fret range on that string is <= 14 (playable in 1-2 positions)

    Args:
        tab_notes: List of tab notes
        capo_fret: Capo position
        config: Fusion configuration (for scale pattern scoring)

    Returns:
        Tab notes with corrected string assignments
    """
    if len(tab_notes) < 3:
        return tab_notes

    sorted_notes = sorted(tab_notes, key=lambda n: n.timestamp)

    # Build segments of consecutive non-chord notes with < 1s gaps
    segments: list[list[int]] = []  # list of index lists
    current_segment: list[int] = []

    for idx, note in enumerate(sorted_notes):
        if note.is_part_of_chord:
            if len(current_segment) >= 3:
                segments.append(current_segment)
            current_segment = []
            continue
        if current_segment:
            prev_idx = current_segment[-1]
            if note.timestamp - sorted_notes[prev_idx].timestamp > 1.5:
                if len(current_segment) >= 3:
                    segments.append(current_segment)
                current_segment = []
        current_segment.append(idx)

    if len(current_segment) >= 3:
        segments.append(current_segment)

    # For each segment, find the best single string
    for seg_indices in segments:
        seg_notes = [sorted_notes[i] for i in seg_indices]
        midi_notes = [n.midi_note for n in seg_notes]

        # Guard 1: Only correct segments that use 2-3 distinct strings.
        # 1 string = already correct, 4+ strings = arpeggio/fingerpicking.
        current_strings = set(n.string for n in seg_notes)
        if len(current_strings) < 2 or len(current_strings) > 3:
            continue

        # Guard 2: Check for scale-like motion (small pitch intervals).
        # Arpeggios have large intervals; scales have small ones.
        intervals = [abs(midi_notes[i] - midi_notes[i-1])
                     for i in range(1, len(midi_notes))]
        if intervals:
            median_interval = sorted(intervals)[len(intervals) // 2]
            if median_interval > 4:  # More than a major third = not scalar
                continue

        best_string = None
        best_score = float('-inf')
        best_frets: list[Optional[int]] = []

        for string_num in range(1, 7):
            open_midi = STANDARD_TUNING[string_num]
            frets: list[Optional[int]] = []
            for midi in midi_notes:
                fret = midi - open_midi
                if capo_fret <= fret <= MAX_FRET:
                    frets.append(fret)
                else:
                    frets.append(None)

            valid_frets = [f for f in frets if f is not None]
            if not valid_frets:
                continue

            coverage = len(valid_frets) / len(midi_notes)
            if coverage < 0.9:
                continue

            fret_range = max(valid_frets) - min(valid_frets)
            if fret_range > 14:
                continue

            avg_fret = sum(valid_frets) / len(valid_frets)

            # Score: full coverage strongly preferred, then minimize range,
            # then prefer lower average fret position
            score = coverage * 100 - fret_range * 1.0 - avg_fret * 0.1

            # Scale pattern bonus: if the frets align with a recognized
            # scale pattern at the average position, boost the score
            if (config is not None
                    and config.chord_shape_config.scale_pattern_enabled):
                positions = [
                    Position(string=string_num, fret=f)
                    for f in valid_frets
                ]
                scale_score = score_positions_against_scale(
                    positions, round(avg_fret), config.playing_style
                )
                score += scale_score * 5.0  # Scale pattern fit bonus

            if score > best_score:
                best_score = score
                best_string = string_num
                best_frets = frets

        if best_string is None:
            continue

        # Check how many notes are already on the best string
        already_correct = sum(1 for n in seg_notes if n.string == best_string)
        if already_correct == len(seg_notes):
            continue  # All notes already on best string

        # Reassign notes to the best string
        for i, idx in enumerate(seg_indices):
            fret = best_frets[i]
            if fret is not None:
                sorted_notes[idx].string = best_string
                sorted_notes[idx].fret = fret

    return sorted_notes


def _detect_techniques(
    tab_notes: list[TabNote],
    config: FusionConfig
) -> list[TabNote]:
    """Detect playing techniques from note patterns.

    Detects:
    - Hammer-ons: ascending notes on same string in quick succession
    - Pull-offs: descending notes on same string in quick succession
    - Slides: notes on same string with continuous pitch change

    Args:
        tab_notes: List of tab notes
        config: Fusion configuration

    Returns:
        Tab notes with technique annotations
    """
    if len(tab_notes) < 2:
        return tab_notes

    # Sort by timestamp
    sorted_notes = sorted(tab_notes, key=lambda n: n.timestamp)

    for i in range(1, len(sorted_notes)):
        prev = sorted_notes[i - 1]
        curr = sorted_notes[i]

        # Skip if different strings or too far apart in time
        if prev.string != curr.string:
            continue
        time_gap = curr.timestamp - prev.timestamp
        if prev.end_time:
            time_gap = curr.timestamp - prev.end_time
        if time_gap > config.hammer_on_max_gap:
            continue

        # Same string, quick succession - check for technique
        if isinstance(prev.fret, int) and isinstance(curr.fret, int):
            fret_diff = curr.fret - prev.fret

            if fret_diff > 0:
                # Ascending - hammer-on
                curr.technique = "hammer-on"
            elif fret_diff < 0:
                # Descending - pull-off
                curr.technique = "pull-off"

            # Check for slide (larger fret jump with sustained note)
            if abs(fret_diff) >= config.slide_pitch_threshold:
                # Could be a slide if previous note was sustained
                if prev.end_time and prev.end_time >= curr.timestamp - 0.02:
                    curr.technique = "slide"

    return sorted_notes


def find_nearest_observation(
    observations: dict[float, HandObservation],
    timestamp: float,
    tolerance: float = 0.1
) -> HandObservation | None:
    """Find the video observation nearest to a timestamp.

    Args:
        observations: Dict mapping timestamps to HandObservation
        timestamp: Target timestamp in seconds
        tolerance: Maximum time difference in seconds

    Returns:
        Nearest HandObservation within tolerance, or None
    """
    if not observations:
        return None

    nearest_ts = min(observations.keys(), key=lambda t: abs(t - timestamp))
    if abs(nearest_ts - timestamp) <= tolerance:
        return observations[nearest_ts]
    return None


def match_video_to_candidates(
    observation: HandObservation,
    fretboard: FretboardGeometry,
    candidates: list[Position],
    frame_width: int = 640,
    frame_height: int = 480
) -> Position | None:
    """Try to match video finger positions to audio candidates.

    Args:
        observation: Hand detection from video
        fretboard: Detected fretboard geometry
        candidates: Candidate positions from audio analysis
        frame_width: Video frame width for coordinate conversion
        frame_height: Video frame height for coordinate conversion

    Returns:
        Matching Position if found, None otherwise
    """
    for finger in observation.fingers:
        # Convert normalized coordinates to pixel coordinates
        finger_x = finger.x * frame_width
        finger_y = finger.y * frame_height

        video_pos = map_finger_to_position(finger_x, finger_y, fretboard)
        if video_pos is None:
            continue

        # Check if video position matches any audio candidate
        for candidate in candidates:
            if candidate.string == video_pos.string and candidate.fret == video_pos.fret:
                return candidate

    return None


def has_open_string_candidate(candidates: list[Position]) -> Position | None:
    """Check if fret 0 (open string) is among the candidates.

    Args:
        candidates: List of possible positions

    Returns:
        Position with fret 0 if found, None otherwise
    """
    for candidate in candidates:
        if candidate.fret == 0:
            return candidate
    return None


def fuse_audio_video(
    detected_notes: list[DetectedNote],
    video_observations: dict[float, HandObservation],
    fretboard: FretboardGeometry | None,
    capo_fret: int = 0,
    config: Optional[FusionConfig] = None,
    muted_notes: list[MutedNote] | None = None
) -> list[TabNote]:
    """Combine audio and video signals for tab generation.

    Uses enhanced fusion logic:
    1. Group notes into chords for coordinated position selection
    2. Match video finger positions to audio candidates
    3. Use finger pressing state to distinguish fretted vs open notes
    4. Apply temporal smoothing for position continuity
    5. Detect playing techniques (hammer-ons, pull-offs, slides)

    Args:
        detected_notes: Notes detected from audio analysis
        video_observations: Dict mapping timestamp to HandObservation
        fretboard: Detected fretboard geometry (None if not detected)
        capo_fret: Fret where capo is placed (0 = no capo)
        config: Fusion configuration

    Returns:
        List of TabNote objects with confidence scores
    """
    if config is None:
        config = FusionConfig()

    # If no fretboard detected, fall back to audio-only
    if fretboard is None:
        return fuse_audio_only(detected_notes, capo_fret, config)

    if not detected_notes:
        return []

    # Pre-filter to remove ghost notes and chord fragment re-detections
    if config.enable_prefiltering:
        detected_notes = _prefilter_notes(detected_notes, config)

    # Group notes into chords
    chords = group_notes_into_chords(detected_notes, config.chord_time_tolerance)

    # Limit chord sizes (trim oversized chords by amplitude)
    chords = _limit_chord_sizes(chords, config)

    # Two-pass anchor system (same as fuse_audio_only)
    chord_anchors = {}
    for idx, chord_notes_group in enumerate(chords):
        if len(chord_notes_group) >= 3:
            cc = []
            for note in chord_notes_group:
                candidates = get_candidate_positions(note.midi_note, capo_fret)
                if candidates:
                    cc.append((note, candidates))
            if len(cc) >= 3:
                positions = _optimize_chord_positions(cc, None, config, hand_position_fret=None)
                valid = [p for p in positions if p]
                if valid:
                    frets = [p.fret for p in valid if p.fret > 0]
                    if frets:
                        chord_anchors[idx] = (sum(frets) / len(frets), len(valid))

    def _get_nearest_anchor(chord_idx: int) -> Optional[float]:
        if not chord_anchors:
            return None
        best_anchor = None
        best_score = float('-inf')
        for k, (avg_fret, num_notes) in chord_anchors.items():
            distance = abs(k - chord_idx)
            if distance > 15:
                continue
            score = num_notes * 2.0 - distance * 0.5
            if score > best_score:
                best_score = score
                best_anchor = avg_fret
        return best_anchor

    tab_notes = []
    previous_position = None
    hand_position_fret = None

    for i, chord_notes_group in enumerate(chords):
        chord_id = str(uuid4()) if len(chord_notes_group) > 1 else None
        chord_timestamp = chord_notes_group[0].start_time

        # Get video observation for this chord
        video_obs = find_nearest_observation(
            video_observations, chord_timestamp, config.video_match_tolerance
        )

        # Get all candidates for each note in the chord
        chord_candidates = []
        for note in chord_notes_group:
            candidates = get_candidate_positions(note.midi_note, capo_fret)
            if candidates:
                chord_candidates.append((note, candidates))

        if not chord_candidates:
            continue

        # Use anchor-based hand position
        anchor = _get_nearest_anchor(i)
        effective_hand_pos = hand_position_fret
        if anchor is not None:
            if effective_hand_pos is None:
                effective_hand_pos = anchor
            else:
                effective_hand_pos = effective_hand_pos * 0.3 + anchor * 0.7

        # Try video matching first for each note
        video_matches = {}  # note_index -> (Position, video_confidence)
        used_strings_video = set()
        if video_obs and fretboard:
            for idx, (note, candidates) in enumerate(chord_candidates):
                match, v_conf = match_video_to_candidates_enhanced(
                    video_obs, fretboard, candidates, used_strings_video
                )
                if match:
                    video_matches[idx] = (match, v_conf)
                    used_strings_video.add(match.string)

        # Process single notes
        if len(chord_candidates) == 1:
            note, candidates = chord_candidates[0]
            if 0 in video_matches:
                # Video confirmed a position
                position, v_conf = video_matches[0]
                confidence = min(1.0, note.confidence + config.video_match_boost)
                video_matched = True
            elif video_obs:
                # Video observation exists but no match
                open_string = has_open_string_candidate(candidates)
                has_pressing_finger = len(video_obs.pressing_fingers) > 0

                if open_string and not has_pressing_finger:
                    position = open_string
                    confidence = config.open_string_confidence
                    video_matched = False
                    v_conf = 0.0
                elif open_string and has_pressing_finger:
                    position = _select_best_position(
                        candidates, previous_position, config,
                        hand_position_fret=effective_hand_pos
                    )
                    if position is None:
                        continue
                    confidence = note.confidence - config.no_match_penalty
                    video_matched = False
                    v_conf = 0.0
                else:
                    position = _select_best_position(
                        candidates, previous_position, config,
                        hand_position_fret=effective_hand_pos
                    )
                    if position is None:
                        continue
                    confidence = note.confidence
                    video_matched = False
                    v_conf = 0.0
            else:
                # No video observation
                position = _select_best_position(
                    candidates, previous_position, config,
                    hand_position_fret=effective_hand_pos
                )
                if position is None:
                    continue
                confidence = note.confidence
                video_matched = False
                v_conf = 0.0

            tab_note = TabNote(
                id=str(uuid4()),
                timestamp=note.start_time,
                string=position.string,
                fret=position.fret,
                confidence=confidence,
                confidence_level=get_confidence_level(confidence),
                midi_note=note.midi_note,
                end_time=note.end_time,
                is_part_of_chord=chord_id is not None,
                chord_id=chord_id,
                video_matched=video_matched,
                audio_confidence=note.confidence,
                video_confidence=v_conf if 0 in video_matches else 0.0,
            )
            tab_notes.append(tab_note)
            previous_position = position
            if hand_position_fret is None:
                hand_position_fret = float(position.fret)
            else:
                hand_position_fret = hand_position_fret * 0.7 + position.fret * 0.3

        else:
            # Multiple notes - optimize as chord
            chord_hand_pos = effective_hand_pos

            selected_positions = _optimize_chord_positions(
                chord_candidates, previous_position, config,
                hand_position_fret=chord_hand_pos
            )

            for idx, ((note, _), position) in enumerate(zip(chord_candidates, selected_positions)):
                # Override with video match if available
                if idx in video_matches:
                    v_pos, v_conf = video_matches[idx]
                    position = v_pos
                    confidence = min(1.0, note.confidence + config.video_match_boost)
                    video_matched = True
                elif video_obs and position:
                    # Video obs exists but no match for this note
                    confidence = note.confidence
                    video_matched = False
                    v_conf = 0.0
                elif position:
                    confidence = note.confidence
                    video_matched = False
                    v_conf = 0.0
                else:
                    continue

                if position:
                    tab_note = _create_tab_note(
                        note, position, chord_id,
                        audio_confidence=note.confidence,
                        video_confidence=v_conf if idx in video_matches else 0.0,
                        video_matched=video_matched,
                    )
                    tab_notes.append(tab_note)

            # Update previous position and hand position from chord center
            valid_positions = list(selected_positions)
            # Include video overrides
            for idx_vm in video_matches:
                if idx_vm < len(valid_positions):
                    valid_positions[idx_vm] = video_matches[idx_vm][0]
            valid_positions = [p for p in valid_positions if p]
            if valid_positions:
                avg_fret = sum(p.fret for p in valid_positions) / len(valid_positions)
                previous_position = min(valid_positions, key=lambda p: abs(p.fret - avg_fret))
                if hand_position_fret is None:
                    hand_position_fret = avg_fret
                else:
                    hand_position_fret = hand_position_fret * 0.4 + avg_fret * 0.6

    # Post-processing: correct slide/legato positions
    tab_notes = _correct_slide_positions(tab_notes, capo_fret)

    # Post-filter: remove duplicate positions and low-confidence strays
    tab_notes = _postfilter_tab_notes(tab_notes, config)

    # Detect techniques
    tab_notes = _detect_techniques(tab_notes, config)

    # Add muted notes as "X" fret TabNotes
    if muted_notes:
        muted_tab = _create_muted_tab_notes(
            muted_notes, video_observations=video_observations, config=config
        )
        tab_notes.extend(muted_tab)
        tab_notes.sort(key=lambda n: n.timestamp)

    return tab_notes


def match_video_to_candidates_enhanced(
    observation: HandObservation,
    fretboard: FretboardGeometry,
    candidates: list[Position],
    used_strings: set[int],
) -> tuple[Optional[Position], float]:
    """Enhanced video-to-audio candidate matching.

    Uses:
    - Finger pressing state (z-depth)
    - Extended finger detection
    - Confidence-weighted matching

    Args:
        observation: Hand detection from video
        fretboard: Detected fretboard geometry
        candidates: Candidate positions from audio analysis
        used_strings: Strings already assigned to other notes

    Returns:
        Tuple of (matched Position or None, confidence)
    """
    best_match = None
    best_confidence = 0.0

    # Get actual frame dimensions from fretboard geometry
    frame_width = fretboard.frame_width
    frame_height = fretboard.frame_height

    # Filter to pressing fingers only (more likely to be fretting)
    pressing_fingers = observation.get_pressing_finger_positions()
    if not pressing_fingers:
        # Fall back to all extended fingers
        pressing_fingers = [f for f in observation.fingers if f.is_extended]

    for finger in pressing_fingers:
        # Convert normalized coordinates to pixel coordinates
        finger_x = finger.x * frame_width
        finger_y = finger.y * frame_height

        # Map finger to fretboard position
        video_pos = map_finger_to_position(
            finger_x, finger_y, fretboard,
            finger_z=finger.z,
            finger_id=finger.finger_id
        )
        if video_pos is None:
            continue

        # Skip if string already used
        if video_pos.string in used_strings:
            continue

        # Check if video position matches any audio candidate
        for candidate in candidates:
            if candidate.string == video_pos.string and candidate.fret == video_pos.fret:
                # Direct match!
                match_confidence = video_pos.confidence

                # Boost confidence if finger appears to be pressing
                if finger.z < -0.03:
                    match_confidence = min(1.0, match_confidence + 0.1)

                if match_confidence > best_confidence:
                    best_match = candidate
                    best_confidence = match_confidence
                break

            # Check for near-miss (adjacent fret)
            if (candidate.string == video_pos.string and
                abs(candidate.fret - video_pos.fret) == 1):
                # Possible match with lower confidence
                near_confidence = video_pos.confidence * 0.7
                if near_confidence > best_confidence and best_match is None:
                    best_match = candidate
                    best_confidence = near_confidence

    return best_match, best_confidence
