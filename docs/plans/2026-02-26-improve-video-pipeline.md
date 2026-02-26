# Improve Video Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the video pipeline help rather than hurt transcription accuracy by bringing `fuse_audio_video` to parity with `fuse_audio_only`'s filtering and position-selection quality.

**Architecture:** The `fuse_audio_video` function currently skips all the filtering and optimization that `fuse_audio_only` uses (prefiltering, chord size limiting, anchor-based position selection, slide correction, post-filtering). Rather than duplicating logic, we'll refactor to share the common pipeline stages, then layer video matching on top as a position refinement/confidence boost.

**Tech Stack:** Python, pytest

---

## Problem Analysis

When the server processes a video, it attempts fretboard detection. If a fretboard is found (even with low confidence), `process_job` routes to `fuse_audio_video` instead of `fuse_audio_only`. The two functions produce dramatically different results:

| Feature | `fuse_audio_only` | `fuse_audio_video` |
|---|---|---|
| Pre-filtering (ghost notes, chord fragments) | Yes | **No** |
| Chord size limiting | Yes | **No** |
| Two-pass anchor system | Yes | **No** (simple per-note) |
| Chord position optimization | Yes | **No** (greedy, no region scoring) |
| Slide correction (3-pass) | Yes | **No** |
| Post-filtering (dedup, low-conf, open-string) | Yes | **No** |
| **Result on sample video** | **41 notes, F1=91.4%** | **65 notes, F1=45.7%** |

The fix: refactor `fuse_audio_video` to use the same pipeline as `fuse_audio_only` as its backbone, with video observations used as an additional signal during position selection (boosting confidence when video confirms, or preferring video-matched positions).

---

### Task 1: Add pre-filtering and chord limiting to `fuse_audio_video`

**Files:**
- Modify: `tabvision-server/app/fusion_engine.py:964-1112` (the `fuse_audio_video` function)
- Test: `tabvision-server/tests/test_fusion.py`

**Step 1: Write the failing test**

Add a test class `TestFuseAudioVideoFiltering` in `tabvision-server/tests/test_fusion.py`:

```python
class TestFuseAudioVideoFiltering:
    """Tests that fuse_audio_video applies the same filtering as fuse_audio_only."""

    def _make_fretboard(self):
        """Create a minimal FretboardGeometry for testing."""
        return FretboardGeometry(
            nut_x=50, bridge_x=600,
            top_string_y=100, bottom_string_y=300,
            fret_positions=[50 + i * 30 for i in range(25)],
            string_positions=[100 + i * 40 for i in range(6)],
            num_strings=6, num_frets=24,
            detection_confidence=0.8,
            frame_width=640, frame_height=480,
        )

    def test_ghost_notes_filtered(self):
        """Ghost notes (low amplitude overlapping loud notes) should be removed."""
        # Two overlapping notes: one loud, one very quiet (ghost)
        loud_note = DetectedNote(
            start_time=1.0, end_time=2.0, midi_note=64,
            confidence=0.9, amplitude=0.8,
        )
        ghost_note = DetectedNote(
            start_time=1.0, end_time=1.5, midi_note=76,  # octave harmonic
            confidence=0.5, amplitude=0.15,  # very quiet
        )
        fretboard = self._make_fretboard()
        result = fuse_audio_video(
            [loud_note, ghost_note], {}, fretboard, capo_fret=0
        )
        # Ghost note should be filtered out
        assert len(result) == 1
        assert result[0].midi_note == 64

    def test_chord_size_limited(self):
        """Chords should be limited to max_chord_size (default 3)."""
        # Create 5 simultaneous notes
        notes = [
            DetectedNote(
                start_time=1.0, end_time=2.0, midi_note=midi,
                confidence=0.8, amplitude=0.5 + i * 0.1,
            )
            for i, midi in enumerate([40, 45, 50, 55, 60])
        ]
        fretboard = self._make_fretboard()
        result = fuse_audio_video(notes, {}, fretboard, capo_fret=0)
        # Should be limited to 3 notes
        assert len(result) <= 3
```

**Step 2: Run test to verify it fails**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestFuseAudioVideoFiltering -v`
Expected: FAIL — `fuse_audio_video` currently does not filter ghost notes or limit chord sizes.

**Step 3: Implement pre-filtering and chord limiting in `fuse_audio_video`**

In `fusion_engine.py`, modify `fuse_audio_video` (around line 997-1001) to add pre-filtering and chord limiting before the main loop:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestFuseAudioVideoFiltering -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py -v`
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add tabvision-server/app/fusion_engine.py tabvision-server/tests/test_fusion.py
git commit -m "feat: add pre-filtering and chord limiting to fuse_audio_video"
```

---

### Task 2: Add slide correction and post-filtering to `fuse_audio_video`

**Files:**
- Modify: `tabvision-server/app/fusion_engine.py:1109-1112` (end of `fuse_audio_video`)
- Test: `tabvision-server/tests/test_fusion.py`

**Step 1: Write the failing test**

Add tests to `TestFuseAudioVideoFiltering`:

```python
    def test_slide_positions_corrected(self):
        """Consecutive semitone notes on different strings should be corrected to same string."""
        # Two notes a semitone apart, both playable on string 5
        note1 = DetectedNote(
            start_time=1.0, end_time=1.3, midi_note=55,  # G3 = s5f10 or s4f5
            confidence=0.9, amplitude=0.7,
        )
        note2 = DetectedNote(
            start_time=1.3, end_time=1.6, midi_note=56,  # G#3 = s5f11 or s4f6
            confidence=0.9, amplitude=0.7,
        )
        fretboard = self._make_fretboard()
        result = fuse_audio_video(
            [note1, note2], {}, fretboard, capo_fret=0
        )
        assert len(result) == 2
        # Both should end up on the same string after slide correction
        assert result[0].string == result[1].string

    def test_low_confidence_singletons_filtered(self):
        """Low-confidence non-chord notes should be removed by post-filtering."""
        notes = [
            DetectedNote(
                start_time=1.0, end_time=2.0, midi_note=64,
                confidence=0.45, amplitude=0.5,  # low conf after normalization
            ),
        ]
        fretboard = self._make_fretboard()
        config = FusionConfig()
        result = fuse_audio_video(notes, {}, fretboard, capo_fret=0, config=config)
        # Should be removed by post-filter (confidence < 0.6, not in chord)
        assert len(result) == 0

    def test_duplicate_positions_deduped(self):
        """Same string+fret within 0.3s should be deduped."""
        # Two notes that map to the same position within 0.3s
        note1 = DetectedNote(
            start_time=1.0, end_time=1.5, midi_note=64,
            confidence=0.9, amplitude=0.7,
        )
        note2 = DetectedNote(
            start_time=1.15, end_time=1.6, midi_note=64,
            confidence=0.8, amplitude=0.6,
        )
        fretboard = self._make_fretboard()
        result = fuse_audio_video(
            [note1, note2], {}, fretboard, capo_fret=0
        )
        # Should be deduped to 1 note
        assert len(result) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestFuseAudioVideoFiltering::test_slide_positions_corrected tests/test_fusion.py::TestFuseAudioVideoFiltering::test_low_confidence_singletons_filtered tests/test_fusion.py::TestFuseAudioVideoFiltering::test_duplicate_positions_deduped -v`
Expected: FAIL — `fuse_audio_video` currently does not apply slide correction or post-filtering.

**Step 3: Add slide correction and post-filtering to end of `fuse_audio_video`**

In `fusion_engine.py`, replace the end of `fuse_audio_video` (lines ~1109-1112):

**Before:**
```python
    # Detect techniques
    tab_notes = _detect_techniques(tab_notes, config)

    return tab_notes
```

**After:**
```python
    # Post-processing: correct slide/legato positions
    tab_notes = _correct_slide_positions(tab_notes, capo_fret)

    # Post-filter: remove duplicate positions and low-confidence strays
    tab_notes = _postfilter_tab_notes(tab_notes, config)

    # Detect techniques
    tab_notes = _detect_techniques(tab_notes, config)

    return tab_notes
```

**Step 4: Run test to verify it passes**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestFuseAudioVideoFiltering -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py -v`
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add tabvision-server/app/fusion_engine.py tabvision-server/tests/test_fusion.py
git commit -m "feat: add slide correction and post-filtering to fuse_audio_video"
```

---

### Task 3: Add anchor-based position selection to `fuse_audio_video`

The current `fuse_audio_video` uses a simple per-note `_select_best_position` without hand position tracking or anchor-based context. This means when video doesn't match (which is most of the time for this sample), positions are chosen poorly.

**Files:**
- Modify: `tabvision-server/app/fusion_engine.py:1005-1107` (the main loop of `fuse_audio_video`)
- Test: `tabvision-server/tests/test_fusion.py`

**Step 1: Write the failing test**

```python
    def test_position_selection_uses_anchors(self):
        """Position selection should use anchor system for consistent hand position."""
        # Create a 3-note chord to establish anchor at fret ~5 region
        # Then a single note that could be fret 0 (open) or fret 5 (same region)
        # Without anchors, it picks fret 0 (lowest). With anchors, prefers fret 5 region.
        chord_notes = [
            DetectedNote(start_time=1.0, end_time=2.0, midi_note=60, confidence=0.9, amplitude=0.7),
            DetectedNote(start_time=1.0, end_time=2.0, midi_note=64, confidence=0.9, amplitude=0.7),
            DetectedNote(start_time=1.0, end_time=2.0, midi_note=67, confidence=0.9, amplitude=0.7),
        ]
        # Single note after chord: MIDI 45 = A2 = s5f0 (open) or other positions
        single_note = DetectedNote(
            start_time=2.5, end_time=3.0, midi_note=45, confidence=0.9, amplitude=0.7,
        )
        fretboard = self._make_fretboard()
        all_notes = chord_notes + [single_note]
        result_av = fuse_audio_video(all_notes, {}, fretboard, capo_fret=0)
        result_ao = fuse_audio_only(all_notes, capo_fret=0)
        # Both should produce the same number of notes (within 1)
        assert abs(len(result_av) - len(result_ao)) <= 1
```

**Step 2: Run test to verify it fails**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestFuseAudioVideoFiltering::test_position_selection_uses_anchors -v`
Expected: Likely FAIL or different note counts.

**Step 3: Refactor `fuse_audio_video` to use anchor-based position selection**

Replace the main loop in `fuse_audio_video` (lines ~1000-1107) with anchor-based logic that mirrors `fuse_audio_only`. The key insight: when video doesn't have an observation or doesn't match, fall back to the same anchor+optimization logic. When video does match, use the video-confirmed position but still within the anchor framework.

The refactored `fuse_audio_video` should:

1. Compute chord anchors (same as `fuse_audio_only`)
2. For each chord event:
   a. Try video matching first for each note
   b. For notes where video matched: use video position, boost confidence
   c. For notes where video didn't match: use `_select_best_position` or `_optimize_chord_positions` with anchor context (same as `fuse_audio_only`)
   d. Track `previous_position` and `hand_position_fret` (same as `fuse_audio_only`)
3. Let small chords (2 notes) self-determine position (same as `fuse_audio_only`)

Here is the replacement code for the main loop body:

```python
    # Two-pass anchor system (same as fuse_audio_only)
    chord_anchors = {}
    for i, chord_notes_group in enumerate(chords):
        if len(chord_notes_group) >= 3:
            chord_candidates = []
            for note in chord_notes_group:
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

        # Position selection: use audio-only logic as backbone, with video overrides
        if len(chord_candidates) == 1:
            note, candidates = chord_candidates[0]
            if 0 in video_matches:
                position, v_conf = video_matches[0]
                confidence = min(1.0, note.confidence + config.video_match_boost)
                video_matched = True
            else:
                position = _select_best_position(
                    candidates, previous_position, config,
                    hand_position_fret=effective_hand_pos
                )
                confidence = note.confidence
                video_matched = False
                v_conf = 0.0
            if position:
                tab_note = _create_tab_note(
                    note, position, chord_id,
                    audio_confidence=note.confidence,
                    video_confidence=v_conf,
                    video_matched=video_matched,
                )
                tab_notes.append(tab_note)
                previous_position = position
                if hand_position_fret is None:
                    hand_position_fret = float(position.fret)
                else:
                    hand_position_fret = hand_position_fret * 0.7 + position.fret * 0.3
        else:
            # Multiple notes — optimize as chord
            chord_hand_pos = effective_hand_pos
            if len(chord_candidates) <= 2 and chord_hand_pos is not None:
                chord_hand_pos = None  # Let small chords self-determine

            selected_positions = _optimize_chord_positions(
                chord_candidates, previous_position, config,
                hand_position_fret=chord_hand_pos
            )

            # Override with video matches where available
            for idx, (note, _) in enumerate(chord_candidates):
                if idx in video_matches:
                    v_pos, v_conf = video_matches[idx]
                    position = v_pos
                    confidence = min(1.0, note.confidence + config.video_match_boost)
                    video_matched = True
                else:
                    position = selected_positions[idx]
                    confidence = note.confidence
                    video_matched = False
                    v_conf = 0.0

                if position:
                    tab_note = _create_tab_note(
                        note, position, chord_id,
                        audio_confidence=note.confidence,
                        video_confidence=v_conf,
                        video_matched=video_matched,
                    )
                    tab_notes.append(tab_note)

            # Update previous position and hand position from chord center
            valid_positions = [selected_positions[j] for j in range(len(chord_candidates))
                             if selected_positions[j]]
            # Include video overrides
            for idx in video_matches:
                if idx < len(selected_positions):
                    valid_positions[idx] = video_matches[idx][0]
            valid_positions = [p for p in valid_positions if p]
            if valid_positions:
                avg_fret = sum(p.fret for p in valid_positions) / len(valid_positions)
                previous_position = min(valid_positions, key=lambda p: abs(p.fret - avg_fret))
                if hand_position_fret is None:
                    hand_position_fret = avg_fret
                else:
                    hand_position_fret = hand_position_fret * 0.4 + avg_fret * 0.6
```

**Step 4: Run test to verify it passes**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestFuseAudioVideoFiltering -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py -v`
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add tabvision-server/app/fusion_engine.py tabvision-server/tests/test_fusion.py
git commit -m "feat: add anchor-based position selection to fuse_audio_video"
```

---

### Task 4: End-to-end validation via server

Validate that the server now produces comparable results to the direct evaluation.

**Step 1: Run the direct evaluation for baseline**

Run: `cd tabvision-server && python evaluate_transcription.py`
Expected: F1=91.4%, 41 notes, 37 TP

**Step 2: Start the server and submit the sample video**

```bash
cd tabvision-server
source venv/bin/activate
python run.py &
sleep 3
# Submit video
JOB_ID=$(curl -s -X POST http://localhost:5000/jobs -F "video=@../sample-video.mp4" -F "capo_fret=0" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job: $JOB_ID"
# Poll until complete
for i in $(seq 1 120); do
  STATUS=$(curl -s http://localhost:5000/jobs/$JOB_ID | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
  if [ "$STATUS" = "completed" ]; then echo "Done!"; break; fi
  if [ "$STATUS" = "failed" ]; then echo "Failed!"; break; fi
  sleep 5
done
# Get result
curl -s http://localhost:5000/jobs/$JOB_ID/result > /tmp/server_result.json
python3 -c "import json; r=json.load(open('/tmp/server_result.json')); print(f'Notes: {len(r[\"notes\"])}')"
```

**Step 3: Compare server result to ground truth**

Run the evaluation comparison using the server output. The server should now produce approximately the same number of notes (~41) as the direct evaluation. The F1 score should be similar (~90%+).

If the server produces significantly more notes (e.g., >50), investigate whether `fretboard` detection is triggering `fuse_audio_video` and whether the video observations are helping or hurting. Consider adding a minimum fretboard confidence threshold in `process_job` (e.g., only use video if `fretboard.detection_confidence > 0.5`).

**Step 4: If needed — add fretboard confidence gate in process_job**

If the video pipeline is consistently finding low-quality fretboards that route to `fuse_audio_video` without actually helping, add a confidence gate in `processing.py`:

```python
# Only use video fusion if fretboard detection is reliable
if fretboard and fretboard.detection_confidence > 0.5 and video_observations:
    tab_notes = fuse_audio_video(...)
else:
    tab_notes = fuse_audio_only(...)
```

**Step 5: Kill the server**

```bash
pkill -f "python run.py"
```

**Step 6: Commit any additional fixes**

```bash
git add -A
git commit -m "fix: tune fuse_audio_video for parity with audio-only pipeline"
```

---

## Risk Notes

- **Video matching may override good audio-only positions**: If the video detects a finger at the wrong fret, it could override a correct audio-derived position. Mitigation: video only overrides when it matches an audio candidate exactly, so it can't introduce entirely wrong positions.
- **FretboardGeometry constructor**: The test helper `_make_fretboard()` must match the actual `FretboardGeometry` dataclass fields. Check the dataclass definition before writing tests.
- **Existing `fuse_audio_video` tests**: Some existing tests may depend on the old behavior (no filtering). These may need confidence or amplitude adjustments (same pattern as `test_fuse_multiple_notes` in previous work).
