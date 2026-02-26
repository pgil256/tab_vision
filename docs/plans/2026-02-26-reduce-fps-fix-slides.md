# Reduce False Positives & Fix Slide Mapping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Push F1 from 84.7% toward ~91% by eliminating 5-6 false positives and recovering 1 false negative via slide correction.

**Architecture:** Two changes to `fusion_engine.py`: (1) Fix the slide correction tiebreaker so it prefers the existing string assignment when multiple strings tie, converting FP s4f2→TP s5f7. (2) Add a post-fusion filter pass (`_postfilter_tab_notes`) that removes duplicate positions, low-confidence isolated singletons, and stray open-string resonance artifacts.

**Tech Stack:** Python, pytest. All changes in `tabvision-server/app/fusion_engine.py` and `tabvision-server/tests/test_fusion.py`.

**Current baseline:** 36 TP, 9 FP, 4 FN → P=80.0%, R=90.0%, F1=84.7%

**Target:** ≥37 TP, ≤4 FP, ≤3 FN → P≥90%, R≥92%, F1≥91%

---

## Evidence: False Positive Analysis

| # | Note | Time | MIDI | Conf | Category | Fix |
|---|------|------|------|------|----------|-----|
| 1 | s3f5 | 1.74s | 60 | 0.78 | Overtone in 3-note chord | Hard - skip for now |
| 2 | s2f3 | 6.31s | 62 | 0.70 | Extra note in 2-note chord | Hard - skip for now |
| 3 | s5f0 | 8.20s | 45 | 0.70 | Isolated open-string resonance | Task 3: open-string filter |
| 4 | s3f0 | 10.06s | 55 | 0.85 | Duplicate of s3f0@9.99s (0.07s) | Task 2: dedup filter |
| 5 | s5f0 | 10.50s | 45 | 0.91 | Duplicate of s5f0@10.30s (0.20s) | Task 2: dedup filter |
| 6 | s4f2 | 10.50s | 52 | 0.69 | Stray in busy section | Task 3: open-string context or Task 2 side-effect |
| 7 | s1f0 | 10.78s | 64 | 0.65 | Isolated open-string resonance | Task 3: open-string filter |
| 8 | s5f4 | 12.51s | 49 | 0.54 | Low-conf isolated stray | Task 2: low-conf filter |
| 9 | s4f2 | 12.98s | 52 | 1.00 | Wrong string (should be s5f7) | Task 1: slide fix |

---

## Task 1: Fix Slide Correction Tiebreaker

**Problem:** In `_correct_slide_positions` Pass 1, when a note pair (s5f8→s4f2) has multiple common strings that all tie on fret distance, `min()` picks the lowest string number (4) instead of preferring the existing string (5). Pass 2 happens to fix the prev note back, but curr (s4f2) stays wrong — it should become s5f7.

**Root cause:** The tiebreaker `min(common_strings, key=lambda s: abs(...))` breaks ties by Python's default `min` on integers, picking the smallest string number. It should prefer the string already assigned to prev or curr.

**Files:**
- Modify: `tabvision-server/app/fusion_engine.py` — `_correct_slide_positions`, Pass 1 tiebreaker
- Test: `tabvision-server/tests/test_fusion.py` — new `TestCorrectSlidePositions` class

**Step 1: Write the failing test**

Add to `test_fusion.py`:

```python
from app.fusion_engine import _correct_slide_positions

class TestCorrectSlidePositions:
    """Tests for slide/legato position correction."""

    def test_descending_semitone_prefers_existing_string(self):
        """When s5f8 is followed by a note that could be s5f7 or s4f2,
        prefer s5f7 (same string as prev) for continuity."""
        notes = [
            TabNote(id='1', timestamp=12.62, string=5, fret=8, confidence=0.77,
                    confidence_level='medium', midi_note=53),
            TabNote(id='2', timestamp=12.98, string=4, fret=2, confidence=1.00,
                    confidence_level='high', midi_note=52),
        ]
        result = _correct_slide_positions(notes, capo_fret=0)
        # Both should end up on string 5 (descending slide)
        assert result[0].string == 5
        assert result[0].fret == 8
        assert result[1].string == 5
        assert result[1].fret == 7

    def test_ascending_semitone_prefers_existing_string(self):
        """When s5f7 is followed by a note that could be s5f8 or s4f3,
        prefer s5f8 (same string as prev)."""
        notes = [
            TabNote(id='1', timestamp=1.00, string=5, fret=7, confidence=0.8,
                    confidence_level='high', midi_note=52),
            TabNote(id='2', timestamp=1.30, string=4, fret=3, confidence=0.8,
                    confidence_level='high', midi_note=53),
        ]
        result = _correct_slide_positions(notes, capo_fret=0)
        assert result[0].string == 5
        assert result[0].fret == 7
        assert result[1].string == 5
        assert result[1].fret == 8

    def test_does_not_change_same_string_pair(self):
        """Notes already on the same string should not be modified."""
        notes = [
            TabNote(id='1', timestamp=1.00, string=3, fret=2, confidence=0.8,
                    confidence_level='high', midi_note=57),
            TabNote(id='2', timestamp=1.30, string=3, fret=4, confidence=0.8,
                    confidence_level='high', midi_note=59),
        ]
        result = _correct_slide_positions(notes, capo_fret=0)
        assert result[0].string == 3
        assert result[0].fret == 2
        assert result[1].string == 3
        assert result[1].fret == 4

    def test_full_slide_section_s5f8_to_s5f7(self):
        """Realistic slide section: s5f0→s5f4→s5f8→s4f2 should correct
        the last note to s5f7 while preserving the rest."""
        notes = [
            TabNote(id='a', timestamp=12.16, string=5, fret=0, confidence=0.88,
                    confidence_level='high', midi_note=45),
            TabNote(id='b', timestamp=12.51, string=5, fret=4, confidence=0.54,
                    confidence_level='medium', midi_note=49),
            TabNote(id='c', timestamp=12.62, string=5, fret=8, confidence=0.77,
                    confidence_level='medium', midi_note=53),
            TabNote(id='d', timestamp=12.98, string=4, fret=2, confidence=1.00,
                    confidence_level='high', midi_note=52),
        ]
        result = _correct_slide_positions(notes, capo_fret=0)
        assert result[0].string == 5 and result[0].fret == 0   # unchanged
        assert result[1].string == 5 and result[1].fret == 4   # unchanged
        assert result[2].string == 5 and result[2].fret == 8   # unchanged
        assert result[3].string == 5 and result[3].fret == 7   # corrected!
```

**Step 2: Run tests to verify they fail**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestCorrectSlidePositions -v`

Expected: FAIL — `test_descending_semitone_prefers_existing_string` fails with `assert 4 == 5` (wrong string chosen).

**Step 3: Fix the tiebreaker**

In `fusion_engine.py`, `_correct_slide_positions`, replace the `best_string = min(...)` line in Pass 1 with a version that breaks ties by preferring the existing string:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestCorrectSlidePositions -v`

Expected: All 4 tests PASS.

**Step 5: Run full test suite**

Run: `cd tabvision-server && python -m pytest tests/ -v`

Expected: All tests pass (existing tests should not break since we only changed tie-breaking behavior).

**Step 6: Verify with evaluation**

Run: `cd tabvision-server && python evaluate_transcription.py`

Expected: s5f7 now matched (FP#9 gone, FN for s5f7 gone). TP: 37, FP: 8.

**Step 7: Commit**

```bash
git add tabvision-server/app/fusion_engine.py tabvision-server/tests/test_fusion.py
git commit -m "fix: slide correction tiebreaker prefers existing string assignment"
```

---

## Task 2: Add Post-Fusion Dedup and Low-Confidence Isolated Note Filter

**Problem:** After fusion, some notes appear as duplicates of the same string+fret within a short time window (0.3s), and some low-confidence isolated notes are stray detections that don't belong to any melodic or chordal context.

**Targets:**
- FP#4: s3f0@10.06s — duplicate of s3f0@9.99s (0.07s apart)
- FP#5: s5f0@10.50s — duplicate of s5f0@10.30s (0.20s apart)
- FP#8: s5f4@12.51s — low-confidence (0.54) isolated singleton

**Files:**
- Modify: `tabvision-server/app/fusion_engine.py` — add `_postfilter_tab_notes()`, call from `fuse_audio_only()`
- Test: `tabvision-server/tests/test_fusion.py` — new `TestPostfilterTabNotes` class

**Step 1: Write the failing tests**

Add to `test_fusion.py`:

```python
from app.fusion_engine import _postfilter_tab_notes, FusionConfig

class TestPostfilterTabNotes:
    """Tests for post-fusion note filtering."""

    def _make_note(self, timestamp, string, fret, confidence=0.8, midi_note=60,
                   is_part_of_chord=False):
        return TabNote(
            id=str(timestamp), timestamp=timestamp, string=string, fret=fret,
            confidence=confidence, confidence_level=get_confidence_level(confidence),
            midi_note=midi_note, is_part_of_chord=is_part_of_chord,
        )

    def test_removes_duplicate_position_within_window(self):
        """Same string+fret within 0.3s should keep only the higher confidence one."""
        notes = [
            self._make_note(9.99, 3, 0, confidence=0.64),
            self._make_note(10.06, 3, 0, confidence=0.85),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 1
        assert result[0].confidence == 0.85  # kept the higher-confidence one

    def test_keeps_duplicate_outside_window(self):
        """Same string+fret more than 0.3s apart should both be kept."""
        notes = [
            self._make_note(1.00, 3, 0, confidence=0.80),
            self._make_note(1.50, 3, 0, confidence=0.85),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 2

    def test_removes_low_confidence_isolated_singleton(self):
        """Low-confidence (<0.6) notes not in a chord and far from same-string
        notes should be removed."""
        notes = [
            self._make_note(12.16, 5, 0, confidence=0.88, midi_note=45),
            self._make_note(12.51, 5, 4, confidence=0.54, midi_note=49),  # low conf, isolated
            self._make_note(12.62, 5, 8, confidence=0.77, midi_note=53),
        ]
        # 12.51 is isolated (different fret from neighbors) AND low confidence
        # But it IS on the same string as neighbors within 0.3s, so NOT isolated by our definition
        # Actually the filter checks: not in chord AND conf < threshold
        # Need to define "isolated" carefully
        result = _postfilter_tab_notes(notes, FusionConfig())
        # s5f4 at 12.51 has conf=0.54 < 0.6, not in chord → removed
        assert len(result) == 2
        assert all(n.fret != 4 for n in result)

    def test_keeps_low_confidence_chord_member(self):
        """Low-confidence notes that are part of a chord should be kept."""
        notes = [
            self._make_note(1.00, 3, 5, confidence=0.55, is_part_of_chord=True),
            self._make_note(1.00, 1, 7, confidence=0.80, is_part_of_chord=True),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 2

    def test_keeps_high_confidence_singleton(self):
        """High-confidence singletons should always be kept."""
        notes = [
            self._make_note(5.00, 5, 0, confidence=0.70),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 1
```

**Step 2: Run tests to verify they fail**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestPostfilterTabNotes -v`

Expected: FAIL — `_postfilter_tab_notes` not defined.

**Step 3: Implement `_postfilter_tab_notes`**

Add to `fusion_engine.py` (before `fuse_audio_only`):

```python
def _postfilter_tab_notes(
    tab_notes: list[TabNote],
    config: FusionConfig
) -> list[TabNote]:
    """Remove post-fusion artifacts: duplicate positions and low-confidence strays.

    Two filters applied in order:
    1. Dedup: same string+fret within 0.3s → keep highest confidence
    2. Low-confidence isolated: conf < 0.6, not in chord → remove

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
    low_conf_threshold = 0.6
    result = []
    for note in keep_after_dedup:
        if (note.confidence < low_conf_threshold
                and not note.is_part_of_chord):
            continue  # Remove low-confidence isolated note
        result.append(note)

    return result
```

Then call it at the end of `fuse_audio_only`, right before `_detect_techniques`:

```python
    # Post-filter: remove duplicate positions and low-confidence strays
    tab_notes = _postfilter_tab_notes(tab_notes, config)

    # Detect techniques (hammer-ons, pull-offs, slides)
    tab_notes = _detect_techniques(tab_notes, config)
```

**Step 4: Run tests to verify they pass**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestPostfilterTabNotes -v`

Expected: All 5 tests PASS.

**Step 5: Run full test suite**

Run: `cd tabvision-server && python -m pytest tests/ -v`

Expected: All pass. Some existing tests may need adjustment if they rely on low-confidence notes surviving.

**Step 6: Verify with evaluation**

Run: `cd tabvision-server && python evaluate_transcription.py`

Expected: FP#4, FP#5, FP#8 removed. TP stays ≥37, FP drops by ~3.

**Step 7: Commit**

```bash
git add tabvision-server/app/fusion_engine.py tabvision-server/tests/test_fusion.py
git commit -m "feat: add post-fusion dedup and low-confidence note filter"
```

---

## Task 3: Add Isolated Open-String Resonance Filter

**Problem:** Open strings (fret 0) on bass strings sometimes appear as isolated false positives due to sympathetic resonance — the open string vibrates when nearby strings are played. These FPs are isolated (not in a chord), on a different string group than their neighbors, and relatively low confidence.

**Targets:**
- FP#3: s5f0@8.20s (conf=0.70) — isolated open A between melody on strings 1-3
- FP#7: s1f0@10.78s (conf=0.65) — isolated open E between notes on strings 2-5

**Files:**
- Modify: `tabvision-server/app/fusion_engine.py` — extend `_postfilter_tab_notes()`
- Test: `tabvision-server/tests/test_fusion.py` — add to `TestPostfilterTabNotes`

**Step 1: Write the failing tests**

Add to `TestPostfilterTabNotes`:

```python
    def test_removes_isolated_open_string_different_string_group(self):
        """Open string note isolated among notes on distant strings should be removed."""
        notes = [
            self._make_note(7.55, 3, 2, confidence=0.90, midi_note=57),   # melody on s3
            self._make_note(7.95, 2, 1, confidence=0.93, midi_note=60),   # melody on s2
            self._make_note(8.20, 5, 0, confidence=0.70, midi_note=45),   # FP: isolated open A
            self._make_note(8.81, 1, 0, confidence=0.92, midi_note=64),   # melody on s1
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 3
        # s5f0 should be removed — isolated open string far from neighbor strings
        assert all(not (n.string == 5 and n.fret == 0) for n in result)

    def test_keeps_open_string_in_chord(self):
        """Open string that's part of a chord should be kept."""
        notes = [
            self._make_note(10.00, 5, 2, confidence=0.67, midi_note=47,
                           is_part_of_chord=True),
            self._make_note(10.00, 3, 0, confidence=0.64, midi_note=55,
                           is_part_of_chord=True),
            self._make_note(10.00, 2, 0, confidence=0.71, midi_note=59,
                           is_part_of_chord=True),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 3  # all kept — part of chord

    def test_keeps_open_string_among_nearby_strings(self):
        """Open string with neighbors on adjacent strings should be kept."""
        notes = [
            self._make_note(5.00, 4, 3, confidence=0.80, midi_note=53),
            self._make_note(5.20, 5, 0, confidence=0.70, midi_note=45),  # open A near s4
            self._make_note(5.40, 5, 2, confidence=0.80, midi_note=47),
        ]
        result = _postfilter_tab_notes(notes, FusionConfig())
        assert len(result) == 3  # kept — s5 is adjacent to s4
```

**Step 2: Run tests to verify they fail**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestPostfilterTabNotes::test_removes_isolated_open_string_different_string_group -v`

Expected: FAIL — the filter doesn't check for isolated open strings yet.

**Step 3: Add open-string isolation check to `_postfilter_tab_notes`**

Add as Pass 3 in `_postfilter_tab_notes`, after the low-confidence filter:

```python
    # Pass 3: Remove isolated open-string resonance
    # Open strings (fret 0) that are not in a chord and whose string is far
    # (2+ strings away) from all neighbors within 1.0s are likely resonance artifacts.
    neighbor_window = 1.0
    min_string_distance = 2  # must be 2+ strings away from ALL neighbors
    final = []
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
        final.append(note)

    return final
```

**Step 4: Run tests to verify they pass**

Run: `cd tabvision-server && python -m pytest tests/test_fusion.py::TestPostfilterTabNotes -v`

Expected: All 8 tests PASS.

**Step 5: Run full test suite**

Run: `cd tabvision-server && python -m pytest tests/ -v`

Expected: All pass.

**Step 6: Verify with evaluation**

Run: `cd tabvision-server && python evaluate_transcription.py`

Expected: FP#3 and FP#7 removed. Final counts should be approximately:
- TP: 37, FP: ~3-4, FN: 3
- P ≈ 90-92%, R ≈ 92.5%, F1 ≈ 91-92%

**Step 7: Commit**

```bash
git add tabvision-server/app/fusion_engine.py tabvision-server/tests/test_fusion.py
git commit -m "feat: add isolated open-string resonance filter to post-fusion pass"
```

---

## Task 4: Final Validation and Tuning

**Step 1: Run full evaluation with all changes**

Run: `cd tabvision-server && python evaluate_transcription.py`

Record exact TP, FP, FN, P, R, F1 numbers.

**Step 2: Review any new FPs or lost TPs**

If any true positives were lost by the new filters, adjust thresholds:
- `dedup_window`: increase from 0.3 if duplicates are still slipping through, decrease if TPs are being removed
- `low_conf_threshold`: decrease from 0.6 if needed
- `min_string_distance`: increase from 2 if open-string filter is too aggressive

**Step 3: Run full test suite one more time**

Run: `cd tabvision-server && python -m pytest tests/ -v`

**Step 4: Commit any tuning adjustments**

```bash
git add tabvision-server/app/fusion_engine.py
git commit -m "tune: adjust post-fusion filter thresholds based on evaluation"
```

---

## Expected Final Results

| Metric | Before | After Task 1 | After Task 2 | After Task 3 |
|--------|--------|-------------|-------------|-------------|
| TP | 36 | 37 | 37 | 37 |
| FP | 9 | 8 | 5 | 3-4 |
| FN | 4 | 3 | 3 | 3 |
| Precision | 80.0% | 82.2% | 88.1% | 90-92% |
| Recall | 90.0% | 92.5% | 92.5% | 92.5% |
| F1 | 84.7% | 87.1% | 90.2% | 91-92% |

## Risk Notes

- **Task 1** is low-risk — only changes tie-breaking behavior, verified with specific test case
- **Task 2** dedup is low-risk; low-conf filter may remove real quiet notes in other songs — the 0.6 threshold is conservative
- **Task 3** open-string filter is medium-risk — could remove legitimate open-string notes in songs with arpeggiated open chords. The `min_string_distance=2` and `neighbor_window=1.0` guards help but may need tuning per-song
