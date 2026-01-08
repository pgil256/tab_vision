# Phase 1: Audio Pipeline Design

**Date:** 2026-01-08
**Status:** Approved
**Author:** Claude + User

## Overview

Replace the fake processing in tabvision-server with real audio analysis using Basic Pitch. This phase implements audio-only tab generation with a simple lowest-fret heuristic. Video analysis will be added in Phase 2.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Processing model | Background thread with polling | Frontend already polls; avoids Redis/Celery complexity |
| MIDI-to-guitar mapping | Prefer lowest fret | Simple baseline; goal is context-aware positioning later |
| Polyphonic handling | Output all detected notes | Preserves data; frontend handles multiple notes at same timestamp |
| Confidence scoring | Pass through Basic Pitch value | Simple and transparent; thresholds already in frontend |

## Architecture

```
POST /jobs (upload)
       │
       ▼
┌─────────────────────────────────────────────┐
│  Background Thread                          │
│                                             │
│  ┌───────────────────────┐                  │
│  │ 1. audio_pipeline.py  │ ◄── Phase 1     │
│  │    - ffmpeg extract   │                  │
│  │    - Basic Pitch      │                  │
│  │    - returns notes    │                  │
│  │      with candidates  │                  │
│  └───────────┬───────────┘                  │
│              │                              │
│              │  ┌───────────────────────┐   │
│              │  │ 2. video_pipeline.py  │   │
│              │  │    - MediaPipe hands  │ ◄── Phase 2 (future)
│              │  │    - fretboard detect │   │
│              │  │    - finger positions │   │
│              │  └───────────┬───────────┘   │
│              │              │               │
│              ▼              ▼               │
│  ┌─────────────────────────────────────┐   │
│  │ 3. fusion_engine.py                 │   │
│  │    Phase 1: just pick lowest fret   │   │
│  │    Phase 2: combine audio + video   │ ◄── Evolves
│  └───────────┬─────────────────────────┘   │
│              ▼                              │
│  ┌───────────────────────┐                  │
│  │ 4. Build TabDocument  │                  │
│  └───────────────────────┘                  │
└─────────────────────────────────────────────┘
       │
       ▼ (updates Job status/progress)
GET /jobs/:id → polls status
GET /jobs/:id/result → returns TabDocument
```

## New Files

| File | Purpose |
|------|---------|
| `app/audio_pipeline.py` | ffmpeg extraction + Basic Pitch analysis |
| `app/guitar_mapping.py` | MIDI note to fret/string mapping |
| `app/fusion_engine.py` | Combine signals into TabNotes |
| `app/processing.py` | Background thread orchestration |

## Modified Files

| File | Changes |
|------|---------|
| `app/routes.py` | Launch background thread instead of fake completion |
| `requirements.txt` | Add basic-pitch, ffmpeg-python, numpy |

---

## Module Details

### 1. Audio Pipeline (`app/audio_pipeline.py`)

**Step 1: Extract audio with ffmpeg**
```python
def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio track from video as WAV (mono, 22050 Hz)."""
    # Basic Pitch expects mono audio, 22050 Hz works well
    # Output: /tmp/{job_id}_audio.wav
```

**Step 2: Run Basic Pitch**
```python
def analyze_pitch(audio_path: str) -> list[DetectedNote]:
    """Run Basic Pitch on audio file."""
    # Returns list of:
    #   - start_time (seconds)
    #   - end_time (seconds)
    #   - midi_note (int, e.g., 69 = A4)
    #   - confidence (0.0-1.0)
```

**Output format (intermediate):**
```python
@dataclass
class DetectedNote:
    start_time: float      # seconds
    end_time: float        # seconds (for sustain, optional use)
    midi_note: int         # MIDI number (21-108 typical for guitar)
    confidence: float      # 0.0-1.0 from Basic Pitch
```

**Progress updates:**
- "extracting_audio" at 0.1
- "analyzing_audio" at 0.2-0.5

---

### 2. Guitar Mapping (`app/guitar_mapping.py`)

**Standard tuning reference (MIDI note numbers):**
```python
STANDARD_TUNING = {
    6: 40,  # Low E (E2)
    5: 45,  # A (A2)
    4: 50,  # D (D3)
    3: 55,  # G (G3)
    2: 59,  # B (B3)
    1: 64,  # High E (E4)
}
MAX_FRET = 24
```

**Core function:**
```python
def get_candidate_positions(midi_note: int, capo_fret: int = 0) -> list[Position]:
    """Return all valid fret/string combos for a MIDI note."""
    # For each string, check if (midi_note - open_string_midi)
    # is in range [capo_fret, MAX_FRET]
    # Returns list of (string, fret) tuples, sorted by fret ascending
```

**Example:** MIDI 69 (A4) with no capo returns:
- String 1, fret 5 (lowest, will be chosen)
- String 2, fret 10
- String 3, fret 14

**Pick best position (Phase 1 heuristic):**
```python
def pick_lowest_fret(candidates: list[Position]) -> Position:
    """Simple heuristic: return position with lowest fret number."""
    return min(candidates, key=lambda p: p.fret)
```

**Handle out-of-range notes:**
- If MIDI note is too low for guitar (below E2/40), skip
- If MIDI note has no valid position, return None

---

### 3. Fusion Engine (`app/fusion_engine.py`)

**Phase 1 implementation:**
```python
def fuse_audio_only(
    detected_notes: list[DetectedNote],
    capo_fret: int
) -> list[TabNote]:
    """Convert detected audio notes to TabNotes using lowest-fret heuristic."""
    tab_notes = []
    for note in detected_notes:
        candidates = get_candidate_positions(note.midi_note, capo_fret)
        if not candidates:
            continue  # Skip notes outside guitar range

        position = pick_lowest_fret(candidates)
        tab_notes.append(TabNote(
            id=str(uuid4()),
            timestamp=note.start_time,
            string=position.string,
            fret=position.fret,
            confidence=note.confidence,
            confidence_level=get_confidence_level(note.confidence),
            detected_midi_note=note.midi_note,  # preserve for debugging
        ))
    return tab_notes
```

---

### 4. Processing Orchestration (`app/processing.py`)

```python
def process_job(job_id: str):
    """Background thread entry point."""
    job = job_storage.get(job_id)
    try:
        # Stage 1: Extract audio
        update_job(job, "extracting_audio", 0.1)
        audio_path = extract_audio(job.video_path)

        # Stage 2: Analyze with Basic Pitch
        update_job(job, "analyzing_audio", 0.3)
        detected_notes = analyze_pitch(audio_path)

        # Stage 3: Fuse (Phase 1: audio only)
        update_job(job, "fusing", 0.7)
        tab_notes = fuse_audio_only(detected_notes, job.capo_fret)

        # Stage 4: Build and save TabDocument
        update_job(job, "complete", 1.0)
        save_result(job, tab_notes)

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
```

---

### 5. Routes Integration (`app/routes.py`)

**Replace fake processing with:**
```python
from threading import Thread
from app.processing import process_job

@bp.route('/jobs', methods=['POST'])
def create_job():
    # ... existing file validation and save ...

    job = Job.create(video_path=video_path, capo_fret=capo_fret)
    job_storage.save(job)

    # Launch background processing
    thread = Thread(target=process_job, args=(job.id,))
    thread.daemon = True  # Don't block app shutdown
    thread.start()

    return jsonify({'job_id': job.id}), 201
```

**Result endpoint:**
```python
@bp.route('/jobs/<job_id>/result', methods=['GET'])
def get_result(job_id):
    job = job_storage.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.status != 'completed':
        return jsonify({'error': 'Job not completed'}), 400

    # Load saved result instead of generating fake data
    result = load_result(job)
    return jsonify(result), 200
```

---

## Dependencies

**requirements.txt additions:**
```
basic-pitch>=0.3.0
ffmpeg-python>=0.2.0
numpy>=1.24.0
```

**System dependency:**
```bash
# Ubuntu/Debian/WSL
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## Error Handling

| Error | Handling |
|-------|----------|
| ffmpeg not found | Fail job with "ffmpeg not installed" |
| No audio track | Fail job with "Video has no audio" |
| Basic Pitch fails | Fail job with exception details |
| No notes detected | Complete with empty notes array (valid result) |
| File I/O errors | Fail job, log traceback |

**Cleanup:**
- Delete temp audio file after processing
- Keep video file for potential Phase 2 reprocessing
- Results JSON persists until job is deleted

**Thread safety:**
- JobStorage uses a simple dict - safe since each job has one writer thread
- For production scaling, would need proper locking or database

---

## Testing Strategy

**Unit tests:**

`tests/test_guitar_mapping.py`:
- MIDI → position candidates for known notes
- Capo offset calculations
- Out-of-range notes return empty list
- Lowest-fret selection

`tests/test_audio_pipeline.py`:
- Audio extraction with sample video (mock ffmpeg for CI)
- Basic Pitch integration with short audio clip
- DetectedNote dataclass

`tests/test_fusion.py`:
- fuse_audio_only produces valid TabNotes
- Confidence level mapping
- Empty input handling

**Integration test:**

`tests/test_processing.py`:
- Full pipeline with real short video (~5 seconds)
- Job state transitions: pending → processing → completed
- Progress updates occur
- Result JSON is valid TabDocument

**Test fixtures:**
- `tests/fixtures/test_video.mp4` - short guitar clip

**Running tests:**
```bash
pytest tests/ -v                    # All tests
pytest tests/test_guitar_mapping.py # Just mapping tests
pytest tests/ -k "not integration"  # Skip slow integration tests
```

---

## Implementation Order

1. `guitar_mapping.py` + tests (no external deps, easy to verify)
2. `audio_pipeline.py` + tests (requires ffmpeg, Basic Pitch)
3. `fusion_engine.py` + tests (combines above)
4. `processing.py` (orchestration)
5. Update `routes.py` (integration)
6. End-to-end testing with real video

---

## Future: Phase 2 Integration Points

The architecture supports adding video analysis:

1. Add `video_pipeline.py` alongside `audio_pipeline.py`
2. Expand `fusion_engine.py` to accept both audio and video signals
3. Upgrade fusion heuristic from "lowest fret" to "audio+video agreement"
4. Add "analyzing_video" stage between "analyzing_audio" and "fusing"
