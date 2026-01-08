# Phase 2: Video Pipeline Design

**Date:** 2026-01-08
**Status:** Draft
**Author:** Claude + User

## Overview

Add video analysis using MediaPipe Hands and OpenCV to detect finger positions on the fretboard. Combined with audio analysis from Phase 1, this enables higher-confidence tab generation through audio-video agreement.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Hand tracking | MediaPipe Hands | Fast, accurate, CPU-friendly, well-documented |
| Fretboard detection | Edge detection + Hough lines | Works without ML model, handles standard guitar necks |
| Frame extraction | At note onset timestamps | Reduces processing; only analyze frames where notes detected |
| Coordinate mapping | Homography transform | Maps finger pixels to fret/string grid |

## Architecture

```
                    Audio Pipeline (Phase 1)
                           │
                    detected_notes[]
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Video Pipeline (Phase 2)                                   │
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │ video_pipeline.py   │    │ fretboard_detection.py      │ │
│  │                     │    │                             │ │
│  │ 1. Extract frames   │    │ 1. Detect fretboard edges   │ │
│  │    at onset times   │    │ 2. Find fret lines (Hough)  │ │
│  │                     │    │ 3. Find string lines        │ │
│  │ 2. MediaPipe Hands  │    │ 4. Build coordinate grid    │ │
│  │    → fingertips     │    │                             │ │
│  └──────────┬──────────┘    └──────────────┬──────────────┘ │
│             │                              │                │
│             │    finger_positions[]        │  fretboard     │
│             │                              │  geometry      │
│             └──────────────┬───────────────┘                │
│                            │                                │
│                            ▼                                │
│             ┌──────────────────────────────┐                │
│             │ map_finger_to_position()     │                │
│             │ → (string, fret) or None     │                │
│             └──────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                           │
                    video_observations[]
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Fusion Engine (Updated)                                    │
│                                                             │
│  For each detected_note:                                    │
│    1. Get audio candidates (from guitar_mapping)            │
│    2. Get video observation (if available)                  │
│    3. If video agrees with audio candidate → high confidence│
│    4. If video disagrees → medium confidence, use audio     │
│    5. If no video data → use audio only (as before)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## New Files

| File | Purpose | Agent |
|------|---------|-------|
| `app/video_pipeline.py` | Frame extraction + MediaPipe hand detection | Agent 2 |
| `app/fretboard_detection.py` | Fretboard geometry detection | Agent 3 |

## Modified Files

| File | Changes | Agent |
|------|---------|-------|
| `app/fusion_engine.py` | Add video signal fusion | Sync point |
| `app/processing.py` | Call video pipeline | Sync point |
| `requirements.txt` | Add mediapipe, opencv-python | Agent 2 |

---

## Module Details

### 1. Video Pipeline (`app/video_pipeline.py`) - Agent 2

**Data structures:**
```python
@dataclass
class FingerPosition:
    """Position of a fingertip in frame coordinates."""
    finger_id: int      # 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
    x: float            # Normalized x coordinate (0-1)
    y: float            # Normalized y coordinate (0-1)
    z: float            # Depth estimate (relative)

@dataclass
class HandObservation:
    """Hand detection result for a single frame."""
    timestamp: float
    fingers: list[FingerPosition]
    is_left_hand: bool
```

**Functions:**
```python
def extract_frame(video_path: str, timestamp: float) -> np.ndarray:
    """Extract a single frame from video at given timestamp."""
    # Use OpenCV VideoCapture
    # Seek to timestamp, read frame
    # Return BGR numpy array

def detect_hand_landmarks(frame: np.ndarray) -> HandObservation | None:
    """Detect hand and finger positions using MediaPipe."""
    # Initialize MediaPipe Hands (cached)
    # Process frame
    # Return fingertip landmarks (tips only: indices 4, 8, 12, 16, 20)
    # Return None if no hand detected

def analyze_video_at_timestamps(
    video_path: str,
    timestamps: list[float]
) -> dict[float, HandObservation]:
    """Analyze video frames at given timestamps."""
    # For each timestamp:
    #   1. Extract frame
    #   2. Detect hand landmarks
    #   3. Store observation
    # Return dict mapping timestamp → observation
```

**MediaPipe setup:**
```python
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
```

---

### 2. Fretboard Detection (`app/fretboard_detection.py`) - Agent 3

**Data structures:**
```python
@dataclass
class FretboardGeometry:
    """Detected fretboard geometry."""
    # Corner points of fretboard region (for homography)
    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]

    # Detected fret positions (x coordinates in normalized space)
    fret_positions: list[float]  # Index 0 = nut, 1 = fret 1, etc.

    # Detected string positions (y coordinates in normalized space)
    string_positions: list[float]  # Index 0 = string 6 (low E), 5 = string 1

@dataclass
class VideoPosition:
    """A fret/string position detected from video."""
    string: int         # 1-6
    fret: int           # 0-24
    confidence: float   # 0-1
```

**Functions:**
```python
def detect_fretboard(frame: np.ndarray) -> FretboardGeometry | None:
    """Detect fretboard region and geometry in frame."""
    # 1. Convert to grayscale
    # 2. Edge detection (Canny)
    # 3. Hough line transform
    # 4. Identify roughly horizontal lines (frets)
    # 5. Identify roughly vertical lines (strings - optional)
    # 6. Find bounding quadrilateral
    # Return None if detection fails

def map_finger_to_position(
    finger: FingerPosition,
    geometry: FretboardGeometry
) -> VideoPosition | None:
    """Map a finger position to fret/string coordinates."""
    # 1. Check if finger is within fretboard region
    # 2. Apply homography to get normalized coordinates
    # 3. Find nearest fret (by x position)
    # 4. Find nearest string (by y position)
    # 5. Calculate confidence based on proximity
    # Return None if finger not on fretboard
```

**OpenCV operations:**
```python
import cv2
import numpy as np

def detect_fretboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )
    # Filter and cluster lines...
```

---

### 3. Updated Fusion Engine (`app/fusion_engine.py`) - Sync Point

**New function:**
```python
def fuse_audio_video(
    detected_notes: list[DetectedNote],
    video_observations: dict[float, HandObservation],
    fretboard: FretboardGeometry | None,
    capo_fret: int
) -> list[TabNote]:
    """Combine audio and video signals for tab generation."""
    tab_notes = []

    for note in detected_notes:
        # Get audio candidates
        candidates = get_candidate_positions(note.midi_note, capo_fret)
        if not candidates:
            continue

        # Try to get video observation at this timestamp
        video_obs = find_nearest_observation(video_observations, note.start_time)
        video_position = None

        if video_obs and fretboard:
            # Map fingers to fret/string positions
            for finger in video_obs.fingers:
                pos = map_finger_to_position(finger, fretboard)
                if pos and pos in candidates:
                    video_position = pos
                    break

        # Determine final position and confidence
        if video_position:
            # Video agrees with an audio candidate
            position = video_position
            confidence = min(1.0, note.confidence + 0.2)  # Boost
        else:
            # Fall back to lowest-fret heuristic
            position = pick_lowest_fret(candidates)
            confidence = note.confidence

        tab_notes.append(TabNote(
            id=str(uuid4()),
            timestamp=note.start_time,
            string=position.string,
            fret=position.fret,
            confidence=confidence,
            confidence_level=get_confidence_level(confidence),
        ))

    return tab_notes
```

---

### 4. Updated Processing (`app/processing.py`) - Sync Point

```python
def process_job(job_id: str, job_storage, results_folder: str):
    job = job_storage.get(job_id)
    try:
        # Stage 1: Extract audio
        update_job(job, "extracting_audio", 0.1)
        audio_path = extract_audio(job.video_path, ...)

        # Stage 2: Analyze audio
        update_job(job, "analyzing_audio", 0.3)
        detected_notes = analyze_pitch(audio_path)

        # Stage 3: Analyze video (NEW)
        update_job(job, "analyzing_video", 0.5)
        timestamps = [n.start_time for n in detected_notes]
        video_observations = analyze_video_at_timestamps(job.video_path, timestamps)
        fretboard = detect_fretboard_from_video(job.video_path)  # First frame

        # Stage 4: Fuse signals
        update_job(job, "fusing", 0.8)
        tab_notes = fuse_audio_video(
            detected_notes,
            video_observations,
            fretboard,
            job.capo_fret
        )

        # Stage 5: Save result
        update_job(job, "complete", 1.0)
        save_result(job, tab_notes, results_folder)

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
```

---

## Dependencies

**requirements.txt additions:**
```
mediapipe>=0.10.0
opencv-python>=4.8.0
```

---

## Testing Strategy

### Agent 2 Tests (`tests/test_video_pipeline.py`)

```python
def test_extract_frame_at_timestamp():
    """Frame extraction returns valid image."""

def test_detect_hand_landmarks_with_hand():
    """MediaPipe detects hand in test image."""

def test_detect_hand_landmarks_no_hand():
    """Returns None when no hand present."""

def test_analyze_video_at_timestamps():
    """Batch analysis returns observations dict."""
```

### Agent 3 Tests (`tests/test_fretboard_detection.py`)

```python
def test_detect_fretboard_with_guitar():
    """Detects fretboard in test image."""

def test_detect_fretboard_no_guitar():
    """Returns None when no fretboard visible."""

def test_map_finger_on_fretboard():
    """Maps finger position to correct fret/string."""

def test_map_finger_off_fretboard():
    """Returns None when finger not on fretboard."""
```

### Integration Tests (Sync Point)

```python
def test_fuse_audio_video_agreement():
    """High confidence when audio and video agree."""

def test_fuse_audio_video_disagreement():
    """Falls back to audio when video disagrees."""

def test_fuse_audio_only_no_video():
    """Works without video observations."""
```

---

## Test Fixtures Needed

| Fixture | Description | Agent |
|---------|-------------|-------|
| `test_hand.jpg` | Image with hand visible | Agent 2 |
| `test_no_hand.jpg` | Image without hand | Agent 2 |
| `test_fretboard.jpg` | Image with guitar fretboard | Agent 3 |
| `test_no_fretboard.jpg` | Image without guitar | Agent 3 |
| `test_guitar_playing.mp4` | Short clip of guitar playing | Sync point |

---

## Implementation Order

### Parallel Phase (Agents 2 & 3)

**Agent 2:**
1. Add mediapipe, opencv-python to requirements.txt
2. Implement `video_pipeline.py` with frame extraction
3. Add MediaPipe hand detection
4. Write tests with fixture images
5. Verify standalone functionality

**Agent 3:**
1. Implement `fretboard_detection.py` with edge detection
2. Add Hough line transform for fret detection
3. Implement coordinate mapping
4. Write tests with fixture images
5. Verify standalone functionality

### Sync Phase

1. Merge Agent 2 & 3 work
2. Update `fusion_engine.py` with `fuse_audio_video()`
3. Update `processing.py` to call video pipeline
4. Integration testing
5. E2E test with real guitar video

---

## Assumptions & Constraints

- Guitar neck roughly horizontal in frame
- Right-handed playing (fretting hand on left side of frame)
- Single guitar visible
- Reasonable lighting (not backlit)
- Camera angle shows fretboard clearly
- Frame rate >= 15 fps for note alignment

---

## Error Handling

| Error | Handling |
|-------|----------|
| No hand detected | Continue with audio-only |
| No fretboard detected | Continue with audio-only |
| OpenCV/MediaPipe failure | Log warning, continue with audio-only |
| Video file corrupted | Fail job with error message |

The video pipeline is designed to be optional - if it fails, the system gracefully falls back to Phase 1 audio-only behavior.
