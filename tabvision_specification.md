# TabVision

**Automatic Guitar Tab Transcription from Video**

A desktop application that analyzes video recordings of guitar playing and generates accurate tablature by combining audio pitch detection with visual finger tracking.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Features](#core-features)
3. [System Architecture](#system-architecture)
4. [Tech Stack](#tech-stack)
5. [Data Models](#data-models)
6. [Phased Build Plan](#phased-build-plan)
7. [Technical Risks & Mitigations](#technical-risks--mitigations)

---

## Overview

### The Problem

Writing guitar tabs manually is tedious. Existing transcription tools either rely solely on audio (which can't distinguish between the same note played at different fret positions) or require expensive professional software.

### The Solution

TabVision uses a multi-modal approach:
- **Audio analysis** detects which pitches are being played and when
- **Video analysis** confirms which fret/string position was actually used
- **Fusion engine** combines both signals for accurate transcription with confidence scoring

### Target User

Guitarists who want to transcribe their own playing quickly and accurately.

---

## Core Features

### Input
- Upload video files (MP4, MOV)
- In-app webcam recording
- Maximum duration: ~5 minutes
- Requirements: Guitar neck visible and roughly centered in frame, horizontal orientation

### Processing
- Cloud-based async processing
- Progress tracking with status updates
- Stages: audio extraction → pitch detection → video analysis → fusion

### Output
- Interactive tab editor synced to video playback
- Confidence highlighting (green/yellow/red)
- Direct fret number editing for corrections
- Export to plain text (Ultimate Guitar format) and PDF

### Assumptions
- Standard tuning (EADGBE)
- Clean guitar audio (no backing track/vocals)
- Right-handed playing (left-handed support planned for later)
- User specifies capo position if applicable

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ELECTRON APP                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         React Frontend                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │   │
│  │  │ Video Input  │  │ Tab Editor   │  │ Export Panel             │   │   │
│  │  │ - Webcam     │  │ - Synced     │  │ - Plain text             │   │   │
│  │  │ - Upload     │  │   playback   │  │ - PDF                    │   │   │
│  │  │ - Preview    │  │ - Confidence │  │                          │   │   │
│  │  │              │  │   colors     │  │                          │   │   │
│  │  │              │  │ - Direct     │  │                          │   │   │
│  │  │              │  │   editing    │  │                          │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                         Electron Main Process                               │
│                         - File system access                                │
│                         - Video encoding (ffmpeg)                           │
│                         - API communication                                 │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ HTTPS
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLOUD BACKEND (Railway/Fly.io)                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Flask API                                    │   │
│  │  POST /jobs         - Upload video, create job                       │   │
│  │  GET  /jobs/:id     - Poll job status                                │   │
│  │  GET  /jobs/:id/result - Download completed tab data                 │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│  ┌──────────────────────────────────▼──────────────────────────────────┐   │
│  │                      Processing Pipeline                             │   │
│  │                                                                      │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │  │ Video/Audio │    │ Audio       │    │ Video                   │  │   │
│  │  │ Splitter    │───▶│ Pipeline    │    │ Pipeline                │  │   │
│  │  │ (ffmpeg)    │    │             │    │                         │  │   │
│  │  └─────────────┘    │ Basic Pitch │    │ MediaPipe Hands         │  │   │
│  │        │            │ ──────────▶ │    │ ──────────────────────▶ │  │   │
│  │        │            │ Pitch/onset │    │ Finger landmarks        │  │   │
│  │        │            │ detection   │    │                         │  │   │
│  │        │            └──────┬──────┘    │ Fretboard detection     │  │   │
│  │        │                   │           │ ──────────────────────▶ │  │   │
│  │        └───────────────────┼──────────▶│ Fret geometry mapping   │  │   │
│  │                            │           └───────────┬─────────────┘  │   │
│  │                            │                       │                │   │
│  │                            ▼                       ▼                │   │
│  │                     ┌──────────────────────────────────────┐        │   │
│  │                     │         Fusion Engine                │        │   │
│  │                     │  - Match pitches to fret positions   │        │   │
│  │                     │  - Resolve ambiguities               │        │   │
│  │                     │  - Calculate confidence scores       │        │   │
│  │                     │  - Handle open strings (fret 0)      │        │   │
│  │                     │  - Detect muted notes (X)            │        │   │
│  │                     └──────────────────┬───────────────────┘        │   │
│  │                                        │                            │   │
│  │                                        ▼                            │   │
│  │                              TabDocument JSON                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Storage                                                             │   │
│  │  - Job queue (Redis or SQLite for MVP)                               │   │
│  │  - Video file storage (local disk or S3)                             │   │
│  │  - Results cache                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Details

#### Audio Pipeline
1. Extract audio track from video using ffmpeg
2. Run through Basic Pitch (Spotify's polyphonic pitch detection)
3. Get list of (timestamp, MIDI note, confidence) tuples
4. Map each MIDI note to candidate fret/string positions

#### Video Pipeline
1. Extract frames at audio onset timestamps (not every frame)
2. Run MediaPipe Hands to get finger landmark positions
3. Detect fretboard geometry (one-time at video start, with minor tracking corrections):
   - Edge detection to find neck outline
   - Fret wire detection (vertical lines)
   - Perspective correction to normalize coordinate space
4. Map finger landmarks to fret/string positions using detected geometry

#### Fusion Engine
1. For each detected pitch from audio:
   - Get candidate fret/string positions (same note can be played multiple places)
   - Get observed finger position from video at that timestamp
   - Match audio candidate to video observation
2. Handle edge cases:
   - Open strings: pitch detected but no finger → fret 0
   - Muted notes: finger detected but no pitch → mark as X
3. Calculate confidence:
   - High (green): audio and video agree clearly
   - Medium (yellow): plausible but some ambiguity
   - Low (red): conflict between audio/video or missing data

---

## Tech Stack

### Frontend (Electron + React)

| Purpose | Library | Notes |
|---------|---------|-------|
| Framework | React 18 | Industry standard, large ecosystem |
| State management | Zustand | Simpler than Redux, good for complex state |
| Video player | Custom HTML5 video | Need fine-grained control for timestamp syncing |
| Tab rendering | Canvas or SVG | Pixel-level control for click targets |
| Styling | Tailwind CSS | Fast iteration |
| PDF export | jsPDF or pdfmake | Client-side generation |
| Webcam | navigator.mediaDevices | Built into Electron |
| Video encoding | ffmpeg-static | Bundled with Electron |

### Backend (Python + Flask)

| Purpose | Library | Notes |
|---------|---------|-------|
| Framework | Flask | Simple, lightweight |
| Async jobs | RQ (Redis Queue) or Celery | Background processing |
| Audio ML | basic-pitch | Spotify's polyphonic pitch detection |
| Video ML | mediapipe | Google's hand tracking |
| Video processing | opencv-python | Frame extraction, image processing |
| Audio extraction | ffmpeg-python or pydub | Split audio from video |
| Fretboard geometry | numpy, opencv | Edge detection, perspective transforms |
| Storage | SQLite (MVP) → PostgreSQL | Job tracking |
| File storage | Local disk (MVP) → S3/R2 | Video uploads |

### Infrastructure

| Purpose | Choice | Notes |
|---------|--------|-------|
| Hosting | Railway or Fly.io | Simple deployment, reasonable pricing |
| File storage (scale) | Cloudflare R2 or AWS S3 | When needed |
| Domain | Any registrar | Namecheap, Cloudflare, etc. |

---

## Data Models

### TabDocument

The core data structure returned by the backend and manipulated by the frontend editor.

```typescript
interface TabDocument {
  id: string;
  createdAt: string;              // ISO timestamp
  duration: number;               // video duration in seconds
  capoFret: number;               // 0 = no capo
  tuning: string[];               // ["E", "A", "D", "G", "B", "E"] for standard
  
  notes: TabNote[];
}

interface TabNote {
  id: string;
  timestamp: number;              // seconds from video start
  
  // What was detected
  string: 1 | 2 | 3 | 4 | 5 | 6;  // 1 = high E, 6 = low E
  fret: number | "X";             // 0 = open, "X" = muted
  
  // Confidence scoring
  confidence: number;             // 0.0 - 1.0
  confidenceLevel: "high" | "medium" | "low";  // >0.8, 0.5-0.8, <0.5
  
  // Editor state
  isEdited: boolean;
  originalFret?: number | "X";    // preserve original if user edits
  
  // Debug data (optional)
  detectedPitch?: number;         // Hz
  detectedMidiNote?: number;      // MIDI note number
}
```

### Job (Backend)

```python
@dataclass
class Job:
    id: str                       # UUID
    status: str                   # pending | processing | completed | failed
    created_at: datetime
    updated_at: datetime
    
    # Input
    video_path: str               # uploaded file location
    capo_fret: int                # 0 = no capo
    
    # Progress tracking
    progress: float               # 0.0 - 1.0
    current_stage: str            # uploading | extracting_audio | 
                                  # analyzing_audio | analyzing_video | 
                                  # fusing | complete
    
    # Output
    result_path: str | None       # TabDocument JSON path when complete
    error_message: str | None     # if failed
```

### API Endpoints

```
POST /jobs
  Body: multipart/form-data with video file + capo_fret
  Returns: { job_id: string }

GET /jobs/:id
  Returns: { 
    status: string, 
    progress: float, 
    current_stage: string,
    error_message?: string 
  }

GET /jobs/:id/result
  Returns: TabDocument JSON (only when status == "completed")
```

---

## Phased Build Plan

### Phase 0: Skeleton (Week 1)

**Goal**: End-to-end hello world—upload a video, get a dummy response back.

- [ ] Electron + React boilerplate with hot reload working
- [ ] Basic UI: upload button, status display, placeholder editor area
- [ ] Flask backend with `/jobs` endpoints (returns fake data, no real processing)
- [ ] File upload flow working
- [ ] Deploy backend to Railway
- [ ] Verify Electron app can communicate with deployed backend

**Deliverable**: Upload a video → see fake TabDocument rendered.

---

### Phase 1: Audio Pipeline (Weeks 2-3)

**Goal**: Given a video, extract pitches and timestamps.

- [ ] ffmpeg integration: extract audio track from uploaded video
- [ ] Basic Pitch integration: process audio, get pitch/onset data
- [ ] Build MIDI-to-guitar mapping: for each detected note, list candidate fret/string positions
- [ ] Implement "best guess" fingering heuristic (prefer lower positions, common chord shapes)
- [ ] Output preliminary TabDocument based on audio alone
- [ ] Display results in frontend (read-only, no editing yet)

**Deliverable**: Upload video → see tab output based purely on audio analysis.

---

### Phase 2: Video Pipeline (Weeks 4-6)

**Goal**: Detect fretboard geometry and finger positions.

- [ ] Frame extraction at configurable intervals (start with every onset timestamp)
- [ ] MediaPipe Hands integration: get 21 finger landmarks per frame
- [ ] Fretboard detection algorithm:
  - [ ] Canny edge detection to find neck edges
  - [ ] Hough line transform to detect fret wires
  - [ ] Perspective correction to normalize to rectangular space
  - [ ] Store fret positions as percentage along neck length
- [ ] Initial detection on first frame, lightweight tracking for subsequent frames
- [ ] Map finger landmarks (specifically fingertips) to fret number using detected geometry
- [ ] Estimate string position from fingertip Y-coordinate relative to neck width

**Deliverable**: Given a video frame, output "finger detected at fret N, string M area" with reasonable accuracy.

---

### Phase 3: Fusion (Weeks 7-8)

**Goal**: Combine audio and video data for accurate transcription.

- [ ] Timestamp synchronization between audio onsets and video frames
- [ ] For each audio-detected pitch:
  - [ ] Generate candidate fret/string positions
  - [ ] Look up finger position from corresponding video frame
  - [ ] Match candidates to observation
  - [ ] Score confidence based on agreement
- [ ] Open string handling:
  - [ ] Pitch detected at timestamp
  - [ ] No finger in fret range for that pitch
  - [ ] → Output fret 0 on appropriate string
- [ ] Muted note detection:
  - [ ] Finger clearly on fretboard
  - [ ] No corresponding pitch detected
  - [ ] → Output "X" on detected string
- [ ] Confidence calculation:
  - [ ] High (>0.8): audio pitch matches video finger position cleanly
  - [ ] Medium (0.5-0.8): plausible match with some ambiguity
  - [ ] Low (<0.5): conflict or missing data

**Deliverable**: Accurate tab output with confidence scores for each note.

---

### Phase 4: Editor UI (Weeks 9-10)

**Goal**: Interactive tab editor with video synchronization.

- [ ] Tab rendering component:
  - [ ] Standard 6-line tab display (horizontal scrolling for long pieces)
  - [ ] Color-coded notes based on confidence level
  - [ ] Click targets on each note position
- [ ] Video player integration:
  - [ ] Click any note → video seeks to that timestamp
  - [ ] Scrub/play video → current position highlighted in tab view
  - [ ] Visual indicator (vertical line) showing playback position
- [ ] Editing functionality:
  - [ ] Click a note to select it
  - [ ] Type fret number to replace value
  - [ ] Tab/arrow keys to navigate between notes
  - [ ] Track edited notes (visual indicator, preserve original value)
- [ ] Undo/redo stack

**Deliverable**: Full editing workflow with video sync.

---

### Phase 5: Recording & Export (Weeks 11-12)

**Goal**: Complete feature set for v1.

- [ ] Webcam recording:
  - [ ] Device selection dropdown
  - [ ] Live preview stream
  - [ ] Record/stop controls
  - [ ] Encode recorded stream to MP4
- [ ] Capo input:
  - [ ] Dropdown or number input before processing
  - [ ] Adjust fret calculations accordingly
- [ ] Export - Plain text:
  - [ ] Generate Ultimate Guitar-style format
  - [ ] Copy to clipboard or save as .txt
- [ ] Export - PDF:
  - [ ] Basic formatting with song title, tab content
  - [ ] Save dialog
- [ ] Polish:
  - [ ] Loading states and spinners
  - [ ] Error handling with user-friendly messages
  - [ ] Progress bar during processing (with stage labels)
  - [ ] Settings persistence (last used capo, etc.)

**Deliverable**: Feature-complete MVP.

---

### Phase 6: Polish & Portfolio-Ready (Weeks 13-14)

**Goal**: Make it presentation-worthy.

- [ ] Onboarding/welcome screen explaining how to position camera
- [ ] Sample video included for first-time demo
- [ ] App icon and branding
- [ ] Demo video (screen recording of full workflow)
- [ ] README with:
  - [ ] Project overview and screenshots
  - [ ] Architecture explanation
  - [ ] Technical challenges and how you solved them
  - [ ] Installation instructions
- [ ] Code cleanup:
  - [ ] Consistent formatting
  - [ ] Comments on complex sections
  - [ ] Remove dead code
- [ ] Blog post / write-up about the technical challenges
- [ ] Performance profiling and optimization pass
- [ ] Edge case testing with various guitar types, lighting conditions

**Deliverable**: Portfolio-ready project with documentation.

---

## Technical Risks & Mitigations

| Risk | Severity | Likelihood | Mitigation Strategy |
|------|----------|------------|---------------------|
| **Fretboard detection fails on varied lighting/angles** | High | Medium | Require consistent camera setup for MVP. Add "draw box around fretboard" fallback if automatic detection fails. Consider calibration step. |
| **Basic Pitch struggles with guitar timbre** | Medium | Low | Test early with real guitar recordings. Preprocess audio (noise gate, EQ). Evaluate alternatives (Omnizart, CREPE) if needed. |
| **MediaPipe loses tracking when fingers overlap** | Medium | Medium | Use audio as primary source when video is ambiguous. Flag as low confidence rather than guessing. |
| **Video/audio sync drift over long recordings** | Medium | Low | Use video's embedded audio track, not separate recording. Sync to detected onsets, not wall clock time. Re-sync periodically. |
| **Processing time too long for good UX** | Medium | Medium | Process only frames near audio onsets, not every frame. Downsample video resolution. Show granular progress updates. |
| **Large video uploads fail or timeout** | Low | Medium | Chunked upload with resume capability. Client-side compression before upload. Enforce 5-minute limit strictly. |
| **Electron app bundle too large** | Low | High | Accept ~150-200MB as reasonable tradeoff. Optimize with electron-builder if needed. |
| **Polyphonic passages (fast arpeggios) overwhelm system** | Medium | Medium | Increase frame analysis rate for dense passages. Accept lower confidence on very fast playing. |

---

## Future Enhancements (Post-MVP)

- Left-handed player support (video mirroring)
- Custom tuning support
- Multi-track support (rhythm + lead)
- Playback audio synthesis
- Direct Ultimate Guitar upload integration (if API becomes available)
- Mobile app version
- Batch processing for multiple videos
- Collaborative editing / sharing
- Training custom ML models on user-corrected data

---

## Development Notes

### Local Development Setup

```bash
# Frontend (Electron + React)
cd tabvision-client
npm install
npm run dev

# Backend (Flask)
cd tabvision-server
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
flask run
```

### Key Dependencies to Install

```bash
# Backend
pip install flask basic-pitch mediapipe opencv-python ffmpeg-python numpy

# Frontend
npm install electron react zustand tailwindcss
npm install -g electron-builder  # for packaging
```

### Testing Strategy

- **Unit tests**: Fusion logic, fret mapping calculations
- **Integration tests**: Full pipeline with sample videos
- **Manual testing**: Various guitars, lighting conditions, playing styles
- **Build test dataset**: Record yourself playing known passages, verify output accuracy
