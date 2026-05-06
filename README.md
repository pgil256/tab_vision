# TabVision

**Automatic guitar tab transcription from video.**

TabVision analyzes video recordings of guitar playing and generates tablature by fusing audio pitch detection with visual finger tracking. The multi-modal approach resolves a limitation audio-only tools share: the same pitch can live at several positions on the fretboard, and only the video knows which one you actually played.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3-000000?logo=flask&logoColor=white)
![Electron](https://img.shields.io/badge/Electron-28-47848F?logo=electron&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-0097A7)
![Basic Pitch](https://img.shields.io/badge/Basic%20Pitch-Spotify-1DB954?logo=spotify&logoColor=white)

## How it works

```
Electron Desktop App (React)              Flask Backend (cloud)
├── Upload or webcam capture              ├── POST /jobs           (upload video)
├── Interactive tab editor                ├── GET  /jobs/:id       (poll status)
└── Export (text / PDF)                   └── GET  /jobs/:id/result
            │                                        │
            └───────── HTTPS ────────────────────────┘
                                                     │
                      Processing pipeline ───────────┘
                      ├── Audio:  ffmpeg → Basic Pitch → MIDI → pitch events
                      ├── Video:  MediaPipe Hands → fretboard geometry → fret/string
                      └── Fusion: align signals → TabDocument + confidence scores
```

**Output:** an interactive tab editor synced to video playback with per-note confidence highlighting (green / yellow / red). Users can correct notes inline and export to plain-text tab (Ultimate Guitar format) or PDF.

## Tech stack

| Layer       | Tech                                                                 |
|-------------|-----------------------------------------------------------------------|
| Desktop app | Electron 28, React 18, Zustand, Tailwind CSS                         |
| Backend     | Python 3.11, Flask, async job queue                                  |
| Audio       | [Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify's polyphonic pitch detector), ffmpeg |
| Video       | MediaPipe Hands, OpenCV                                              |
| Fusion      | Custom scoring — combines audio pitch candidates with hand-position evidence |

## Getting started

### Backend (Flask)

```bash
cd tabvision-server
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python run.py              # dev server on :5000
pytest tests/ -v           # run tests
```

Key dependencies: `flask`, `basic-pitch`, `mediapipe`, `opencv-python`, `ffmpeg-python`, `numpy`.

### Frontend (Electron + React)

```bash
cd tabvision-client
npm install
npm run dev                # hot-reload development
npm run build              # production build
```

To package a distributable:

```bash
npm install -g electron-builder
npm run dist
```

## Core data model

**`TabDocument`** (frontend): an array of `TabNote` objects —
```ts
interface TabNote {
  timestamp: number;            // seconds into the video
  string: 1 | 2 | 3 | 4 | 5 | 6;
  fret: number | "X";
  confidence: number;           // 0.0 – 1.0
  confidenceLevel: "high" | "medium" | "low";
}
```

**`Job`** (backend): tracks processing state —
```python
status: "pending" | "processing" | "completed" | "failed"
progress: float                  # 0.0 – 1.0
current_stage: "uploading" | "extracting_audio" | "analyzing_audio" |
               "analyzing_video" | "fusing" | "complete"
```

## Assumptions & constraints

- Standard tuning (EADGBE) only
- Video ≤ ~5 minutes per job
- Guitar neck must be visible and roughly centered, horizontal orientation
- Webcam capture works but file upload (MP4 / MOV) gives better results

## Status

End-to-end pipeline (audio → video → fusion) and the editor UI are working. Webcam capture, capo input, and text/PDF export are wired in via the desktop app. First-time users see a welcome screen with camera-positioning tips, accessible later via the **Tips** button in the header.

Remaining polish (app icon, demo video, performance pass) is tracked in [`tabvision_specification.md`](./tabvision_specification.md) under Phase 6.
