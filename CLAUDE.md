# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TabVision is a desktop application for automatic guitar tab transcription from video. It uses a multi-modal approach combining audio pitch detection with visual finger tracking to generate accurate tablature with confidence scoring.

## Architecture

```
Electron App (React Frontend)          Flask Backend (Cloud)
├── Video input (upload/webcam)        ├── POST /jobs (upload video)
├── Interactive tab editor             ├── GET /jobs/:id (poll status)
└── Export (text/PDF)                  └── GET /jobs/:id/result
         │                                      │
         │ HTTPS                                │
         └──────────────────────────────────────┘
                                                │
                    Processing Pipeline ────────┘
                    ├── Audio Pipeline (ffmpeg → Basic Pitch → MIDI mapping)
                    ├── Video Pipeline (MediaPipe Hands → fretboard geometry)
                    └── Fusion Engine (combine signals → TabDocument JSON)
```

**Frontend**: `tabvision-client/` - Electron + React 18 + Zustand + Tailwind CSS
**Backend**: `tabvision-server/` - Python Flask + Basic Pitch + MediaPipe + OpenCV

## Development Commands

### Frontend (Electron + React)
```bash
cd tabvision-client
npm install
npm run dev              # Development with hot reload
npm run build            # Build production app
npm install -g electron-builder  # For packaging
```

### Backend (Flask)
```bash
cd tabvision-server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
flask run                # Start dev server
```

### Key Backend Dependencies
```bash
pip install flask basic-pitch mediapipe opencv-python ffmpeg-python numpy
```

## Core Data Structures

**TabDocument** (frontend): Array of `TabNote` objects with `timestamp`, `string` (1-6), `fret` (number or "X"), `confidence` (0.0-1.0), and `confidenceLevel` ("high"/"medium"/"low").

**Job** (backend): Tracks processing state with `status` (pending/processing/completed/failed), `progress` (0.0-1.0), and `current_stage` (uploading/extracting_audio/analyzing_audio/analyzing_video/fusing/complete).

## Processing Pipeline Details

1. **Audio Pipeline**: Extract audio via ffmpeg → Basic Pitch for polyphonic pitch detection → Map MIDI notes to candidate fret/string positions
2. **Video Pipeline**: Extract frames at onset timestamps → MediaPipe for finger landmarks → Detect fretboard geometry (edge detection, Hough transform) → Map fingers to fret/string
3. **Fusion Engine**: Match audio candidates to video observations → Score confidence → Handle open strings (fret 0) and muted notes (X)

## Assumptions & Constraints

- Standard tuning (EADGBE) only
- Clean guitar audio (no backing track/vocals)
- Right-handed playing
- Guitar neck must be visible and roughly centered, horizontal orientation
- Maximum video duration: ~5 minutes
- User specifies capo position if applicable

## Confidence Scoring

- **High (green, >0.8)**: Audio and video agree clearly
- **Medium (yellow, 0.5-0.8)**: Plausible match with some ambiguity
- **Low (red, <0.5)**: Conflict between audio/video or missing data
