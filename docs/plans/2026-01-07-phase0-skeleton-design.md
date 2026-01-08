# Phase 0: Skeleton Design

## Overview

End-to-end hello world: upload a video, get a fake TabDocument response back. Local development only (no deployment).

## Tech Choices

- **Frontend**: Electron Forge + Vite + React 18 + TypeScript
- **Backend**: Flask + Python
- **State Management**: Zustand
- **Job Storage**: In-memory Python dict (no persistence)

## Project Structure

```
tab_vision/
├── tabvision-client/          # Electron + React frontend
│   ├── src/
│   │   ├── main/              # Electron main process
│   │   │   └── index.ts
│   │   ├── preload/           # Preload scripts (IPC bridge)
│   │   │   └── index.ts
│   │   └── renderer/          # React app
│   │       ├── App.tsx
│   │       ├── components/
│   │       ├── types/         # TabDocument, TabNote interfaces
│   │       └── index.tsx
│   ├── package.json
│   └── forge.config.ts
│
├── tabvision-server/          # Flask backend
│   ├── app/
│   │   ├── __init__.py        # Flask app factory
│   │   ├── routes.py          # /jobs endpoints
│   │   ├── models.py          # Job dataclass
│   │   └── storage.py         # In-memory job storage
│   ├── requirements.txt
│   └── run.py                 # Entry point
│
├── CLAUDE.md
└── tabvision_specification.md
```

## Backend API

### POST /jobs
- Accepts: multipart/form-data with video file + capo_fret (int)
- Saves video to uploads/ directory
- Creates job with status "pending", immediately sets to "completed" (fake processing)
- Returns: `{ "job_id": "<uuid>" }`

### GET /jobs/<job_id>
- Returns job status and progress
- Response: `{ "status": "completed", "progress": 1.0, "current_stage": "complete" }`

### GET /jobs/<job_id>/result
- Returns fake TabDocument JSON (only if status == "completed")
- Fake data: 5-10 hardcoded notes at different timestamps

### Configuration
- CORS enabled for localhost
- Uploads stored in `uploads/` folder (gitignored)

## Frontend UI

Three-panel layout:

```
┌─────────────────────────────────────────────────────┐
│  TabVision                              [Settings]  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │     [Upload Video]  or  drag & drop        │   │
│  │     Status: Idle / Uploading / Processing  │   │
│  │     Progress: ████████░░ 80%               │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Placeholder: Tab Editor                    │   │
│  │  (shows fake TabDocument data once ready)   │   │
│  │  e|---0---2---3---|                        │   │
│  │  B|---1---3---0---|                        │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## State (Zustand)

```typescript
interface AppState {
  currentJobId: string | null;
  jobStatus: 'idle' | 'uploading' | 'processing' | 'completed' | 'failed';
  progress: number;
  tabDocument: TabDocument | null;
}
```

## Data Flow

1. User clicks "Upload Video" or drags file onto drop zone
2. Frontend validates file type (MP4, MOV) and shows "Uploading..."
3. POST /jobs with FormData (video file + capo_fret=0)
4. Backend saves file, creates job, returns job_id
5. Frontend starts polling GET /jobs/:id every 1 second
6. When status == "completed", fetch GET /jobs/:id/result
7. Store TabDocument in Zustand, render in placeholder editor

## Fake TabDocument

```json
{
  "id": "abc-123",
  "createdAt": "2024-01-07T12:00:00Z",
  "duration": 30.0,
  "capoFret": 0,
  "tuning": ["E", "A", "D", "G", "B", "E"],
  "notes": [
    { "id": "n1", "timestamp": 0.5, "string": 1, "fret": 0, "confidence": 0.9, "confidenceLevel": "high", "isEdited": false },
    { "id": "n2", "timestamp": 1.0, "string": 2, "fret": 1, "confidence": 0.7, "confidenceLevel": "medium", "isEdited": false }
  ]
}
```

## Dev Workflow

- Terminal 1: `cd tabvision-server && flask run` (port 5000)
- Terminal 2: `cd tabvision-client && npm run dev` (Electron with HMR)

## Deliverable

Upload a video → see fake TabDocument rendered in placeholder editor.
