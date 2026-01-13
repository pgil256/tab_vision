# TabVision Improvements Design

## Problem Statement

Three issues identified:
1. **Accuracy** - Transcription produces wrong frets and wrong strings
2. **UI** - Horizontal scroller is awkward; vertical tab layout preferred
3. **Recording** - No way to record video in-app; must upload files

## Design

### Phase 1: Diagnostics System

Add pipeline diagnostics to understand accuracy failures before attempting fixes.

**New file: `tabvision-server/app/diagnostics.py`**

```python
class PipelineDiagnostics:
    def __init__(self):
        self.data = {
            "audio_pipeline": {},
            "video_pipeline": {},
            "fusion": {}
        }
```

**Collected data:**
```json
{
  "audio_pipeline": {
    "notes_detected": 45,
    "sample_notes": ["...first 5 with timestamps, MIDI, confidence..."]
  },
  "video_pipeline": {
    "fretboard_detected": true,
    "fretboard_geometry": {"...or null..."},
    "frames_analyzed": 45,
    "frames_with_hands": 12,
    "hand_detection_rate": 0.27
  },
  "fusion": {
    "video_matched": 8,
    "audio_only_fallback": 37,
    "open_string_detected": 3
  }
}
```

**Backend changes:**
- `processing.py` creates diagnostics object, passes through pipeline
- Each stage logs relevant metrics
- Saved as `{job_id}_diagnostics.json` alongside result
- New endpoint: `GET /jobs/{id}/diagnostics`

**Frontend changes:**
- Fetch diagnostics after processing completes
- Display summary panel: fretboard detected, hand detection rate, video-matched vs fallback counts

---

### Phase 2: Accuracy Fixes

Scope depends on Phase 1 diagnostics findings. Potential fixes:

- **Fretboard detection failing** → improve edge detection, add fallback heuristics
- **Low hand detection rate** → adjust MediaPipe parameters, handle low-resolution video better
- **Fusion logic issues** → fix candidate matching algorithm, improve open string detection

---

### Phase 3: Vertical Tab UI

**New file: `tabvision-client/src/components/VerticalTabCanvas.tsx`**

**Layout:**
```
┌─────────────────────────────────────────┐
│  Time: 0:05                             │
│  ┌─────────────────────────────────┐    │
│  │ e │ 0 │   │ 2 │   │ 3 │   │     │    │
│  │ B │ 1 │   │ 3 │   │ 0 │   │     │    │
│  │ G │ 0 │   │ 2 │   │ 0 │   │     │    │
│  │ D │ 2 │   │ 0 │   │ 2 │   │     │    │
│  │ A │ 3 │   │   │   │ 3 │   │     │    │
│  │ E │   │   │   │   │   │   │     │    │
│  └─────────────────────────────────┘    │
│  ═══════════════════════════════════    │  ← Playback line (fixed center)
│  ┌─────────────────────────────────┐    │
│  │ e │   │ 5 │   │ 7 │   │   │     │    │
│  │ B │   │ 5 │   │ 8 │   │   │     │    │
│  │ ... upcoming notes ...          │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**Key characteristics:**
- Time axis vertical (top = past, bottom = future)
- Each row is a time slice showing all 6 strings horizontally
- Playback line fixed at vertical center
- Content scrolls upward as time progresses
- String labels (e B G D A E) as fixed header on left

**Rendering:**
- Canvas-based for performance
- Calculate visible time range from scroll offset
- Only render notes within visible range

**Interaction:**
- Click to select notes
- Up/down arrows navigate by time
- Left/right arrows navigate by string
- Number keys to edit fret, delete for X
- Undo/redo support

---

### Phase 4: Recording Feature

**Modified:** `UploadPanel.tsx` becomes tabbed container

**New file:** `tabvision-client/src/components/RecordPanel.tsx`

**UI Layout:**
```
┌─────────────────────────────────────────┐
│  [ Upload File ]  [ Record Video ]      │  ← Tabs
├─────────────────────────────────────────┤
│  Camera: [ Built-in Webcam      ▼]      │
│  ┌─────────────────────────────────┐    │
│  │        Live Preview             │    │
│  │        (webcam feed)            │    │
│  └─────────────────────────────────┘    │
│         [ ● Start Recording ]           │
└─────────────────────────────────────────┘
```

**Recording flow:**
1. User selects camera from dropdown (via `enumerateDevices()`)
2. Preview shows live webcam feed
3. Click "Start Recording" → 3-2-1 countdown overlay
4. Recording indicator appears, button becomes "Stop"
5. User plays guitar
6. Click "Stop" → blob created, auto-submitted to `/jobs`
7. Normal processing flow continues

**State additions to appStore:**
- `recordingState`: 'idle' | 'previewing' | 'countdown' | 'recording'
- `selectedCameraId`: string
- `availableCameras`: MediaDeviceInfo[]

**Technology:** Browser MediaDevices API (getUserMedia, MediaRecorder)

---

## Implementation Order

1. **Phase 1: Diagnostics** - understand what's failing
2. **Phase 2: Accuracy fixes** - targeted fixes based on diagnostics
3. **Phase 3: Vertical Tab UI** - new component
4. **Phase 4: Recording** - camera picker, preview, countdown, record

## Files to Create/Modify

| File | Action |
|------|--------|
| `tabvision-server/app/diagnostics.py` | Create |
| `tabvision-server/app/processing.py` | Modify - add diagnostics collection |
| `tabvision-server/app/routes.py` | Modify - add diagnostics endpoint |
| `tabvision-client/src/components/DiagnosticsPanel.tsx` | Create |
| `tabvision-client/src/components/VerticalTabCanvas.tsx` | Create |
| `tabvision-client/src/components/RecordPanel.tsx` | Create |
| `tabvision-client/src/components/UploadPanel.tsx` | Modify - add tabs |
| `tabvision-client/src/store/appStore.ts` | Modify - add recording state |
| `tabvision-client/src/api/client.ts` | Modify - add diagnostics fetch |
