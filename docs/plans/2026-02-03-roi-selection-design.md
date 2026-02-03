# ROI Selection Feature Design

**Date:** 2026-02-03
**Status:** Approved

## Overview

Add a region-of-interest (ROI) selection step after video upload. Users draw a bounding box around the fretboard and left hand to focus the analysis on the relevant portion of the video.

## User Flow

```
1. User uploads video (drag-drop or file picker)
   ↓
2. ROI Selection Screen appears
   - Video player with scrubbing controls
   - Hint text: "Draw a box around the fretboard and left hand"
   - User seeks to find good frame
   - User clicks and drags to draw rectangle
   - Hint fades after box is drawn
   - "Process" button (disabled until box drawn)
   - "Back" button to re-upload different video
   ↓
3. User clicks "Process"
   ↓
4. Processing begins with ROI coordinates sent to backend
   ↓
5. Results shown in tab editor
```

**State Transitions:**
- `idle` → (upload) → `roi_selection` → (process) → `processing` → `completed`

## UI Component: ROISelector

```
┌─────────────────────────────────────────────────────────┐
│  ┌─────────────────────────────────────────────────┐    │
│  │                                                 │    │
│  │              VIDEO FRAME                        │    │
│  │                                                 │    │
│  │         ┌───────────────────┐                   │    │
│  │         │   ROI BOX         │ ← cyan border,    │    │
│  │         │   (user drawn)    │   10% white fill  │    │
│  │         └───────────────────┘                   │    │
│  │                                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ◄──●────────────────────────────────────────────► 0:32 │
│     ▲ scrubber (seek to any frame)                      │
│                                                         │
│  "Draw a box around the fretboard and left hand"        │
│     ▲ hint (fades after box drawn)                      │
│                                                         │
│  [ ← Back ]                              [ Process → ]  │
│                                          (disabled      │
│                                           until drawn)  │
└─────────────────────────────────────────────────────────┘
```

**Interaction:**
- Click and drag to draw rectangle
- Click again to redraw (replaces previous box)
- Drawing a box is required before Process button is enabled

**Visual Feedback:**
- Cyan border with 10% white fill
- Hint text fades after box is drawn

## Data Flow

### Frontend State (appStore)

```typescript
// New fields
roi: {
  x1: number;  // 0-1 normalized (left)
  y1: number;  // 0-1 normalized (top)
  x2: number;  // 0-1 normalized (right)
  y2: number;  // 0-1 normalized (bottom)
} | null;

// Updated status type
status: 'idle' | 'roi_selection' | 'uploading' | 'processing' | 'completed' | 'failed';
```

### API Change - POST /jobs

```
// Current
FormData: { video, capo_fret }

// New
FormData: { video, capo_fret, roi_x1, roi_y1, roi_x2, roi_y2 }
```

All ROI values are normalized 0-1 (percentage of video dimensions).

### Backend Job Model

```python
@dataclass
class Job:
    # ... existing fields ...
    roi_x1: float  # 0-1
    roi_y1: float  # 0-1
    roi_x2: float  # 0-1
    roi_y2: float  # 0-1
```

## Backend Processing Changes

### Frame Extraction with ROI

```python
def extract_frame(video_path: str, timestamp: float, roi: dict = None) -> np.ndarray:
    # ... existing frame extraction ...
    frame = cap.read()

    if roi:
        h, w = frame.shape[:2]
        x1 = int(roi['x1'] * w)
        y1 = int(roi['y1'] * h)
        x2 = int(roi['x2'] * w)
        y2 = int(roi['y2'] * h)
        frame = frame[y1:y2, x1:x2]

    return frame
```

### Impact

- MediaPipe hand detection runs on smaller, focused image → faster, more accurate
- Fretboard detection has less noise from background → better line detection
- All coordinates returned are relative to the cropped frame
- No coordinate translation needed - positions are relative to fretboard within ROI

### Unaffected Components

- Audio pipeline (audio extraction and pitch detection unchanged)
- Fusion logic (works with whatever video pipeline returns)
- Tab editor (displays results same as before)

## Validation

### Frontend

- Minimum box size: 10% of video dimensions in each direction
- If box too small, show brief error message and let user redraw
- Box must be fully within video bounds (clamp coordinates to 0-1)

### Backend

- If ROI values missing or invalid, return 400 error
- Validate: 0 ≤ x1 < x2 ≤ 1 and 0 ≤ y1 < y2 ≤ 1

## Files to Modify

| File | Change |
|------|--------|
| `tabvision-client/src/store/appStore.ts` | Add `roi` state, `roi_selection` status |
| `tabvision-client/src/components/ROISelector.tsx` | New component |
| `tabvision-client/src/components/UploadPanel.tsx` | Transition to ROI selection after upload |
| `tabvision-client/src/App.tsx` | Render ROISelector when in `roi_selection` state |
| `tabvision-client/src/api/client.ts` | Add ROI params to uploadVideo |
| `tabvision-server/app/models.py` | Add ROI fields to Job |
| `tabvision-server/app/routes.py` | Extract/validate ROI from request |
| `tabvision-server/app/video_pipeline.py` | Crop frames using ROI |
| `tabvision-server/app/fretboard_detection.py` | Crop frames using ROI |
| `tabvision-server/app/processing.py` | Pass ROI through pipeline |
