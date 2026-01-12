# Phase 4: Editor UI Design

**Date:** 2026-01-12
**Status:** Approved
**Author:** Claude + User

## Overview

Build an interactive tab editor with canvas-based rendering, video synchronization, and inline editing. Replaces the basic text-based TabEditor with a professional editing experience.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tab rendering | Canvas | Performance with 1000+ notes, precise click targets, smooth scrolling |
| Layout | Stacked (video top, tab below) | Maximizes horizontal space for tab, works on all screen sizes |
| Scroll behavior | Auto-scroll with manual override | Intuitive playback following, user can explore freely |
| Edit interaction | Click + type | Fastest workflow for corrections |

---

## Component Architecture

```
App.tsx
├── Header
├── Main
│   ├── VideoPlayer (new)
│   │   ├── HTML5 <video> element
│   │   ├── Custom controls (play/pause, seek bar, time display)
│   │   └── Exposes: currentTime, duration, seek(), play(), pause()
│   │
│   ├── TabCanvas (new - replaces TabEditor)
│   │   ├── <canvas> element for tab rendering
│   │   ├── Playback position indicator (vertical line)
│   │   ├── Note selection highlight
│   │   └── Handles: click-to-select, keyboard input, scroll
│   │
│   └── TabToolbar (new)
│       ├── Undo/Redo buttons
│       ├── "Follow playback" toggle
│       └── Export button (placeholder)
│
└── Footer
```

---

## State Management

### New Zustand Store Fields

```typescript
interface AppState {
  // Existing fields...

  // Playback state
  currentTime: number;              // Video playback position in seconds
  duration: number;                 // Video duration in seconds
  isPlaying: boolean;               // Video play/pause state

  // Editor state
  selectedNoteId: string | null;    // Currently selected note
  isFollowingPlayback: boolean;     // Auto-scroll enabled
  pendingFretInput: string;         // Digits being typed (e.g., "1" before "12")

  // Edit history
  editHistory: EditAction[];        // Stack of edits for undo/redo
  editHistoryIndex: number;         // Current position (-1 = no edits)
}

interface EditAction {
  noteId: string;
  previousFret: number | "X";
  newFret: number | "X";
}
```

### New Actions

```typescript
// Playback
setCurrentTime: (time: number) => void;
setDuration: (duration: number) => void;
setIsPlaying: (playing: boolean) => void;

// Selection
selectNote: (noteId: string | null) => void;
selectAdjacentNote: (direction: 'left' | 'right' | 'up' | 'down') => void;

// Editing
updateNoteFret: (noteId: string, newFret: number | "X") => void;
setPendingFretInput: (input: string) => void;
commitPendingEdit: () => void;

// Undo/Redo
undo: () => void;
redo: () => void;
canUndo: () => boolean;
canRedo: () => boolean;

// Auto-scroll
setFollowingPlayback: (following: boolean) => void;
```

---

## Canvas Rendering

### Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  0.0s      0.5s      1.0s      1.5s      2.0s      2.5s      3.0s  ...  │  ← Time axis (20px height)
├──────────────────────────────────────────────────────────────────────────┤
│ e│  ──────5───────────3─────────────────────12──────────────────────────│  ← String 1 (24px height)
│ B│  ────────────5───────────────8───────────────────────────────────────│  ← String 2
│ G│  ──────────────────────5─────────────────────────2───────────────────│  ← String 3
│ D│  ────────────────────────────────────0───────────────────────────────│  ← String 4
│ A│  ──0─────────────────────────────────────────────────────────────────│  ← String 5
│ E│  ────────────────────────────────────────────────────────────────────│  ← String 6
│                           ▲                                              │
│                           │ Playback indicator (red line)                │
└──────────────────────────────────────────────────────────────────────────┘
```

### Dimensions

| Element | Size |
|---------|------|
| Canvas height | 200px (fixed) |
| Canvas width | `duration × pixelsPerSecond` |
| Pixels per second | 50px (configurable) |
| String spacing | 24px |
| Time axis height | 20px |
| String label width | 20px |
| Note hitbox | 20px × 20px |

### Rendering Layers (draw order)

1. **Background**: Dark gray (#1f2937)
2. **String lines**: Horizontal gray lines
3. **Time markers**: Vertical tick marks every second, labels every 5 seconds
4. **Notes**: Fret numbers with confidence-based colors
5. **Selection highlight**: Blue rounded rect behind selected note
6. **Playback indicator**: Red vertical line at current time
7. **Edit indicator**: Small dot on edited notes

### Note Colors

| Confidence | Fill Color | Text Color |
|------------|------------|------------|
| High (>0.8) | #22c55e (green-500) | white |
| Medium (0.5-0.8) | #eab308 (yellow-500) | black |
| Low (<0.5) | #ef4444 (red-500) | white |
| Selected | #3b82f6 (blue-500) border | unchanged |
| Edited | Small white dot indicator | unchanged |

---

## Video Sync

### Video → Tab

```typescript
// VideoPlayer component
<video
  ref={videoRef}
  onTimeUpdate={() => setCurrentTime(videoRef.current.currentTime)}
  onLoadedMetadata={() => setDuration(videoRef.current.duration)}
  onPlay={() => setIsPlaying(true)}
  onPause={() => setIsPlaying(false)}
/>
```

### Tab → Video

```typescript
// TabCanvas click handler
const handleNoteClick = (note: TabNote) => {
  selectNote(note.id);
  videoRef.current.currentTime = note.timestamp;
};
```

### Auto-scroll Logic

```typescript
// In TabCanvas, on currentTime change
useEffect(() => {
  if (!isFollowingPlayback) return;

  const indicatorX = currentTime * pixelsPerSecond;
  const viewportLeft = scrollContainer.scrollLeft;
  const viewportWidth = scrollContainer.clientWidth;
  const targetX = indicatorX - viewportWidth * 0.2; // Keep indicator 20% from left

  scrollContainer.scrollTo({ left: Math.max(0, targetX), behavior: 'smooth' });
}, [currentTime, isFollowingPlayback]);

// Disable auto-follow on manual scroll
const handleScroll = () => {
  if (isUserScrolling) {
    setFollowingPlayback(false);
  }
};
```

---

## Editing

### Click + Type Workflow

1. **Click note** → `selectNote(note.id)`
2. **Type digit** → `setPendingFretInput(pending + digit)`
3. **After 500ms timeout** → `commitPendingEdit()`
4. **Or press Tab/Enter/Arrow** → `commitPendingEdit()` then navigate

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| 0-9 | Append digit to pending fret input |
| Enter | Commit edit, deselect |
| Escape | Cancel edit, deselect |
| Tab | Commit edit, select next note |
| Shift+Tab | Commit edit, select previous note |
| Arrow Left | Select previous note (by timestamp) |
| Arrow Right | Select next note (by timestamp) |
| Arrow Up | Select note on string above (same time) |
| Arrow Down | Select note on string below (same time) |
| Delete/Backspace | Set fret to "X" |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |

### Undo/Redo Implementation

```typescript
const updateNoteFret = (noteId: string, newFret: number | "X") => {
  const note = tabDocument.notes.find(n => n.id === noteId);
  if (!note) return;

  const previousFret = note.fret;

  // Update note
  note.fret = newFret;
  note.isEdited = true;
  note.originalFret = note.originalFret ?? previousFret;

  // Add to history (truncate any redo stack)
  const newHistory = editHistory.slice(0, editHistoryIndex + 1);
  newHistory.push({ noteId, previousFret, newFret });

  set({
    editHistory: newHistory,
    editHistoryIndex: newHistory.length - 1,
  });
};

const undo = () => {
  if (editHistoryIndex < 0) return;

  const action = editHistory[editHistoryIndex];
  const note = tabDocument.notes.find(n => n.id === action.noteId);
  if (note) {
    note.fret = action.previousFret;
    // Reset isEdited if back to original
    if (note.fret === note.originalFret) {
      note.isEdited = false;
    }
  }

  set({ editHistoryIndex: editHistoryIndex - 1 });
};

const redo = () => {
  if (editHistoryIndex >= editHistory.length - 1) return;

  const action = editHistory[editHistoryIndex + 1];
  const note = tabDocument.notes.find(n => n.id === action.noteId);
  if (note) {
    note.fret = action.newFret;
    note.isEdited = true;
  }

  set({ editHistoryIndex: editHistoryIndex + 1 });
};
```

---

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `src/components/VideoPlayer.tsx` | HTML5 video with custom controls |
| `src/components/TabCanvas.tsx` | Canvas rendering and interaction |
| `src/components/TabToolbar.tsx` | Undo/redo buttons, follow toggle |
| `src/hooks/useTabRenderer.ts` | Canvas drawing logic |
| `src/hooks/useNoteInteraction.ts` | Click detection, keyboard handling |

### Modified Files

| File | Changes |
|------|---------|
| `src/store/appStore.ts` | Add playback, selection, edit history state |
| `src/App.tsx` | New stacked layout with video + canvas |

### Deleted Files

| File | Reason |
|------|--------|
| `src/components/TabEditor.tsx` | Replaced by TabCanvas |

---

## Implementation Order

1. **Store updates** - Add new state fields and actions
2. **VideoPlayer** - Basic video element with time sync
3. **TabCanvas static** - Render notes without interaction
4. **Playback indicator** - Red line synced to video time
5. **Auto-scroll** - Follow playback with manual override
6. **Click-to-select** - Note selection and video seek
7. **Keyboard editing** - Type to change fret values
8. **Undo/redo** - Edit history stack
9. **TabToolbar** - UI controls for undo/redo/follow

---

## Testing Plan

### Manual Testing

- [ ] Upload sample video, verify notes render correctly
- [ ] Play video, verify playback indicator moves smoothly
- [ ] Verify auto-scroll keeps indicator visible
- [ ] Manual scroll disables auto-follow
- [ ] Click "Follow playback" re-enables auto-scroll
- [ ] Click note seeks video to that timestamp
- [ ] Click note selects it (blue highlight)
- [ ] Type digits to change fret value
- [ ] Arrow keys navigate between notes
- [ ] Ctrl+Z undoes edit
- [ ] Ctrl+Shift+Z redoes edit
- [ ] Edited notes show indicator
- [ ] Performance acceptable with 1000+ notes

### Edge Cases

- [ ] Very long video (5 minutes) - canvas width handles correctly
- [ ] Dense notes (many per second) - click targets don't overlap badly
- [ ] No notes in view - empty state handled
- [ ] Video without audio track - graceful handling

---

## Future Considerations (Not in Phase 4)

- Export to text/PDF (Phase 5)
- Zoom in/out on tab view
- Multiple selection for batch editing
- Copy/paste notes
- Add/delete notes manually
