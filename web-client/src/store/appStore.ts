// tabvision-client/src/store/appStore.ts
import { create } from 'zustand';
import { TabDocument, TabNote } from '../types/tab';
import type { AccuracyMode, Instrument, PlayingStyle, Tone, UploadRoi } from '../api/client';
import {
  PersistedSession,
  clearSession,
  loadSession,
  persistSession,
} from '../utils/editPersistence';

type JobStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed';

// Only the fields an edit mutates are snapshotted, so undo/redo restore them
// exactly (including isEdited / originalFret bookkeeping) rather than trying to
// recompute derived flags.
type NoteMutableFields = Pick<TabNote, 'string' | 'fret' | 'isEdited' | 'originalFret'>;

type EditAction =
  | { kind: 'position'; noteId: string; before: NoteMutableFields; after: NoteMutableFields }
  | { kind: 'delete'; note: TabNote; index: number }
  | { kind: 'insert'; note: TabNote; index: number };

// Standard-tuning open-string MIDI, keyed by the client's string number
// (1 = high E … 6 = low E — see tab_events_to_tab_document `6 - string_idx`).
const STRING_OPEN_MIDI: Record<number, number> = { 1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40 };
const MIN_STRING = 1;
const MAX_STRING = 6;
const MAX_FRET = 24;

/**
 * Fret on `toString` that sounds the same pitch as `fret` on `fromString`
 * (standard tuning). A capo shifts every string equally, so it cancels out of
 * the difference and this is capo-independent. Returns `null` when the pitch is
 * not playable on the target string (fret would be < 0 or > MAX_FRET); a muted
 * "X" moves across strings unchanged.
 */
export function pitchPreservingFret(
  fromString: number,
  toString: number,
  fret: number | 'X',
): number | 'X' | null {
  if (fret === 'X') return 'X';
  const next = fret + STRING_OPEN_MIDI[fromString] - STRING_OPEN_MIDI[toString];
  if (next < 0 || next > MAX_FRET) return null;
  return next;
}

interface AppState {
  // Job state
  currentJobId: string | null;
  jobStatus: JobStatus;
  progress: number;
  currentStage: string;
  // Whether the server pipeline runs the video stack for this job; true until
  // the job status says otherwise (v0 always runs video).
  pipelineVideoEnabled: boolean;
  tabDocument: TabDocument | null;
  errorMessage: string | null;
  videoUrl: string | null;
  // B5 — a persisted (edited) session found in localStorage on mount, offered
  // for restore. Null once restored/discarded or when none exists.
  restorable: PersistedSession | null;

  // Playback state
  currentTime: number;
  duration: number;
  isPlaying: boolean;

  // Editor state
  selectedNoteId: string | null;
  isFollowingPlayback: boolean;
  pendingFretInput: string;

  // UI state
  zoomLevel: number;
  capoFretInput: number;
  instrumentInput: Instrument;
  toneInput: Tone;
  styleInput: PlayingStyle;
  accuracyModeInput: AccuracyMode;
  roiEnabled: boolean;
  roiInput: UploadRoi;
  isVideoCollapsed: boolean;
  showShortcutsModal: boolean;
  playbackRate: number;

  // Edit history
  editHistory: EditAction[];
  editHistoryIndex: number;

  // Job actions
  setJobId: (id: string) => void;
  setStatus: (status: JobStatus) => void;
  setProgress: (progress: number, stage: string) => void;
  setPipelineVideoEnabled: (enabled: boolean) => void;
  setTabDocument: (doc: TabDocument) => void;
  setError: (message: string) => void;
  setVideoUrl: (url: string | null) => void;
  reset: () => void;

  // B5 — edit persistence / restore
  loadPersistedSession: () => void;
  restorePersistedSession: () => void;
  discardPersistedSession: () => void;

  // Playback actions
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setIsPlaying: (playing: boolean) => void;

  // Selection actions
  selectNote: (noteId: string | null) => void;
  selectAdjacentNote: (direction: 'left' | 'right' | 'up' | 'down') => void;

  // Editing actions
  updateNoteFret: (noteId: string, newFret: number | "X") => void;
  updateNotePosition: (noteId: string, newString: number, newFret: number | "X") => void;
  moveNoteString: (direction: 'up' | 'down') => void;
  deleteNote: (noteId: string) => void;
  insertNote: (opts: { timestamp: number; string: number; fret?: number | "X" }) => void;
  setPendingFretInput: (input: string) => void;
  commitPendingEdit: () => void;

  // Undo/Redo actions
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;

  // UI actions
  setFollowingPlayback: (following: boolean) => void;
  setZoomLevel: (zoom: number) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  resetZoom: () => void;
  setCapoFretInput: (fret: number) => void;
  setInstrumentInput: (instrument: Instrument) => void;
  setToneInput: (tone: Tone) => void;
  setStyleInput: (style: PlayingStyle) => void;
  setAccuracyModeInput: (mode: AccuracyMode) => void;
  setRoiEnabled: (enabled: boolean) => void;
  setRoiInput: (roi: UploadRoi) => void;
  setVideoCollapsed: (collapsed: boolean) => void;
  toggleVideoCollapsed: () => void;
  setShowShortcutsModal: (show: boolean) => void;
  setPlaybackRate: (rate: number) => void;
}

const initialState = {
  // Job state
  currentJobId: null as string | null,
  jobStatus: 'idle' as JobStatus,
  progress: 0,
  currentStage: '',
  pipelineVideoEnabled: true,
  tabDocument: null as TabDocument | null,
  errorMessage: null as string | null,
  videoUrl: null as string | null,
  restorable: null as PersistedSession | null,

  // Playback state
  currentTime: 0,
  duration: 0,
  isPlaying: false,

  // Editor state
  selectedNoteId: null as string | null,
  isFollowingPlayback: true,
  pendingFretInput: '',

  // UI state
  zoomLevel: 1.0,
  capoFretInput: 0,
  instrumentInput: 'acoustic' as Instrument,
  toneInput: 'clean' as Tone,
  styleInput: 'mixed' as PlayingStyle,
  accuracyModeInput: 'accurate' as AccuracyMode,
  roiEnabled: false,
  roiInput: { x1: 0, y1: 0, x2: 1, y2: 1 } as UploadRoi,
  isVideoCollapsed: false,
  showShortcutsModal: false,
  playbackRate: 1.0,

  // Edit history
  editHistory: [] as EditAction[],
  editHistoryIndex: -1,
};

export const useAppStore = create<AppState>((set, get) => ({
  ...initialState,

  // Job actions
  setJobId: (id) => set({ currentJobId: id }),

  setStatus: (status) => set({ jobStatus: status }),

  setProgress: (progress, stage) => set({ progress, currentStage: stage }),

  setPipelineVideoEnabled: (enabled) => set({ pipelineVideoEnabled: enabled }),

  setTabDocument: (doc) => {
    set({ tabDocument: doc, jobStatus: 'completed', restorable: null });
    persistSession(doc, get().currentJobId); // survive a refresh from the first render
  },

  setError: (message) => set({ errorMessage: message, jobStatus: 'failed' }),

  setVideoUrl: (url) => set({ videoUrl: url }),

  reset: () => {
    clearSession(); // "New transcription" discards the autosaved session
    set(initialState);
  },

  // B5 — offer to restore an autosaved edited session on a fresh mount. Only
  // when nothing is loaded yet, so it never overrides a live job.
  loadPersistedSession: () => {
    if (get().tabDocument) return;
    const session = loadSession();
    if (session) set({ restorable: session });
  },

  restorePersistedSession: () => {
    const session = get().restorable;
    if (!session) return;
    set({
      tabDocument: session.doc,
      currentJobId: session.jobId,
      jobStatus: 'completed',
      restorable: null,
      editHistory: [],
      editHistoryIndex: -1,
    });
  },

  discardPersistedSession: () => {
    clearSession();
    set({ restorable: null });
  },

  // Playback actions
  setCurrentTime: (time) => set({ currentTime: time }),

  // MediaRecorder clips report Infinity until seeked; never let a non-finite
  // duration into the store (it would freeze the tab canvas layout).
  setDuration: (duration) => set({ duration: Number.isFinite(duration) && duration > 0 ? duration : 0 }),

  setIsPlaying: (playing) => set({ isPlaying: playing }),

  // Selection actions
  selectNote: (noteId) => set({ selectedNoteId: noteId, pendingFretInput: '' }),

  selectAdjacentNote: (direction) => {
    const { tabDocument, selectedNoteId } = get();
    if (!tabDocument || !selectedNoteId) return;

    const notes = tabDocument.notes;
    const currentNote = notes.find(n => n.id === selectedNoteId);
    if (!currentNote) return;

    const sortedByTime = [...notes].sort((a, b) => a.timestamp - b.timestamp);
    const currentIndex = sortedByTime.findIndex(n => n.id === selectedNoteId);

    let nextNote: TabNote | undefined;

    if (direction === 'left') {
      nextNote = sortedByTime[currentIndex - 1];
    } else if (direction === 'right') {
      nextNote = sortedByTime[currentIndex + 1];
    } else if (direction === 'up' || direction === 'down') {
      // Find notes at similar timestamp (within 50ms)
      const nearbyNotes = notes.filter(
        n => Math.abs(n.timestamp - currentNote.timestamp) < 0.05 && n.id !== currentNote.id
      );
      if (direction === 'up') {
        // Find note on higher string (lower string number)
        nextNote = nearbyNotes
          .filter(n => n.string < currentNote.string)
          .sort((a, b) => b.string - a.string)[0];
      } else {
        // Find note on lower string (higher string number)
        nextNote = nearbyNotes
          .filter(n => n.string > currentNote.string)
          .sort((a, b) => a.string - b.string)[0];
      }
    }

    if (nextNote) {
      set({ selectedNoteId: nextNote.id, pendingFretInput: '' });
    }
  },

  // Editing actions
  // Fret-only edit: unchanged string. Thin wrapper over updateNotePosition so
  // number entry, mute (X), and commit all funnel through one history path.
  updateNoteFret: (noteId, newFret) => {
    const { tabDocument, updateNotePosition } = get();
    const note = tabDocument?.notes.find(n => n.id === noteId);
    if (!note) return;
    updateNotePosition(noteId, note.string, newFret);
  },

  // The general position edit (B3): change string and/or fret. Records a
  // before/after snapshot so undo/redo restore string, fret and the edited
  // bookkeeping exactly.
  updateNotePosition: (noteId, newString, newFret) => {
    const { tabDocument, editHistory, editHistoryIndex } = get();
    if (!tabDocument) return;

    const noteIndex = tabDocument.notes.findIndex(n => n.id === noteId);
    if (noteIndex === -1) return;

    const note = tabDocument.notes[noteIndex];
    if (note.string === newString && note.fret === newFret) return; // no-op

    const before: NoteMutableFields = {
      string: note.string,
      fret: note.fret,
      isEdited: note.isEdited,
      originalFret: note.originalFret,
    };
    const updated: TabNote = {
      ...note,
      string: newString as TabNote['string'],
      fret: newFret,
      isEdited: true,
      originalFret: note.originalFret ?? note.fret,
    };
    const after: NoteMutableFields = {
      string: updated.string,
      fret: updated.fret,
      isEdited: updated.isEdited,
      originalFret: updated.originalFret,
    };

    const updatedNotes = [...tabDocument.notes];
    updatedNotes[noteIndex] = updated;

    const newHistory = editHistory.slice(0, editHistoryIndex + 1);
    newHistory.push({ kind: 'position', noteId, before, after });

    set({
      tabDocument: { ...tabDocument, notes: updatedNotes },
      editHistory: newHistory,
      editHistoryIndex: newHistory.length - 1,
    });
    persistSession(get().tabDocument!, get().currentJobId);
  },

  // Move the selected note to the adjacent string, keeping its pitch by
  // recomputing the fret (B3 — the fix for the dominant wrong-string error).
  // 'up' is toward string 1 (high E); the move is a no-op when the pitch is
  // unplayable on the target string or the note is at the edge.
  moveNoteString: (direction) => {
    const { tabDocument, selectedNoteId, updateNotePosition } = get();
    if (!tabDocument || !selectedNoteId) return;
    const note = tabDocument.notes.find(n => n.id === selectedNoteId);
    if (!note) return;
    const target = direction === 'up' ? note.string - 1 : note.string + 1;
    if (target < MIN_STRING || target > MAX_STRING) return;
    const nextFret = pitchPreservingFret(note.string, target, note.fret);
    if (nextFret === null) return; // pitch not reachable on the target string
    updateNotePosition(selectedNoteId, target, nextFret);
  },

  // True removal (B3) — distinct from mute (fret = "X").
  deleteNote: (noteId) => {
    const { tabDocument, editHistory, editHistoryIndex, selectedNoteId } = get();
    if (!tabDocument) return;
    const index = tabDocument.notes.findIndex(n => n.id === noteId);
    if (index === -1) return;

    const removed = tabDocument.notes[index];
    const updatedNotes = tabDocument.notes.filter(n => n.id !== noteId);
    const newHistory = editHistory.slice(0, editHistoryIndex + 1);
    newHistory.push({ kind: 'delete', note: removed, index });

    set({
      tabDocument: { ...tabDocument, notes: updatedNotes },
      editHistory: newHistory,
      editHistoryIndex: newHistory.length - 1,
      selectedNoteId: selectedNoteId === noteId ? null : selectedNoteId,
    });
    persistSession(get().tabDocument!, get().currentJobId);
  },

  // Insert a new note (B3), kept in timestamp order, and select it for editing.
  insertNote: ({ timestamp, string, fret = 0 }) => {
    const { tabDocument, editHistory, editHistoryIndex } = get();
    if (!tabDocument) return;

    const note: TabNote = {
      id: `insert-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp,
      string: Math.max(MIN_STRING, Math.min(MAX_STRING, string)) as TabNote['string'],
      fret,
      confidence: 1,
      confidenceLevel: 'high',
      isEdited: true,
    };

    const updatedNotes = [...tabDocument.notes];
    let index = updatedNotes.findIndex(n => n.timestamp > timestamp);
    if (index === -1) index = updatedNotes.length;
    updatedNotes.splice(index, 0, note);

    const newHistory = editHistory.slice(0, editHistoryIndex + 1);
    newHistory.push({ kind: 'insert', note, index });

    set({
      tabDocument: { ...tabDocument, notes: updatedNotes },
      editHistory: newHistory,
      editHistoryIndex: newHistory.length - 1,
      selectedNoteId: note.id,
    });
    persistSession(get().tabDocument!, get().currentJobId);
  },

  setPendingFretInput: (input) => set({ pendingFretInput: input }),

  commitPendingEdit: () => {
    const { selectedNoteId, pendingFretInput, updateNoteFret } = get();
    if (!selectedNoteId || !pendingFretInput) {
      set({ pendingFretInput: '' });
      return;
    }

    const fretValue = parseInt(pendingFretInput, 10);
    if (!isNaN(fretValue) && fretValue >= 0 && fretValue <= 24) {
      updateNoteFret(selectedNoteId, fretValue);
    }
    set({ pendingFretInput: '' });
  },

  // Undo/Redo actions — dispatch on the action kind. position restores a
  // field snapshot; delete/insert are inverses (re-insert at index / remove).
  undo: () => {
    const { tabDocument, editHistory, editHistoryIndex, selectedNoteId } = get();
    if (!tabDocument || editHistoryIndex < 0) return;

    const action = editHistory[editHistoryIndex];
    let notes = tabDocument.notes;
    let selected = selectedNoteId;

    if (action.kind === 'position') {
      const i = notes.findIndex(n => n.id === action.noteId);
      if (i === -1) return;
      notes = [...notes];
      notes[i] = { ...notes[i], ...action.before };
    } else if (action.kind === 'delete') {
      notes = [...notes];
      notes.splice(Math.min(action.index, notes.length), 0, action.note);
      selected = action.note.id;
    } else {
      // insert → remove
      notes = notes.filter(n => n.id !== action.note.id);
      if (selected === action.note.id) selected = null;
    }

    set({
      tabDocument: { ...tabDocument, notes },
      editHistoryIndex: editHistoryIndex - 1,
      selectedNoteId: selected,
    });
    persistSession(get().tabDocument!, get().currentJobId);
  },

  redo: () => {
    const { tabDocument, editHistory, editHistoryIndex, selectedNoteId } = get();
    if (!tabDocument || editHistoryIndex >= editHistory.length - 1) return;

    const action = editHistory[editHistoryIndex + 1];
    let notes = tabDocument.notes;
    let selected = selectedNoteId;

    if (action.kind === 'position') {
      const i = notes.findIndex(n => n.id === action.noteId);
      if (i === -1) return;
      notes = [...notes];
      notes[i] = { ...notes[i], ...action.after };
    } else if (action.kind === 'delete') {
      notes = notes.filter(n => n.id !== action.note.id);
      if (selected === action.note.id) selected = null;
    } else {
      // insert → re-insert
      notes = [...notes];
      notes.splice(Math.min(action.index, notes.length), 0, action.note);
      selected = action.note.id;
    }

    set({
      tabDocument: { ...tabDocument, notes },
      editHistoryIndex: editHistoryIndex + 1,
      selectedNoteId: selected,
    });
    persistSession(get().tabDocument!, get().currentJobId);
  },

  canUndo: () => get().editHistoryIndex >= 0,

  canRedo: () => {
    const { editHistory, editHistoryIndex } = get();
    return editHistoryIndex < editHistory.length - 1;
  },

  // UI actions
  setFollowingPlayback: (following) => set({ isFollowingPlayback: following }),

  setZoomLevel: (zoom) => set({ zoomLevel: Math.max(0.25, Math.min(4.0, zoom)) }),

  zoomIn: () => {
    const { zoomLevel } = get();
    const nextZoom = Math.min(4.0, Math.round((zoomLevel + 0.25) * 100) / 100);
    set({ zoomLevel: nextZoom });
  },

  zoomOut: () => {
    const { zoomLevel } = get();
    const nextZoom = Math.max(0.25, Math.round((zoomLevel - 0.25) * 100) / 100);
    set({ zoomLevel: nextZoom });
  },

  resetZoom: () => set({ zoomLevel: 1.0 }),

  setCapoFretInput: (fret) => set({ capoFretInput: Math.max(0, Math.min(12, fret)) }),

  setInstrumentInput: (instrument) => set({ instrumentInput: instrument }),

  setToneInput: (tone) => set({ toneInput: tone }),

  setStyleInput: (style) => set({ styleInput: style }),

  setAccuracyModeInput: (mode) => set({ accuracyModeInput: mode }),

  setRoiEnabled: (enabled) => set({ roiEnabled: enabled }),

  setRoiInput: (roi) => set({
    roiInput: {
      x1: Math.max(0, Math.min(1, roi.x1)),
      y1: Math.max(0, Math.min(1, roi.y1)),
      x2: Math.max(0, Math.min(1, roi.x2)),
      y2: Math.max(0, Math.min(1, roi.y2)),
    },
  }),

  setVideoCollapsed: (collapsed) => set({ isVideoCollapsed: collapsed }),

  toggleVideoCollapsed: () => set((state) => ({ isVideoCollapsed: !state.isVideoCollapsed })),

  setShowShortcutsModal: (show) => set({ showShortcutsModal: show }),

  setPlaybackRate: (rate) => set({ playbackRate: rate }),
}));
