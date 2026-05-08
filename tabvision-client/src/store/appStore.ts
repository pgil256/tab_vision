// tabvision-client/src/store/appStore.ts
import { create } from 'zustand';
import { TabDocument, TabNote } from '../types/tab';

type JobStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed';

interface EditAction {
  noteId: string;
  previousFret: number | "X";
  newFret: number | "X";
}

interface AppState {
  // Job state
  currentJobId: string | null;
  jobStatus: JobStatus;
  progress: number;
  currentStage: string;
  tabDocument: TabDocument | null;
  errorMessage: string | null;
  videoUrl: string | null;

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
  setTabDocument: (doc: TabDocument) => void;
  setError: (message: string) => void;
  setVideoUrl: (url: string | null) => void;
  reset: () => void;

  // Playback actions
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setIsPlaying: (playing: boolean) => void;

  // Selection actions
  selectNote: (noteId: string | null) => void;
  selectAdjacentNote: (direction: 'left' | 'right' | 'up' | 'down') => void;

  // Editing actions
  updateNoteFret: (noteId: string, newFret: number | "X") => void;
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
  tabDocument: null as TabDocument | null,
  errorMessage: null as string | null,
  videoUrl: null as string | null,

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

  setTabDocument: (doc) => set({ tabDocument: doc, jobStatus: 'completed' }),

  setError: (message) => set({ errorMessage: message, jobStatus: 'failed' }),

  setVideoUrl: (url) => set({ videoUrl: url }),

  reset: () => set(initialState),

  // Playback actions
  setCurrentTime: (time) => set({ currentTime: time }),

  setDuration: (duration) => set({ duration }),

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
  updateNoteFret: (noteId, newFret) => {
    const { tabDocument, editHistory, editHistoryIndex } = get();
    if (!tabDocument) return;

    const noteIndex = tabDocument.notes.findIndex(n => n.id === noteId);
    if (noteIndex === -1) return;

    const note = tabDocument.notes[noteIndex];
    const previousFret = note.fret;

    // Don't record if no change
    if (previousFret === newFret) return;

    // Update note
    const updatedNotes = [...tabDocument.notes];
    updatedNotes[noteIndex] = {
      ...note,
      fret: newFret,
      isEdited: true,
      originalFret: note.originalFret ?? previousFret,
    };

    // Add to history (truncate any redo stack)
    const newHistory = editHistory.slice(0, editHistoryIndex + 1);
    newHistory.push({ noteId, previousFret, newFret });

    set({
      tabDocument: { ...tabDocument, notes: updatedNotes },
      editHistory: newHistory,
      editHistoryIndex: newHistory.length - 1,
    });
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

  // Undo/Redo actions
  undo: () => {
    const { tabDocument, editHistory, editHistoryIndex } = get();
    if (!tabDocument || editHistoryIndex < 0) return;

    const action = editHistory[editHistoryIndex];
    const noteIndex = tabDocument.notes.findIndex(n => n.id === action.noteId);
    if (noteIndex === -1) return;

    const note = tabDocument.notes[noteIndex];
    const updatedNotes = [...tabDocument.notes];
    updatedNotes[noteIndex] = {
      ...note,
      fret: action.previousFret,
      isEdited: action.previousFret !== note.originalFret,
    };

    set({
      tabDocument: { ...tabDocument, notes: updatedNotes },
      editHistoryIndex: editHistoryIndex - 1,
    });
  },

  redo: () => {
    const { tabDocument, editHistory, editHistoryIndex } = get();
    if (!tabDocument || editHistoryIndex >= editHistory.length - 1) return;

    const action = editHistory[editHistoryIndex + 1];
    const noteIndex = tabDocument.notes.findIndex(n => n.id === action.noteId);
    if (noteIndex === -1) return;

    const note = tabDocument.notes[noteIndex];
    const updatedNotes = [...tabDocument.notes];
    updatedNotes[noteIndex] = {
      ...note,
      fret: action.newFret,
      isEdited: true,
    };

    set({
      tabDocument: { ...tabDocument, notes: updatedNotes },
      editHistoryIndex: editHistoryIndex + 1,
    });
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

  setVideoCollapsed: (collapsed) => set({ isVideoCollapsed: collapsed }),

  toggleVideoCollapsed: () => set((state) => ({ isVideoCollapsed: !state.isVideoCollapsed })),

  setShowShortcutsModal: (show) => set({ showShortcutsModal: show }),

  setPlaybackRate: (rate) => set({ playbackRate: rate }),
}));
