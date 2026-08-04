// tabvision-client/src/store/appStore.ts
import { create } from 'zustand';
import { EditableTechnique, TabDocument, TabNote } from '../types/tab';
import type { AccuracyMode, Instrument, PlayingStyle, Tone, UploadRoi } from '../api/client';
import {
  bankGoldSession as apiBankGoldSession,
  getPersonalIngestAvailable,
} from '../api/client';
import {
  PersistedSession,
  clearSession,
  loadSession,
  persistSession,
} from '../utils/editPersistence';
import { MAX_FRET, MAX_STRING, MIN_STRING, openStringMidi, TuningId } from '../utils/pitch';
import { deleteRecordingBlob, loadRecordingBlob } from '../utils/blobStore';

type JobStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed';
export type InputMediaKind = 'audio' | 'video';

// Speed-vs-accuracy slider. The server pipeline currently has two modes
// ('fast' | 'accurate'); the notches bucket into those, so the middle steps
// are forward-compatible UI rather than distinct pipelines today.
export const SPEED_ACCURACY_LABELS = [
  'Fastest',
  'Fast',
  'Balanced',
  'Accurate',
  'Most accurate',
] as const;
export const SPEED_ACCURACY_MAX = SPEED_ACCURACY_LABELS.length - 1;

export function speedAccuracyToMode(notch: number): AccuracyMode {
  return notch <= 1 ? 'fast' : 'accurate';
}

// What the user hears during playback: the original recording, the pluck-synth
// rendition of the (edited) notes, or both layered.
export type AuditionMode = 'original' | 'synth' | 'both';

// How the notes are displayed: the scrollable timeline canvas (editing) or the
// sheet-style score view (reading / printing).
export type ViewMode = 'timeline' | 'score';

// Only the fields an edit mutates are snapshotted, so undo/redo restore them
// exactly (including isEdited / originalFret bookkeeping) rather than trying to
// recompute derived flags. timestamp/endTime joined for drag-retiming (M4);
// history is in-memory only, so extending the snapshot needs no migration.
type NoteMutableFields = Pick<
  TabNote,
  | 'string'
  | 'fret'
  | 'isEdited'
  | 'originalFret'
  | 'timestamp'
  | 'endTime'
  | 'confidence'
  | 'confidenceLevel'
  | 'technique'
  | 'pitchBend'
>;

type EditAction =
  | { kind: 'position'; noteId: string; before: NoteMutableFields; after: NoteMutableFields }
  | {
      kind: 'batch-position';
      changes: { noteId: string; before: NoteMutableFields; after: NoteMutableFields }[];
    }
  | { kind: 'delete'; note: TabNote; index: number }
  | { kind: 'insert'; note: TabNote; index: number };

/** Keyboard horizontal nudge. Small enough for onset correction, but still
 * visible at the timeline's base density (4 px at 80 px/s). */
export const NOTE_NUDGE_SECONDS = 0.05;

/**
 * Fret on `toString` that sounds the same pitch as `fret` on `fromString`,
 * under the document's tuning (`tuningMidi` low-to-high; absent = standard).
 * A capo shifts every string equally, so it cancels out of the difference and
 * this is capo-independent. Returns `null` when the pitch is not playable on
 * the target string (fret would be < 0 or > MAX_FRET); a muted "X" moves
 * across strings unchanged.
 */
export function pitchPreservingFret(
  fromString: number,
  toString: number,
  fret: number | 'X',
  tuningMidi?: number[],
): number | 'X' | null {
  if (fret === 'X') return 'X';
  const next =
    fret + openStringMidi(fromString, tuningMidi) - openStringMidi(toString, tuningMidi);
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
  // Actual submitted file kind. Unlike pipelineVideoEnabled, this stays
  // accurate while the initial upload is still waiting for server status.
  inputMediaKind: InputMediaKind;
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
  /** All selected notes. `selectedNoteId` remains the primary selection for
   * single-note editors such as fret entry and candidate cycling. */
  selectedNoteIds: string[];
  selectionAnchorId: string | null;
  isFollowingPlayback: boolean;
  pendingFretInput: string;
  // Bumped by jumpToNextConfidence so the active view scrolls the newly
  // selected note into view (a counter, since the same note can be re-focused).
  noteFocusNonce: number;

  // Review mode (2026-07-20 assisted program): step through the
  // lowest-confidence notes and cycle each one's ranked pitch-preserving
  // alternatives (server-computed min-marginal candidates).
  reviewActive: boolean;
  reviewIds: string[];
  reviewIndex: number;

  // UI state
  zoomLevel: number;
  capoFretInput: number;
  tuningInput: TuningId;
  instrumentInput: Instrument;
  toneInput: Tone;
  styleInput: PlayingStyle;
  speedAccuracyInput: number;
  roiEnabled: boolean;
  roiInput: UploadRoi;
  isVideoCollapsed: boolean;
  showShortcutsModal: boolean;
  playbackRate: number;
  auditionMode: AuditionMode;
  viewMode: ViewMode;

  // Edit history
  editHistory: EditAction[];
  editHistoryIndex: number;

  // Job actions
  setJobId: (id: string) => void;
  setStatus: (status: JobStatus) => void;
  setProgress: (progress: number, stage: string) => void;
  setPipelineVideoEnabled: (enabled: boolean) => void;
  setInputMediaKind: (kind: InputMediaKind) => void;
  setTabDocument: (doc: TabDocument) => void;
  setError: (message: string) => void;
  setVideoUrl: (url: string | null) => void;
  reset: () => void;

  // B5 — edit persistence / restore
  loadPersistedSession: () => void;
  restorePersistedSession: () => void;
  discardPersistedSession: () => void;

  // Gold-session banking (local-only; SPEC §1.5 carve-out 2026-08-02).
  // Only the studio backend advertises personal_ingest, so the deployed
  // site never shows the feature.
  personalIngestAvailable: boolean;
  goldBankStatus: 'idle' | 'banking' | 'done' | 'error';
  goldBankMessage: string | null;
  checkPersonalIngest: () => Promise<void>;
  bankGoldSession: () => Promise<void>;

  // Playback actions
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setIsPlaying: (playing: boolean) => void;

  // Selection actions
  selectNote: (noteId: string | null) => void;
  toggleNoteSelection: (noteId: string) => void;
  selectNoteRange: (noteId: string) => void;
  selectAdjacentNote: (direction: 'left' | 'right' | 'up' | 'down') => void;
  jumpToNextConfidence: (level: 'medium' | 'low') => void;

  // Review actions
  startReview: () => void;
  exitReview: () => void;
  reviewNext: () => void;
  reviewPrev: () => void;
  cycleNoteCandidate: (direction: 1 | -1) => void;

  // Editing actions
  updateNoteFret: (noteId: string, newFret: number | "X") => void;
  updateNotePosition: (noteId: string, newString: number, newFret: number | "X") => void;
  applyNoteDrag: (
    noteId: string,
    next: { timestamp: number; string: number; fret: number | "X" },
  ) => void;
  moveNoteString: (direction: 'up' | 'down') => void;
  moveSelectedNotes: (direction: 'left' | 'right' | 'up' | 'down') => void;
  setSelectedTechnique: (
    technique: EditableTechnique | null,
    pitchBend?: number,
  ) => void;
  deleteNote: (noteId: string) => void;
  insertNote: (opts: { timestamp: number; string: number; fret?: number | "X" }) => void;
  setPendingFretInput: (input: string) => void;
  commitPendingEdit: () => void;
  setDocumentTitle: (title: string) => void;

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
  setTuningInput: (tuning: TuningId) => void;
  setInstrumentInput: (instrument: Instrument) => void;
  setToneInput: (tone: Tone) => void;
  setStyleInput: (style: PlayingStyle) => void;
  setSpeedAccuracyInput: (notch: number) => void;
  setRoiEnabled: (enabled: boolean) => void;
  setRoiInput: (roi: UploadRoi) => void;
  setVideoCollapsed: (collapsed: boolean) => void;
  toggleVideoCollapsed: () => void;
  setShowShortcutsModal: (show: boolean) => void;
  setPlaybackRate: (rate: number) => void;
  setAuditionMode: (mode: AuditionMode) => void;
  setViewMode: (mode: ViewMode) => void;
}

const initialState = {
  // Job state
  currentJobId: null as string | null,
  jobStatus: 'idle' as JobStatus,
  progress: 0,
  currentStage: '',
  pipelineVideoEnabled: true,
  inputMediaKind: 'video' as InputMediaKind,
  tabDocument: null as TabDocument | null,
  errorMessage: null as string | null,
  videoUrl: null as string | null,
  restorable: null as PersistedSession | null,
  personalIngestAvailable: false,
  goldBankStatus: 'idle' as 'idle' | 'banking' | 'done' | 'error',
  goldBankMessage: null as string | null,

  // Playback state
  currentTime: 0,
  duration: 0,
  isPlaying: false,

  // Editor state
  selectedNoteId: null as string | null,
  selectedNoteIds: [] as string[],
  selectionAnchorId: null as string | null,
  isFollowingPlayback: true,
  pendingFretInput: '',
  noteFocusNonce: 0,

  // Review mode
  reviewActive: false,
  reviewIds: [] as string[],
  reviewIndex: -1,

  // UI state
  zoomLevel: 1.0,
  capoFretInput: 0,
  tuningInput: 'standard' as TuningId,
  instrumentInput: 'acoustic' as Instrument,
  toneInput: 'clean' as Tone,
  styleInput: 'mixed' as PlayingStyle,
  speedAccuracyInput: 3, // 'Accurate' — matches the old default mode
  roiEnabled: false,
  roiInput: { x1: 0, y1: 0, x2: 1, y2: 1 } as UploadRoi,
  isVideoCollapsed: false,
  showShortcutsModal: false,
  playbackRate: 1.0,
  auditionMode: 'original' as AuditionMode,
  viewMode: 'timeline' as ViewMode,

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

  setInputMediaKind: (kind) => set({ inputMediaKind: kind }),

  setTabDocument: (doc) => {
    set({
      tabDocument: doc,
      jobStatus: 'completed',
      restorable: null,
      selectedNoteId: null,
      selectedNoteIds: [],
      selectionAnchorId: null,
    });
    persistSession(doc, get().currentJobId); // survive a refresh from the first render
  },

  setError: (message) => set({ errorMessage: message, jobStatus: 'failed' }),

  setVideoUrl: (url) => set({ videoUrl: url }),

  reset: () => {
    const jobId = get().currentJobId;
    if (jobId) void deleteRecordingBlob(jobId); // best-effort cleanup
    clearSession(); // "New transcription" discards the autosaved session
    // The backend's personal-ingest capability doesn't change per job.
    set({ ...initialState, personalIngestAvailable: get().personalIngestAvailable });
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
      selectedNoteId: null,
      selectedNoteIds: [],
      selectionAnchorId: null,
    });
    // Recover the recording from IndexedDB so the restored session keeps real
    // playback. Async and fail-open: no blob just means no video pane (the
    // synth transport still plays the notes).
    const jobId = session.jobId;
    if (jobId) {
      void loadRecordingBlob(jobId).then(blob => {
        if (blob && get().currentJobId === jobId) {
          set({ videoUrl: URL.createObjectURL(blob) });
        }
      });
    }
  },

  discardPersistedSession: () => {
    const jobId = get().restorable?.jobId;
    if (jobId) void deleteRecordingBlob(jobId); // best-effort cleanup
    clearSession();
    set({ restorable: null });
  },

  // Gold-session banking — the corrected document is ground truth for the
  // take, so the whole (reviewed) note list is sent, edits and all.
  checkPersonalIngest: async () => {
    set({ personalIngestAvailable: await getPersonalIngestAvailable() });
  },

  bankGoldSession: async () => {
    const { currentJobId, tabDocument, goldBankStatus } = get();
    if (!currentJobId || !tabDocument || goldBankStatus === 'banking') return;
    set({ goldBankStatus: 'banking', goldBankMessage: null });
    try {
      const summary = await apiBankGoldSession(
        currentJobId,
        tabDocument.notes.map((note) => ({
          timestamp: note.timestamp,
          string: note.string,
          fret: note.fret,
        })),
      );
      const frames =
        summary.frames_written > 0 ? `${summary.frames_written} frames` : 'no frames (audio-only)';
      set({
        goldBankStatus: 'done',
        goldBankMessage: `Banked ${summary.notes} notes → ${frames} + ${summary.prior_labels} prior labels`,
      });
    } catch (error) {
      set({
        goldBankStatus: 'error',
        goldBankMessage: error instanceof Error ? error.message : 'Failed to bank the gold session',
      });
    }
  },

  // Playback actions
  setCurrentTime: (time) => set({ currentTime: time }),

  // MediaRecorder clips report Infinity until seeked; never let a non-finite
  // duration into the store (it would freeze the tab canvas layout).
  setDuration: (duration) => set({ duration: Number.isFinite(duration) && duration > 0 ? duration : 0 }),

  setIsPlaying: (playing) => set({ isPlaying: playing }),

  // Selection actions. A plain selection replaces the group; Ctrl/Cmd-click
  // toggles one note and Shift-click selects the inclusive time-ordered range.
  selectNote: (noteId) => set({
    selectedNoteId: noteId,
    selectedNoteIds: noteId ? [noteId] : [],
    selectionAnchorId: noteId,
    pendingFretInput: '',
  }),

  toggleNoteSelection: (noteId) => {
    const { tabDocument, selectedNoteIds, reviewActive, selectNote } = get();
    if (reviewActive) {
      selectNote(noteId);
      return;
    }
    if (!tabDocument?.notes.some(n => n.id === noteId)) return;
    const selected = selectedNoteIds.includes(noteId)
      ? selectedNoteIds.filter(id => id !== noteId)
      : [...selectedNoteIds, noteId];
    const fallback = selected[selected.length - 1] ?? null;
    set({
      selectedNoteIds: selected,
      selectedNoteId: selected.includes(noteId) ? noteId : fallback,
      selectionAnchorId: selected.includes(noteId) ? noteId : fallback,
      pendingFretInput: '',
    });
  },

  selectNoteRange: (noteId) => {
    const { tabDocument, selectedNoteId, selectionAnchorId, reviewActive, selectNote } = get();
    if (reviewActive) {
      selectNote(noteId);
      return;
    }
    if (!tabDocument) return;
    const ordered = [...tabDocument.notes].sort(
      (a, b) => a.timestamp - b.timestamp || a.string - b.string || a.id.localeCompare(b.id)
    );
    const anchorId = selectionAnchorId ?? selectedNoteId ?? noteId;
    const anchorIndex = ordered.findIndex(n => n.id === anchorId);
    const targetIndex = ordered.findIndex(n => n.id === noteId);
    if (targetIndex === -1) return;
    const start = anchorIndex === -1 ? targetIndex : Math.min(anchorIndex, targetIndex);
    const end = anchorIndex === -1 ? targetIndex : Math.max(anchorIndex, targetIndex);
    set({
      selectedNoteId: noteId,
      selectedNoteIds: ordered.slice(start, end + 1).map(n => n.id),
      selectionAnchorId: anchorIndex === -1 ? noteId : anchorId,
      pendingFretInput: '',
    });
  },

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
      set({
        selectedNoteId: nextNote.id,
        selectedNoteIds: [nextNote.id],
        selectionAnchorId: nextNote.id,
        pendingFretInput: '',
      });
    }
  },

  // Toolbar confidence pills: jump to the next note still at the clicked
  // level (fixing a note promotes it to high, so it drops out of the cycle).
  // Anchored on the current selection when it's part of the cycle, otherwise
  // on its timestamp / the playhead; wraps around at the end.
  jumpToNextConfidence: (level) => {
    const { tabDocument, selectedNoteId, currentTime, noteFocusNonce } = get();
    if (!tabDocument) return;
    const targets = tabDocument.notes
      .filter(n => n.confidenceLevel === level)
      .sort((a, b) => a.timestamp - b.timestamp || a.string - b.string);
    if (!targets.length) return;

    const current = selectedNoteId
      ? tabDocument.notes.find(n => n.id === selectedNoteId)
      : undefined;
    const currentIdx = current ? targets.findIndex(n => n.id === current.id) : -1;
    const next =
      currentIdx !== -1
        ? targets[(currentIdx + 1) % targets.length]
        : targets.find(n => n.timestamp > (current ? current.timestamp : currentTime)) ??
          targets[0];

    set({
      selectedNoteId: next.id,
      selectedNoteIds: [next.id],
      selectionAnchorId: next.id,
      pendingFretInput: '',
      // Detach auto-follow so the scroll-to-note isn't fought by playback.
      isFollowingPlayback: false,
      noteFocusNonce: noteFocusNonce + 1,
    });
  },

  // Review actions — the review queue is the phase-6 design at its measured
  // level: lowest-confidence unedited notes first (the Viterbi string-flip
  // margin drives `confidence`), a 30-note budget (the offline 60-second
  // replay), and candidate cycling instead of free-form re-entry.
  startReview: () => {
    const { tabDocument, setFollowingPlayback } = get();
    if (!tabDocument) return;
    const REVIEW_BUDGET = 30;
    const ids = [...tabDocument.notes]
      .filter(n => !n.isEdited && n.fret !== 'X')
      .sort((a, b) => a.confidence - b.confidence)
      .slice(0, REVIEW_BUDGET)
      .map(n => n.id);
    if (!ids.length) return;
    setFollowingPlayback(false);
    set({
      reviewActive: true,
      reviewIds: ids,
      reviewIndex: 0,
      selectedNoteId: ids[0],
      selectedNoteIds: [ids[0]],
      selectionAnchorId: ids[0],
      pendingFretInput: '',
    });
  },

  exitReview: () => set({ reviewActive: false, reviewIds: [], reviewIndex: -1 }),

  reviewNext: () => {
    const { reviewActive, reviewIds, reviewIndex, exitReview } = get();
    if (!reviewActive) return;
    const next = reviewIndex + 1;
    if (next >= reviewIds.length) {
      exitReview();
      return;
    }
    set({
      reviewIndex: next,
      selectedNoteId: reviewIds[next],
      selectedNoteIds: [reviewIds[next]],
      selectionAnchorId: reviewIds[next],
      pendingFretInput: '',
    });
  },

  reviewPrev: () => {
    const { reviewActive, reviewIds, reviewIndex } = get();
    if (!reviewActive) return;
    const prev = Math.max(0, reviewIndex - 1);
    set({
      reviewIndex: prev,
      selectedNoteId: reviewIds[prev],
      selectedNoteIds: [reviewIds[prev]],
      selectionAnchorId: reviewIds[prev],
      pendingFretInput: '',
    });
  },

  // Cycle the selected note through its server-ranked pitch-preserving
  // alternatives. The list always contains the emitted position, so cycling
  // can never change pitch or land on an unplayable position.
  cycleNoteCandidate: (direction) => {
    const { tabDocument, selectedNoteId, updateNotePosition } = get();
    if (!tabDocument || !selectedNoteId) return;
    const note = tabDocument.notes.find(n => n.id === selectedNoteId);
    if (!note || note.fret === 'X') return;
    const candidates = note.candidates ?? [];
    if (candidates.length < 2) return;
    const current = candidates.findIndex(
      c => c.string === note.string && c.fret === note.fret
    );
    const base = current === -1 ? 0 : current;
    const next = candidates[(base + direction + candidates.length) % candidates.length];
    if (next.string < MIN_STRING || next.string > MAX_STRING) return;
    updateNotePosition(selectedNoteId, next.string, next.fret);
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
    // Re-entering the unchanged position on a medium/low note is a "confirm":
    // it still promotes the note to high confidence below. Only skip when
    // nothing at all would change.
    const samePosition = note.string === newString && note.fret === newFret;
    if (samePosition && note.confidenceLevel === 'high') return; // no-op

    const before: NoteMutableFields = {
      string: note.string,
      fret: note.fret,
      isEdited: note.isEdited,
      originalFret: note.originalFret,
      timestamp: note.timestamp,
      endTime: note.endTime,
      confidence: note.confidence,
      confidenceLevel: note.confidenceLevel,
    };
    const updated: TabNote = {
      ...note,
      string: newString as TabNote['string'],
      fret: newFret,
      isEdited: true,
      originalFret: note.originalFret ?? note.fret,
      // A fixed note is user-verified — its confidence color turns green.
      confidence: 1,
      confidenceLevel: 'high',
    };
    const after: NoteMutableFields = {
      string: updated.string,
      fret: updated.fret,
      isEdited: updated.isEdited,
      originalFret: updated.originalFret,
      timestamp: updated.timestamp,
      endTime: updated.endTime,
      confidence: updated.confidence,
      confidenceLevel: updated.confidenceLevel,
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

  // Mouse-drag commit (M4): one history entry for retime + restring together,
  // so a single Ctrl+Z reverses the whole gesture. endTime shifts with the
  // timestamp to preserve the note's duration. The notes array is deliberately
  // left unsorted after a retime — every consumer sorts on read, and
  // re-sorting here would break the index semantics of delete/insert history
  // entries.
  applyNoteDrag: (noteId, next) => {
    const { tabDocument, editHistory, editHistoryIndex } = get();
    if (!tabDocument) return;

    const noteIndex = tabDocument.notes.findIndex(n => n.id === noteId);
    if (noteIndex === -1) return;
    const note = tabDocument.notes[noteIndex];

    const newTimestamp = Math.max(0, next.timestamp);
    const delta = newTimestamp - note.timestamp;
    if (delta === 0 && note.string === next.string && note.fret === next.fret) return;

    const before: NoteMutableFields = {
      string: note.string,
      fret: note.fret,
      isEdited: note.isEdited,
      originalFret: note.originalFret,
      timestamp: note.timestamp,
      endTime: note.endTime,
      confidence: note.confidence,
      confidenceLevel: note.confidenceLevel,
    };
    const updated: TabNote = {
      ...note,
      string: next.string as TabNote['string'],
      fret: next.fret,
      timestamp: newTimestamp,
      endTime: typeof note.endTime === 'number' ? note.endTime + delta : note.endTime,
      isEdited: true,
      originalFret: note.originalFret ?? note.fret,
      // A fixed note is user-verified — its confidence color turns green.
      confidence: 1,
      confidenceLevel: 'high',
    };
    const after: NoteMutableFields = {
      string: updated.string,
      fret: updated.fret,
      isEdited: updated.isEdited,
      originalFret: updated.originalFret,
      timestamp: updated.timestamp,
      endTime: updated.endTime,
      confidence: updated.confidence,
      confidenceLevel: updated.confidenceLevel,
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

  // Move the selection to the adjacent string while keeping pitch. 'up' is
  // toward string 1 (high E); any unplayable member makes the move a no-op.
  moveNoteString: (direction) => get().moveSelectedNotes(direction),

  // Move the current selection as one unit. Vertical moves preserve every
  // note's pitch; horizontal moves nudge onset and end time by 50 ms. If any
  // note would cross a string/time boundary or become unplayable, the entire
  // move is rejected. One history entry means one undo restores the group.
  moveSelectedNotes: (direction) => {
    const {
      tabDocument,
      selectedNoteId,
      selectedNoteIds,
      editHistory,
      editHistoryIndex,
    } = get();
    if (!tabDocument) return;
    const ids = selectedNoteIds.length
      ? selectedNoteIds
      : (selectedNoteId ? [selectedNoteId] : []);
    if (!ids.length) return;

    const selected = new Set(ids);
    const selectedNotes = tabDocument.notes.filter(note => selected.has(note.id));
    if (selectedNotes.length !== selected.size) return;

    const changes: Extract<EditAction, { kind: 'batch-position' }>['changes'] = [];
    const replacements = new Map<string, TabNote>();
    const horizontalDelta =
      direction === 'left'
        ? -NOTE_NUDGE_SECONDS
        : direction === 'right'
          ? NOTE_NUDGE_SECONDS
          : 0;

    for (const note of selectedNotes) {
      let nextString = note.string;
      let nextFret = note.fret;
      let nextTimestamp = note.timestamp;
      let nextEndTime = note.endTime;

      if (direction === 'up' || direction === 'down') {
        const target = direction === 'up' ? note.string - 1 : note.string + 1;
        if (target < MIN_STRING || target > MAX_STRING) return;
        const fret = pitchPreservingFret(
          note.string,
          target,
          note.fret,
          tabDocument.tuningMidi,
        );
        if (fret === null) return;
        nextString = target as TabNote['string'];
        nextFret = fret;
      } else {
        nextTimestamp = note.timestamp + horizontalDelta;
        if (nextTimestamp < 0 || nextTimestamp > tabDocument.duration) return;
        nextEndTime = typeof note.endTime === 'number'
          ? note.endTime + horizontalDelta
          : note.endTime;
      }

      const before: NoteMutableFields = {
        string: note.string,
        fret: note.fret,
        isEdited: note.isEdited,
        originalFret: note.originalFret,
        timestamp: note.timestamp,
        endTime: note.endTime,
        confidence: note.confidence,
        confidenceLevel: note.confidenceLevel,
      };
      const updated: TabNote = {
        ...note,
        string: nextString,
        fret: nextFret,
        timestamp: nextTimestamp,
        endTime: nextEndTime,
        isEdited: true,
        originalFret: note.originalFret ?? note.fret,
        confidence: 1,
        confidenceLevel: 'high',
      };
      const after: NoteMutableFields = {
        string: updated.string,
        fret: updated.fret,
        isEdited: updated.isEdited,
        originalFret: updated.originalFret,
        timestamp: updated.timestamp,
        endTime: updated.endTime,
        confidence: updated.confidence,
        confidenceLevel: updated.confidenceLevel,
      };
      changes.push({ noteId: note.id, before, after });
      replacements.set(note.id, updated);
    }

    const updatedNotes = tabDocument.notes.map(note => replacements.get(note.id) ?? note);
    const newHistory = editHistory.slice(0, editHistoryIndex + 1);
    newHistory.push({ kind: 'batch-position', changes });
    set({
      tabDocument: { ...tabDocument, notes: updatedNotes },
      editHistory: newHistory,
      editHistoryIndex: newHistory.length - 1,
    });
    persistSession(get().tabDocument!, get().currentJobId);
  },

  // Expressive markings apply to the current selection as one undoable edit.
  // Slides are stored on the destination note and connect from the preceding
  // note on that string. Bends carry their upward amount in semitones.
  setSelectedTechnique: (technique, pitchBend) => {
    const {
      tabDocument,
      selectedNoteId,
      selectedNoteIds,
      editHistory,
      editHistoryIndex,
    } = get();
    if (!tabDocument) return;

    const ids = selectedNoteIds.length
      ? selectedNoteIds
      : (selectedNoteId ? [selectedNoteId] : []);
    if (!ids.length) return;

    const selected = new Set(ids);
    const selectedNotes = tabDocument.notes.filter(note => selected.has(note.id));
    // A muted string has no pitched gesture to slide or bend. Keep group edits
    // atomic instead of silently annotating only part of the selection.
    if (selectedNotes.length !== selected.size || selectedNotes.some(note => note.fret === 'X')) {
      return;
    }

    const nextPitchBend = technique === 'bend'
      ? Math.max(0.25, Math.min(12, pitchBend ?? 2))
      : undefined;
    const changes: Extract<EditAction, { kind: 'batch-position' }>['changes'] = [];
    const replacements = new Map<string, TabNote>();

    for (const note of selectedNotes) {
      const sameTechnique = (note.technique ?? null) === technique;
      const sameBend = technique !== 'bend' || note.pitchBend === nextPitchBend;
      if (sameTechnique && sameBend) continue;

      const before: NoteMutableFields = {
        string: note.string,
        fret: note.fret,
        isEdited: note.isEdited,
        originalFret: note.originalFret,
        timestamp: note.timestamp,
        endTime: note.endTime,
        confidence: note.confidence,
        confidenceLevel: note.confidenceLevel,
        technique: note.technique,
        pitchBend: note.pitchBend,
      };
      const updated: TabNote = {
        ...note,
        technique: technique ?? undefined,
        pitchBend: nextPitchBend,
        isEdited: true,
      };
      const after: NoteMutableFields = {
        string: updated.string,
        fret: updated.fret,
        isEdited: updated.isEdited,
        originalFret: updated.originalFret,
        timestamp: updated.timestamp,
        endTime: updated.endTime,
        confidence: updated.confidence,
        confidenceLevel: updated.confidenceLevel,
        technique: updated.technique,
        pitchBend: updated.pitchBend,
      };
      changes.push({ noteId: note.id, before, after });
      replacements.set(note.id, updated);
    }

    if (!changes.length) return;
    const updatedNotes = tabDocument.notes.map(note => replacements.get(note.id) ?? note);
    const newHistory = editHistory.slice(0, editHistoryIndex + 1);
    newHistory.push({ kind: 'batch-position', changes });
    set({
      tabDocument: { ...tabDocument, notes: updatedNotes },
      editHistory: newHistory,
      editHistoryIndex: newHistory.length - 1,
    });
    persistSession(get().tabDocument!, get().currentJobId);
  },

  // True removal (B3) — distinct from mute (fret = "X").
  deleteNote: (noteId) => {
    const { tabDocument, editHistory, editHistoryIndex, selectedNoteId, selectedNoteIds } = get();
    if (!tabDocument) return;
    const index = tabDocument.notes.findIndex(n => n.id === noteId);
    if (index === -1) return;

    const removed = tabDocument.notes[index];
    const updatedNotes = tabDocument.notes.filter(n => n.id !== noteId);
    const newHistory = editHistory.slice(0, editHistoryIndex + 1);
    newHistory.push({ kind: 'delete', note: removed, index });
    const remainingSelection = selectedNoteIds.filter(id => id !== noteId);
    const nextPrimary = selectedNoteId === noteId
      ? (remainingSelection[remainingSelection.length - 1] ?? null)
      : selectedNoteId;

    set({
      tabDocument: { ...tabDocument, notes: updatedNotes },
      editHistory: newHistory,
      editHistoryIndex: newHistory.length - 1,
      selectedNoteId: nextPrimary,
      selectedNoteIds: remainingSelection,
      selectionAnchorId: nextPrimary,
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
      selectedNoteIds: [note.id],
      selectionAnchorId: note.id,
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

  // Piece title (score header + export filenames). Not an undoable edit —
  // it's document metadata, but it persists with the session like note edits.
  setDocumentTitle: (title) => {
    const { tabDocument } = get();
    if (!tabDocument) return;
    const trimmed = title.trim();
    if ((tabDocument.title ?? '') === trimmed) return;
    set({ tabDocument: { ...tabDocument, title: trimmed || undefined } });
    persistSession(get().tabDocument!, get().currentJobId);
  },

  // Undo/Redo actions — dispatch on the action kind. position restores a
  // field snapshot; delete/insert are inverses (re-insert at index / remove).
  undo: () => {
    const { tabDocument, editHistory, editHistoryIndex, selectedNoteId, selectedNoteIds } = get();
    if (!tabDocument || editHistoryIndex < 0) return;

    const action = editHistory[editHistoryIndex];
    let notes = tabDocument.notes;
    let selected = selectedNoteId;
    let selection = selectedNoteIds;

    if (action.kind === 'position') {
      const i = notes.findIndex(n => n.id === action.noteId);
      if (i === -1) return;
      notes = [...notes];
      notes[i] = { ...notes[i], ...action.before };
    } else if (action.kind === 'batch-position') {
      const snapshots = new Map(action.changes.map(change => [change.noteId, change.before]));
      if (action.changes.some(change => !notes.some(note => note.id === change.noteId))) return;
      notes = notes.map(note => {
        const snapshot = snapshots.get(note.id);
        return snapshot ? { ...note, ...snapshot } : note;
      });
    } else if (action.kind === 'delete') {
      notes = [...notes];
      notes.splice(Math.min(action.index, notes.length), 0, action.note);
      selected = action.note.id;
      selection = [action.note.id];
    } else {
      // insert → remove
      notes = notes.filter(n => n.id !== action.note.id);
      if (selected === action.note.id) selected = null;
      selection = selection.filter(id => id !== action.note.id);
    }

    set({
      tabDocument: { ...tabDocument, notes },
      editHistoryIndex: editHistoryIndex - 1,
      selectedNoteId: selected,
      selectedNoteIds: selection,
      selectionAnchorId: selected,
    });
    persistSession(get().tabDocument!, get().currentJobId);
  },

  redo: () => {
    const { tabDocument, editHistory, editHistoryIndex, selectedNoteId, selectedNoteIds } = get();
    if (!tabDocument || editHistoryIndex >= editHistory.length - 1) return;

    const action = editHistory[editHistoryIndex + 1];
    let notes = tabDocument.notes;
    let selected = selectedNoteId;
    let selection = selectedNoteIds;

    if (action.kind === 'position') {
      const i = notes.findIndex(n => n.id === action.noteId);
      if (i === -1) return;
      notes = [...notes];
      notes[i] = { ...notes[i], ...action.after };
    } else if (action.kind === 'batch-position') {
      const snapshots = new Map(action.changes.map(change => [change.noteId, change.after]));
      if (action.changes.some(change => !notes.some(note => note.id === change.noteId))) return;
      notes = notes.map(note => {
        const snapshot = snapshots.get(note.id);
        return snapshot ? { ...note, ...snapshot } : note;
      });
    } else if (action.kind === 'delete') {
      notes = notes.filter(n => n.id !== action.note.id);
      if (selected === action.note.id) selected = null;
      selection = selection.filter(id => id !== action.note.id);
    } else {
      // insert → re-insert
      notes = [...notes];
      notes.splice(Math.min(action.index, notes.length), 0, action.note);
      selected = action.note.id;
      selection = [action.note.id];
    }

    set({
      tabDocument: { ...tabDocument, notes },
      editHistoryIndex: editHistoryIndex + 1,
      selectedNoteId: selected,
      selectedNoteIds: selection,
      selectionAnchorId: selected,
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

  setTuningInput: (tuning) => set({ tuningInput: tuning }),

  setInstrumentInput: (instrument) => set({ instrumentInput: instrument }),

  setToneInput: (tone) => set({ toneInput: tone }),

  setStyleInput: (style) => set({ styleInput: style }),

  setSpeedAccuracyInput: (notch) =>
    set({ speedAccuracyInput: Math.max(0, Math.min(SPEED_ACCURACY_MAX, Math.round(notch))) }),

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

  setAuditionMode: (mode) => set({ auditionMode: mode }),

  setViewMode: (mode) => set({ viewMode: mode }),
}));
