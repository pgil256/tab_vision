// Single keyboard dispatcher (M7): absorbs TabCanvas's editor keydown handler
// and VideoPlayer's Space handler, which used to be two competing window
// listeners. Mounted once in App. Space routes through the audition transport
// so it works with or without a video element; every other binding is
// unchanged (Ctrl+Z/⇧Z, Ctrl+=/−, arrows ± Shift, R/N/P/C, Esc, Enter,
// Del/Backspace, X, I/Insert, digit entry with the 500 ms commit timeout).

import { useEffect, useRef } from 'react';
import { useAppStore } from '../store/appStore';
import { audition } from '../utils/auditionEngine';

export function useEditorHotkeys() {
  const fretInputTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Never capture typing targets.
      const target = e.target as HTMLElement | null;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        (target && target.isContentEditable)
      ) {
        return;
      }

      const s = useAppStore.getState();
      // The hook is always mounted (unlike TabCanvas), so editor keys are
      // gated on a completed job.
      if (s.jobStatus !== 'completed') return;

      // Playback toggle — transport-aware (video or synth-only).
      if (e.code === 'Space' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        audition.togglePlay();
        return;
      }

      // Undo/Redo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        if (e.shiftKey) {
          s.redo();
        } else {
          s.undo();
        }
        return;
      }

      // Zoom shortcuts
      if ((e.ctrlKey || e.metaKey) && (e.key === '=' || e.key === '+')) {
        e.preventDefault();
        s.zoomIn();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key === '-') {
        e.preventDefault();
        s.zoomOut();
        return;
      }

      // Navigation
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        s.commitPendingEdit();
        if (s.selectedNoteIds.length > 1) {
          s.moveSelectedNotes('left');
        } else {
          s.selectAdjacentNote('left');
        }
        return;
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        s.commitPendingEdit();
        if (s.selectedNoteIds.length > 1) {
          s.moveSelectedNotes('right');
        } else {
          s.selectAdjacentNote('right');
        }
        return;
      }
      if (e.key === 'Tab') {
        e.preventDefault();
        s.commitPendingEdit();
        s.selectAdjacentNote('right');
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        s.commitPendingEdit();
        // Shift+Up moves the selected note to the adjacent (higher) string,
        // fret recomputed to keep the pitch — the fix for the #1 wrong-string
        // error. Plain Up navigates.
        if (s.selectedNoteIds.length > 1) {
          s.moveSelectedNotes('up');
        } else if (e.shiftKey) {
          s.moveNoteString('up');
        } else {
          s.selectAdjacentNote('up');
        }
        return;
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        s.commitPendingEdit();
        if (s.selectedNoteIds.length > 1) {
          s.moveSelectedNotes('down');
        } else if (e.shiftKey) {
          s.moveNoteString('down');
        } else {
          s.selectAdjacentNote('down');
        }
        return;
      }

      // Review mode (2026-07-20): R toggles, N/P step the queue, C cycles the
      // selected note through its ranked pitch-preserving alternatives.
      if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        s.commitPendingEdit();
        if (s.reviewActive) {
          s.exitReview();
        } else {
          s.startReview();
        }
        return;
      }
      if ((e.key === 'c' || e.key === 'C') && s.selectedNoteId) {
        e.preventDefault();
        s.setPendingFretInput('');
        s.cycleNoteCandidate(e.shiftKey ? -1 : 1);
        return;
      }
      if ((e.key === 'n' || e.key === 'N') && s.reviewActive) {
        e.preventDefault();
        s.commitPendingEdit();
        s.reviewNext();
        return;
      }
      if ((e.key === 'p' || e.key === 'P') && s.reviewActive) {
        e.preventDefault();
        s.commitPendingEdit();
        s.reviewPrev();
        return;
      }

      // Escape — leaves review mode first, then clears selection.
      if (e.key === 'Escape') {
        e.preventDefault();
        s.setPendingFretInput('');
        if (s.reviewActive) {
          s.exitReview();
          return;
        }
        s.selectNote(null);
        return;
      }

      // Enter
      if (e.key === 'Enter') {
        e.preventDefault();
        s.commitPendingEdit();
        s.selectNote(null);
        return;
      }

      // Delete/Backspace -> true removal (B3). Mute lives on 'x'.
      if ((e.key === 'Delete' || e.key === 'Backspace') && s.selectedNoteId) {
        e.preventDefault();
        s.setPendingFretInput('');
        s.deleteNote(s.selectedNoteId);
        return;
      }

      // Expressive markings. Slides are attached to the destination note;
      // bends default to the most common whole-step amount. Repeating the
      // shortcut clears that marking from the full selection.
      if ((e.key === 's' || e.key === 'S') && s.selectedNoteId) {
        e.preventDefault();
        s.setPendingFretInput('');
        const ids = new Set(s.selectedNoteIds.length ? s.selectedNoteIds : [s.selectedNoteId]);
        const notes = s.tabDocument?.notes.filter(note => ids.has(note.id)) ?? [];
        const allSlides = notes.length > 0 && notes.every(note => note.technique === 'slide');
        s.setSelectedTechnique(allSlides ? null : 'slide');
        return;
      }
      if ((e.key === 'b' || e.key === 'B') && s.selectedNoteId) {
        e.preventDefault();
        s.setPendingFretInput('');
        const ids = new Set(s.selectedNoteIds.length ? s.selectedNoteIds : [s.selectedNoteId]);
        const notes = s.tabDocument?.notes.filter(note => ids.has(note.id)) ?? [];
        const allBends = notes.length > 0 && notes.every(note => note.technique === 'bend');
        s.setSelectedTechnique(allBends ? null : 'bend', 2);
        return;
      }

      // 'x' -> mute the selected note (fret = "X"), distinct from delete.
      if ((e.key === 'x' || e.key === 'X') && s.selectedNoteId) {
        e.preventDefault();
        s.setPendingFretInput('');
        s.updateNoteFret(s.selectedNoteId, 'X');
        return;
      }

      // 'i' / Insert -> insert a note at the playhead (B3), on the selected
      // note's string (or the G string by default), open, and select it.
      if (e.key === 'i' || e.key === 'Insert') {
        e.preventDefault();
        const anchor = s.selectedNoteId
          ? s.tabDocument?.notes.find(n => n.id === s.selectedNoteId)
          : undefined;
        s.insertNote({ timestamp: s.currentTime, string: anchor?.string ?? 3, fret: 0 });
        return;
      }

      // Number input (0-9)
      if (s.selectedNoteId && /^[0-9]$/.test(e.key)) {
        e.preventDefault();
        const newInput = s.pendingFretInput + e.key;
        const fretValue = parseInt(newInput, 10);

        if (fretValue > 24) {
          s.setPendingFretInput(e.key);
        } else {
          s.setPendingFretInput(newInput);
        }

        if (fretInputTimeoutRef.current) {
          clearTimeout(fretInputTimeoutRef.current);
        }
        fretInputTimeoutRef.current = window.setTimeout(() => {
          useAppStore.getState().commitPendingEdit();
        }, 500);

        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      if (fretInputTimeoutRef.current) clearTimeout(fretInputTimeoutRef.current);
    };
  }, []);
}
