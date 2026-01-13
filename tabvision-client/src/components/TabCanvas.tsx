// tabvision-client/src/components/TabCanvas.tsx
import React, { useRef, useEffect, useCallback, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { TabNote } from '../types/tab';

// Layout constants
const PIXELS_PER_SECOND = 50;
const STRING_HEIGHT = 24;
const TIME_AXIS_HEIGHT = 24;
const STRING_LABEL_WIDTH = 24;
const CANVAS_PADDING = 16;
const NOTE_SIZE = 20;
const CANVAS_HEIGHT = TIME_AXIS_HEIGHT + (STRING_HEIGHT * 6) + CANVAS_PADDING * 2;

const STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E'];

// Colors
const COLORS = {
  background: '#1f2937',
  stringLine: '#4b5563',
  timeMarker: '#6b7280',
  timeText: '#9ca3af',
  noteHigh: '#22c55e',
  noteMedium: '#eab308',
  noteLow: '#ef4444',
  noteSelected: '#3b82f6',
  noteEdited: '#ffffff',
  playbackIndicator: '#ef4444',
  noteTextLight: '#ffffff',
  noteTextDark: '#000000',
};

interface NoteHitbox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  note: TabNote;
}

interface TabCanvasProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

export function TabCanvas({ videoRef }: TabCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const noteHitboxesRef = useRef<NoteHitbox[]>([]);
  const isUserScrollingRef = useRef(false);
  const scrollTimeoutRef = useRef<number | null>(null);
  const fretInputTimeoutRef = useRef<number | null>(null);

  const {
    tabDocument,
    jobStatus,
    currentTime,
    duration,
    selectedNoteId,
    isFollowingPlayback,
    pendingFretInput,
    selectNote,
    selectAdjacentNote,
    setCurrentTime,
    setFollowingPlayback,
    setPendingFretInput,
    commitPendingEdit,
    updateNoteFret,
    undo,
    redo,
  } = useAppStore();

  // Calculate canvas width based on duration
  const canvasWidth = Math.max(
    800,
    duration > 0 ? duration * PIXELS_PER_SECOND + STRING_LABEL_WIDTH + CANVAS_PADDING * 2 : 800
  );

  // Get note color based on confidence
  const getNoteColor = useCallback((note: TabNote, isSelected: boolean): string => {
    if (isSelected) return COLORS.noteSelected;
    if (note.confidenceLevel === 'high') return COLORS.noteHigh;
    if (note.confidenceLevel === 'medium') return COLORS.noteMedium;
    return COLORS.noteLow;
  }, []);

  // Get text color for note (contrast with background)
  const getNoteTextColor = useCallback((note: TabNote): string => {
    if (note.confidenceLevel === 'medium') return COLORS.noteTextDark;
    return COLORS.noteTextLight;
  }, []);

  // Convert timestamp to X position
  const timestampToX = useCallback((timestamp: number): number => {
    return STRING_LABEL_WIDTH + CANVAS_PADDING + timestamp * PIXELS_PER_SECOND;
  }, []);

  // Convert string number (1-6) to Y position
  const stringToY = useCallback((stringNum: number): number => {
    return TIME_AXIS_HEIGHT + CANVAS_PADDING + (stringNum - 1) * STRING_HEIGHT + STRING_HEIGHT / 2;
  }, []);

  // Draw the canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !tabDocument) return;

    // Clear canvas
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw string lines
    ctx.strokeStyle = COLORS.stringLine;
    ctx.lineWidth = 1;
    for (let i = 1; i <= 6; i++) {
      const y = stringToY(i);
      ctx.beginPath();
      ctx.moveTo(STRING_LABEL_WIDTH, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw string labels
    ctx.fillStyle = COLORS.timeText;
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < 6; i++) {
      const y = stringToY(i + 1);
      ctx.fillText(STRING_NAMES[i], STRING_LABEL_WIDTH / 2, y);
    }

    // Draw time markers
    ctx.fillStyle = COLORS.timeMarker;
    ctx.strokeStyle = COLORS.timeMarker;
    ctx.lineWidth = 1;
    const maxTime = duration > 0 ? duration : (tabDocument.duration || 60);
    for (let t = 0; t <= maxTime; t++) {
      const x = timestampToX(t);
      // Tick mark every second
      ctx.beginPath();
      ctx.moveTo(x, TIME_AXIS_HEIGHT);
      ctx.lineTo(x, TIME_AXIS_HEIGHT + 4);
      ctx.stroke();

      // Label every 5 seconds
      if (t % 5 === 0) {
        ctx.fillStyle = COLORS.timeText;
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${t}s`, x, TIME_AXIS_HEIGHT / 2);
      }
    }

    // Clear hitboxes and rebuild
    noteHitboxesRef.current = [];

    // Draw notes
    const sortedNotes = [...tabDocument.notes].sort((a, b) => a.timestamp - b.timestamp);
    for (const note of sortedNotes) {
      const x = timestampToX(note.timestamp);
      const y = stringToY(note.string);
      const isSelected = note.id === selectedNoteId;

      // Store hitbox
      noteHitboxesRef.current.push({
        id: note.id,
        x: x - NOTE_SIZE / 2,
        y: y - NOTE_SIZE / 2,
        width: NOTE_SIZE,
        height: NOTE_SIZE,
        note,
      });

      // Draw note background
      const color = getNoteColor(note, isSelected);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x - NOTE_SIZE / 2, y - NOTE_SIZE / 2, NOTE_SIZE, NOTE_SIZE, 4);
      ctx.fill();

      // Draw selection border
      if (isSelected) {
        ctx.strokeStyle = COLORS.noteTextLight;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw edited indicator (small dot)
      if (note.isEdited && !isSelected) {
        ctx.fillStyle = COLORS.noteEdited;
        ctx.beginPath();
        ctx.arc(x + NOTE_SIZE / 2 - 3, y - NOTE_SIZE / 2 + 3, 2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Draw fret number
      const fretText = note.fret === 'X' ? 'X' : note.fret.toString();
      ctx.fillStyle = isSelected ? COLORS.noteTextLight : getNoteTextColor(note);
      ctx.font = 'bold 11px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // Show pending input if this note is selected and has pending input
      if (isSelected && pendingFretInput) {
        ctx.fillText(pendingFretInput, x, y);
      } else {
        ctx.fillText(fretText, x, y);
      }
    }

    // Draw playback indicator
    if (currentTime > 0 || duration > 0) {
      const indicatorX = timestampToX(currentTime);
      ctx.strokeStyle = COLORS.playbackIndicator;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(indicatorX, TIME_AXIS_HEIGHT);
      ctx.lineTo(indicatorX, canvas.height);
      ctx.stroke();
    }
  }, [
    tabDocument,
    duration,
    currentTime,
    selectedNoteId,
    pendingFretInput,
    timestampToX,
    stringToY,
    getNoteColor,
    getNoteTextColor,
  ]);

  // Redraw on state changes
  useEffect(() => {
    draw();
  }, [draw]);

  // Auto-scroll to follow playback
  useEffect(() => {
    if (!isFollowingPlayback || !containerRef.current || isUserScrollingRef.current) return;

    const indicatorX = timestampToX(currentTime);
    const container = containerRef.current;
    const viewportWidth = container.clientWidth;
    const targetScrollLeft = indicatorX - viewportWidth * 0.2;

    container.scrollTo({
      left: Math.max(0, targetScrollLeft),
      behavior: 'smooth',
    });
  }, [currentTime, isFollowingPlayback, timestampToX]);

  // Handle scroll - disable auto-follow when user scrolls manually
  const handleScroll = useCallback(() => {
    isUserScrollingRef.current = true;
    setFollowingPlayback(false);

    // Reset the flag after scrolling stops
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }
    scrollTimeoutRef.current = window.setTimeout(() => {
      isUserScrollingRef.current = false;
    }, 150);
  }, [setFollowingPlayback]);

  // Handle canvas click
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    // Check if click is on a note
    for (const hitbox of noteHitboxesRef.current) {
      if (
        clickX >= hitbox.x &&
        clickX <= hitbox.x + hitbox.width &&
        clickY >= hitbox.y &&
        clickY <= hitbox.y + hitbox.height
      ) {
        selectNote(hitbox.id);
        // Seek video to note timestamp
        if (videoRef.current) {
          videoRef.current.currentTime = hitbox.note.timestamp;
          setCurrentTime(hitbox.note.timestamp);
        }
        return;
      }
    }

    // Click on empty area - deselect and seek
    selectNote(null);
    const clickedTime = (clickX - STRING_LABEL_WIDTH - CANVAS_PADDING) / PIXELS_PER_SECOND;
    if (clickedTime >= 0 && clickedTime <= duration && videoRef.current) {
      videoRef.current.currentTime = clickedTime;
      setCurrentTime(clickedTime);
    }
  }, [selectNote, setCurrentTime, duration, videoRef]);

  // Handle keyboard input
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      // Undo/Redo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        if (e.shiftKey) {
          redo();
        } else {
          undo();
        }
        return;
      }

      // Navigation
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        commitPendingEdit();
        selectAdjacentNote('left');
        return;
      }
      if (e.key === 'ArrowRight' || e.key === 'Tab') {
        e.preventDefault();
        commitPendingEdit();
        selectAdjacentNote('right');
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        commitPendingEdit();
        selectAdjacentNote('up');
        return;
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        commitPendingEdit();
        selectAdjacentNote('down');
        return;
      }

      // Escape - deselect
      if (e.key === 'Escape') {
        e.preventDefault();
        setPendingFretInput('');
        selectNote(null);
        return;
      }

      // Enter - commit and deselect
      if (e.key === 'Enter') {
        e.preventDefault();
        commitPendingEdit();
        selectNote(null);
        return;
      }

      // Delete/Backspace - set to X (muted)
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedNoteId) {
        e.preventDefault();
        setPendingFretInput('');
        updateNoteFret(selectedNoteId, 'X');
        return;
      }

      // Number input (0-9)
      if (selectedNoteId && /^[0-9]$/.test(e.key)) {
        e.preventDefault();
        const newInput = pendingFretInput + e.key;

        // Validate: max fret 24
        const fretValue = parseInt(newInput, 10);
        if (fretValue > 24) {
          // If invalid, start fresh with this digit
          setPendingFretInput(e.key);
        } else {
          setPendingFretInput(newInput);
        }

        // Clear any existing timeout
        if (fretInputTimeoutRef.current) {
          clearTimeout(fretInputTimeoutRef.current);
        }

        // Auto-commit after 500ms of no input
        fretInputTimeoutRef.current = window.setTimeout(() => {
          commitPendingEdit();
        }, 500);

        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    selectedNoteId,
    pendingFretInput,
    selectNote,
    selectAdjacentNote,
    setPendingFretInput,
    commitPendingEdit,
    updateNoteFret,
    undo,
    redo,
  ]);

  // Cleanup timeouts
  useEffect(() => {
    return () => {
      if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
      if (fretInputTimeoutRef.current) clearTimeout(fretInputTimeoutRef.current);
    };
  }, []);

  if (jobStatus !== 'completed' || !tabDocument) {
    return (
      <div className="border border-gray-700 rounded-lg p-8 text-center text-gray-500">
        <p>Upload a video to see the tab here</p>
      </div>
    );
  }

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-700 flex justify-between items-center">
        <div>
          <h2 className="text-lg font-semibold">Tab Editor</h2>
          <div className="text-sm text-gray-400">
            {tabDocument.notes.length} notes | Click note to edit | Arrow keys to navigate
          </div>
        </div>
        <div className="flex gap-4 text-sm">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded" style={{ backgroundColor: COLORS.noteHigh }}></span>
            High
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded" style={{ backgroundColor: COLORS.noteMedium }}></span>
            Medium
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded" style={{ backgroundColor: COLORS.noteLow }}></span>
            Low
          </span>
        </div>
      </div>

      {/* Canvas container with scroll */}
      <div
        ref={containerRef}
        className="overflow-x-auto bg-gray-900"
        onScroll={handleScroll}
      >
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={CANVAS_HEIGHT}
          onClick={handleClick}
          className="cursor-pointer"
          style={{ display: 'block' }}
        />
      </div>

      {/* Footer info */}
      <div className="px-4 py-2 border-t border-gray-700 text-xs text-gray-500 flex justify-between">
        <span>
          Tuning: {tabDocument.tuning?.join(' ') || 'E B G D A E'} | Capo: {tabDocument.capoFret ?? 'None'}
        </span>
        <span>
          {selectedNoteId ? 'Editing - type fret number, Esc to cancel' : 'Click a note to edit'}
        </span>
      </div>
    </div>
  );
}
