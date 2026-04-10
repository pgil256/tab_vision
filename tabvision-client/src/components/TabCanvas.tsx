// tabvision-client/src/components/TabCanvas.tsx
import React, { useRef, useEffect, useCallback, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { TabNote } from '../types/tab';

// Base layout constants (scaled by zoom)
const BASE_PPS = 60; // pixels per second
const STRING_HEIGHT = 28;
const TIME_AXIS_HEIGHT = 28;
const STRING_LABEL_WIDTH = 28;
const CANVAS_PADDING = 12;
const NOTE_SIZE = 22;
const CANVAS_HEIGHT = TIME_AXIS_HEIGHT + (STRING_HEIGHT * 6) + CANVAS_PADDING * 2;

const STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E'];

// Design system colors
const COLORS = {
  background: '#0b0e14',
  surface: '#131620',
  stringLine: 'rgba(148, 163, 184, 0.12)',
  stringLineHighlight: 'rgba(148, 163, 184, 0.06)',
  timeMarker: 'rgba(148, 163, 184, 0.15)',
  timeMajorMarker: 'rgba(148, 163, 184, 0.25)',
  timeText: '#64748b',
  labelText: '#94a3b8',
  noteHigh: '#10b981',
  noteHighBg: 'rgba(16, 185, 129, 0.15)',
  noteMedium: '#f59e0b',
  noteMediumBg: 'rgba(245, 158, 11, 0.15)',
  noteLow: '#f43f5e',
  noteLowBg: 'rgba(244, 63, 94, 0.15)',
  noteSelected: '#6366f1',
  noteSelectedBg: 'rgba(99, 102, 241, 0.2)',
  noteSelectedGlow: 'rgba(99, 102, 241, 0.4)',
  noteEdited: '#ffffff',
  noteMuted: '#94a3b8',
  playbackLine: '#6366f1',
  playbackLineGlow: 'rgba(99, 102, 241, 0.3)',
  noteText: '#ffffff',
  noteTextDark: '#1a1a1a',
  beatLine: 'rgba(99, 102, 241, 0.06)',
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
  const [hoveredNoteId, setHoveredNoteId] = useState<string | null>(null);

  const {
    tabDocument,
    jobStatus,
    currentTime,
    duration,
    selectedNoteId,
    isFollowingPlayback,
    pendingFretInput,
    zoomLevel,
    selectNote,
    selectAdjacentNote,
    setCurrentTime,
    setFollowingPlayback,
    setPendingFretInput,
    commitPendingEdit,
    updateNoteFret,
    undo,
    redo,
    zoomIn,
    zoomOut,
  } = useAppStore();

  // Zoom-adjusted pixels per second
  const pps = BASE_PPS * zoomLevel;

  // Calculate canvas width based on duration
  const canvasWidth = Math.max(
    800,
    duration > 0 ? duration * pps + STRING_LABEL_WIDTH + CANVAS_PADDING * 2 : 800
  );

  // Get note color
  const getNoteColor = useCallback((note: TabNote, isSelected: boolean): string => {
    if (isSelected) return COLORS.noteSelected;
    if (note.fret === 'X') return COLORS.noteMuted;
    if (note.confidenceLevel === 'high') return COLORS.noteHigh;
    if (note.confidenceLevel === 'medium') return COLORS.noteMedium;
    return COLORS.noteLow;
  }, []);

  const getNoteBgColor = useCallback((note: TabNote, isSelected: boolean): string => {
    if (isSelected) return COLORS.noteSelectedBg;
    if (note.confidenceLevel === 'high') return COLORS.noteHighBg;
    if (note.confidenceLevel === 'medium') return COLORS.noteMediumBg;
    return COLORS.noteLowBg;
  }, []);

  const getNoteTextColor = useCallback((note: TabNote): string => {
    if (note.confidenceLevel === 'medium') return COLORS.noteTextDark;
    return COLORS.noteText;
  }, []);

  // Convert timestamp to X position
  const timestampToX = useCallback((timestamp: number): number => {
    return STRING_LABEL_WIDTH + CANVAS_PADDING + timestamp * pps;
  }, [pps]);

  // Convert string number (1-6) to Y position
  const stringToY = useCallback((stringNum: number): number => {
    return TIME_AXIS_HEIGHT + CANVAS_PADDING + (stringNum - 1) * STRING_HEIGHT + STRING_HEIGHT / 2;
  }, []);

  // Draw the canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !tabDocument) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    canvas.style.width = `${canvasWidth}px`;
    canvas.style.height = `${CANVAS_HEIGHT}px`;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, canvasWidth, CANVAS_HEIGHT);

    // Draw alternating string row backgrounds
    for (let i = 0; i < 6; i++) {
      if (i % 2 === 0) {
        ctx.fillStyle = COLORS.stringLineHighlight;
        const y = TIME_AXIS_HEIGHT + CANVAS_PADDING + i * STRING_HEIGHT - STRING_HEIGHT / 2;
        ctx.fillRect(STRING_LABEL_WIDTH, y, canvasWidth - STRING_LABEL_WIDTH, STRING_HEIGHT);
      }
    }

    // Draw string lines
    ctx.strokeStyle = COLORS.stringLine;
    ctx.lineWidth = 1;
    for (let i = 1; i <= 6; i++) {
      const y = stringToY(i);
      ctx.beginPath();
      ctx.moveTo(STRING_LABEL_WIDTH, y);
      ctx.lineTo(canvasWidth, y);
      ctx.stroke();
    }

    // Draw string labels
    ctx.fillStyle = COLORS.labelText;
    ctx.font = '600 12px "SF Mono", "Cascadia Code", "Fira Code", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < 6; i++) {
      const y = stringToY(i + 1);
      ctx.fillText(STRING_NAMES[i], STRING_LABEL_WIDTH / 2, y);
    }

    // Draw time markers
    const maxTime = duration > 0 ? duration : (tabDocument.duration || 60);
    const majorInterval = zoomLevel < 0.5 ? 10 : zoomLevel < 1 ? 5 : zoomLevel < 2 ? 5 : 1;

    for (let t = 0; t <= maxTime; t++) {
      const x = timestampToX(t);
      const isMajor = t % majorInterval === 0;

      // Tick marks
      ctx.strokeStyle = isMajor ? COLORS.timeMajorMarker : COLORS.timeMarker;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, TIME_AXIS_HEIGHT - (isMajor ? 8 : 4));
      ctx.lineTo(x, TIME_AXIS_HEIGHT);
      ctx.stroke();

      // Vertical beat grid lines
      if (isMajor) {
        ctx.strokeStyle = COLORS.beatLine;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, TIME_AXIS_HEIGHT);
        ctx.lineTo(x, CANVAS_HEIGHT);
        ctx.stroke();
      }

      // Time labels
      if (isMajor || (zoomLevel >= 2 && t % 1 === 0)) {
        ctx.fillStyle = COLORS.timeText;
        ctx.font = '10px "SF Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const label = t >= 60 ? `${Math.floor(t / 60)}:${(t % 60).toString().padStart(2, '0')}` : `${t}s`;
        ctx.fillText(label, x, TIME_AXIS_HEIGHT / 2);
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
      const isHovered = note.id === hoveredNoteId;

      // Store hitbox
      noteHitboxesRef.current.push({
        id: note.id,
        x: x - NOTE_SIZE / 2,
        y: y - NOTE_SIZE / 2,
        width: NOTE_SIZE,
        height: NOTE_SIZE,
        note,
      });

      const color = getNoteColor(note, isSelected);

      // Draw glow effect for selected note
      if (isSelected) {
        ctx.shadowColor = COLORS.noteSelectedGlow;
        ctx.shadowBlur = 12;
        ctx.fillStyle = COLORS.noteSelectedBg;
        ctx.beginPath();
        ctx.roundRect(x - NOTE_SIZE / 2 - 2, y - NOTE_SIZE / 2 - 2, NOTE_SIZE + 4, NOTE_SIZE + 4, 6);
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // Draw note background
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x - NOTE_SIZE / 2, y - NOTE_SIZE / 2, NOTE_SIZE, NOTE_SIZE, 5);
      ctx.fill();

      // Draw hover effect
      if (isHovered && !isSelected) {
        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Draw selection ring
      if (isSelected) {
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw edited indicator (top-right dot)
      if (note.isEdited && !isSelected) {
        ctx.fillStyle = COLORS.noteEdited;
        ctx.beginPath();
        ctx.arc(x + NOTE_SIZE / 2 - 2, y - NOTE_SIZE / 2 + 2, 2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Draw technique indicator
      if (note.fret !== 'X' && note.technique) {
        const technique = note.technique;
        if (technique === 'bend') {
          // Small bend arrow
          ctx.strokeStyle = 'rgba(255,255,255,0.6)';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(x + NOTE_SIZE / 2 + 2, y + 2);
          ctx.lineTo(x + NOTE_SIZE / 2 + 2, y - 4);
          ctx.lineTo(x + NOTE_SIZE / 2 + 5, y - 1);
          ctx.stroke();
        }
      }

      // Draw fret number text
      const fretText = note.fret === 'X' ? 'X' : note.fret.toString();
      ctx.fillStyle = isSelected ? COLORS.noteText : getNoteTextColor(note);
      ctx.font = 'bold 11px "SF Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // Show pending input if selected
      if (isSelected && pendingFretInput) {
        ctx.fillStyle = COLORS.noteText;
        ctx.fillText(pendingFretInput + '_', x, y);
      } else {
        ctx.fillText(fretText, x, y);
      }
    }

    // Draw playback indicator
    if (currentTime >= 0) {
      const indicatorX = timestampToX(currentTime);

      // Glow effect
      const gradient = ctx.createLinearGradient(indicatorX - 8, 0, indicatorX + 8, 0);
      gradient.addColorStop(0, 'transparent');
      gradient.addColorStop(0.5, COLORS.playbackLineGlow);
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.fillRect(indicatorX - 8, TIME_AXIS_HEIGHT, 16, CANVAS_HEIGHT - TIME_AXIS_HEIGHT);

      // Line
      ctx.strokeStyle = COLORS.playbackLine;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(indicatorX, TIME_AXIS_HEIGHT);
      ctx.lineTo(indicatorX, CANVAS_HEIGHT);
      ctx.stroke();

      // Top triangle
      ctx.fillStyle = COLORS.playbackLine;
      ctx.beginPath();
      ctx.moveTo(indicatorX - 5, TIME_AXIS_HEIGHT);
      ctx.lineTo(indicatorX + 5, TIME_AXIS_HEIGHT);
      ctx.lineTo(indicatorX, TIME_AXIS_HEIGHT + 6);
      ctx.closePath();
      ctx.fill();
    }
  }, [
    canvasWidth,
    tabDocument,
    duration,
    currentTime,
    selectedNoteId,
    hoveredNoteId,
    pendingFretInput,
    zoomLevel,
    timestampToX,
    stringToY,
    getNoteColor,
    getNoteBgColor,
    getNoteTextColor,
    pps,
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
    const targetScrollLeft = indicatorX - viewportWidth * 0.3;

    container.scrollTo({
      left: Math.max(0, targetScrollLeft),
      behavior: 'smooth',
    });
  }, [currentTime, isFollowingPlayback, timestampToX]);

  // Handle scroll - disable auto-follow when user scrolls manually
  const handleScroll = useCallback(() => {
    isUserScrollingRef.current = true;
    setFollowingPlayback(false);

    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }
    scrollTimeoutRef.current = window.setTimeout(() => {
      isUserScrollingRef.current = false;
    }, 150);
  }, [setFollowingPlayback]);

  // Handle mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      if (e.deltaY < 0) {
        zoomIn();
      } else {
        zoomOut();
      }
    }
  }, [zoomIn, zoomOut]);

  // Handle canvas click
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const scaleX = (canvasWidth * dpr) / rect.width;
    const scaleY = (CANVAS_HEIGHT * dpr) / rect.height;
    const clickX = (e.clientX - rect.left) * scaleX / dpr;
    const clickY = (e.clientY - rect.top) * scaleY / dpr;

    // Check if click is on a note
    for (const hitbox of noteHitboxesRef.current) {
      if (
        clickX >= hitbox.x &&
        clickX <= hitbox.x + hitbox.width &&
        clickY >= hitbox.y &&
        clickY <= hitbox.y + hitbox.height
      ) {
        selectNote(hitbox.id);
        if (videoRef.current) {
          videoRef.current.currentTime = hitbox.note.timestamp;
          setCurrentTime(hitbox.note.timestamp);
        }
        return;
      }
    }

    // Click on empty area - deselect and seek
    selectNote(null);
    const clickedTime = (clickX - STRING_LABEL_WIDTH - CANVAS_PADDING) / pps;
    if (clickedTime >= 0 && clickedTime <= duration && videoRef.current) {
      videoRef.current.currentTime = clickedTime;
      setCurrentTime(clickedTime);
    }
  }, [selectNote, setCurrentTime, duration, videoRef, canvasWidth, pps]);

  // Handle mouse move for hover effects
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const scaleX = (canvasWidth * dpr) / rect.width;
    const scaleY = (CANVAS_HEIGHT * dpr) / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX / dpr;
    const mouseY = (e.clientY - rect.top) * scaleY / dpr;

    let found = false;
    for (const hitbox of noteHitboxesRef.current) {
      if (
        mouseX >= hitbox.x &&
        mouseX <= hitbox.x + hitbox.width &&
        mouseY >= hitbox.y &&
        mouseY <= hitbox.y + hitbox.height
      ) {
        if (hoveredNoteId !== hitbox.id) {
          setHoveredNoteId(hitbox.id);
        }
        found = true;
        break;
      }
    }
    if (!found && hoveredNoteId) {
      setHoveredNoteId(null);
    }
  }, [hoveredNoteId, canvasWidth]);

  // Handle keyboard input
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
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

      // Zoom shortcuts
      if ((e.ctrlKey || e.metaKey) && (e.key === '=' || e.key === '+')) {
        e.preventDefault();
        zoomIn();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key === '-') {
        e.preventDefault();
        zoomOut();
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

      // Escape
      if (e.key === 'Escape') {
        e.preventDefault();
        setPendingFretInput('');
        selectNote(null);
        return;
      }

      // Enter
      if (e.key === 'Enter') {
        e.preventDefault();
        commitPendingEdit();
        selectNote(null);
        return;
      }

      // Delete/Backspace -> muted
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
        const fretValue = parseInt(newInput, 10);

        if (fretValue > 24) {
          setPendingFretInput(e.key);
        } else {
          setPendingFretInput(newInput);
        }

        if (fretInputTimeoutRef.current) {
          clearTimeout(fretInputTimeoutRef.current);
        }
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
    zoomIn,
    zoomOut,
  ]);

  // Cleanup timeouts
  useEffect(() => {
    return () => {
      if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
      if (fretInputTimeoutRef.current) clearTimeout(fretInputTimeoutRef.current);
    };
  }, []);

  if (jobStatus !== 'completed' || !tabDocument) {
    return null;
  }

  return (
    <div
      className="h-full flex flex-col"
      style={{ background: COLORS.background }}
    >
      {/* Scrollable canvas */}
      <div
        ref={containerRef}
        className="flex-1 overflow-x-auto overflow-y-hidden"
        onScroll={handleScroll}
        onWheel={handleWheel}
      >
        <canvas
          ref={canvasRef}
          onClick={handleClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredNoteId(null)}
          className="cursor-pointer block"
          style={{
            width: `${canvasWidth}px`,
            height: `${CANVAS_HEIGHT}px`,
          }}
        />
      </div>
    </div>
  );
}
