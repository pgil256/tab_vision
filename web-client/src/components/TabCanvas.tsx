// tabvision-client/src/components/TabCanvas.tsx
import React, { useRef, useEffect, useCallback, useMemo, useState } from 'react';
import { pitchPreservingFret, useAppStore } from '../store/appStore';
import { TabNote } from '../types/tab';
import { BeatGrid, getBeatGrid } from '../utils/beatGrid';

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
  measureLine: 'rgba(148, 163, 184, 0.28)',
  beatGridLine: 'rgba(148, 163, 184, 0.10)',
  subdivisionLine: 'rgba(148, 163, 184, 0.05)',
  measureText: '#818cf8',
  activeRing: 'rgba(255, 255, 255, 0.9)',
  reviewRing: '#38bdf8',
};

interface NoteHitbox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  note: TabNote;
}

// Geometry shared by the draw helpers below, so each stays a pure
// (ctx, layout, state) function.
interface Layout {
  pps: number;
  canvasWidth: number;
  safeDuration: number;
  zoomLevel: number;
  timestampToX: (t: number) => number;
  stringToY: (s: number) => number;
}

function getNoteColor(note: TabNote, isSelected: boolean): string {
  if (isSelected) return COLORS.noteSelected;
  if (note.fret === 'X') return COLORS.noteMuted;
  if (note.confidenceLevel === 'high') return COLORS.noteHigh;
  if (note.confidenceLevel === 'medium') return COLORS.noteMedium;
  return COLORS.noteLow;
}

function getNoteTextColor(note: TabNote): string {
  if (note.confidenceLevel === 'medium') return COLORS.noteTextDark;
  return COLORS.noteText;
}

function drawStringsAndLabels(ctx: CanvasRenderingContext2D, l: Layout) {
  // Alternating string row backgrounds
  for (let i = 0; i < 6; i++) {
    if (i % 2 === 0) {
      ctx.fillStyle = COLORS.stringLineHighlight;
      const y = TIME_AXIS_HEIGHT + CANVAS_PADDING + i * STRING_HEIGHT - STRING_HEIGHT / 2;
      ctx.fillRect(STRING_LABEL_WIDTH, y, l.canvasWidth - STRING_LABEL_WIDTH, STRING_HEIGHT);
    }
  }

  ctx.strokeStyle = COLORS.stringLine;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i <= 6; i++) {
    const y = l.stringToY(i);
    ctx.moveTo(STRING_LABEL_WIDTH, y);
    ctx.lineTo(l.canvasWidth, y);
  }
  ctx.stroke();

  ctx.fillStyle = COLORS.labelText;
  ctx.font = '600 12px "SF Mono", "Cascadia Code", "Fira Code", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < 6; i++) {
    ctx.fillText(STRING_NAMES[i], STRING_LABEL_WIDTH / 2, l.stringToY(i + 1));
  }
}

/** One batched vertical-line pass: same style, many x positions. */
function strokeVerticals(
  ctx: CanvasRenderingContext2D,
  xs: number[],
  y0: number,
  y1: number,
  style: string,
  width = 1
) {
  if (xs.length === 0) return;
  ctx.strokeStyle = style;
  ctx.lineWidth = width;
  ctx.beginPath();
  for (const x of xs) {
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
  }
  ctx.stroke();
}

function drawTimeGrid(
  ctx: CanvasRenderingContext2D,
  l: Layout,
  beatGrid: BeatGrid | null
) {
  // Seconds tick axis: always drawn for orientation and click-to-seek.
  const majorInterval = l.zoomLevel < 2 ? (l.zoomLevel < 0.5 ? 10 : 5) : 1;
  const minorTicks: number[] = [];
  const majorTicks: number[] = [];
  for (let t = 0; t <= l.safeDuration; t++) {
    (t % majorInterval === 0 ? majorTicks : minorTicks).push(l.timestampToX(t));
  }
  strokeVerticals(ctx, minorTicks, TIME_AXIS_HEIGHT - 4, TIME_AXIS_HEIGHT, COLORS.timeMarker);
  strokeVerticals(ctx, majorTicks, TIME_AXIS_HEIGHT - 8, TIME_AXIS_HEIGHT, COLORS.timeMajorMarker);

  ctx.fillStyle = COLORS.timeText;
  ctx.font = '10px "SF Mono", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let t = 0; t <= l.safeDuration; t++) {
    if (t % majorInterval !== 0 && l.zoomLevel < 2) continue;
    const label =
      t >= 60 ? `${Math.floor(t / 60)}:${(t % 60).toString().padStart(2, '0')}` : `${t}s`;
    ctx.fillText(label, l.timestampToX(t), TIME_AXIS_HEIGHT / 2 - 4);
  }

  if (!beatGrid) {
    // No detected grid (old document or detection failed): legacy faint
    // verticals on the major seconds only.
    const xs: number[] = [];
    for (let t = 0; t <= l.safeDuration; t += majorInterval) {
      xs.push(l.timestampToX(t));
    }
    strokeVerticals(ctx, xs, TIME_AXIS_HEIGHT, CANVAS_HEIGHT, COLORS.beatLine);
    return;
  }

  // Detected grid: measure lines (every beatsPerBar-th tracked beat, first
  // beat treated as a downbeat), beat lines, and zoom-gated interpolated
  // subdivisions (8ths at >= 2x, 16ths at >= 3x). Batched per style — this
  // redraws on every timeupdate.
  const { beatTimes, beatsPerBar } = beatGrid;
  const measureXs: number[] = [];
  const beatXs: number[] = [];
  const subdivisionXs: number[] = [];
  for (let i = 0; i < beatTimes.length; i++) {
    const x = l.timestampToX(beatTimes[i]);
    (i % beatsPerBar === 0 ? measureXs : beatXs).push(x);
    if (l.zoomLevel >= 2 && i < beatTimes.length - 1) {
      const next = beatTimes[i + 1];
      const fractions = l.zoomLevel >= 3 ? [0.25, 0.5, 0.75] : [0.5];
      for (const f of fractions) {
        subdivisionXs.push(l.timestampToX(beatTimes[i] + (next - beatTimes[i]) * f));
      }
    }
  }
  strokeVerticals(ctx, subdivisionXs, TIME_AXIS_HEIGHT, CANVAS_HEIGHT, COLORS.subdivisionLine);
  strokeVerticals(ctx, beatXs, TIME_AXIS_HEIGHT, CANVAS_HEIGHT, COLORS.beatGridLine);
  strokeVerticals(ctx, measureXs, TIME_AXIS_HEIGHT, CANVAS_HEIGHT, COLORS.measureLine);

  // Measure numbers in the lower half of the axis strip.
  ctx.fillStyle = COLORS.measureText;
  ctx.font = '600 9px "SF Mono", monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  for (let m = 0; m < measureXs.length; m++) {
    ctx.fillText(`${m + 1}`, measureXs[m] + 3, TIME_AXIS_HEIGHT - 2);
  }
}

interface NotesState {
  notes: TabNote[];
  selectedNoteId: string | null;
  hoveredNoteId: string | null;
  pendingFretInput: string;
  reviewActive: boolean;
  reviewIds: string[];
  currentTime: number;
  /** Note whose in-flight drag targets an unplayable pitch (red ring). */
  invalidNoteId: string | null;
}

// In-flight mouse drag (M4): preview state only — the store is written once,
// on pointer-up, via applyNoteDrag. `invalid` means the vertical component
// was rejected (pitch unplayable on the target string) and the note is shown
// at its original string with a red ring; the horizontal retime still applies.
interface DragPreview {
  noteId: string;
  timestamp: number;
  string: number;
  fret: number | 'X';
  invalid: boolean;
}

interface DragGesture {
  pointerId: number;
  noteId: string;
  startX: number;
  startY: number;
  orig: { timestamp: number; string: number; fret: number | 'X' };
  dragging: boolean;
}

const DRAG_THRESHOLD_PX = 4;

const NOTE_FONT = 'bold 11px "SF Mono", monospace';

/** Draws duration tails then note glyphs; returns the rebuilt hitboxes. */
function drawNotes(
  ctx: CanvasRenderingContext2D,
  l: Layout,
  s: NotesState
): NoteHitbox[] {
  const hitboxes: NoteHitbox[] = [];

  // Pass 1 — duration tails, under every glyph so long notes read as bars.
  ctx.globalAlpha = 0.35;
  for (const note of s.notes) {
    if (note.fret === 'X') continue;
    if (typeof note.endTime !== 'number' || note.endTime <= note.timestamp) continue;
    const x = l.timestampToX(note.timestamp);
    const xEnd = l.timestampToX(note.endTime);
    const y = l.stringToY(note.string);
    ctx.fillStyle = getNoteColor(note, note.id === s.selectedNoteId);
    ctx.beginPath();
    ctx.roundRect(x, y - 3, Math.max(2, xEnd - x), 6, 3);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Pass 2 — glyphs.
  ctx.font = NOTE_FONT;
  for (const note of s.notes) {
    const x = l.timestampToX(note.timestamp);
    const y = l.stringToY(note.string);
    const isSelected = note.id === s.selectedNoteId;
    const isHovered = note.id === s.hoveredNoteId;
    const fretText = note.fret === 'X' ? 'X' : note.fret.toString();
    const pendingText = isSelected && s.pendingFretInput ? s.pendingFretInput + '_' : null;
    // Two-digit frets get a wider glyph instead of a cramped square.
    const textWidth = ctx.measureText(pendingText ?? fretText).width;
    const w = Math.max(NOTE_SIZE, textWidth + 10);
    // The playhead is crossing this note: brighten it so the eye can follow.
    const activeEnd = Math.max(note.endTime ?? 0, note.timestamp + 0.15);
    const isActive = s.currentTime >= note.timestamp && s.currentTime <= activeEnd;

    hitboxes.push({
      id: note.id,
      x: x - w / 2,
      y: y - NOTE_SIZE / 2,
      width: w,
      height: NOTE_SIZE,
      note,
    });

    const color = getNoteColor(note, isSelected);

    if (isSelected) {
      ctx.shadowColor = COLORS.noteSelectedGlow;
      ctx.shadowBlur = 12;
      ctx.fillStyle = COLORS.noteSelectedBg;
      ctx.beginPath();
      ctx.roundRect(x - w / 2 - 2, y - NOTE_SIZE / 2 - 2, w + 4, NOTE_SIZE + 4, 6);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    ctx.fillStyle = color;
    if (isActive && !isSelected) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
    }
    ctx.beginPath();
    ctx.roundRect(x - w / 2, y - NOTE_SIZE / 2, w, NOTE_SIZE, 5);
    ctx.fill();
    ctx.shadowBlur = 0;

    if (isActive) {
      ctx.strokeStyle = COLORS.activeRing;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else if (isHovered && !isSelected) {
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    if (isSelected) {
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Drag targeting an unplayable pitch: dashed red ring at the original
    // string until the pointer reaches a playable one.
    if (note.id === s.invalidNoteId) {
      ctx.strokeStyle = COLORS.noteLow;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.roundRect(x - w / 2 - 3, y - NOTE_SIZE / 2 - 3, w + 6, NOTE_SIZE + 6, 7);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Edited indicator (top-right dot)
    if (note.isEdited && !isSelected) {
      ctx.fillStyle = COLORS.noteEdited;
      ctx.beginPath();
      ctx.arc(x + w / 2 - 2, y - NOTE_SIZE / 2 + 2, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Review mode: dashed ring around every queued note so the user can see
    // where the review will travel.
    if (s.reviewActive && !isSelected && s.reviewIds.includes(note.id)) {
      ctx.strokeStyle = COLORS.reviewRing;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.roundRect(x - w / 2 - 3, y - NOTE_SIZE / 2 - 3, w + 6, NOTE_SIZE + 6, 7);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Selected note with ranked alternatives: show "pos k/m" under the note
    // so C-cycling has visible state.
    if (isSelected && note.fret !== 'X' && (note.candidates?.length ?? 0) > 1) {
      const candidates = note.candidates!;
      const current = candidates.findIndex(
        c => c.string === note.string && c.fret === note.fret
      );
      ctx.fillStyle = COLORS.reviewRing;
      ctx.font = '600 9px "SF Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(
        `${current === -1 ? '·' : current + 1}/${candidates.length}`,
        x,
        y + NOTE_SIZE / 2 + 3
      );
      ctx.font = NOTE_FONT;
    }

    // Technique indicator
    if (note.fret !== 'X' && note.technique === 'bend') {
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x + w / 2 + 2, y + 2);
      ctx.lineTo(x + w / 2 + 2, y - 4);
      ctx.lineTo(x + w / 2 + 5, y - 1);
      ctx.stroke();
    }

    // Fret number text
    ctx.fillStyle = isSelected ? COLORS.noteText : getNoteTextColor(note);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(pendingText ?? fretText, x, y);
  }

  return hitboxes;
}

function drawPlayhead(ctx: CanvasRenderingContext2D, l: Layout, currentTime: number) {
  if (currentTime < 0) return;
  const indicatorX = l.timestampToX(currentTime);

  const gradient = ctx.createLinearGradient(indicatorX - 8, 0, indicatorX + 8, 0);
  gradient.addColorStop(0, 'transparent');
  gradient.addColorStop(0.5, COLORS.playbackLineGlow);
  gradient.addColorStop(1, 'transparent');
  ctx.fillStyle = gradient;
  ctx.fillRect(indicatorX - 8, TIME_AXIS_HEIGHT, 16, CANVAS_HEIGHT - TIME_AXIS_HEIGHT);

  ctx.strokeStyle = COLORS.playbackLine;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(indicatorX, TIME_AXIS_HEIGHT);
  ctx.lineTo(indicatorX, CANVAS_HEIGHT);
  ctx.stroke();

  ctx.fillStyle = COLORS.playbackLine;
  ctx.beginPath();
  ctx.moveTo(indicatorX - 5, TIME_AXIS_HEIGHT);
  ctx.lineTo(indicatorX + 5, TIME_AXIS_HEIGHT);
  ctx.lineTo(indicatorX, TIME_AXIS_HEIGHT + 6);
  ctx.closePath();
  ctx.fill();
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
  // Timestamp of the last wheel / pointer input on the scroll container. A
  // scroll event only counts as a MANUAL scroll (and detaches follow) when it
  // arrives right after real input — the auto-follow's own scrollTo() also
  // fires scroll events, and treating those as manual made follow-playback
  // cancel itself on its first tick.
  const lastUserInputAtRef = useRef(0);
  const [hoveredNoteId, setHoveredNoteId] = useState<string | null>(null);
  // Drag gesture + preview. The ref is authoritative (pointer-up commits from
  // it); the state mirror only exists to trigger redraws per move event.
  const dragGestureRef = useRef<DragGesture | null>(null);
  const dragPreviewRef = useRef<DragPreview | null>(null);
  const [dragPreview, setDragPreview] = useState<DragPreview | null>(null);

  const {
    tabDocument,
    jobStatus,
    currentTime,
    duration,
    selectedNoteId,
    isFollowingPlayback,
    pendingFretInput,
    zoomLevel,
    reviewActive,
    reviewIds,
    reviewIndex,
    noteFocusNonce,
    selectNote,
    setCurrentTime,
    setFollowingPlayback,
    applyNoteDrag,
    zoomIn,
    zoomOut,
  } = useAppStore();

  // Layout duration MUST be finite (see safeDuration below); computed here
  // because the density scale needs it.
  const docDuration =
    tabDocument && Number.isFinite(tabDocument.duration) && tabDocument.duration > 0
      ? tabDocument.duration
      : 60;

  // Density-aware base scale: fast runs pack same-string glyphs closer than a
  // glyph width at BASE_PPS, so the densest same-string pair sets the base
  // pixels-per-second. Near-simultaneous pairs (<30ms) are ignored — they are
  // detection duplicates, not playable successions — and the boost is capped
  // so the canvas never outgrows the browser's canvas size limit.
  const densityPps = useMemo(() => {
    if (!tabDocument) return BASE_PPS;
    const byString = new Map<number, number[]>();
    for (const n of tabDocument.notes) {
      const times = byString.get(n.string);
      if (times) times.push(n.timestamp);
      else byString.set(n.string, [n.timestamp]);
    }
    let minGap = Infinity;
    for (const times of byString.values()) {
      times.sort((a, b) => a - b);
      for (let i = 1; i < times.length; i++) {
        const gap = times[i] - times[i - 1];
        if (gap > 0.03 && gap < minGap) minGap = gap;
      }
    }
    if (!Number.isFinite(minGap)) return BASE_PPS;
    const neededPx = NOTE_SIZE + 6; // adjacent glyph centers, incl. 2-digit frets
    const maxPps = Math.max(BASE_PPS, 24000 / docDuration);
    return Math.min(maxPps, Math.max(BASE_PPS, neededPx / minGap));
  }, [tabDocument, docDuration]);

  // Zoom-adjusted pixels per second
  const pps = densityPps * zoomLevel;

  // Layout duration MUST be finite. Recorded (MediaRecorder) clips report a
  // video duration of Infinity until the element is seeked — that would make
  // the canvas infinitely wide AND make the time-marker loop below never
  // terminate, freezing the page. Fall back to the tab document's own
  // duration, which is always finite.
  const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : docDuration;

  // Calculate canvas width based on duration
  const canvasWidth = Math.max(
    800,
    safeDuration * pps + STRING_LABEL_WIDTH + CANVAS_PADDING * 2
  );

  // Convert timestamp to X position
  const timestampToX = useCallback((timestamp: number): number => {
    return STRING_LABEL_WIDTH + CANVAS_PADDING + timestamp * pps;
  }, [pps]);

  // Convert string number (1-6) to Y position
  const stringToY = useCallback((stringNum: number): number => {
    return TIME_AXIS_HEIGHT + CANVAS_PADDING + (stringNum - 1) * STRING_HEIGHT + STRING_HEIGHT / 2;
  }, []);

  // Detected beat grid (null on old documents or when detection failed —
  // drawTimeGrid then falls back to the plain seconds grid).
  const beatGrid = useMemo(() => getBeatGrid(tabDocument), [tabDocument]);

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

    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, canvasWidth, CANVAS_HEIGHT);

    const layout: Layout = {
      pps,
      canvasWidth,
      safeDuration,
      zoomLevel,
      timestampToX,
      stringToY,
    };

    drawStringsAndLabels(ctx, layout);
    drawTimeGrid(ctx, layout, beatGrid);

    // Substitute the in-flight drag preview so the note (and its duration
    // tail) follows the pointer without touching the store.
    const preview = dragPreviewRef.current;
    let notes = [...tabDocument.notes].sort((a, b) => a.timestamp - b.timestamp);
    if (preview) {
      notes = notes.map(n => {
        if (n.id !== preview.noteId) return n;
        const delta = preview.timestamp - n.timestamp;
        return {
          ...n,
          timestamp: preview.timestamp,
          endTime: typeof n.endTime === 'number' ? n.endTime + delta : n.endTime,
          string: preview.string as TabNote['string'],
          fret: preview.fret,
        };
      });
    }

    noteHitboxesRef.current = drawNotes(ctx, layout, {
      notes,
      selectedNoteId,
      hoveredNoteId,
      pendingFretInput,
      reviewActive,
      reviewIds,
      currentTime,
      invalidNoteId: preview?.invalid ? preview.noteId : null,
    });
    drawPlayhead(ctx, layout, currentTime);
  }, [
    canvasWidth,
    tabDocument,
    safeDuration,
    currentTime,
    selectedNoteId,
    hoveredNoteId,
    pendingFretInput,
    zoomLevel,
    reviewActive,
    reviewIds,
    beatGrid,
    dragPreview,
    timestampToX,
    stringToY,
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

  // Handle scroll - disable auto-follow when the user scrolls manually.
  // Programmatic scrolls (auto-follow, review centering) fire this too, so
  // only scrolls arriving right after wheel/pointer input detach follow.
  const handleScroll = useCallback(() => {
    if (performance.now() - lastUserInputAtRef.current > 300) return;
    isUserScrollingRef.current = true;
    setFollowingPlayback(false);

    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }
    scrollTimeoutRef.current = window.setTimeout(() => {
      isUserScrollingRef.current = false;
    }, 150);
  }, [setFollowingPlayback]);

  // Handle mouse wheel zoom; plain wheel marks user input for handleScroll.
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      if (e.deltaY < 0) {
        zoomIn();
      } else {
        zoomOut();
      }
      return;
    }
    lastUserInputAtRef.current = performance.now();
  }, [zoomIn, zoomOut]);

  // Pointer input on the scroll area (scrollbar drags, touch pans) also
  // counts as user input for the manual-scroll detection above.
  const markUserInput = useCallback(() => {
    lastUserInputAtRef.current = performance.now();
  }, []);

  // Canvas-space coordinates for a pointer event (same math the old click
  // handler used; DPR cancels out of the hitbox space).
  const canvasCoords = useCallback(
    (e: { clientX: number; clientY: number }): { x: number; y: number } | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      return {
        x: ((e.clientX - rect.left) * canvasWidth) / rect.width,
        y: ((e.clientY - rect.top) * CANVAS_HEIGHT) / rect.height,
      };
    },
    [canvasWidth]
  );

  const hitTest = useCallback((x: number, y: number): NoteHitbox | null => {
    for (const hitbox of noteHitboxesRef.current) {
      if (
        x >= hitbox.x &&
        x <= hitbox.x + hitbox.width &&
        y >= hitbox.y &&
        y <= hitbox.y + hitbox.height
      ) {
        return hitbox;
      }
    }
    return null;
  }, []);

  const clearDrag = useCallback(() => {
    dragGestureRef.current = null;
    dragPreviewRef.current = null;
    setDragPreview(null);
  }, []);

  // Pointer state machine (M4): press on a note arms a drag; moving past the
  // threshold previews retime (horizontal) + pitch-preserving restring
  // (vertical); release commits once via applyNoteDrag. A press that never
  // crosses the threshold is the old click (select / seek). Drag is disabled
  // in review mode — review owns navigation and scroll.
  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (e.button !== 0) return;
      const coords = canvasCoords(e);
      if (!coords) return;
      const hit = hitTest(coords.x, coords.y);
      if (!hit || reviewActive || hit.note.fret === 'X') return;
      dragGestureRef.current = {
        pointerId: e.pointerId,
        noteId: hit.id,
        startX: coords.x,
        startY: coords.y,
        orig: {
          timestamp: hit.note.timestamp,
          string: hit.note.string,
          fret: hit.note.fret,
        },
        dragging: false,
      };
      try {
        e.currentTarget.setPointerCapture(e.pointerId);
      } catch {
        // Untracked pointer id (synthetic events); drag still works while the
        // pointer stays over the canvas.
      }
    },
    [canvasCoords, hitTest, reviewActive]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const coords = canvasCoords(e);
      if (!coords) return;
      const gesture = dragGestureRef.current;

      if (!gesture) {
        // Plain hover
        const hit = hitTest(coords.x, coords.y);
        const id = hit?.id ?? null;
        if (id !== hoveredNoteId) setHoveredNoteId(id);
        return;
      }

      if (!gesture.dragging) {
        const dist = Math.hypot(coords.x - gesture.startX, coords.y - gesture.startY);
        if (dist < DRAG_THRESHOLD_PX) return;
        gesture.dragging = true;
      }

      const dt = (coords.x - gesture.startX) / pps;
      const timestamp = Math.max(0, Math.min(safeDuration, gesture.orig.timestamp + dt));
      const stringDelta = Math.round((coords.y - gesture.startY) / STRING_HEIGHT);
      const targetString = Math.max(1, Math.min(6, gesture.orig.string + stringDelta));

      let string = gesture.orig.string;
      let fret = gesture.orig.fret;
      let invalid = false;
      if (targetString !== gesture.orig.string) {
        const nextFret = pitchPreservingFret(
          gesture.orig.string,
          targetString,
          gesture.orig.fret,
          useAppStore.getState().tabDocument?.tuningMidi,
        );
        if (nextFret === null) {
          invalid = true; // vertical rejected, horizontal still applies
        } else {
          string = targetString;
          fret = nextFret;
        }
      }

      const preview: DragPreview = { noteId: gesture.noteId, timestamp, string, fret, invalid };
      dragPreviewRef.current = preview;
      setDragPreview(preview);
    },
    [canvasCoords, hitTest, hoveredNoteId, pps, safeDuration]
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const coords = canvasCoords(e);
      const gesture = dragGestureRef.current;
      const preview = dragPreviewRef.current;

      if (gesture?.dragging && preview) {
        applyNoteDrag(gesture.noteId, {
          timestamp: preview.timestamp,
          string: preview.string,
          fret: preview.fret,
        });
        selectNote(gesture.noteId);
        clearDrag();
        return;
      }
      clearDrag();

      // Click semantics (unchanged): select + seek on a note, deselect + seek
      // on empty canvas.
      if (!coords) return;
      const hit = hitTest(coords.x, coords.y);
      if (hit) {
        selectNote(hit.id);
        if (videoRef.current) {
          videoRef.current.currentTime = hit.note.timestamp;
        }
        setCurrentTime(hit.note.timestamp);
        return;
      }
      selectNote(null);
      const clickedTime = (coords.x - STRING_LABEL_WIDTH - CANVAS_PADDING) / pps;
      if (clickedTime >= 0 && clickedTime <= safeDuration) {
        if (videoRef.current) {
          videoRef.current.currentTime = clickedTime;
        }
        setCurrentTime(clickedTime);
      }
    },
    [
      applyNoteDrag,
      canvasCoords,
      clearDrag,
      hitTest,
      pps,
      safeDuration,
      selectNote,
      setCurrentTime,
      videoRef,
    ]
  );

  // Keyboard handling lives in hooks/useEditorHotkeys.ts (M7), mounted once
  // in App — one window listener for the whole editor.

  // Review mode: keep the note under review in view. Runs on queue-step (and
  // on entering review); manual scrolling stays free between steps.
  useEffect(() => {
    if (!reviewActive || !selectedNoteId || !containerRef.current || !tabDocument) return;
    const note = tabDocument.notes.find(n => n.id === selectedNoteId);
    if (!note) return;
    const container = containerRef.current;
    const noteX = timestampToX(note.timestamp);
    container.scrollTo({
      left: Math.max(0, noteX - container.clientWidth / 2),
      behavior: 'smooth',
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reviewActive, reviewIndex, selectedNoteId]);

  // Toolbar pill jump: center the newly focused note whenever the nonce
  // bumps. Nonce-keyed (not selection-keyed) so ordinary clicks/arrow moves
  // never yank the scroll position.
  useEffect(() => {
    if (!noteFocusNonce || !selectedNoteId || !containerRef.current || !tabDocument) return;
    const note = tabDocument.notes.find(n => n.id === selectedNoteId);
    if (!note) return;
    const container = containerRef.current;
    container.scrollTo({
      left: Math.max(0, timestampToX(note.timestamp) - container.clientWidth / 2),
      behavior: 'smooth',
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [noteFocusNonce]);

  // Cleanup timeouts
  useEffect(() => {
    return () => {
      if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
    };
  }, []);

  if (jobStatus !== 'completed' || !tabDocument) {
    return null;
  }

  const reviewNote =
    reviewActive && selectedNoteId
      ? tabDocument.notes.find(n => n.id === selectedNoteId)
      : undefined;
  const reviewAlternatives = reviewNote?.candidates?.length ?? 0;

  return (
    <div
      className="h-full flex flex-col"
      style={{ background: COLORS.background }}
    >
      {reviewActive && (
        <div
          data-testid="review-banner"
          className="flex items-center gap-3 px-3 py-1.5 text-xs font-medium"
          style={{ background: 'rgba(56, 189, 248, 0.12)', color: '#38bdf8' }}
        >
          <span>
            Review {Math.min(reviewIndex + 1, reviewIds.length)}/{reviewIds.length} — lowest-confidence notes
          </span>
          <span style={{ color: '#7dd3fc' }}>
            {reviewAlternatives > 1
              ? `C cycle position (${reviewAlternatives} options) · Shift+C back`
              : 'no ranked alternatives for this note — edit directly or N to skip'}
          </span>
          <span className="ml-auto" style={{ color: '#7dd3fc' }}>
            N next · P previous · R/Esc done
          </span>
        </div>
      )}
      {/* Scrollable canvas */}
      <div
        ref={containerRef}
        className="flex-1 overflow-x-auto overflow-y-hidden"
        onScroll={handleScroll}
        onWheel={handleWheel}
        // Scrollbar drags hit the container itself (canvas clicks hit the
        // canvas and must NOT detach follow — they seek, and follow should
        // keep tracking from the new position).
        onPointerDown={(e) => {
          if (e.target === containerRef.current) markUserInput();
        }}
      >
        <canvas
          ref={canvasRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          // pointercancel on the canvas means the browser took the pointer
          // for a native touch pan — that is a manual scroll.
          onPointerCancel={() => {
            clearDrag();
            markUserInput();
          }}
          onPointerLeave={() => setHoveredNoteId(null)}
          className="block"
          style={{
            width: `${canvasWidth}px`,
            height: `${CANVAS_HEIGHT}px`,
            cursor: dragPreview
              ? 'grabbing'
              : hoveredNoteId && !reviewActive
                ? 'grab'
                : 'pointer',
            touchAction: 'pan-x',
          }}
        />
      </div>
    </div>
  );
}
