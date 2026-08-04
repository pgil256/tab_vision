// Score view: the tab document rendered as sheet music (white page, systems
// of measures, MuseScore-style playback cursor) instead of the timeline grid.
//
// Playback is unchanged — the audition engine keeps driving currentTime
// (video clock or internal synth transport), and this view just maps time →
// (system, x). Printing uses the browser's print-to-PDF: index.css hides the
// app chrome and lets the page column flow across paper pages.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { TabNote } from '../types/tab';
import { audition } from '../utils/auditionEngine';
import { getBeatGrid } from '../utils/beatGrid';
import {
  describeSelection,
  describeTablature,
  isFlaggedNote,
} from '../utils/noteAccessibility';
import {
  bendLabel,
  isBend,
  isSlide,
  previousNoteById,
  slideDirection,
} from '../utils/noteTechniques';
import {
  ScoreSystem,
  buildMeasures,
  buildSystems,
  systemIndexAt,
} from '../utils/scoreLayout';

// SVG viewBox geometry (rendered at 100% width of the page column, so screen
// pixels track these units closely).
const VIEW_W = 800;
const LABEL_W = 34; // string-letter gutter at the left of each system
const CONTENT_W = VIEW_W - LABEL_W - 6;
const LINE_GAP = 14;
const STAFF_TOP = 30; // room for the measure number above the staff
const STAFF_H = LINE_GAP * 5;
const SYSTEM_H = STAFF_TOP + STAFF_H + 22;

const STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E'];

// Ink on paper; the accent matches the app's indigo for cursor + selection.
const INK = '#1a1a1a';
const INK_FAINT = '#8a8a8a';
const PAPER = '#ffffff';
const ACCENT = '#4f46e5';
const CURSOR_FILL = 'rgba(255, 112, 72, 0.18)';

interface PlacedNote {
  note: TabNote;
  x: number;
  y: number;
  /** Glyph halo width (two-digit frets are wider). */
  w: number;
  /** Same-system source geometry for an incoming slide. */
  slideFrom?: { x: number; y: number; w: number };
  slideDirection?: -1 | 0 | 1;
}

function stringY(stringNum: number): number {
  return STAFF_TOP + (stringNum - 1) * LINE_GAP;
}

/** Time → x inside one system: equal-width measures, proportional within. */
function timeToX(system: ScoreSystem, t: number): number {
  const measureW = CONTENT_W / system.measures.length;
  for (let i = 0; i < system.measures.length; i++) {
    const m = system.measures[i];
    if (t <= m.end || i === system.measures.length - 1) {
      const frac = Math.max(0, Math.min(1, (t - m.start) / (m.end - m.start)));
      // Inset note placement so glyphs don't sit on the barlines.
      return LABEL_W + measureW * i + 10 + frac * (measureW - 20);
    }
  }
  return LABEL_W;
}

/** Inverse of timeToX for click-to-seek. */
function xToTime(system: ScoreSystem, x: number): number {
  const measureW = CONTENT_W / system.measures.length;
  const i = Math.max(
    0,
    Math.min(system.measures.length - 1, Math.floor((x - LABEL_W) / measureW))
  );
  const m = system.measures[i];
  const frac = Math.max(0, Math.min(1, (x - LABEL_W - measureW * i - 10) / (measureW - 20)));
  return m.start + frac * (m.end - m.start);
}

interface SystemProps {
  system: ScoreSystem;
  notes: PlacedNote[];
  systemIndex: number;
  systemCount: number;
  selectedNoteIds: string[];
  /** currentTime when the playhead is inside this system, else -1 (keeps
   * React.memo effective: only the active system re-renders per tick). */
  cursorTime: number;
  onSeek: (t: number) => void;
  onSelect: (note: TabNote, mode: 'replace' | 'toggle' | 'range') => void;
}

const System = React.memo(function System({
  system,
  notes,
  systemIndex,
  systemCount,
  selectedNoteIds,
  cursorTime,
  onSeek,
  onSelect,
}: SystemProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const measureW = CONTENT_W / system.measures.length;

  const handleClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const x = ((e.clientX - rect.left) * VIEW_W) / rect.width;
      const y = ((e.clientY - rect.top) * SYSTEM_H) / rect.height;
      // Note hit first, then plain seek.
      for (const p of notes) {
        if (Math.abs(x - p.x) <= Math.max(10, p.w / 2 + 3) && Math.abs(y - p.y) <= 8) {
          onSelect(
            p.note,
            e.shiftKey ? 'range' : (e.ctrlKey || e.metaKey ? 'toggle' : 'replace'),
          );
          return;
        }
      }
      onSeek(xToTime(system, x));
    },
    [notes, onSeek, onSelect, system]
  );

  const cursorX = cursorTime >= 0 ? timeToX(system, cursorTime) : null;
  const selectedIds = new Set(selectedNoteIds);
  const firstMeasure = system.measures[0]?.number ?? 1;
  const lastMeasure = system.measures[system.measures.length - 1]?.number ?? firstMeasure;
  const flaggedCount = notes.reduce(
    (count, placed) => count + (isFlaggedNote(placed.note) ? 1 : 0),
    0,
  );
  const systemSummary = `Score system ${systemIndex + 1} of ${systemCount}, measures ${firstMeasure} through ${lastMeasure}, ${notes.length} notes, ${flaggedCount} flagged for review.`;

  return (
    <svg
      ref={svgRef}
      className="score-system block w-full"
      viewBox={`0 0 ${VIEW_W} ${SYSTEM_H}`}
      onClick={handleClick}
      role="img"
      aria-label={systemSummary}
      style={{ cursor: 'pointer' }}
    >
      <title>{systemSummary}</title>
      {/* Measure number at the system start */}
      <text
        x={LABEL_W}
        y={STAFF_TOP - 12}
        fontSize={11}
        fontFamily="Georgia, 'Times New Roman', serif"
        fontStyle="italic"
        fill={INK_FAINT}
      >
        {system.measures[0].number}
      </text>

      {/* String letters */}
      {STRING_NAMES.map((name, i) => (
        <text
          key={name + i}
          x={LABEL_W - 12}
          y={stringY(i + 1) + 3.5}
          fontSize={11}
          fontFamily="'SF Mono', 'Cascadia Code', monospace"
          textAnchor="middle"
          fill={INK_FAINT}
        >
          {name}
        </text>
      ))}

      {/* Staff lines */}
      {Array.from({ length: 6 }, (_, i) => (
        <line
          key={i}
          x1={LABEL_W}
          y1={stringY(i + 1)}
          x2={LABEL_W + CONTENT_W}
          y2={stringY(i + 1)}
          stroke={INK}
          strokeWidth={0.7}
        />
      ))}

      {/* Barlines (including system start/end) */}
      {Array.from({ length: system.measures.length + 1 }, (_, i) => (
        <line
          key={i}
          x1={LABEL_W + measureW * i}
          y1={stringY(1)}
          x2={LABEL_W + measureW * i}
          y2={stringY(6)}
          stroke={INK}
          strokeWidth={i === 0 || i === system.measures.length ? 1.4 : 0.8}
        />
      ))}

      {/* Slides: a conventional slash from the prior same-string fret into
          the marked destination. A compact slash is retained at a system
          break where the source note lives on the previous line. */}
      {notes.map(p => {
        if (!isSlide(p.note)) return null;
        const rising = (p.slideDirection ?? 0) >= 0;
        const endX = p.x - p.w / 2 - 2;
        const proposedStart = p.slideFrom
          ? p.slideFrom.x + p.slideFrom.w / 2 + 2
          : endX - 8;
        const startX = endX - proposedStart < 5 ? endX - 8 : proposedStart;
        return (
          <line
            key={`slide-${p.note.id}`}
            data-technique="slide"
            x1={startX}
            y1={p.y + (rising ? 4 : -4)}
            x2={endX}
            y2={p.y + (rising ? -4 : 4)}
            stroke={INK}
            strokeWidth={1.2}
            strokeLinecap="round"
          />
        );
      })}

      {/* Playback cursor (screen only; hidden in print via CSS) */}
      {cursorX != null && (
        <g className="score-cursor">
          <rect
            x={cursorX - 9}
            y={STAFF_TOP - 8}
            width={18}
            height={STAFF_H + 16}
            rx={3}
            fill={CURSOR_FILL}
          />
          <line
            x1={cursorX}
            y1={STAFF_TOP - 8}
            x2={cursorX}
            y2={STAFF_TOP + STAFF_H + 8}
            stroke={ACCENT}
            strokeWidth={1.5}
          />
        </g>
      )}

      {/* Notes: white halo so the number sits "on" the line, then the glyph. */}
      {notes.map(p => {
        const isSelected = selectedIds.has(p.note.id);
        const activeEnd = Math.max(p.note.endTime ?? 0, p.note.timestamp + 0.15);
        const isActive =
          cursorTime >= 0 && cursorTime >= p.note.timestamp && cursorTime <= activeEnd;
        const fill = isSelected || isActive ? ACCENT : INK;
        return (
          <g key={p.note.id}>
            <rect
              x={p.x - p.w / 2}
              y={p.y - 6}
              width={p.w}
              height={12}
              fill={PAPER}
            />
            {isSelected && (
              <rect
                x={p.x - p.w / 2 - 2}
                y={p.y - 8}
                width={p.w + 4}
                height={16}
                rx={3}
                fill="none"
                stroke={ACCENT}
                strokeWidth={1}
              />
            )}
            {isBend(p.note) && p.note.fret !== 'X' ? (
              <g data-technique="bend">
                <path
                  d={`M ${p.x + p.w / 2 + 1} ${p.y + 2} Q ${p.x + p.w / 2 + 2} ${p.y - 6} ${p.x + p.w / 2 + 9} ${p.y - 10}`}
                  fill="none"
                  stroke={fill}
                  strokeWidth={1.1}
                  strokeLinecap="round"
                />
                <path
                  d={`M ${p.x + p.w / 2 + 9} ${p.y - 10} l -4 1 m 4 -1 l -1 4`}
                  fill="none"
                  stroke={fill}
                  strokeWidth={1.1}
                  strokeLinecap="round"
                />
                <text
                  x={p.x + p.w / 2 + 12}
                  y={p.y - 8}
                  fontSize={11}
                  fontFamily="'SF Mono', 'Cascadia Code', monospace"
                  fill={fill}
                >
                  {bendLabel(p.note)}
                </text>
              </g>
            ) : null}
            <text
              x={p.x}
              y={p.y + 3.5}
              fontSize={11}
              fontWeight={isActive || isSelected ? 700 : 600}
              fontFamily="'SF Mono', 'Cascadia Code', monospace"
              textAnchor="middle"
              fill={fill}
            >
              {p.note.fret === 'X' ? 'x' : p.note.fret}
            </text>
          </g>
        );
      })}
    </svg>
  );
});

export function ScoreView() {
  const tabDocument = useAppStore(s => s.tabDocument);
  const jobStatus = useAppStore(s => s.jobStatus);
  const currentTime = useAppStore(s => s.currentTime);
  const duration = useAppStore(s => s.duration);
  const isPlaying = useAppStore(s => s.isPlaying);
  const isFollowingPlayback = useAppStore(s => s.isFollowingPlayback);
  const selectedNoteId = useAppStore(s => s.selectedNoteId);
  const selectedNoteIds = useAppStore(s => s.selectedNoteIds);
  const noteFocusNonce = useAppStore(s => s.noteFocusNonce);
  const selectNote = useAppStore(s => s.selectNote);
  const toggleNoteSelection = useAppStore(s => s.toggleNoteSelection);
  const selectNoteRange = useAppStore(s => s.selectNoteRange);
  const setCurrentTime = useAppStore(s => s.setCurrentTime);
  const setFollowingPlayback = useAppStore(s => s.setFollowingPlayback);
  const setDocumentTitle = useAppStore(s => s.setDocumentTitle);

  // Title editing: view text swaps to an input on click; blur/Enter commits,
  // Esc cancels. The global hotkey dispatcher already ignores inputs.
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState('');

  const scrollRef = useRef<HTMLDivElement>(null);
  const systemRefs = useRef<(HTMLDivElement | null)[]>([]);
  // Distinguishes our own smooth-scroll from user scrolling, so auto-follow
  // isn't cancelled by its own scroll events.
  const programmaticScrollUntil = useRef(0);

  const safeDuration =
    Number.isFinite(duration) && duration > 0
      ? duration
      : (tabDocument?.duration ?? 0);

  const beatGrid = useMemo(() => getBeatGrid(tabDocument), [tabDocument]);

  const systems = useMemo(
    () => buildSystems(buildMeasures(safeDuration, beatGrid)),
    [safeDuration, beatGrid]
  );

  // Notes bucketed per system, positioned once (layout is time-pure).
  const placedBySystem = useMemo(() => {
    const out: PlacedNote[][] = systems.map(() => []);
    if (!tabDocument) return out;
    const sorted = [...tabDocument.notes].sort((a, b) => a.timestamp - b.timestamp);
    const placed = new Map<string, { systemIndex: number; value: PlacedNote }>();
    for (const note of sorted) {
      const i = systemIndexAt(systems, note.timestamp);
      if (i < 0) continue;
      const digits = note.fret === 'X' ? 1 : String(note.fret).length;
      const value: PlacedNote = {
        note,
        x: timeToX(systems[i], note.timestamp),
        y: stringY(note.string),
        w: digits * 7 + 5,
      };
      out[i].push(value);
      placed.set(note.id, { systemIndex: i, value });
    }

    // Enrich slide destinations after every note has geometry, including the
    // case where their source falls in the previous printed system.
    const previous = previousNoteById(sorted);
    for (const note of sorted) {
      if (!isSlide(note)) continue;
      const current = placed.get(note.id);
      if (!current) continue;
      const prior = previous.get(note.id);
      current.value.slideDirection = slideDirection(prior, note);
      const source = prior ? placed.get(prior.id) : undefined;
      if (source && source.systemIndex === current.systemIndex) {
        current.value.slideFrom = {
          x: source.value.x,
          y: source.value.y,
          w: source.value.w,
        };
      }
    }
    return out;
  }, [systems, tabDocument]);

  const activeSystem = systemIndexAt(systems, currentTime);

  const handleSeek = useCallback(
    (t: number) => {
      selectNote(null);
      audition.seek(t);
    },
    [selectNote]
  );

  const handleSelect = useCallback(
    (note: TabNote, mode: 'replace' | 'toggle' | 'range') => {
      if (mode === 'range') {
        selectNoteRange(note.id);
      } else if (mode === 'toggle') {
        toggleNoteSelection(note.id);
      } else {
        selectNote(note.id);
      }
      audition.seek(note.timestamp);
      setCurrentTime(note.timestamp);
    },
    [selectNote, selectNoteRange, setCurrentTime, toggleNoteSelection]
  );

  // While the score is showing, the browser-tab title carries the piece title
  // — it's what Chrome uses as the default "Save as PDF" filename.
  useEffect(() => {
    const prev = document.title;
    if (tabDocument?.title) document.title = tabDocument.title;
    return () => {
      document.title = prev;
    };
  }, [tabDocument?.title]);

  // Follow playback: keep the active system in view. Only fires when the
  // cursor crosses into a new system, so manual reading scroll stays free
  // between jumps.
  useEffect(() => {
    if (!isPlaying || !isFollowingPlayback) return;
    const el = systemRefs.current[activeSystem];
    if (!el) return;
    programmaticScrollUntil.current = performance.now() + 600;
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }, [activeSystem, isPlaying, isFollowingPlayback]);

  // Toolbar pill jump: bring the focused note's system into view. Nonce-keyed
  // so ordinary note clicks never yank the reading scroll.
  useEffect(() => {
    if (!noteFocusNonce || !selectedNoteId || !tabDocument) return;
    const note = tabDocument.notes.find(n => n.id === selectedNoteId);
    if (!note) return;
    const i = systemIndexAt(systems, note.timestamp);
    const el = i >= 0 ? systemRefs.current[i] : null;
    if (!el) return;
    programmaticScrollUntil.current = performance.now() + 600;
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [noteFocusNonce]);

  const handleScroll = useCallback(() => {
    if (performance.now() < programmaticScrollUntil.current) return;
    setFollowingPlayback(false);
  }, [setFollowingPlayback]);

  if (jobStatus !== 'completed' || !tabDocument) return null;

  const tuningLabel = tabDocument.tuning?.length
    ? tabDocument.tuning.join(' ')
    : 'E A D G B E';
  const infoParts = [
    beatGrid
      ? `♩ = ${Math.round(beatGrid.tempoBpm)} · ${beatGrid.beatsPerBar}/4`
      : 'free time',
    `Tuning: ${tuningLabel}`,
  ];
  if (tabDocument.capoFret > 0) infoParts.push(`Capo ${tabDocument.capoFret}`);
  const scoreSummary = describeTablature('score', tabDocument.notes);
  const selectionAnnouncement = describeSelection(
    tabDocument.notes,
    selectedNoteId,
    selectedNoteIds,
  );

  return (
    <div
      ref={scrollRef}
      className="score-scroll h-full overflow-y-auto"
      onScroll={handleScroll}
      role="region"
      aria-label={scoreSummary}
      style={{ background: 'var(--bg-base)' }}
    >
      <div className="score-page-wrap flex justify-center px-4 py-6">
        <div
          className="score-page w-full"
          style={{
            maxWidth: '900px',
            background: PAPER,
            color: INK,
            borderRadius: '4px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.55)',
            padding: '48px 40px 56px',
          }}
        >
          {/* Sheet header — the title is click-to-edit */}
          <div className="text-center" style={{ marginBottom: '28px' }}>
            {editingTitle ? (
              <input
                autoFocus
                value={titleDraft}
                placeholder="Untitled"
                onChange={e => setTitleDraft(e.target.value)}
                onBlur={() => {
                  setDocumentTitle(titleDraft);
                  setEditingTitle(false);
                }}
                onKeyDown={e => {
                  if (e.key === 'Enter') e.currentTarget.blur();
                  if (e.key === 'Escape') {
                    setTitleDraft(tabDocument?.title ?? '');
                    setEditingTitle(false);
                  }
                }}
                style={{
                  display: 'block',
                  width: '100%',
                  margin: 0,
                  padding: '0 0 2px',
                  border: 'none',
                  borderBottom: `1px dashed ${INK_FAINT}`,
                  outline: 'none',
                  background: 'transparent',
                  textAlign: 'center',
                  fontFamily: "Georgia, 'Times New Roman', serif",
                  fontSize: '22px',
                  fontWeight: 600,
                  color: INK,
                }}
              />
            ) : (
              <h2
                className="score-title group cursor-text"
                title="Click to rename"
                tabIndex={0}
                aria-label={`${tabDocument?.title || 'TabVision Transcription'}. Activate to rename.`}
                onClick={() => {
                  setTitleDraft(tabDocument?.title ?? '');
                  setEditingTitle(true);
                }}
                onKeyDown={event => {
                  if (event.key !== 'Enter' && event.key !== ' ') return;
                  event.preventDefault();
                  setTitleDraft(tabDocument?.title ?? '');
                  setEditingTitle(true);
                }}
                style={{
                  margin: 0,
                  fontFamily: "Georgia, 'Times New Roman', serif",
                  fontSize: '22px',
                  fontWeight: 600,
                  color: tabDocument?.title ? INK : INK_FAINT,
                }}
              >
                {tabDocument?.title || 'TabVision Transcription'}
                <svg
                  className="score-title-pencil print-hide"
                  aria-hidden="true"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={1.5}
                  style={{
                    display: 'inline-block',
                    width: '13px',
                    height: '13px',
                    marginLeft: '8px',
                    verticalAlign: 'baseline',
                    color: INK_FAINT,
                  }}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L6.832 19.82a4.5 4.5 0 01-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 011.13-1.897L16.863 4.487z"
                  />
                </svg>
              </h2>
            )}
            <p
              style={{
                margin: '6px 0 0',
                fontFamily: "Georgia, 'Times New Roman', serif",
                fontStyle: 'italic',
                fontSize: '12px',
                color: INK_FAINT,
              }}
            >
              {infoParts.join('   ·   ')}
            </p>
          </div>

          {/* Systems */}
          {systems.map((system, i) => (
            <div
              key={i}
              ref={el => {
                systemRefs.current[i] = el;
              }}
              style={{ marginBottom: '14px' }}
            >
              <System
                system={system}
                notes={placedBySystem[i]}
                systemIndex={i}
                systemCount={systems.length}
                selectedNoteIds={selectedNoteIds}
                cursorTime={i === activeSystem ? currentTime : -1}
                onSeek={handleSeek}
                onSelect={handleSelect}
              />
            </div>
          ))}

          {/* Footer */}
          <p
            style={{
              margin: '24px 0 0',
              textAlign: 'center',
              fontFamily: "Georgia, 'Times New Roman', serif",
              fontStyle: 'italic',
              fontSize: '11px',
              color: INK_FAINT,
            }}
          >
            Transcribed with TabVision
          </p>
          <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
            {selectionAnnouncement}
          </div>
        </div>
      </div>
    </div>
  );
}
