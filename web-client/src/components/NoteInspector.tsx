import React from 'react';
import { useAppStore } from '../store/appStore';
import { noteNameForMidi } from '../utils/audioAnalysis';
import { midiPitchForNote, openStringMidi } from '../utils/pitch';
import type { TabNote } from '../types/tab';

const CONFIDENCE_COPY = {
  high: 'Confident',
  medium: 'Check',
  low: 'Review',
} as const;

/** Open-string letter for a string number (1 = high E … 6 = low E), taken from
 * the document's tuning when it has one. Matches the timeline's string
 * labels for standard tuning. */
function stringLetter(string: number, tuningMidi?: number[]): string {
  return noteNameForMidi(openStringMidi(string, tuningMidi)).replace(/-?\d+$/, '');
}

function formatClock(seconds: number): string {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  return `${Math.floor(safe / 60)}:${(safe % 60).toFixed(2).padStart(5, '0')}`;
}

function describeTiming(note: TabNote): string {
  const start = formatClock(note.timestamp);
  if (note.endTime === undefined || note.endTime <= note.timestamp) return start;
  return `${start} · ${(note.endTime - note.timestamp).toFixed(2)}s long`;
}

/** Fills the band under the toolbar: the selected note's identity and its
 * ranked pitch-preserving alternatives, or the document summary when nothing
 * is selected. Everything here was previously either canvas-only (the "2/4"
 * candidate counter), hotkey-only (C cycles positions), or hidden behind the
 * details popover. */
export function NoteInspector() {
  const tabDocument = useAppStore(s => s.tabDocument);
  const selectedNoteId = useAppStore(s => s.selectedNoteId);
  const selectedNoteIds = useAppStore(s => s.selectedNoteIds);
  const updateNotePosition = useAppStore(s => s.updateNotePosition);
  const startReview = useAppStore(s => s.startReview);

  if (!tabDocument) return null;

  const tuningMidi = tabDocument.tuningMidi;
  const notes = tabDocument.notes;
  const note = selectedNoteId ? notes.find(n => n.id === selectedNoteId) : undefined;

  // --- Nothing selected: document summary + the way in to the review queue.
  if (!note) {
    // The confidence split is the toolbar row directly above — don't repeat it.
    const flagged = notes.reduce(
      (count, n) => count + (n.confidenceLevel !== 'high' && !n.isEdited && n.fret !== 'X' ? 1 : 0),
      0,
    );
    const metadata = tabDocument.metadata;
    const facts: Array<[string, string]> = [
      ['Tuning', tabDocument.tuning?.length ? tabDocument.tuning.join(' ') : 'Standard'],
      ['Capo', tabDocument.capoFret ? `Fret ${tabDocument.capoFret}` : 'None'],
    ];
    if (metadata?.pipelineVersion) {
      facts.push(['Pipeline', `${metadata.pipelineVersion} · ${metadata.audioBackend || 'audio'}`]);
    }
    if (metadata?.positionPrior) {
      facts.push(['Prior', `${metadata.positionPrior} · video ${metadata.videoEnabled ? 'on' : 'off'}`]);
    }

    return (
      <div className="note-inspector note-inspector--idle" data-testid="note-inspector">
        <p className="note-inspector__hint">
          <strong>Select a note</strong>
          <span>to see its position, timing and alternatives</span>
        </p>
        <dl className="inspector-facts">
          {facts.map(([term, value]) => (
            <div key={term}>
              <dt>{term}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        {flagged > 0 && (
          <button
            className="btn btn-ghost inspector-review-cta"
            onClick={startReview}
            data-tooltip="Step through the flagged notes · R"
          >
            Review {flagged} flagged
            <span className="kbd">R</span>
          </button>
        )}
      </div>
    );
  }

  // --- Multiple notes: the group-edit affordances, since per-note detail
  // would be ambiguous.
  if (selectedNoteIds.length > 1) {
    const edited = selectedNoteIds.reduce(
      (count, id) => count + (notes.find(n => n.id === id)?.isEdited ? 1 : 0),
      0,
    );
    return (
      <div className="note-inspector note-inspector--group" data-testid="note-inspector">
        <p className="note-inspector__hint">
          <strong>{selectedNoteIds.length} notes selected</strong>
          <span>{edited > 0 ? `${edited} already edited` : 'none edited yet'}</span>
        </p>
        <dl className="inspector-facts">
          <div><dt>Move</dt><dd>↑ ↓ one string · ← → 50 ms</dd></div>
          <div><dt>Select</dt><dd>Ctrl-click toggles · Shift-click a range</dd></div>
          <div><dt>Clear</dt><dd>Esc</dd></div>
        </dl>
      </div>
    );
  }

  // --- One note: identity, timing, confidence, and its ranked alternatives.
  const muted = note.fret === 'X';
  const pitch = muted
    ? null
    : noteNameForMidi(
        midiPitchForNote(note.string, note.fret as number, tabDocument.capoFret, tuningMidi),
      );
  const candidates = note.candidates ?? [];
  const activeIndex = candidates.findIndex(c => c.string === note.string && c.fret === note.fret);

  return (
    <div className="note-inspector" data-testid="note-inspector">
      <span
        className={`inspector-badge inspector-badge--${note.confidenceLevel}`}
        role="img"
        aria-label={muted ? 'Muted note' : `Fret ${note.fret}`}
      >
        {note.fret}
      </span>

      <div className="inspector-identity">
        <strong>
          {stringLetter(note.string, tuningMidi)} string
          {muted ? <i>· muted</i> : pitch && <em>{pitch}</em>}
        </strong>
        <small>{describeTiming(note)}</small>
      </div>

      <span className={`inspector-confidence inspector-confidence--${note.confidenceLevel}`}>
        <i />
        {CONFIDENCE_COPY[note.confidenceLevel]}
        <b>{Math.round(note.confidence * 100)}%</b>
      </span>

      {/* No revert affordance here: the store records originalFret but not the
          original string, so "restore the transcribed fret" would change pitch
          on any note whose string was also edited. Undo is the correct path. */}
      {note.isEdited && <span className="inspector-edited">edited</span>}

      {/* A muted note has no pitch to preserve, and the store refuses to cycle
          candidates on one — so there is nothing to offer. */}
      {!muted && (
      <div className="inspector-alts">
        <span className="toolbar-group__label">Same pitch</span>
        {candidates.length > 1 ? (
          <>
            <div className="inspector-alts__chips" role="group" aria-label="Alternative positions">
              {candidates.map((candidate, index) => {
                const isActive = index === activeIndex;
                return (
                  <button
                    key={`${candidate.string}-${candidate.fret}`}
                    className={`alt-chip ${isActive ? 'is-active' : ''}`}
                    onClick={() => updateNotePosition(note.id, candidate.string, candidate.fret)}
                    aria-pressed={isActive}
                    aria-label={`Move to string ${candidate.string}, fret ${candidate.fret} — ranked ${index + 1} of ${candidates.length}`}
                    data-tooltip={`Rank ${index + 1} of ${candidates.length}`}
                  >
                    {stringLetter(candidate.string, tuningMidi)}
                    <b>{candidate.fret}</b>
                  </button>
                );
              })}
            </div>
            <small>C cycles</small>
          </>
        ) : (
          <small className="inspector-alts__empty">
            No ranked alternatives — drag the note or type a fret to correct it.
          </small>
        )}
      </div>
      )}
    </div>
  );
}
