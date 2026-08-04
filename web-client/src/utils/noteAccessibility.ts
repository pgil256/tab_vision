import { TabNote } from '../types/tab';

const CONFIDENCE_LABELS: Record<TabNote['confidenceLevel'], string> = {
  high: 'high confidence',
  medium: 'check confidence',
  low: 'low confidence',
};

function formatSeconds(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  return `${safeSeconds.toFixed(1)} seconds`;
}

export function isFlaggedNote(note: TabNote): boolean {
  return note.confidenceLevel !== 'high' && !note.isEdited && note.fret !== 'X';
}

export function describeNote(note: TabNote): string {
  const fret = note.fret === 'X' ? 'muted note' : `fret ${note.fret}`;
  const details = [
    `String ${note.string}`,
    fret,
    `at ${formatSeconds(note.timestamp)}`,
    CONFIDENCE_LABELS[note.confidenceLevel],
  ];

  if (note.isEdited) details.push('edited');
  if (note.technique) details.push(note.technique);
  return details.join(', ');
}

export function describeSelection(
  notes: TabNote[],
  selectedNoteId: string | null,
  selectedNoteIds: string[],
): string {
  if (selectedNoteIds.length > 1) {
    return `${selectedNoteIds.length} notes selected.`;
  }

  const note = selectedNoteId ? notes.find(candidate => candidate.id === selectedNoteId) : undefined;
  return note ? `Selected note: ${describeNote(note)}.` : 'No note selected.';
}

export function describeTablature(view: 'timeline' | 'score', notes: TabNote[]): string {
  const flagged = notes.reduce((count, note) => count + (isFlaggedNote(note) ? 1 : 0), 0);
  const viewLabel = view === 'timeline' ? 'Interactive tablature timeline' : 'Printable tablature score';
  return `${viewLabel}, ${notes.length} notes, ${flagged} flagged for review.`;
}
