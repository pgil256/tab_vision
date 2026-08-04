import type { TabNote } from '../types/tab';

export const DEFAULT_BEND_SEMITONES = 2;

export function isSlide(note: TabNote): boolean {
  return note.technique === 'slide' || note.technique?.startsWith('slide-') === true;
}

export function isBend(note: TabNote): boolean {
  return note.technique === 'bend';
}

export function bendSemitones(note: TabNote): number {
  return Math.max(0.25, note.pitchBend ?? DEFAULT_BEND_SEMITONES);
}

/** Short, musician-facing bend label used beside tab glyphs. */
export function bendLabel(note: TabNote): string {
  const semitones = bendSemitones(note);
  if (Math.abs(semitones - 1) < 0.01) return '½';
  if (Math.abs(semitones - 2) < 0.01) return 'full';
  if (Math.abs(semitones - 3) < 0.01) return '1½';
  return `${Number(semitones.toFixed(2))}st`;
}

/** -1 = descending, 0 = same/unknown, 1 = ascending. */
export function slideDirection(from: TabNote | undefined, to: TabNote): -1 | 0 | 1 {
  if (!from || typeof from.fret !== 'number' || typeof to.fret !== 'number') return 0;
  return Math.sign(to.fret - from.fret) as -1 | 0 | 1;
}

/** Slides are stored on their destination note. This map locates the note
 * immediately before each note on the same string without coupling renderers
 * and exporters to array order. */
export function previousNoteById(notes: TabNote[]): Map<string, TabNote> {
  const previous = new Map<string, TabNote>();
  const lastByString = new Map<number, TabNote>();
  const ordered = [...notes].sort(
    (a, b) => a.timestamp - b.timestamp || a.string - b.string || a.id.localeCompare(b.id),
  );

  for (const note of ordered) {
    const prior = lastByString.get(note.string);
    if (prior && prior.timestamp < note.timestamp) previous.set(note.id, prior);
    lastByString.set(note.string, note);
  }
  return previous;
}
