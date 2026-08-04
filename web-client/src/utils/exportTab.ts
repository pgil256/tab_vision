// tabvision-client/src/utils/exportTab.ts
import { TabDocument, TabNote } from '../types/tab';
import {
  bendSemitones,
  isBend,
  isSlide,
  previousNoteById,
  slideDirection,
} from './noteTechniques';

function noteToken(note: TabNote, previous: TabNote | undefined): string {
  if (note.fret === 'X') return 'x';

  if (isBend(note)) {
    const amount = bendSemitones(note);
    const target = note.fret + amount;
    return Number.isInteger(target)
      ? `${note.fret}b${target}`
      : `${note.fret}b(${Number(amount.toFixed(2))})`;
  }

  if (isSlide(note)) {
    const direction = slideDirection(previous, note);
    return `${direction > 0 ? '/' : direction < 0 ? '\\' : 's'}${note.fret}`;
  }

  return note.fret.toString();
}

/**
 * Export a TabDocument to standard text tablature format.
 *
 * Output example:
 *   e|---0---2---3---|
 *   B|---1---3---0---|
 *   G|---0---2---0---|
 *   D|---2---0---0---|
 *   A|---3-------2---|
 *   E|-----------3---|
 */
export function exportToTextTab(doc: TabDocument): string {
  const stringLabels = ['e', 'B', 'G', 'D', 'A', 'E'];
  const notes = [...doc.notes].sort((a, b) => a.timestamp - b.timestamp);
  const previous = previousNoteById(notes);

  if (notes.length === 0) {
    return stringLabels.map(label => `${label}|${'---'.repeat(4)}|`).join('\n');
  }

  // Group notes into time columns (within 50ms tolerance)
  const columns: TabNote[][] = [];
  let currentColumn: TabNote[] = [];
  let lastTime = -1;

  for (const note of notes) {
    if (lastTime < 0 || Math.abs(note.timestamp - lastTime) < 0.05) {
      currentColumn.push(note);
    } else {
      if (currentColumn.length > 0) columns.push(currentColumn);
      currentColumn = [note];
    }
    lastTime = note.timestamp;
  }
  if (currentColumn.length > 0) columns.push(currentColumn);

  // Build tab lines - split into rows of ~60 columns for readability
  const COLUMNS_PER_ROW = 60;
  const rows: string[] = [];

  for (let rowStart = 0; rowStart < columns.length; rowStart += COLUMNS_PER_ROW) {
    const rowColumns = columns.slice(rowStart, rowStart + COLUMNS_PER_ROW);
    const columnWidths = rowColumns.map(col => {
      const longest = col.reduce(
        (width, note) => Math.max(width, noteToken(note, previous.get(note.id)).length),
        0,
      );
      // Keep the legacy three-character cell for ordinary one/two-digit frets;
      // expressive tokens expand only the column that needs the room.
      return Math.max(3, longest + 1);
    });
    const lines: string[] = [];

    for (let s = 0; s < 6; s++) {
      const stringNum = (s + 1) as 1 | 2 | 3 | 4 | 5 | 6;
      let line = `${stringLabels[s]}|`;

      for (let columnIndex = 0; columnIndex < rowColumns.length; columnIndex++) {
        const col = rowColumns[columnIndex];
        const width = columnWidths[columnIndex];
        const note = col.find(n => n.string === stringNum);
        if (note) {
          const token = noteToken(note, previous.get(note.id));
          line += `${'-'.repeat(width - token.length - 1)}${token}-`;
        } else {
          line += '-'.repeat(width);
        }
      }

      line += '|';
      lines.push(line);
    }

    rows.push(lines.join('\n'));
  }

  // Header
  let header = 'TabVision Transcription\n';
  header += `Tuning: ${doc.tuning?.join(' ') || 'E A D G B E'}`;
  if (doc.capoFret > 0) header += ` | Capo: Fret ${doc.capoFret}`;
  header += `\n${doc.notes.length} notes detected\n`;
  if (notes.some(note => isSlide(note) || isBend(note))) {
    header += 'Techniques: / upward slide, \\ downward slide, b target-pitch bend\n';
  }
  header += '='.repeat(40) + '\n\n';

  return header + rows.join('\n\n');
}
