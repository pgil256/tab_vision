/**
 * B3 headless verification — exercises the string-level correction store logic
 * without a browser (the project has no test runner; this mirrors the existing
 * standalone `scripts/*.mjs` verification pattern).
 *
 * Run:  npx tsx scripts/verify-b3-editing.mts
 */
import { useAppStore, pitchPreservingFret } from '../src/store/appStore';
import type { TabDocument, TabNote } from '../src/types/tab';

let failures = 0;
function check(name: string, cond: boolean) {
  if (cond) {
    console.log(`  ok  ${name}`);
  } else {
    failures++;
    console.error(`FAIL  ${name}`);
  }
}

function note(id: string, string: TabNote['string'], fret: number | 'X', t: number): TabNote {
  return { id, string, fret, timestamp: t, confidence: 1, confidenceLevel: 'high', isEdited: false };
}

function freshDoc(): TabDocument {
  return {
    id: 'doc', createdAt: '', duration: 10, capoFret: 0,
    tuning: ['E', 'B', 'G', 'D', 'A', 'E'],
    // B string (2) fret 5 = E4 (MIDI 64); low E (6) fret 3 = G2; open D (4).
    notes: [note('n1', 2, 5, 1.0), note('n2', 6, 3, 2.0), note('n3', 4, 0, 3.0)],
  };
}

const s = useAppStore.getState;
function reset() {
  useAppStore.setState({ tabDocument: freshDoc(), selectedNoteId: null, editHistory: [], editHistoryIndex: -1 });
}

// --- pitchPreservingFret (pure) ---
check('preserve: B/5 -> high-E(1) = fret 0', pitchPreservingFret(2, 1, 5) === 0);
check('preserve: B/5 -> G(3) = fret 9', pitchPreservingFret(2, 3, 5) === 9);
check('preserve: muted stays muted', pitchPreservingFret(2, 1, 'X') === 'X');
check('preserve: unplayable -> null (D open up to G)', pitchPreservingFret(4, 3, 0) === null);

// --- moveNoteString up preserves pitch, undo/redo ---
reset();
useAppStore.getState().selectNote('n1');
useAppStore.getState().moveNoteString('up'); // B(2)/5 -> highE(1)/0
{
  const n1 = s().tabDocument!.notes.find((n) => n.id === 'n1')!;
  check('move up: string 2 -> 1', n1.string === 1);
  check('move up: fret 5 -> 0 (pitch kept)', n1.fret === 0);
  check('move up: marked edited', n1.isEdited === true);
  check('move up: originalFret recorded', n1.originalFret === 5);
}
useAppStore.getState().undo();
{
  const n1 = s().tabDocument!.notes.find((n) => n.id === 'n1')!;
  check('undo move: string back to 2', n1.string === 2);
  check('undo move: fret back to 5', n1.fret === 5);
  check('undo move: isEdited restored to false', n1.isEdited === false);
}
useAppStore.getState().redo();
check('redo move: string 1 again', s().tabDocument!.notes.find((n) => n.id === 'n1')!.string === 1);

// --- moveNoteString down at edge / unplayable is a no-op ---
reset();
useAppStore.getState().selectNote('n3'); // D(4)/0
useAppStore.getState().moveNoteString('up'); // D open -> G string would be fret -5 => unplayable
check('unplayable move is no-op (string unchanged)', s().tabDocument!.notes.find((n) => n.id === 'n3')!.string === 4);
check('unplayable move recorded no history', s().editHistoryIndex === -1);

// --- delete + undo restores at index ---
reset();
useAppStore.getState().selectNote('n2');
useAppStore.getState().deleteNote('n2');
check('delete: removed', s().tabDocument!.notes.findIndex((n) => n.id === 'n2') === -1);
check('delete: count 2', s().tabDocument!.notes.length === 2);
check('delete: selection cleared', s().selectedNoteId === null);
useAppStore.getState().undo();
{
  const notes = s().tabDocument!.notes;
  check('undo delete: count 3', notes.length === 3);
  check('undo delete: restored at original index 1', notes[1].id === 'n2');
}

// --- insert + select + undo ---
reset();
useAppStore.getState().insertNote({ timestamp: 1.5, string: 3, fret: 7 });
{
  const notes = s().tabDocument!.notes;
  check('insert: count 4', notes.length === 4);
  check('insert: kept in timestamp order', notes.map((n) => n.timestamp).join(',') === '1,1.5,2,3');
  const inserted = notes.find((n) => n.timestamp === 1.5)!;
  check('insert: on requested string/fret', inserted.string === 3 && inserted.fret === 7);
  check('insert: selected', s().selectedNoteId === inserted.id);
}
useAppStore.getState().undo();
check('undo insert: count 3', s().tabDocument!.notes.length === 3);

// --- updateNoteFret still works (delegates through position) ---
reset();
useAppStore.getState().updateNoteFret('n2', 7);
check('updateNoteFret: fret set', s().tabDocument!.notes.find((n) => n.id === 'n2')!.fret === 7);
check('updateNoteFret: string unchanged', s().tabDocument!.notes.find((n) => n.id === 'n2')!.string === 6);

console.log(failures === 0 ? '\nALL B3 CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
