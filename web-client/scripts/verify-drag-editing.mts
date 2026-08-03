/**
 * M4 headless verification — drag-commit store logic (applyNoteDrag) and the
 * extended undo/redo snapshot (timestamp/endTime join NoteMutableFields).
 *
 * Run:  npx tsx scripts/verify-drag-editing.mts
 */
import { useAppStore } from '../src/store/appStore';
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

function note(id: string, string: TabNote['string'], fret: number | 'X', t: number, end?: number): TabNote {
  return {
    id,
    string,
    fret,
    timestamp: t,
    endTime: end,
    confidence: 1,
    confidenceLevel: 'high',
    isEdited: false,
  };
}

function freshDoc(): TabDocument {
  return {
    id: 'doc',
    createdAt: '',
    duration: 10,
    capoFret: 0,
    tuning: ['E', 'B', 'G', 'D', 'A', 'E'],
    notes: [note('n1', 2, 5, 1.0, 1.4), note('n2', 6, 3, 2.0), note('n3', 4, 0, 3.0, 3.5)],
  };
}

const s = useAppStore.getState;
function reset() {
  useAppStore.setState({
    tabDocument: freshDoc(),
    selectedNoteId: null,
    editHistory: [],
    editHistoryIndex: -1,
  });
}

// --- Combined retime + restring commits as ONE history entry ---
reset();
// Drag n1 (B string fret 5, 1.0-1.4s) to 1.75s on the high-E string (fret 0).
s().applyNoteDrag('n1', { timestamp: 1.75, string: 1, fret: 0 });
{
  const n1 = s().tabDocument!.notes.find(n => n.id === 'n1')!;
  check('drag: timestamp moved to 1.75', n1.timestamp === 1.75);
  check('drag: endTime shifted by the same delta (duration kept)', Math.abs(n1.endTime! - 2.15) < 1e-9);
  check('drag: string moved to 1', n1.string === 1);
  check('drag: fret recomputed to 0', n1.fret === 0);
  check('drag: marked edited', n1.isEdited === true);
  check('drag: originalFret recorded', n1.originalFret === 5);
  check('drag: exactly one history entry', s().editHistory.length === 1);
}

// --- One undo reverses the whole gesture ---
s().undo();
{
  const n1 = s().tabDocument!.notes.find(n => n.id === 'n1')!;
  check('undo: timestamp restored', n1.timestamp === 1.0);
  check('undo: endTime restored', n1.endTime === 1.4);
  check('undo: string restored', n1.string === 2);
  check('undo: fret restored', n1.fret === 5);
  check('undo: isEdited restored', n1.isEdited === false);
  check('undo: originalFret restored', n1.originalFret === undefined);
}
s().redo();
{
  const n1 = s().tabDocument!.notes.find(n => n.id === 'n1')!;
  check('redo: full drag reapplied', n1.timestamp === 1.75 && n1.string === 1 && n1.fret === 0);
}

// --- Retime-only drag on a note without endTime ---
reset();
s().applyNoteDrag('n2', { timestamp: 2.6, string: 6, fret: 3 });
{
  const n2 = s().tabDocument!.notes.find(n => n.id === 'n2')!;
  check('retime-only: timestamp moved', n2.timestamp === 2.6);
  check('retime-only: missing endTime stays undefined', n2.endTime === undefined);
  check('retime-only: string/fret unchanged', n2.string === 6 && n2.fret === 3);
}

// --- Negative timestamps clamp to 0 ---
reset();
s().applyNoteDrag('n1', { timestamp: -0.5, string: 2, fret: 5 });
{
  const n1 = s().tabDocument!.notes.find(n => n.id === 'n1')!;
  check('clamp: negative timestamp clamps to 0', n1.timestamp === 0);
  check('clamp: endTime keeps duration from 0', Math.abs(n1.endTime! - 0.4) < 1e-9);
}

// --- No-op drag records nothing ---
reset();
s().applyNoteDrag('n3', { timestamp: 3.0, string: 4, fret: 0 });
check('no-op drag: history untouched', s().editHistory.length === 0);

// --- Old-style position edit still restores timestamp fields (snapshot grew) ---
reset();
s().updateNotePosition('n1', 3, 9);
s().undo();
{
  const n1 = s().tabDocument!.notes.find(n => n.id === 'n1')!;
  check('position-edit undo keeps timestamp intact', n1.timestamp === 1.0 && n1.endTime === 1.4);
  check('position-edit undo restores string/fret', n1.string === 2 && n1.fret === 5);
}

// --- Interleave with delete/insert history (indices stay valid) ---
reset();
s().applyNoteDrag('n1', { timestamp: 4.5, string: 2, fret: 5 }); // now after n3 in time, array order unchanged
s().deleteNote('n2');
s().undo(); // restore n2 at its recorded index
{
  const ids = s().tabDocument!.notes.map(n => n.id);
  check('delete-undo after retime restores at original index', JSON.stringify(ids) === '["n1","n2","n3"]');
}
s().undo(); // reverse the drag
{
  const n1 = s().tabDocument!.notes.find(n => n.id === 'n1')!;
  check('second undo reverses the drag', n1.timestamp === 1.0);
}

if (failures) {
  console.error(`\n${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('\nALL DRAG CHECKS PASSED');
