/**
 * Headless verification for multi-note selection and atomic arrow movement.
 *
 * Run: npx tsx scripts/verify-group-note-movement.mts
 */
import { NOTE_NUDGE_SECONDS, useAppStore } from '../src/store/appStore';
import type { TabDocument, TabNote } from '../src/types/tab';

let failures = 0;
function check(name: string, condition: boolean) {
  if (condition) {
    console.log(`  ok  ${name}`);
  } else {
    failures++;
    console.error(`FAIL  ${name}`);
  }
}

function note(
  id: string,
  string: TabNote['string'],
  fret: number | 'X',
  timestamp: number,
  endTime?: number,
): TabNote {
  return {
    id,
    string,
    fret,
    timestamp,
    endTime,
    confidence: 0.5,
    confidenceLevel: 'medium',
    isEdited: false,
  };
}

function freshDoc(): TabDocument {
  return {
    id: 'group-doc',
    createdAt: '',
    duration: 10,
    capoFret: 0,
    tuning: ['E', 'B', 'G', 'D', 'A', 'E'],
    notes: [
      note('n1', 2, 5, 1, 1.4),
      note('n2', 3, 9, 2, 2.25),
      note('n3', 4, 0, 3),
      note('n0', 6, 3, 0),
      note('n-end', 6, 3, 10),
    ],
  };
}

const state = useAppStore.getState;
function reset() {
  useAppStore.setState({
    tabDocument: freshDoc(),
    selectedNoteId: null,
    selectedNoteIds: [],
    selectionAnchorId: null,
    reviewActive: false,
    reviewIds: [],
    reviewIndex: -1,
    editHistory: [],
    editHistoryIndex: -1,
  });
}

function select(...ids: string[]) {
  state().selectNote(ids[0] ?? null);
  for (const id of ids.slice(1)) state().toggleNoteSelection(id);
}

// Selection semantics.
reset();
state().selectNote('n1');
state().toggleNoteSelection('n2');
check('toggle builds a two-note group', state().selectedNoteIds.join(',') === 'n1,n2');
check('last toggled note becomes primary', state().selectedNoteId === 'n2');
state().toggleNoteSelection('n2');
check('toggle removes a note', state().selectedNoteIds.join(',') === 'n1');
check('removing primary falls back to remaining note', state().selectedNoteId === 'n1');
state().selectNoteRange('n3');
check('Shift range is inclusive and time ordered', state().selectedNoteIds.join(',') === 'n1,n2,n3');
useAppStore.setState({ reviewActive: true });
state().toggleNoteSelection('n-end');
check('review mode keeps selection singular', state().selectedNoteIds.join(',') === 'n-end');

// Vertical group movement preserves pitch and is one undoable edit.
reset();
select('n1', 'n2');
state().moveSelectedNotes('up');
{
  const n1 = state().tabDocument!.notes.find(n => n.id === 'n1')!;
  const n2 = state().tabDocument!.notes.find(n => n.id === 'n2')!;
  check('vertical move updates every selected string', n1.string === 1 && n2.string === 2);
  check('vertical move preserves both pitches', n1.fret === 0 && n2.fret === 5);
  check('vertical group move records one history entry', state().editHistory.length === 1);
  check('vertical group move keeps the group selected', state().selectedNoteIds.join(',') === 'n1,n2');
}
state().undo();
check(
  'one undo restores the whole vertical group',
  state().tabDocument!.notes.find(n => n.id === 'n1')!.string === 2 &&
    state().tabDocument!.notes.find(n => n.id === 'n2')!.string === 3,
);
state().redo();
check(
  'one redo reapplies the whole vertical group',
  state().tabDocument!.notes.find(n => n.id === 'n1')!.string === 1 &&
    state().tabDocument!.notes.find(n => n.id === 'n2')!.string === 2,
);

// A single invalid note rejects the complete vertical edit.
reset();
select('n1', 'n3'); // n3 is open D and cannot move up to the G string.
state().moveSelectedNotes('up');
check(
  'unplayable vertical member rejects all movement',
  state().tabDocument!.notes.find(n => n.id === 'n1')!.string === 2 &&
    state().tabDocument!.notes.find(n => n.id === 'n3')!.string === 4,
);
check('rejected vertical group records no history', state().editHistory.length === 0);

// Horizontal movement nudges onset and duration tails together.
reset();
select('n1', 'n2');
state().moveSelectedNotes('right');
{
  const n1 = state().tabDocument!.notes.find(n => n.id === 'n1')!;
  const n2 = state().tabDocument!.notes.find(n => n.id === 'n2')!;
  check('horizontal group nudges every onset by 50 ms',
    Math.abs(n1.timestamp - (1 + NOTE_NUDGE_SECONDS)) < 1e-9 &&
    Math.abs(n2.timestamp - (2 + NOTE_NUDGE_SECONDS)) < 1e-9);
  check('horizontal group preserves duration tails', Math.abs(n1.endTime! - 1.45) < 1e-9);
  check('horizontal group move records one history entry', state().editHistory.length === 1);
}
state().undo();
check(
  'one undo restores every horizontal position',
  state().tabDocument!.notes.find(n => n.id === 'n1')!.timestamp === 1 &&
    state().tabDocument!.notes.find(n => n.id === 'n2')!.timestamp === 2,
);

// Timeline boundaries are atomic too.
reset();
select('n0', 'n1');
state().moveSelectedNotes('left');
check(
  'left boundary rejects the complete group',
  state().tabDocument!.notes.find(n => n.id === 'n0')!.timestamp === 0 &&
    state().tabDocument!.notes.find(n => n.id === 'n1')!.timestamp === 1,
);
reset();
select('n1', 'n-end');
state().moveSelectedNotes('right');
check(
  'right boundary rejects the complete group',
  state().tabDocument!.notes.find(n => n.id === 'n1')!.timestamp === 1 &&
    state().tabDocument!.notes.find(n => n.id === 'n-end')!.timestamp === 10,
);

if (failures) {
  console.error(`\n${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('\nALL GROUP NOTE MOVEMENT CHECKS PASSED');
