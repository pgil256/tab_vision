/**
 * Headless verification for manual slide/bend notation and text export.
 *
 * Run: npx tsx scripts/verify-technique-notation.mts
 */
import { useAppStore } from '../src/store/appStore';
import type { TabDocument, TabNote } from '../src/types/tab';
import { exportToTextTab } from '../src/utils/exportTab';

let failures = 0;
function check(name: string, condition: boolean) {
  if (condition) console.log(`  ok  ${name}`);
  else {
    failures++;
    console.error(`FAIL  ${name}`);
  }
}

function note(
  id: string,
  string: TabNote['string'],
  fret: number | 'X',
  timestamp: number,
): TabNote {
  return {
    id,
    string,
    fret,
    timestamp,
    confidence: 0.7,
    confidenceLevel: 'medium',
    isEdited: false,
  };
}

function freshDoc(): TabDocument {
  return {
    id: 'technique-doc',
    createdAt: '',
    duration: 5,
    capoFret: 0,
    tuning: ['E', 'B', 'G', 'D', 'A', 'E'],
    notes: [
      note('start', 2, 5, 0),
      note('slide-up', 2, 7, 1),
      note('slide-down', 2, 3, 2),
      note('bend', 3, 7, 3),
      note('muted', 4, 'X', 4),
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
    editHistory: [],
    editHistoryIndex: -1,
  });
}

reset();
state().selectNote('slide-up');
state().toggleNoteSelection('slide-down');
state().setSelectedTechnique('slide');
check(
  'a slide marking applies to the complete selection',
  state().tabDocument!.notes.slice(1, 3).every(item => item.technique === 'slide'),
);
check('a group technique edit records one history entry', state().editHistory.length === 1);
state().undo();
check(
  'one undo removes every slide in the group',
  state().tabDocument!.notes.slice(1, 3).every(item => item.technique == null),
);
state().redo();
check(
  'one redo restores every slide in the group',
  state().tabDocument!.notes.slice(1, 3).every(item => item.technique === 'slide'),
);

state().selectNote('bend');
state().setSelectedTechnique('bend', 2);
const bend = state().tabDocument!.notes.find(item => item.id === 'bend')!;
check('bend amount is stored in semitones', bend.technique === 'bend' && bend.pitchBend === 2);

state().selectNote('start');
state().toggleNoteSelection('muted');
const historyBeforeMutedEdit = state().editHistory.length;
state().setSelectedTechnique('bend', 1);
check(
  'a muted member rejects the complete group edit',
  state().tabDocument!.notes.find(item => item.id === 'start')!.technique == null &&
    state().editHistory.length === historyBeforeMutedEdit,
);

const text = exportToTextTab(state().tabDocument!);
check('ascending slides export with /', text.includes('/7-'));
check('descending slides export with \\', text.includes('\\3-'));
check('whole-step bends export with target fret', text.includes('7b9-'));
check('expressive text exports include a legend', text.includes('Techniques:'));
const tabLines = text.split('\n').filter(line => /^[eBGDAE]\|/.test(line));
check('expanded technique columns stay aligned', new Set(tabLines.map(line => line.length)).size === 1);

if (failures) {
  console.error(`\n${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('\nALL TECHNIQUE NOTATION CHECKS PASSED');
