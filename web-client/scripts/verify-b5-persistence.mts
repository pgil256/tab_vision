/**
 * B5 headless verification — exercises edit persistence (autosave + restore)
 * without a browser. Installs a minimal in-memory localStorage BEFORE loading
 * the store (dynamic import), since editPersistence.ts guards on
 * `typeof window === 'undefined'` at call time, not at import time.
 *
 * Run:  npx tsx scripts/verify-b5-persistence.mts
 */
const memory = new Map<string, string>();
(globalThis as unknown as { window: unknown }).window = globalThis;
(globalThis as unknown as { localStorage: Storage }).localStorage = {
  getItem: (k: string) => (memory.has(k) ? (memory.get(k) as string) : null),
  setItem: (k: string, v: string) => {
    memory.set(k, v);
  },
  removeItem: (k: string) => {
    memory.delete(k);
  },
  clear: () => memory.clear(),
  key: (i: number) => Array.from(memory.keys())[i] ?? null,
  get length() {
    return memory.size;
  },
} as Storage;

const { useAppStore } = await import('../src/store/appStore');
const { loadSession } = await import('../src/utils/editPersistence');
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
    id: 'doc',
    createdAt: '',
    duration: 10,
    capoFret: 0,
    tuning: ['E', 'B', 'G', 'D', 'A', 'E'],
    // n2: low E (string 6) fret 8 = MIDI 48 -> playable on string 5 (open A=45) at fret 3.
    notes: [note('n1', 2, 5, 1.0), note('n2', 6, 8, 2.0)],
  };
}

const s = useAppStore.getState;
function resetAll() {
  memory.clear();
  useAppStore.setState({
    tabDocument: null,
    currentJobId: null,
    jobStatus: 'idle',
    selectedNoteId: null,
    editHistory: [],
    editHistoryIndex: -1,
    restorable: null,
  });
}

// --- setTabDocument persists immediately ---
resetAll();
useAppStore.getState().setJobId('job-1');
useAppStore.getState().setTabDocument(freshDoc());
{
  const persisted = loadSession();
  check('setTabDocument persists to storage', persisted !== null);
  check('persisted doc has 2 notes', persisted?.doc.notes.length === 2);
  check('persisted jobId matches', persisted?.jobId === 'job-1');
}

// --- editing updates the persisted payload ---
useAppStore.getState().updateNoteFret('n1', 9);
{
  const persisted = loadSession();
  const n1 = persisted?.doc.notes.find((n) => n.id === 'n1');
  check('edit (fret) updates persisted snapshot', n1?.fret === 9);
}

useAppStore.getState().moveNoteString('down'); // no selection -> no-op, but must not throw
useAppStore.getState().selectNote('n2');
useAppStore.getState().moveNoteString('up'); // string 6 -> 5, pitch-preserving
{
  const persisted = loadSession();
  const n2 = persisted?.doc.notes.find((n) => n.id === 'n2');
  check('moveNoteString updates persisted snapshot', n2?.string === 5);
}

useAppStore.getState().deleteNote('n1');
{
  const persisted = loadSession();
  check('deleteNote updates persisted snapshot', persisted?.doc.notes.length === 1);
}

useAppStore.getState().undo(); // restore n1
{
  const persisted = loadSession();
  check('undo updates persisted snapshot', persisted?.doc.notes.length === 2);
}

useAppStore.getState().insertNote({ timestamp: 5, string: 3, fret: 2 });
{
  const persisted = loadSession();
  check('insertNote updates persisted snapshot', persisted?.doc.notes.length === 3);
}

// --- reset() clears the persisted session (mirrors "New transcription") ---
useAppStore.getState().reset();
check('reset() clears persisted storage', loadSession() === null);

// --- loadPersistedSession surfaces restorable only when nothing is loaded ---
resetAll();
useAppStore.getState().setJobId('job-2');
useAppStore.getState().setTabDocument(freshDoc());
useAppStore.setState({ tabDocument: null, jobStatus: 'idle' }); // simulate a fresh mount post-refresh
useAppStore.getState().loadPersistedSession();
{
  const r = s().restorable;
  check('loadPersistedSession surfaces restorable after refresh', r !== null);
  check('restorable carries the autosaved notes', r?.doc.notes.length === 2);
}

resetAll();
useAppStore.getState().setTabDocument(freshDoc()); // a doc IS loaded (live session)
useAppStore.getState().loadPersistedSession();
check('loadPersistedSession is a no-op when a doc is already loaded', s().restorable === null);

// --- restorePersistedSession brings the doc back + resets edit history ---
resetAll();
useAppStore.getState().setJobId('job-3');
useAppStore.getState().setTabDocument(freshDoc());
useAppStore.getState().updateNoteFret('n1', 11); // leave history behind
useAppStore.setState({ tabDocument: null, jobStatus: 'idle', editHistory: [], editHistoryIndex: -1 });
useAppStore.getState().loadPersistedSession();
useAppStore.getState().restorePersistedSession();
{
  const doc = s().tabDocument;
  check('restorePersistedSession loads the doc', doc?.notes.length === 2);
  check('restorePersistedSession restores the edit (fret 11)', doc?.notes.find((n) => n.id === 'n1')?.fret === 11);
  check('restorePersistedSession sets jobStatus completed', s().jobStatus === 'completed');
  check('restorePersistedSession clears restorable', s().restorable === null);
  check('restorePersistedSession resets edit history', s().editHistoryIndex === -1);
}

// --- discardPersistedSession clears storage AND restorable state ---
resetAll();
useAppStore.getState().setTabDocument(freshDoc());
useAppStore.setState({ tabDocument: null, jobStatus: 'idle' });
useAppStore.getState().loadPersistedSession();
check('restorable present before discard', s().restorable !== null);
useAppStore.getState().discardPersistedSession();
check('discardPersistedSession clears restorable', s().restorable === null);
check('discardPersistedSession clears storage', loadSession() === null);

// --- malformed storage content is handled gracefully (no throw, returns null) ---
resetAll();
memory.set('tabvision:session:v1', '{not valid json');
check('malformed JSON in storage does not throw and yields null', loadSession() === null);
memory.set('tabvision:session:v1', JSON.stringify({ jobId: 'x' })); // missing doc.notes
check('storage payload missing notes yields null', loadSession() === null);

console.log(failures === 0 ? '\nALL B5 CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
