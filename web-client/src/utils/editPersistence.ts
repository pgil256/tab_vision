// B5 — localStorage autosave of the (edited) tab document so corrections survive
// a refresh. The app does not re-fetch the job on reload, so we persist the whole
// document (not just a diff) and offer to restore it on the next mount.
import { TabDocument } from '../types/tab';

const KEY = 'tabvision:session:v1';

export interface PersistedSession {
  jobId: string | null;
  doc: TabDocument;
  savedAt: number; // epoch ms
}

// Guard every access — localStorage can be absent (SSR/tests) or throw
// (private mode, quota). Persistence is best-effort; failures are non-fatal.
function storage(): Storage | null {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return null;
    return window.localStorage;
  } catch {
    return null;
  }
}

export function persistSession(doc: TabDocument, jobId: string | null): void {
  const s = storage();
  if (!s) return;
  try {
    const payload: PersistedSession = { jobId, doc, savedAt: Date.now() };
    s.setItem(KEY, JSON.stringify(payload));
  } catch {
    /* quota / serialization — ignore, autosave is best-effort */
  }
}

export function loadSession(): PersistedSession | null {
  const s = storage();
  if (!s) return null;
  try {
    const raw = s.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PersistedSession;
    if (!parsed || !parsed.doc || !Array.isArray(parsed.doc.notes)) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function clearSession(): void {
  const s = storage();
  if (!s) return;
  try {
    s.removeItem(KEY);
  } catch {
    /* ignore */
  }
}
