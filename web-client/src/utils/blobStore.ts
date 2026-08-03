// IndexedDB persistence for the uploaded/recorded media blob, keyed by jobId,
// so a restored localStorage session gets real playback back (blob URLs die
// with the page). Fail-open everywhere: private mode, quota pressure, or
// eviction just means a restore without playback — the synth-only transport
// covers that — so every function swallows its errors.

const DB_NAME = 'tabvision';
const STORE = 'recordings';

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) {
        req.result.createObjectStore(STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function requestToPromise<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** Persist the recording for `jobId`, evicting every other job's blob (the
 * app is single-session, so at most one recording is ever restorable). */
export async function saveRecordingBlob(jobId: string, blob: Blob): Promise<void> {
  try {
    const db = await openDb();
    const tx = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    const keys = await requestToPromise(store.getAllKeys());
    for (const key of keys) {
      if (key !== jobId) store.delete(key);
    }
    store.put(blob, jobId);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
    db.close();
  } catch {
    // fail-open: no persisted playback for this session
  }
}

export async function loadRecordingBlob(jobId: string): Promise<Blob | null> {
  try {
    const db = await openDb();
    const value = await requestToPromise(
      db.transaction(STORE, 'readonly').objectStore(STORE).get(jobId)
    );
    db.close();
    return value instanceof Blob ? value : null;
  } catch {
    return null;
  }
}

export async function deleteRecordingBlob(jobId: string): Promise<void> {
  try {
    const db = await openDb();
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(jobId);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
    db.close();
  } catch {
    // fail-open
  }
}
