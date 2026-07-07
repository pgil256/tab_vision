// B5 — shown on the landing screen when an autosaved edited session is found in
// localStorage (a prior visit that was refreshed/closed). Lets the user bring
// their corrections back instead of losing them.
import { useAppStore } from '../store/appStore';

function timeAgo(ms: number): string {
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (s < 60) return 'just now';
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h} hr ago`;
  const d = Math.floor(h / 24);
  return `${d} day${d === 1 ? '' : 's'} ago`;
}

export function RestoreBanner() {
  const restorable = useAppStore((s) => s.restorable);
  const restore = useAppStore((s) => s.restorePersistedSession);
  const discard = useAppStore((s) => s.discardPersistedSession);

  if (!restorable) return null;
  const count = restorable.doc.notes.length;

  return (
    <div
      className="w-full max-w-md mb-5 rounded-lg px-4 py-3 flex items-center justify-between gap-3 animate-fade-in"
      style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--accent-glow)',
        boxShadow: 'var(--shadow-md)',
      }}
    >
      <div className="flex items-center gap-2 min-w-0">
        <svg
          className="w-4 h-4 shrink-0"
          style={{ color: 'var(--accent-tertiary)' }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          strokeWidth={1.8}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
          />
        </svg>
        <span className="text-xs truncate" style={{ color: 'var(--text-secondary)' }}>
          Unsaved transcription ({count} note{count === 1 ? '' : 's'}, edited{' '}
          {timeAgo(restorable.savedAt)})
        </span>
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        <button className="btn btn-primary text-xs" style={{ padding: '5px 12px' }} onClick={restore}>
          Restore
        </button>
        <button className="btn btn-ghost text-xs" style={{ padding: '5px 10px' }} onClick={discard}>
          Discard
        </button>
      </div>
    </div>
  );
}
