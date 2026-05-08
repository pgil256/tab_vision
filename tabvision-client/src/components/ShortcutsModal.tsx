// tabvision-client/src/components/ShortcutsModal.tsx
import React, { useEffect } from 'react';
import { useAppStore } from '../store/appStore';

interface ShortcutRow {
  keys: string[];
  description: string;
}

const SECTIONS: { title: string; shortcuts: ShortcutRow[] }[] = [
  {
    title: 'Playback',
    shortcuts: [
      { keys: ['Space'], description: 'Play / Pause' },
      { keys: ['Ctrl', '+'], description: 'Zoom in' },
      { keys: ['Ctrl', '-'], description: 'Zoom out' },
    ],
  },
  {
    title: 'Navigation',
    shortcuts: [
      { keys: ['\u2190'], description: 'Previous note' },
      { keys: ['\u2192'], description: 'Next note' },
      { keys: ['\u2191'], description: 'Higher string' },
      { keys: ['\u2193'], description: 'Lower string' },
      { keys: ['Tab'], description: 'Next note' },
    ],
  },
  {
    title: 'Editing',
    shortcuts: [
      { keys: ['0-9'], description: 'Set fret number' },
      { keys: ['Del'], description: 'Mark as muted (X)' },
      { keys: ['Enter'], description: 'Commit edit' },
      { keys: ['Esc'], description: 'Deselect note' },
      { keys: ['Ctrl', 'Z'], description: 'Undo' },
      { keys: ['Ctrl', 'Shift', 'Z'], description: 'Redo' },
    ],
  },
];

export function ShortcutsModal() {
  const { setShowShortcutsModal } = useAppStore();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowShortcutsModal(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setShowShortcutsModal]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center animate-fade-in"
      onClick={() => setShowShortcutsModal(false)}
    >
      {/* Backdrop */}
      <div className="absolute inset-0" style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }} />

      {/* Modal */}
      <div
        className="relative rounded-2xl p-6 w-full max-w-md animate-slide-up"
        style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-default)',
          boxShadow: 'var(--shadow-lg)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
            Keyboard Shortcuts
          </h2>
          <button
            className="btn btn-ghost btn-icon"
            onClick={() => setShowShortcutsModal(false)}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Sections */}
        <div className="space-y-5">
          {SECTIONS.map((section) => (
            <div key={section.title}>
              <h3
                className="text-[11px] font-semibold uppercase tracking-wider mb-2"
                style={{ color: 'var(--text-muted)' }}
              >
                {section.title}
              </h3>
              <div className="space-y-1.5">
                {section.shortcuts.map((shortcut, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between py-1"
                  >
                    <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      {shortcut.description}
                    </span>
                    <div className="flex items-center gap-1">
                      {shortcut.keys.map((key, keyIdx) => (
                        <React.Fragment key={keyIdx}>
                          {keyIdx > 0 && (
                            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>+</span>
                          )}
                          <span className="kbd">{key}</span>
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="mt-5 pt-4" style={{ borderTop: '1px solid var(--border-subtle)' }}>
          <p className="text-[11px] text-center" style={{ color: 'var(--text-muted)' }}>
            Press <span className="kbd">Esc</span> to close
          </p>
        </div>
      </div>
    </div>
  );
}
