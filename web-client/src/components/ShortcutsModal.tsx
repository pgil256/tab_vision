// tabvision-client/src/components/ShortcutsModal.tsx
import React, { useCallback, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useAppStore } from '../store/appStore';

const FOCUSABLE_SELECTOR = [
  'button:not([disabled])',
  '[href]',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

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
    ],
  },
  {
    title: 'Group selection',
    shortcuts: [
      { keys: ['Ctrl/Cmd', 'Click'], description: 'Add or remove a note' },
      { keys: ['Shift', 'Click'], description: 'Select a note range' },
      { keys: ['Arrow keys'], description: 'Move group by one string / 50 ms' },
    ],
  },
  {
    title: 'Editing',
    shortcuts: [
      { keys: ['0-9'], description: 'Set fret number' },
      { keys: ['Shift', '↑'], description: 'Move to higher string (keep pitch)' },
      { keys: ['Shift', '↓'], description: 'Move to lower string (keep pitch)' },
      { keys: ['C'], description: 'Cycle ranked alternative position' },
      { keys: ['Shift', 'C'], description: 'Cycle alternative backwards' },
      { keys: ['S'], description: 'Toggle slide into selected note(s)' },
      { keys: ['B'], description: 'Toggle full-step bend' },
      { keys: ['X'], description: 'Mark as muted (X)' },
      { keys: ['Del'], description: 'Delete note' },
      { keys: ['I'], description: 'Insert note at playhead' },
      { keys: ['Enter'], description: 'Commit edit' },
      { keys: ['Esc'], description: 'Deselect note' },
      { keys: ['Ctrl', 'Z'], description: 'Undo' },
      { keys: ['Ctrl', 'Shift', 'Z'], description: 'Redo' },
    ],
  },
  {
    title: 'Review',
    shortcuts: [
      { keys: ['R'], description: 'Start / end low-confidence review' },
      { keys: ['N'], description: 'Next note in review queue' },
      { keys: ['P'], description: 'Previous note in review queue' },
    ],
  },
];

export function ShortcutsModal() {
  const { setShowShortcutsModal } = useAppStore();
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const closeModal = useCallback(() => setShowShortcutsModal(false), [setShowShortcutsModal]);

  useEffect(() => {
    const previouslyFocused = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const appShell = document.querySelector<HTMLElement>('.app-shell');
    const wasInert = appShell?.inert ?? false;
    const previousAriaHidden = appShell?.getAttribute('aria-hidden');

    if (appShell) {
      appShell.inert = true;
      appShell.setAttribute('aria-hidden', 'true');
    }
    closeButtonRef.current?.focus();

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        closeModal();
        return;
      }

      if (e.key !== 'Tab') return;
      const dialog = dialogRef.current;
      if (!dialog) return;
      const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
      if (focusable.length === 0) {
        e.preventDefault();
        dialog.focus();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const activeElement = document.activeElement;
      if (e.shiftKey && (activeElement === first || !dialog.contains(activeElement))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (activeElement === last || !dialog.contains(activeElement))) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      if (appShell) {
        appShell.inert = wasInert;
        if (previousAriaHidden == null) appShell.removeAttribute('aria-hidden');
        else appShell.setAttribute('aria-hidden', previousAriaHidden);
      }
      window.requestAnimationFrame(() => previouslyFocused?.focus());
    };
  }, [closeModal]);

  return createPortal(
    <div
      className="shortcuts-modal-layer fixed inset-0 z-50 flex items-center justify-center animate-fade-in"
      onClick={closeModal}
    >
      {/* Backdrop */}
      <div className="absolute inset-0" style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }} />

      {/* Modal */}
      <div
        ref={dialogRef}
        className="shortcuts-modal-card relative rounded-2xl p-6 w-full animate-slide-up"
        role="dialog"
        aria-modal="true"
        aria-labelledby="shortcuts-modal-title"
        aria-describedby="shortcuts-modal-description"
        tabIndex={-1}
        style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-default)',
          boxShadow: 'var(--shadow-lg)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <h2 id="shortcuts-modal-title" className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
            Keyboard Shortcuts
          </h2>
          <button
            ref={closeButtonRef}
            className="btn btn-ghost btn-icon"
            onClick={closeModal}
            aria-label="Close keyboard shortcuts"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Sections */}
        <div className="shortcuts-modal-sections">
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
          <p id="shortcuts-modal-description" className="text-[11px] text-center" style={{ color: 'var(--text-muted)' }}>
            Press <span className="kbd">Esc</span> to close
          </p>
        </div>
      </div>
    </div>,
    document.body,
  );
}
