// tabvision-client/src/components/TabToolbar.tsx
import React, { useState, useCallback } from 'react';
import { useAppStore } from '../store/appStore';
import { exportToTextTab } from '../utils/exportTab';

export function TabToolbar() {
  const {
    jobStatus,
    tabDocument,
    isFollowingPlayback,
    editHistoryIndex,
    editHistory,
    zoomLevel,
    setFollowingPlayback,
    undo,
    redo,
    zoomIn,
    zoomOut,
    resetZoom,
  } = useAppStore();

  const [showExportMenu, setShowExportMenu] = useState(false);

  const canUndo = editHistoryIndex >= 0;
  const canRedo = editHistoryIndex < editHistory.length - 1;
  const editCount = editHistory.length;
  const editedNoteCount = tabDocument ? tabDocument.notes.filter(n => n.isEdited).length : 0;

  const handleExportText = useCallback(() => {
    if (!tabDocument) return;
    const text = exportToTextTab(tabDocument);
    // Copy to clipboard
    navigator.clipboard.writeText(text).then(() => {
      setShowExportMenu(false);
    });
  }, [tabDocument]);

  const handleExportDownload = useCallback(() => {
    if (!tabDocument) return;
    const text = exportToTextTab(tabDocument);
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tablature-${tabDocument.id || 'export'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
  }, [tabDocument]);

  if (jobStatus !== 'completed') return null;

  return (
    <div
      className="h-full flex flex-col justify-between px-4 py-3"
      style={{
        background: 'var(--bg-surface)',
        borderLeft: '1px solid var(--border-subtle)',
      }}
    >
      {/* Top section: stats */}
      <div className="space-y-3">
        {/* Confidence stats */}
        {tabDocument && (
          <div className="flex gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full" style={{ background: 'var(--color-success)' }} />
              <span style={{ color: 'var(--text-muted)' }}>
                {tabDocument.notes.filter(n => n.confidenceLevel === 'high').length}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full" style={{ background: 'var(--color-warning)' }} />
              <span style={{ color: 'var(--text-muted)' }}>
                {tabDocument.notes.filter(n => n.confidenceLevel === 'medium').length}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full" style={{ background: 'var(--color-error)' }} />
              <span style={{ color: 'var(--text-muted)' }}>
                {tabDocument.notes.filter(n => n.confidenceLevel === 'low').length}
              </span>
            </div>
            <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              {tabDocument.notes.length} notes
            </span>
          </div>
        )}

        {/* Tuning and capo */}
        {tabDocument && (
          <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
            <span>{tabDocument.tuning?.join(' ') || 'E A D G B E'}</span>
            {tabDocument.capoFret > 0 && (
              <span> &middot; Capo {tabDocument.capoFret}</span>
            )}
          </div>
        )}
      </div>

      {/* Middle section: controls */}
      <div className="flex items-center gap-3 py-2">
        {/* Undo/Redo group */}
        <div
          className="flex items-center rounded-lg overflow-hidden"
          style={{ border: '1px solid var(--border-subtle)' }}
        >
          <button
            onClick={undo}
            disabled={!canUndo}
            className="btn btn-ghost btn-icon"
            title="Undo (Ctrl+Z)"
            style={{ borderRadius: 0, padding: '6px 8px' }}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 15L3 9m0 0l6-6M3 9h12a6 6 0 010 12h-3" />
            </svg>
          </button>
          <div style={{ width: '1px', height: '20px', background: 'var(--border-subtle)' }} />
          <button
            onClick={redo}
            disabled={!canRedo}
            className="btn btn-ghost btn-icon"
            title="Redo (Ctrl+Shift+Z)"
            style={{ borderRadius: 0, padding: '6px 8px' }}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 15l6-6m0 0l-6-6m6 6H9a6 6 0 000 12h3" />
            </svg>
          </button>
        </div>

        {editCount > 0 && (
          <span className="text-[11px] tabular-nums" style={{ color: 'var(--text-muted)' }}>
            {editedNoteCount} edited
          </span>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Zoom controls */}
        <div
          className="flex items-center gap-1 rounded-lg px-1"
          style={{ border: '1px solid var(--border-subtle)' }}
        >
          <button
            onClick={zoomOut}
            disabled={zoomLevel <= 0.25}
            className="btn btn-ghost btn-icon"
            title="Zoom out"
            style={{ padding: '4px' }}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 12h-15" />
            </svg>
          </button>
          <button
            onClick={resetZoom}
            className="text-[11px] px-1.5 py-0.5 rounded transition-colors tabular-nums"
            style={{
              color: zoomLevel !== 1 ? 'var(--accent-primary)' : 'var(--text-muted)',
              minWidth: '36px',
              textAlign: 'center',
            }}
            title="Reset zoom"
          >
            {Math.round(zoomLevel * 100)}%
          </button>
          <button
            onClick={zoomIn}
            disabled={zoomLevel >= 4}
            className="btn btn-ghost btn-icon"
            title="Zoom in"
            style={{ padding: '4px' }}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
          </button>
        </div>

        {/* Follow playback */}
        <button
          onClick={() => setFollowingPlayback(!isFollowingPlayback)}
          className="btn btn-ghost btn-icon"
          title={isFollowingPlayback ? 'Following playback' : 'Follow playback'}
          style={{
            color: isFollowingPlayback ? 'var(--accent-primary)' : 'var(--text-muted)',
            background: isFollowingPlayback ? 'var(--accent-glow)' : 'transparent',
            padding: '6px',
          }}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
            {isFollowingPlayback ? (
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
            )}
          </svg>
        </button>

        {/* Export */}
        <div className="relative">
          <button
            className="btn btn-primary text-xs"
            onClick={() => setShowExportMenu(!showExportMenu)}
            style={{ padding: '6px 12px' }}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
            </svg>
            Export
          </button>

          {showExportMenu && (
            <div
              className="absolute top-full right-0 mt-1 py-1 rounded-lg shadow-lg z-50"
              style={{
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border-default)',
                minWidth: '160px',
              }}
            >
              <button
                className="w-full px-3 py-2 text-xs text-left flex items-center gap-2 transition-colors hover:bg-white/5"
                style={{ color: 'var(--text-secondary)' }}
                onClick={handleExportText}
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9.75a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184" />
                </svg>
                Copy to Clipboard
              </button>
              <button
                className="w-full px-3 py-2 text-xs text-left flex items-center gap-2 transition-colors hover:bg-white/5"
                style={{ color: 'var(--text-secondary)' }}
                onClick={handleExportDownload}
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                </svg>
                Download .txt
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Bottom: hints */}
      <div className="text-[11px] flex gap-3" style={{ color: 'var(--text-muted)' }}>
        <span className="flex items-center gap-1">
          <span className="kbd">0-9</span> edit
        </span>
        <span className="flex items-center gap-1">
          <span className="kbd">Del</span> mute
        </span>
        <span className="flex items-center gap-1">
          <span className="kbd">&larr;&rarr;</span> navigate
        </span>
      </div>
    </div>
  );
}
