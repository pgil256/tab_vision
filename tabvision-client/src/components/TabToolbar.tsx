// tabvision-client/src/components/TabToolbar.tsx
import React from 'react';
import { useAppStore } from '../store/appStore';

export function TabToolbar() {
  const {
    jobStatus,
    isFollowingPlayback,
    editHistoryIndex,
    editHistory,
    setFollowingPlayback,
    undo,
    redo,
  } = useAppStore();

  const canUndo = editHistoryIndex >= 0;
  const canRedo = editHistoryIndex < editHistory.length - 1;

  if (jobStatus !== 'completed') {
    return null;
  }

  return (
    <div className="flex items-center justify-between px-4 py-2 bg-gray-800 rounded-lg">
      {/* Left side - Undo/Redo */}
      <div className="flex items-center gap-2">
        <button
          onClick={undo}
          disabled={!canUndo}
          className={`flex items-center gap-1 px-3 py-1.5 rounded text-sm transition-colors ${
            canUndo
              ? 'bg-gray-700 hover:bg-gray-600 text-white'
              : 'bg-gray-800 text-gray-600 cursor-not-allowed'
          }`}
          title="Undo (Ctrl+Z)"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
          </svg>
          Undo
        </button>

        <button
          onClick={redo}
          disabled={!canRedo}
          className={`flex items-center gap-1 px-3 py-1.5 rounded text-sm transition-colors ${
            canRedo
              ? 'bg-gray-700 hover:bg-gray-600 text-white'
              : 'bg-gray-800 text-gray-600 cursor-not-allowed'
          }`}
          title="Redo (Ctrl+Shift+Z)"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10h-10a8 8 0 00-8 8v2M21 10l-6 6m6-6l-6-6" />
          </svg>
          Redo
        </button>

        {editHistory.length > 0 && (
          <span className="text-xs text-gray-500 ml-2">
            {editHistoryIndex + 1} / {editHistory.length} edits
          </span>
        )}
      </div>

      {/* Right side - Follow playback toggle */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => setFollowingPlayback(!isFollowingPlayback)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-colors ${
            isFollowingPlayback
              ? 'bg-blue-600 hover:bg-blue-500 text-white'
              : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
          }`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {isFollowingPlayback ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
            )}
          </svg>
          {isFollowingPlayback ? 'Following playback' : 'Follow playback'}
        </button>

        {/* Placeholder for future export button */}
        <button
          disabled
          className="flex items-center gap-1 px-3 py-1.5 rounded text-sm bg-gray-800 text-gray-600 cursor-not-allowed"
          title="Export (coming soon)"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Export
        </button>
      </div>
    </div>
  );
}
