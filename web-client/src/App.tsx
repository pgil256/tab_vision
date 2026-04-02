// tabvision-client/src/App.tsx
import React, { useRef } from 'react';
import { UploadPanel } from './components/UploadPanel';
import { VideoPlayer } from './components/VideoPlayer';
import { TabCanvas } from './components/TabCanvas';
import { TabToolbar } from './components/TabToolbar';
import { ShortcutsModal } from './components/ShortcutsModal';
import { useAppStore } from './store/appStore';
import './index.css';

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { jobStatus, videoUrl, showShortcutsModal, setShowShortcutsModal, reset } = useAppStore();

  const showEditor = jobStatus === 'completed';
  const isProcessing = jobStatus === 'uploading' || jobStatus === 'processing';

  return (
    <div className="h-screen flex flex-col overflow-hidden" style={{ background: 'var(--bg-base)' }}>
      {/* Accent gradient bar */}
      <div
        className="h-[2px] shrink-0"
        style={{ background: 'linear-gradient(90deg, var(--accent-primary), var(--accent-secondary), var(--accent-primary))' }}
      />

      {/* Header */}
      <header
        className="shrink-0 flex items-center justify-between px-5 py-3 glass-strong"
        style={{ borderBottom: '1px solid var(--border-subtle)' }}
      >
        <div className="flex items-center gap-3">
          {/* Logo */}
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-sm"
            style={{
              background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
              boxShadow: '0 0 12px var(--accent-glow)',
            }}
          >
            TV
          </div>
          <div>
            <h1 className="text-[15px] font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>
              TabVision
            </h1>
            <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              {showEditor ? 'Editor' : isProcessing ? 'Processing...' : 'Guitar Tab Transcription'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Keyboard shortcuts */}
          <button
            className="btn btn-ghost btn-icon"
            onClick={() => setShowShortcutsModal(true)}
            data-tooltip="Keyboard shortcuts"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
            </svg>
          </button>

          {/* New transcription button (when in editor) */}
          {showEditor && (
            <button
              className="btn btn-secondary text-xs"
              onClick={reset}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
              </svg>
              New
            </button>
          )}
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {/* Upload view */}
        {!showEditor && !isProcessing && jobStatus !== 'failed' && (
          <div className="flex-1 flex items-center justify-center p-8 animate-fade-in">
            <UploadPanel />
          </div>
        )}

        {/* Processing / error view (shows within UploadPanel) */}
        {(isProcessing || jobStatus === 'failed') && (
          <div className="flex-1 flex items-center justify-center p-8 animate-fade-in">
            <UploadPanel />
          </div>
        )}

        {/* Editor view */}
        {showEditor && (
          <div className="flex-1 flex flex-col min-h-0 animate-fade-in">
            {/* Video + Toolbar row */}
            <div className="shrink-0" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
              <div className="flex items-stretch">
                {/* Video player */}
                <div className="shrink-0">
                  {videoUrl && <VideoPlayer videoRef={videoRef} />}
                </div>

                {/* Toolbar (fills remaining space) */}
                <div className="flex-1 min-w-0">
                  <TabToolbar />
                </div>
              </div>
            </div>

            {/* Tab canvas (takes all remaining space) */}
            <div className="flex-1 min-h-0">
              <TabCanvas videoRef={videoRef} />
            </div>
          </div>
        )}
      </main>

      {/* Shortcuts Modal */}
      {showShortcutsModal && <ShortcutsModal />}
    </div>
  );
}

export default App;
