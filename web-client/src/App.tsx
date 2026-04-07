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
      {/* Header */}
      <header
        className="shrink-0 flex items-center justify-between px-5 py-2.5 glass-strong"
        style={{ borderBottom: '1px solid var(--border-subtle)' }}
      >
        <div className="flex items-center gap-3">
          {/* Logo */}
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center text-white"
            style={{
              background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
              boxShadow: '0 0 16px var(--accent-glow)',
            }}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>
              TabVision
            </h1>
            <p className="text-[10px] -mt-0.5" style={{ color: 'var(--text-muted)' }}>
              {showEditor ? 'Editor' : isProcessing ? 'Processing...' : 'Guitar Tab Transcription'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
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
              style={{ padding: '6px 12px' }}
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
          <div className="flex-1 overflow-y-auto flex items-center justify-center p-6 animate-fade-in">
            <UploadPanel />
          </div>
        )}

        {/* Processing / error view */}
        {(isProcessing || jobStatus === 'failed') && (
          <div className="flex-1 overflow-y-auto flex items-center justify-center p-6 animate-fade-in">
            <UploadPanel />
          </div>
        )}

        {/* Editor view */}
        {showEditor && (
          <div className="flex-1 flex flex-col min-h-0 animate-fade-in">
            {/* Video + Toolbar row */}
            <div className="shrink-0" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
              <div className="flex items-stretch">
                <div className="shrink-0">
                  {videoUrl && <VideoPlayer videoRef={videoRef} />}
                </div>
                <div className="flex-1 min-w-0">
                  <TabToolbar />
                </div>
              </div>
            </div>

            {/* Tab canvas */}
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
