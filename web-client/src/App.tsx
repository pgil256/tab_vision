import React, { useCallback, useEffect, useRef, useState } from 'react';
import { UploadPanel } from './components/UploadPanel';
import { RecordPanel } from './components/RecordPanel';
import { VideoPlayer } from './components/VideoPlayer';
import { TabCanvas } from './components/TabCanvas';
import { ScoreView } from './components/ScoreView';
import { TabToolbar } from './components/TabToolbar';
import { NoteInspector } from './components/NoteInspector';
import { ShortcutsModal } from './components/ShortcutsModal';
import { RestoreBanner } from './components/RestoreBanner';
import { useAppStore } from './store/appStore';
import { useAudition } from './hooks/useAudition';
import { useEditorHotkeys } from './hooks/useEditorHotkeys';
import './index.css';

type InputMode = 'upload' | 'record';

const CaptureIcon = ({ mode }: { mode: InputMode }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} aria-hidden="true">
    {mode === 'upload' ? (
      <>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 14.5v3A2.5 2.5 0 0 0 7.5 20h9a2.5 2.5 0 0 0 2.5-2.5v-3" />
      </>
    ) : (
      <>
        <rect x="3" y="6" width="13" height="12" rx="3" />
        <path strokeLinecap="round" strokeLinejoin="round" d="m16 10 4-2v8l-4-2" />
      </>
    )}
  </svg>
);

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [inputMode, setInputMode] = useState<InputMode>('record');
  const [confirmingNewTake, setConfirmingNewTake] = useState(false);
  const { jobStatus, showShortcutsModal, setShowShortcutsModal, reset, viewMode, isVideoCollapsed } = useAppStore();
  const editedNoteCount = useAppStore((state) =>
    state.tabDocument?.notes.reduce((count, note) => count + (note.isEdited ? 1 : 0), 0) ?? 0,
  );
  const appliedEditCount = useAppStore((state) => state.editHistoryIndex + 1);
  const correctionCount = Math.max(editedNoteCount, appliedEditCount);
  const loadPersistedSession = useAppStore((s) => s.loadPersistedSession);
  const checkServiceHealth = useAppStore((s) => s.checkServiceHealth);
  const serviceHealth = useAppStore((s) => s.serviceHealth);

  useAudition(videoRef);
  useEditorHotkeys();

  useEffect(() => {
    loadPersistedSession();
    checkServiceHealth();
  }, [loadPersistedSession, checkServiceHealth]);

  useEffect(() => {
    if (!confirmingNewTake) return;
    const timeout = window.setTimeout(() => setConfirmingNewTake(false), 5000);
    return () => window.clearTimeout(timeout);
  }, [confirmingNewTake]);

  const handleNewTake = useCallback(() => {
    if (correctionCount > 0 && !confirmingNewTake) {
      setConfirmingNewTake(true);
      return;
    }

    setConfirmingNewTake(false);
    reset();
  }, [confirmingNewTake, correctionCount, reset]);

  const showEditor = jobStatus === 'completed';
  const isProcessing = jobStatus === 'uploading' || jobStatus === 'processing';
  const stageLabel = showEditor ? 'Refine' : isProcessing ? 'Analyzing' : 'Capture';

  return (
    <div className={`app-shell ${showEditor ? 'app-shell--editor' : 'app-shell--landing'} print-expand`}>
      <header className="topbar print-hide">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none">
              <path d="M7 8v16M11 5v22M15 10v12M19 7v18M23 11v10M27 9v14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </div>
          <div className="brand-copy">
            <h1>TabVision</h1>
            <p>Transcription studio</p>
          </div>
        </div>

        <nav className="workflow-nav" aria-label="Transcription workflow">
          <span
            className={`workflow-nav__item ${showEditor ? 'is-complete' : 'is-active'}`}
            aria-current={!showEditor ? 'step' : undefined}
          >
            <b>01</b> Capture
          </span>
          <i />
          <span
            className={`workflow-nav__item ${showEditor ? 'is-active' : ''}`}
            aria-current={showEditor ? 'step' : undefined}
          >
            <b>02</b> Refine
          </span>
        </nav>

        <div className="topbar-actions">
          <div className="stage-indicator"><span />{stageLabel}</div>
          <button
            className="btn btn-ghost topbar-shortcuts"
            onClick={() => setShowShortcutsModal(true)}
            data-tooltip="Keyboard shortcuts"
            data-tooltip-placement="bottom-end"
            aria-label="Keyboard shortcuts"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} aria-hidden="true">
              <rect x="3" y="6" width="18" height="12" rx="3" />
              <path strokeLinecap="round" d="M7 10h.01M11 10h.01M15 10h.01M7 14h10" />
            </svg>
            <span>Shortcuts</span>
          </button>
          {showEditor && (
            <button
              className={`btn btn-secondary topbar-new ${confirmingNewTake ? 'is-confirming' : ''}`}
              onClick={handleNewTake}
              aria-label={confirmingNewTake
                ? `Discard ${correctionCount} ${correctionCount === 1 ? 'correction' : 'corrections'}; click again to confirm`
                : 'Start a new take'}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 5v14M5 12h14" />
              </svg>
              {confirmingNewTake
                ? `Discard ${correctionCount} ${correctionCount === 1 ? 'correction' : 'corrections'}?`
                : 'New take'}
            </button>
          )}
        </div>
      </header>

      <main className="app-main print-expand">
        {!showEditor && (
          <div className="landing-viewport animate-fade-in" data-testid="landing-scroll">
            <div className="landing-orb landing-orb--one" />
            <div className="landing-orb landing-orb--two" />
            <div className="landing-layout" data-testid="landing-content">
              <section className="landing-intro">
                <div className="eyebrow"><span />Audio meets vision</div>
                <h2>
                  Your performance,
                  <span> mapped to every string.</span>
                </h2>
                <p className="landing-lede">
                  Turn a guitar take into playable tablature. TabVision listens for every note,
                  watches the fretboard, and gives you a fast visual review pass.
                </p>

                <div className="landing-proof" aria-label="How TabVision works">
                  <div>
                    <span className="proof-number">01</span>
                    <span className="proof-icon proof-icon--audio">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7}><path strokeLinecap="round" d="M4 13v-2m4 6V7m4 13V4m4 13V7m4 6v-2" /></svg>
                    </span>
                    <p><strong>Hear the notes</strong><small>Pitch and timing analysis</small></p>
                  </div>
                  <div>
                    <span className="proof-number">02</span>
                    <span className="proof-icon proof-icon--vision">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7}><path strokeLinecap="round" strokeLinejoin="round" d="M3 12s3.4-5 9-5 9 5 9 5-3.4 5-9 5-9-5-9-5Z" /><circle cx="12" cy="12" r="2.5" /></svg>
                    </span>
                    <p><strong>See the position</strong><small>String and fret tracking</small></p>
                  </div>
                  <div>
                    <span className="proof-number">03</span>
                    <span className="proof-icon proof-icon--edit">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7}><path strokeLinecap="round" strokeLinejoin="round" d="M5 18.5 18.2 5.3a1.8 1.8 0 0 1 2.5 2.5L7.5 21H4v-3.5Z" /><path d="m15.5 8 2.5 2.5" /></svg>
                    </span>
                    <p><strong>Make it yours</strong><small>Correct, audition, export</small></p>
                  </div>
                </div>

                <div className="format-strip">
                  <span>MP4</span><span>MOV</span><span>WAV</span><span>MP3</span>
                  <p>No account · Local editing · MIDI &amp; PDF export</p>
                </div>
              </section>

              <section className="landing-workbench" aria-label="Start a transcription">
                <RestoreBanner />
                <div className="workbench-card">
                  <div className="workbench-topline">
                    <div>
                      <span className="workbench-kicker">New transcription</span>
                      <h3>{inputMode === 'upload' ? 'Import a recording' : 'Capture a live take'}</h3>
                    </div>
                    <span
                      className={`workbench-status workbench-status--${serviceHealth}`}
                      role="status"
                      title={serviceHealth === 'offline'
                        ? 'The transcription service did not answer its health check.'
                        : undefined}
                    >
                      <i />
                      {serviceHealth === 'checking'
                        ? 'Checking service'
                        : serviceHealth === 'online'
                          ? 'Service online'
                          : 'Service unreachable'}
                    </span>
                  </div>

                  {!isProcessing && jobStatus !== 'failed' && (
                    <div className="input-mode-tabs" role="tablist" aria-label="Recording source">
                      {(['upload', 'record'] as const).map((mode) => (
                        <button
                          key={mode}
                          role="tab"
                          aria-selected={inputMode === mode}
                          onClick={() => setInputMode(mode)}
                          className={inputMode === mode ? 'is-active' : ''}
                        >
                          <CaptureIcon mode={mode} />
                          <span>
                            <strong>{mode === 'upload' ? 'Upload file' : 'Record now'}</strong>
                            <small>{mode === 'upload' ? 'Video or audio' : 'Camera or microphone'}</small>
                          </span>
                        </button>
                      ))}
                    </div>
                  )}

                  <div className={`workbench-body ${isProcessing || jobStatus === 'failed' ? 'is-centered' : ''}`}>
                    {isProcessing || jobStatus === 'failed'
                      ? <UploadPanel />
                      : inputMode === 'upload' ? <UploadPanel /> : <RecordPanel />}
                  </div>
                </div>
              </section>
            </div>
          </div>
        )}

        {showEditor && (
          <div className="editor-shell animate-fade-in print-expand">
            <div className={`editor-control-deck print-hide ${isVideoCollapsed ? 'is-source-collapsed' : ''}`}>
              <section className="editor-source-panel" aria-label="Source playback">
                <div className="panel-label">
                  <span>Source</span>
                  <small>Listen &amp; compare</small>
                </div>
                <VideoPlayer videoRef={videoRef} />
              </section>
              <section className="editor-tools-panel" aria-label="Transcription tools">
                <div className="panel-label panel-label--tools">
                  <span>Review desk</span>
                  <small>Correct · audition · export</small>
                </div>
                <TabToolbar />
                <NoteInspector />
              </section>
            </div>

            <section className="editor-canvas-deck print-expand">
              <div className="canvas-heading print-hide">
                <div>
                  <span>{viewMode === 'score' ? 'Printable score' : 'Interactive timeline'}</span>
                  <small>{viewMode === 'score' ? 'Read, title, and print your transcription' : 'Click a note to edit · drag to retime · press ? for shortcuts'}</small>
                </div>
                <span className="canvas-live-label"><i /> Synced to playhead</span>
              </div>
              <div className="canvas-surface print-expand">
                {viewMode === 'score' ? <ScoreView /> : <TabCanvas videoRef={videoRef} />}
              </div>
            </section>
          </div>
        )}
      </main>

      {showShortcutsModal && <ShortcutsModal />}
    </div>
  );
}

export default App;
