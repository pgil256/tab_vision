import React, { useCallback, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { useProcessVideo } from '../utils/useProcessVideo';
import { TranscriptionOptions } from './TranscriptionOptions';
import { AudioReviewPanel } from './AudioReviewPanel';

const ALLOWED_TYPES = [
  'video/mp4',
  'video/quicktime',
  'video/webm', // the recorder's own output
  'audio/webm',
  'audio/wav',
  'audio/x-wav', // some browsers report .wav this way
  'audio/wave',
  'audio/mpeg',
  'audio/mp4',
  'audio/x-m4a', // Safari's .m4a type
];

const PIPELINE_STAGES = [
  { key: 'uploading', label: 'Uploading recording', icon: 'upload' },
  { key: 'extracting_audio', label: 'Extracting audio', icon: 'waveform' },
  { key: 'analyzing_audio', label: 'Analyzing pitch', icon: 'music' },
  { key: 'analyzing_video', label: 'Tracking fingers', icon: 'eye' },
  { key: 'fusing', label: 'Fusing signals', icon: 'merge' },
  { key: 'saving', label: 'Finalizing tab', icon: 'check' },
];

function StageIcon({ icon, active, done }: { icon: string; active: boolean; done: boolean }) {
  const color = done ? 'var(--color-success)' : active ? 'var(--accent-tertiary)' : 'var(--text-muted)';

  const icons: Record<string, React.ReactNode> = {
    upload: (
      <svg className="w-4 h-4" fill="none" stroke={color} viewBox="0 0 24 24" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>
    ),
    waveform: (
      <svg className="w-4 h-4" fill="none" stroke={color} viewBox="0 0 24 24" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l3-3m0 0l3 3m-3-3v12m0-12a9 9 0 11-9 9" />
      </svg>
    ),
    music: (
      <svg className="w-4 h-4" fill="none" stroke={color} viewBox="0 0 24 24" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
      </svg>
    ),
    eye: (
      <svg className="w-4 h-4" fill="none" stroke={color} viewBox="0 0 24 24" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
    merge: (
      <svg className="w-4 h-4" fill="none" stroke={color} viewBox="0 0 24 24" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
      </svg>
    ),
    check: (
      <svg className="w-4 h-4" fill="none" stroke={color} viewBox="0 0 24 24" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  };

  return <>{icons[icon] || null}</>;
}

function getStageIndex(stages: typeof PIPELINE_STAGES, stage: string): number {
  const idx = stages.findIndex(s => s.key === stage);
  return idx >= 0 ? idx : 0;
}

// Guitar pick SVG icon
function GuitarIcon() {
  return (
    <svg className="w-10 h-10" viewBox="0 0 48 48" fill="none">
      {/* Guitar body */}
      <path
        d="M24 6L28 10V16C32 18 36 22 36 28C36 34 30 40 24 40C18 40 12 34 12 28C12 22 16 18 20 16V10L24 6Z"
        fill="url(#guitar-grad)"
        stroke="url(#guitar-stroke)"
        strokeWidth="1.5"
      />
      {/* Strings */}
      <line x1="22" y1="12" x2="22" y2="36" stroke="rgba(255,255,255,0.15)" strokeWidth="0.5" />
      <line x1="24" y1="10" x2="24" y2="36" stroke="rgba(255,255,255,0.2)" strokeWidth="0.5" />
      <line x1="26" y1="12" x2="26" y2="36" stroke="rgba(255,255,255,0.15)" strokeWidth="0.5" />
      {/* Sound hole */}
      <circle cx="24" cy="28" r="4" stroke="rgba(255,255,255,0.2)" strokeWidth="1" fill="none" />
      <defs>
        <linearGradient id="guitar-grad" x1="12" y1="6" x2="36" y2="40" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="rgba(255, 112, 72, 0.34)" />
          <stop offset="100%" stopColor="rgba(255, 154, 98, 0.14)" />
        </linearGradient>
        <linearGradient id="guitar-stroke" x1="12" y1="6" x2="36" y2="40" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="rgba(255, 112, 72, 0.7)" />
          <stop offset="100%" stopColor="rgba(255, 154, 98, 0.32)" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export function UploadPanel() {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  // Audio files pause here for a listen-back/cleanup step before upload.
  const [reviewFile, setReviewFile] = useState<File | null>(null);
  const {
    jobStatus,
    progress,
    currentStage,
    pipelineVideoEnabled,
    inputMediaKind,
    errorMessage,
    setError,
    reset,
  } = useAppStore();
  const processVideo = useProcessVideo();

  const processFile = useCallback(async (file: File) => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError('Please upload a video (MP4, MOV, WEBM) or audio file (WAV, MP3, M4A)');
      return;
    }
    setFileName(file.name);
    // Everything stops at the review stage. For video, cleanup falls back to
    // an audio-only upload (the browser can't re-encode audio back into a
    // video track) — untouched settings upload the original video.
    setReviewFile(file);
  }, [setError]);

  const handleReviewSubmit = useCallback((file: File) => {
    setReviewFile(null);
    setFileName(file.name);
    processVideo(file).catch((err) => {
      setError(err instanceof Error ? err.message : 'Processing failed');
    });
  }, [processVideo, setError]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  }, [processFile]);

  const isProcessing = jobStatus === 'uploading' || jobStatus === 'processing';
  // Audio-only pipelines never run the video stage; don't promise it.
  const pipelineStages = (pipelineVideoEnabled
    ? PIPELINE_STAGES
    : PIPELINE_STAGES.filter(s => s.key !== 'analyzing_video'))
    .map(stage => stage.key === 'uploading'
      ? { ...stage, label: `Uploading ${inputMediaKind}` }
      : stage);
  const currentStageIndex = getStageIndex(pipelineStages, currentStage);

  // === IDLE + file chosen: listen back & clean up before uploading ===
  if (jobStatus === 'idle' && reviewFile) {
    return (
      <AudioReviewPanel
        file={reviewFile}
        onSubmit={handleReviewSubmit}
        onCancel={() => { setReviewFile(null); setFileName(null); }}
        cancelLabel="Choose another file"
        isVideo={reviewFile.type.startsWith('video/')}
      />
    );
  }

  // === IDLE: Upload form ===
  if (jobStatus === 'idle') {
    return (
      <div className="capture-flow capture-flow--upload animate-slide-up">
        <div className="capture-hint">
          <span className="capture-hint__icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 12s3.4-5 9-5 9 5 9 5-3.4 5-9 5-9-5-9-5Z" />
              <circle cx="12" cy="12" r="2.5" />
            </svg>
          </span>
          <p><strong>Video unlocks string tracking</strong><span>A clear view of the fretboard gives the most precise tab.</span></p>
          <span className="capture-hint__badge">Recommended</span>
          </div>

          {/* Drop zone */}
          <div
            className={`drop-zone p-8 text-center group ${isDragging ? 'dragging' : ''}`}
            style={{
              background: isDragging ? 'rgba(255, 112, 72, 0.06)' : 'var(--bg-surface)',
              border: `1.5px dashed ${isDragging ? 'var(--accent-primary)' : 'var(--border-strong)'}`,
            }}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            <div className="animate-float mb-5">
              <div
                className="w-20 h-20 rounded-2xl mx-auto flex items-center justify-center transition-all duration-300 group-hover:scale-105"
                style={{
                  background: 'linear-gradient(135deg, rgba(255, 112, 72, 0.13), rgba(255, 154, 98, 0.06))',
                  border: '1px solid rgba(255, 112, 72, 0.22)',
                  boxShadow: '0 0 30px rgba(255, 112, 72, 0.09)',
                }}
              >
                <GuitarIcon />
              </div>
            </div>

            <p className="text-base font-medium mb-1" style={{ color: 'var(--text-primary)' }}>
              Drop your performance here
            </p>
            <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
              Video or audio up to 5 minutes
            </p>

            <button
              className="btn btn-primary px-6"
              onClick={(e) => { e.stopPropagation(); document.getElementById('file-input')?.click(); }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
              Browse recordings
            </button>

            <input
              id="file-input"
              type="file"
              accept="video/mp4,video/quicktime,video/webm,audio/webm,audio/wav,audio/x-wav,audio/wave,audio/mpeg,audio/mp4,audio/x-m4a,.wav,.mp3,.m4a,.webm"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>

          <TranscriptionOptions />

          {/* Feature highlights */}
          <div className="upload-features">
            <div className="feature-card">
              <div className="feature-icon" style={{ background: 'var(--color-success-soft)' }}>
                <svg className="w-4 h-4" fill="none" stroke="var(--color-success)" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303" />
                </svg>
              </div>
              <div>
                <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>Audio Analysis</p>
                <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Pitch detection</p>
              </div>
            </div>
            <div className="feature-card">
              <div className="feature-icon" style={{ background: 'var(--accent-glow)' }}>
                <svg className="w-4 h-4" fill="none" stroke="var(--accent-tertiary)" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <div>
                <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>Video Tracking</p>
                <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Finger positions</p>
              </div>
            </div>
            <div className="feature-card">
              <div className="feature-icon" style={{ background: 'var(--color-warning-soft)' }}>
                <svg className="w-4 h-4" fill="none" stroke="var(--color-warning)" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                </svg>
              </div>
              <div>
                <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>Smart Fusion</p>
                <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Confidence scores</p>
              </div>
            </div>
          </div>

          {/* Tips */}
          <p className="upload-tip" style={{ color: 'var(--text-muted)' }}>
            Tip: clean audio and a steady fretboard view produce the fastest review pass.
          </p>
      </div>
    );
  }

  // === PROCESSING: Pipeline visualization ===
  if (isProcessing) {
    return (
      <div className="processing-flow animate-slide-up">
        <div
          className="rounded-2xl p-8"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
            boxShadow: 'var(--shadow-lg)',
          }}
        >
          {/* Animated logo */}
          <div className="flex justify-center mb-6">
            <div
              className="w-14 h-14 rounded-2xl flex items-center justify-center animate-pulse-glow"
              style={{
                background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
              }}
            >
              <svg className="w-7 h-7" fill="none" stroke="white" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303" />
              </svg>
            </div>
          </div>

          {/* File name */}
          {fileName && (
            <div className="text-center mb-6">
              <p className="text-xs truncate max-w-[250px] mx-auto" style={{ color: 'var(--text-muted)' }}>
                {fileName}
              </p>
            </div>
          )}

          {/* Stage indicators */}
          <div className="space-y-2.5 mb-6">
            {pipelineStages.map((stage, idx) => {
              const isDone = idx < currentStageIndex || (idx === currentStageIndex && progress >= 0.95);
              const isActive = idx === currentStageIndex && !isDone;
              const isPending = idx > currentStageIndex;

              return (
                <div
                  key={stage.key}
                  className="flex items-center gap-3 transition-all duration-300"
                  style={{
                    opacity: isPending ? 0.3 : 1,
                    animation: isActive ? 'stage-enter 0.3s ease-out' : undefined,
                  }}
                >
                  <div
                    className="w-6 h-6 rounded-md flex items-center justify-center shrink-0 transition-all duration-300"
                    style={{
                      background: isDone
                        ? 'var(--color-success-soft)'
                        : isActive
                        ? 'var(--accent-glow)'
                        : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${
                        isDone
                          ? 'rgba(52, 211, 153, 0.25)'
                          : isActive
                          ? 'var(--border-accent)'
                          : 'var(--border-subtle)'
                      }`,
                    }}
                  >
                    {isDone ? (
                      <svg className="w-3 h-3" fill="none" stroke="var(--color-success)" viewBox="0 0 24 24" strokeWidth={3}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                      </svg>
                    ) : isActive ? (
                      <div
                        className="w-2.5 h-2.5 rounded-full animate-spin-slow"
                        style={{
                          border: '2px solid var(--accent-tertiary)',
                          borderTopColor: 'transparent',
                        }}
                      />
                    ) : (
                      <StageIcon icon={stage.icon} active={false} done={false} />
                    )}
                  </div>

                  <span
                    className="text-sm"
                    style={{
                      color: isDone
                        ? 'var(--color-success)'
                        : isActive
                        ? 'var(--text-primary)'
                        : 'var(--text-muted)',
                      fontWeight: isActive ? 500 : 400,
                    }}
                  >
                    {stage.label}
                    {isActive && <span className="inline-block ml-1 animate-pulse">...</span>}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Progress bar */}
          <div className="progress-bar mb-2">
            <div
              className="progress-bar-fill"
              style={{ width: `${Math.max(progress * 100, 3)}%` }}
            />
          </div>
          <p className="text-xs text-center tabular-nums font-medium" style={{ color: 'var(--text-secondary)' }}>
            {Math.round(progress * 100)}%
          </p>
        </div>
      </div>
    );
  }

  // === FAILED: Error state ===
  if (jobStatus === 'failed') {
    return (
      <div className="processing-flow animate-fade-in">
        <div
          className="rounded-2xl p-8 text-center"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid rgba(251, 113, 133, 0.15)',
            boxShadow: 'var(--shadow-lg)',
          }}
        >
          <div
            className="w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center"
            style={{ background: 'var(--color-error-soft)' }}
          >
            <svg className="w-7 h-7" fill="none" stroke="var(--color-error)" viewBox="0 0 24 24" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
          </div>
          <h3 className="text-base font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
            Processing Failed
          </h3>
          <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
            {errorMessage || 'An unexpected error occurred'}
          </p>
          <button className="btn btn-primary" onClick={reset}>
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return null;
}
