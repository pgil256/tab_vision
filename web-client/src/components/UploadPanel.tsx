// tabvision-client/src/components/UploadPanel.tsx
import React, { useCallback, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { uploadVideo, getJobStatus, getJobResult } from '../api/client';

const ALLOWED_TYPES = ['video/mp4', 'video/quicktime'];

// Pipeline stages with labels and icons
const PIPELINE_STAGES = [
  { key: 'uploading', label: 'Upload', icon: 'upload' },
  { key: 'extracting_audio', label: 'Audio Extract', icon: 'waveform' },
  { key: 'analyzing_audio', label: 'Pitch Analysis', icon: 'music' },
  { key: 'analyzing_video', label: 'Video Analysis', icon: 'eye' },
  { key: 'fusing', label: 'Fusion', icon: 'merge' },
  { key: 'saving', label: 'Finalize', icon: 'check' },
];

function StageIcon({ icon, active, done }: { icon: string; active: boolean; done: boolean }) {
  const color = done ? 'var(--color-success)' : active ? 'var(--accent-primary)' : 'var(--text-muted)';

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

function getStageIndex(stage: string): number {
  const idx = PIPELINE_STAGES.findIndex(s => s.key === stage);
  return idx >= 0 ? idx : 0;
}

export function UploadPanel() {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const {
    jobStatus, progress, currentStage, errorMessage, capoFretInput,
    setJobId, setStatus, setProgress, setTabDocument, setError, setVideoUrl,
    setCapoFretInput, reset,
  } = useAppStore();

  const processFile = useCallback(async (file: File) => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError('Please upload an MP4 or MOV file');
      return;
    }

    setFileName(file.name);
    reset();
    setStatus('uploading');

    // Create a blob URL for the video player
    const videoUrl = URL.createObjectURL(file);
    setVideoUrl(videoUrl);

    try {
      const jobId = await uploadVideo(file, capoFretInput);
      setJobId(jobId);
      setStatus('processing');

      // Poll for status
      const pollInterval = setInterval(async () => {
        try {
          const status = await getJobStatus(jobId);
          setProgress(status.progress, status.current_stage);

          if (status.status === 'completed') {
            clearInterval(pollInterval);
            const result = await getJobResult(jobId);
            setTabDocument(result);
          } else if (status.status === 'failed') {
            clearInterval(pollInterval);
            setError(status.error_message || 'Processing failed');
          }
        } catch (err) {
          clearInterval(pollInterval);
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
      }, 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
  }, [reset, setJobId, setStatus, setProgress, setTabDocument, setError, setVideoUrl, capoFretInput]);

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
  const currentStageIndex = getStageIndex(currentStage);

  // === IDLE: Upload form ===
  if (jobStatus === 'idle') {
    return (
      <div className="w-full max-w-lg animate-slide-up">
        {/* Upload zone */}
        <div
          className="relative rounded-2xl p-10 text-center transition-all duration-300 cursor-pointer group"
          style={{
            background: isDragging ? 'rgba(99, 102, 241, 0.08)' : 'var(--bg-surface)',
            border: isDragging
              ? '2px solid var(--accent-primary)'
              : '2px dashed var(--border-strong)',
            boxShadow: isDragging ? 'var(--shadow-glow)' : 'none',
          }}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          {/* Guitar icon */}
          <div
            className="w-16 h-16 rounded-2xl mx-auto mb-5 flex items-center justify-center transition-transform duration-300 group-hover:scale-110"
            style={{
              background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(59, 130, 246, 0.15))',
              border: '1px solid rgba(99, 102, 241, 0.2)',
            }}
          >
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="var(--accent-primary)">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h1.5C5.496 19.5 6 18.996 6 18.375m-3.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-1.5A1.125 1.125 0 0118 18.375M20.625 4.5H3.375m17.25 0c.621 0 1.125.504 1.125 1.125M20.625 4.5h-1.5C18.504 4.5 18 5.004 18 5.625m3.75 0v1.5c0 .621-.504 1.125-1.125 1.125M3.375 4.5c-.621 0-1.125.504-1.125 1.125M3.375 4.5h1.5C5.496 4.5 6 5.004 6 5.625m-3.75 0v1.5c0 .621.504 1.125 1.125 1.125m0 0h1.5m-1.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125m1.5-3.75C5.496 8.25 6 7.746 6 7.125v-1.5M4.875 8.25C5.496 8.25 6 8.754 6 9.375v1.5m0-5.25v5.25m0-5.25C6 5.004 6.504 4.5 7.125 4.5h9.75c.621 0 1.125.504 1.125 1.125m1.125 2.625h1.5m-1.5 0A1.125 1.125 0 0118 7.125v-1.5m1.125 2.625c-.621 0-1.125.504-1.125 1.125v1.5m2.625-2.625c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125M18 5.625v5.25M7.125 12h9.75m-9.75 0A1.125 1.125 0 016 10.875M7.125 12C6.504 12 6 12.504 6 13.125m0-2.25C6 11.496 5.496 12 4.875 12M18 10.875c0 .621-.504 1.125-1.125 1.125M18 10.875c0 .621.504 1.125 1.125 1.125m-2.25 0c.621 0 1.125.504 1.125 1.125m-12 5.25v-5.25m0 5.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125m-12 0v-1.5c0-.621-.504-1.125-1.125-1.125M18 18.375v-5.25m0 5.25v-1.5c0-.621.504-1.125 1.125-1.125M18 13.125v1.5c0 .621.504 1.125 1.125 1.125M18 13.125c0-.621.504-1.125 1.125-1.125M6 13.125v1.5c0 .621-.504 1.125-1.125 1.125M6 13.125C6 12.504 5.496 12 4.875 12m-1.5 0h1.5m-1.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125M19.125 12h1.5m0 0c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h1.5m14.25 0h1.5" />
            </svg>
          </div>

          <h2 className="text-lg font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
            Drop a guitar video here
          </h2>
          <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
            or click to browse &middot; MP4 / MOV
          </p>

          <button
            className="btn btn-primary"
            onClick={(e) => { e.stopPropagation(); document.getElementById('file-input')?.click(); }}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
            Choose File
          </button>

          <input
            id="file-input"
            type="file"
            accept="video/mp4,video/quicktime"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>

        {/* Capo selector */}
        <div
          className="mt-4 rounded-xl px-5 py-4 flex items-center justify-between"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <div>
            <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Capo Position</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Set to 0 if no capo</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              className="btn btn-ghost btn-icon"
              onClick={() => setCapoFretInput(Math.max(0, capoFretInput - 1))}
              disabled={capoFretInput === 0}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 12h-15" />
              </svg>
            </button>
            <span
              className="w-10 text-center text-lg font-semibold tabular-nums"
              style={{ color: capoFretInput > 0 ? 'var(--accent-primary)' : 'var(--text-muted)' }}
            >
              {capoFretInput}
            </span>
            <button
              className="btn btn-ghost btn-icon"
              onClick={() => setCapoFretInput(Math.min(12, capoFretInput + 1))}
              disabled={capoFretInput === 12}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
              </svg>
            </button>
          </div>
        </div>

        {/* Tips */}
        <div className="mt-4 px-1">
          <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            For best results: Clean guitar audio (no backing track), guitar neck visible and roughly horizontal, standard tuning, max 5 minutes.
          </p>
        </div>
      </div>
    );
  }

  // === PROCESSING: Pipeline visualization ===
  if (isProcessing) {
    return (
      <div className="w-full max-w-md animate-slide-up">
        <div
          className="rounded-2xl p-8"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          {/* File name */}
          {fileName && (
            <div className="flex items-center gap-2 mb-6 pb-4" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
              <svg className="w-4 h-4 shrink-0" fill="none" stroke="var(--text-muted)" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125m-9.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125" />
              </svg>
              <span className="text-sm truncate" style={{ color: 'var(--text-secondary)' }}>{fileName}</span>
            </div>
          )}

          {/* Stage indicators */}
          <div className="space-y-3 mb-6">
            {PIPELINE_STAGES.map((stage, idx) => {
              const isDone = idx < currentStageIndex || (idx === currentStageIndex && progress >= 0.95);
              const isActive = idx === currentStageIndex && !isDone;
              const isPending = idx > currentStageIndex;

              return (
                <div
                  key={stage.key}
                  className="flex items-center gap-3 transition-all duration-300"
                  style={{
                    opacity: isPending ? 0.35 : 1,
                    animation: isActive ? 'stage-enter 0.3s ease-out' : undefined,
                  }}
                >
                  {/* Stage indicator dot */}
                  <div
                    className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 transition-all duration-300"
                    style={{
                      background: isDone
                        ? 'var(--color-success-soft)'
                        : isActive
                        ? 'var(--accent-glow)'
                        : 'rgba(255,255,255,0.04)',
                      border: `1px solid ${
                        isDone
                          ? 'rgba(16, 185, 129, 0.3)'
                          : isActive
                          ? 'rgba(99, 102, 241, 0.3)'
                          : 'var(--border-subtle)'
                      }`,
                    }}
                  >
                    {isDone ? (
                      <svg className="w-3.5 h-3.5" fill="none" stroke="var(--color-success)" viewBox="0 0 24 24" strokeWidth={2.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                      </svg>
                    ) : isActive ? (
                      <div
                        className="w-3 h-3 rounded-full animate-spin-slow"
                        style={{
                          border: '2px solid var(--accent-primary)',
                          borderTopColor: 'transparent',
                        }}
                      />
                    ) : (
                      <StageIcon icon={stage.icon} active={false} done={false} />
                    )}
                  </div>

                  {/* Stage label */}
                  <span
                    className="text-sm font-medium"
                    style={{
                      color: isDone
                        ? 'var(--color-success)'
                        : isActive
                        ? 'var(--text-primary)'
                        : 'var(--text-muted)',
                    }}
                  >
                    {stage.label}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Progress bar */}
          <div className="progress-bar mb-2">
            <div
              className="progress-bar-fill"
              style={{ width: `${Math.max(progress * 100, 2)}%` }}
            />
          </div>
          <p className="text-xs text-center tabular-nums" style={{ color: 'var(--text-muted)' }}>
            {Math.round(progress * 100)}% complete
          </p>
        </div>
      </div>
    );
  }

  // === FAILED: Error state ===
  if (jobStatus === 'failed') {
    return (
      <div className="w-full max-w-md animate-fade-in">
        <div
          className="rounded-2xl p-8 text-center"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid rgba(244, 63, 94, 0.2)',
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
          <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
            {errorMessage || 'An unexpected error occurred'}
          </p>
          <button className="btn btn-primary" onClick={reset}>
            Try Again
          </button>
        </div>
      </div>
    );
  }

  // === COMPLETED (fallback - normally editor shows) ===
  return null;
}
