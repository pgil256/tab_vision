import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CleanupSettings,
  computePeaks,
  dbToGain,
  decodeAudioFile,
  defaultSettings,
  isPassthrough,
  NORMALIZE_PEAK,
  renderCleanedWav,
  windowPeak,
} from '../utils/audioCleanup';
import {
  analyzeTake,
  CLIPPING_WARN_SAMPLES,
  findAutoTrim,
  TUNING_MIN_CONFIDENCE,
  TUNING_MIN_FRAMES,
  TUNING_WARN_CENTS,
} from '../utils/audioAnalysis';
import { TranscriptionOptions } from './TranscriptionOptions';

const MIN_KEEP_SECONDS = 0.25;
const WAVEFORM_BINS = 600;

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `${m}:${sec.toFixed(1).padStart(4, '0')}`;
}

interface AudioReviewPanelProps {
  file: File;
  /** Called with the file to transcribe — cleaned WAV, or the original when nothing changed. */
  onSubmit: (file: File) => void;
  onCancel: () => void;
  cancelLabel: string;
  /**
   * The source is a video file. Its audio track is decoded for preview/cleanup,
   * but applying any cleanup means uploading audio only — the browser can't
   * re-mux processed audio back into the video container — which skips the
   * fretboard-tracking half of the pipeline.
   */
  isVideo?: boolean;
}

// Listen-back + cleanup stage between capturing/choosing a take and sending
// it for transcription. Preview edits run live through WebAudio; the same
// chain is re-rendered offline to a WAV on submit.
export function AudioReviewPanel({ file, onSubmit, onCancel, cancelLabel, isVideo = false }: AudioReviewPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const filterRef = useRef<BiquadFilterNode | null>(null);
  const playStartRef = useRef<{ ctxTime: number; offset: number } | null>(null);
  const rafRef = useRef<number | null>(null);

  const [buffer, setBuffer] = useState<AudioBuffer | null>(null);
  const [decodeError, setDecodeError] = useState<string | null>(null);
  const [settings, setSettings] = useState<CleanupSettings | null>(null);
  const [playing, setPlaying] = useState(false);
  const [playhead, setPlayhead] = useState<number | null>(null);
  const [rendering, setRendering] = useState(false);

  const duration = buffer?.duration ?? 0;

  // Decode once per file.
  useEffect(() => {
    let cancelled = false;
    const ctx = new AudioContext();
    ctxRef.current = ctx;
    decodeAudioFile(ctx, file)
      .then((buf) => {
        if (cancelled) return;
        setBuffer(buf);
        setSettings(defaultSettings(buf.duration));
      })
      .catch(() => {
        if (!cancelled) setDecodeError('This browser could not decode the audio for preview.');
      });
    return () => {
      cancelled = true;
      ctx.close().catch(() => {});
      ctxRef.current = null;
    };
  }, [file]);

  const peaks = useMemo(() => (buffer ? computePeaks(buffer, WAVEFORM_BINS) : null), [buffer]);
  // One-time health scan of the decoded take (clipping, concert-pitch offset)
  // plus the suggested silence trim.
  const analysis = useMemo(() => (buffer ? analyzeTake(buffer) : null), [buffer]);
  const autoTrim = useMemo(() => (buffer ? findAutoTrim(buffer) : null), [buffer]);
  // Peak of the kept window, for the normalize preview gain. windowPeak is a
  // full scan, so only recompute when the trim actually changes.
  const keptPeak = useMemo(
    () => (buffer && settings ? windowPeak(buffer, settings.trimStart, settings.trimEnd) : 0),
    [buffer, settings?.trimStart, settings?.trimEnd],
  );

  const effectiveGain = useCallback(
    (s: CleanupSettings) =>
      dbToGain(s.gainDb) * (s.normalize && keptPeak > 0 ? NORMALIZE_PEAK / (keptPeak * dbToGain(s.gainDb)) : 1),
    [keptPeak],
  );

  const stopPlayback = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.onended = null;
      try {
        sourceRef.current.stop();
      } catch {
        // already stopped
      }
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    gainRef.current?.disconnect();
    gainRef.current = null;
    filterRef.current?.disconnect();
    filterRef.current = null;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    playStartRef.current = null;
    setPlaying(false);
  }, []);

  useEffect(() => stopPlayback, [stopPlayback]);

  const startPlayback = useCallback(async () => {
    const ctx = ctxRef.current;
    if (!ctx || !buffer || !settings) return;
    if (ctx.state === 'suspended') await ctx.resume();
    stopPlayback();

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    let node: AudioNode = source;

    // Filter node always in the chain; 10 Hz is inaudibly transparent when "off"
    // so toggling mid-playback never needs a graph rebuild.
    const hp = ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = settings.highPassHz > 0 ? settings.highPassHz : 10;
    hp.Q.value = 0.707;
    node.connect(hp);
    node = hp;
    filterRef.current = hp;

    const gain = ctx.createGain();
    gain.gain.value = effectiveGain(settings);
    node.connect(gain);
    gain.connect(ctx.destination);
    gainRef.current = gain;

    const keepDur = Math.max(MIN_KEEP_SECONDS, settings.trimEnd - settings.trimStart);
    source.onended = () => stopPlayback();
    source.start(0, settings.trimStart, keepDur);
    sourceRef.current = source;
    playStartRef.current = { ctxTime: ctx.currentTime, offset: settings.trimStart };
    setPlaying(true);

    const tick = () => {
      const start = playStartRef.current;
      const actx = ctxRef.current;
      if (!start || !actx) return;
      setPlayhead(start.offset + (actx.currentTime - start.ctxTime));
      rafRef.current = requestAnimationFrame(tick);
    };
    tick();
  }, [buffer, settings, effectiveGain, stopPlayback]);

  // Live-apply gain / filter changes to a running preview.
  useEffect(() => {
    if (!settings) return;
    if (gainRef.current) gainRef.current.gain.value = effectiveGain(settings);
    if (filterRef.current) {
      filterRef.current.frequency.value = settings.highPassHz > 0 ? settings.highPassHz : 10;
    }
  }, [settings, effectiveGain]);

  const updateSettings = useCallback(
    (patch: Partial<CleanupSettings>, restartsPlayback = false) => {
      if (restartsPlayback) stopPlayback();
      setSettings((prev) => (prev ? { ...prev, ...patch } : prev));
    },
    [stopPlayback],
  );

  // Waveform + trim shading + playhead.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !peaks || !settings || duration === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const g = canvas.getContext('2d');
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const mid = h / 2;
    const startX = (settings.trimStart / duration) * w;
    const endX = (settings.trimEnd / duration) * w;
    const gainScale = Math.min(3, effectiveGain(settings));

    for (let i = 0; i < WAVEFORM_BINS; i++) {
      const x = (i / WAVEFORM_BINS) * w;
      const inKeep = x >= startX && x <= endX;
      const scale = inKeep ? gainScale : 1;
      const top = mid + Math.max(-1, peaks.min[i] * scale) * (mid - 2);
      const bot = mid + Math.min(1, peaks.max[i] * scale) * (mid - 2);
      g.strokeStyle = inKeep ? 'rgba(139, 92, 246, 0.9)' : 'rgba(255, 255, 255, 0.18)';
      g.lineWidth = Math.max(1, w / WAVEFORM_BINS - 0.5);
      g.beginPath();
      g.moveTo(x, top);
      g.lineTo(x, Math.max(bot, top + 1));
      g.stroke();
    }

    // Trimmed-out regions get a darkening overlay + boundary lines.
    g.fillStyle = 'rgba(0, 0, 0, 0.45)';
    if (startX > 0) g.fillRect(0, 0, startX, h);
    if (endX < w) g.fillRect(endX, 0, w - endX, h);
    g.strokeStyle = 'rgba(251, 191, 36, 0.9)';
    g.lineWidth = 1.5;
    for (const x of [startX, endX]) {
      g.beginPath();
      g.moveTo(x, 0);
      g.lineTo(x, h);
      g.stroke();
    }

    if (playhead !== null && playing) {
      const px = (playhead / duration) * w;
      g.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      g.lineWidth = 1.5;
      g.beginPath();
      g.moveTo(px, 0);
      g.lineTo(px, h);
      g.stroke();
    }
  }, [peaks, settings, duration, playhead, playing, effectiveGain]);

  const handleSubmit = useCallback(async () => {
    stopPlayback();
    // No decoded buffer (unsupported format) or untouched settings → send the original.
    if (!buffer || !settings || isPassthrough(settings, duration)) {
      onSubmit(file);
      return;
    }
    setRendering(true);
    try {
      const cleaned = await renderCleanedWav(buffer, settings, file.name);
      onSubmit(cleaned);
    } catch {
      // Rendering failed — fall back to the untouched original rather than dead-ending.
      onSubmit(file);
    } finally {
      setRendering(false);
    }
  }, [buffer, settings, duration, file, onSubmit, stopPlayback]);

  const gainFill = settings ? `${((settings.gainDb + 12) / 24) * 100}%` : '50%';
  const changed = settings !== null && !isPassthrough(settings, duration);

  return (
    <div className="w-full max-w-xl animate-slide-up relative">
      <div className="ambient-bg" />
      <div className="relative z-10">
        <div className="text-center mb-6">
          <div
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full mb-4"
            style={{ background: 'var(--accent-glow)', border: '1px solid var(--border-accent)' }}
          >
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-tertiary)' }} />
            <span className="text-xs font-medium" style={{ color: 'var(--accent-tertiary)' }}>
              Review Your Take
            </span>
          </div>
          <h2 className="text-2xl font-bold mb-2 tracking-tight" style={{ color: 'var(--text-primary)' }}>
            Listen back before transcribing
          </h2>
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Trim silence, boost a quiet take, or cut low rumble — then send it off
          </p>
        </div>

        {/* Waveform */}
        <div
          className="rounded-xl overflow-hidden relative"
          style={{ background: 'rgba(0,0,0,0.35)', border: '1px solid var(--border-subtle)' }}
        >
          {decodeError ? (
            <div className="h-28 flex items-center justify-center px-6 text-center">
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{decodeError}</p>
            </div>
          ) : !buffer ? (
            <div className="h-28 flex items-center justify-center gap-2">
              <div
                className="w-3 h-3 rounded-full animate-spin-slow"
                style={{ border: '2px solid var(--accent-tertiary)', borderTopColor: 'transparent' }}
              />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Decoding audio…</span>
            </div>
          ) : (
            <canvas ref={canvasRef} className="w-full block" style={{ height: '112px' }} />
          )}
        </div>

        {buffer && settings && (
          <>
            {/* Transport row */}
            <div className="mt-3 flex items-center gap-3">
              <button
                className="btn btn-primary btn-icon"
                onClick={playing ? stopPlayback : startPlayback}
                aria-label={playing ? 'Stop preview' : 'Play preview'}
                style={{ padding: '10px' }}
              >
                {playing ? (
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <rect x="6" y="6" width="12" height="12" rx="1" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5.14v13.72a1 1 0 001.5.87l11-6.86a1 1 0 000-1.74l-11-6.86a1 1 0 00-1.5.87z" />
                  </svg>
                )}
              </button>
              <p className="text-xs tabular-nums" style={{ color: 'var(--text-secondary)' }}>
                {playing && playhead !== null ? formatTime(playhead) : formatTime(settings.trimStart)}
                {' / '}
                {formatTime(duration)}
                <span style={{ color: 'var(--text-muted)' }}>
                  {' '}&middot; keeping {formatTime(Math.max(0, settings.trimEnd - settings.trimStart))}
                </span>
              </p>
              {changed && (
                <span
                  className="ml-auto text-[11px] px-2 py-0.5 rounded-full"
                  style={{ background: 'var(--accent-glow)', color: 'var(--accent-tertiary)' }}
                >
                  Will upload cleaned WAV
                </span>
              )}
            </div>

            {/* Take-health warnings: problems no cleanup slider can fix. */}
            {analysis && analysis.clippedSamples >= CLIPPING_WARN_SAMPLES && (
              <div
                className="mt-3 flex items-start gap-2 px-3 py-2 rounded-lg"
                style={{ background: 'var(--color-error-soft)', border: '1px solid rgba(251, 113, 133, 0.25)' }}
              >
                <svg className="w-4 h-4 shrink-0 mt-0.5" fill="none" stroke="var(--color-error)" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                </svg>
                <p className="text-[11px] leading-relaxed" style={{ color: 'var(--color-error)' }}>
                  Clipping detected in {analysis.clippedRuns} spot{analysis.clippedRuns === 1 ? '' : 's'} —
                  the input was too loud and the waveform is flattened. Cleanup can&apos;t undo this
                  and it distorts pitch detection; if this is your take, re-record with lower input gain.
                </p>
              </div>
            )}
            {analysis &&
              analysis.tuningCents !== null &&
              Math.abs(analysis.tuningCents) >= TUNING_WARN_CENTS &&
              analysis.tuningConfidence >= TUNING_MIN_CONFIDENCE &&
              analysis.voicedFrames >= TUNING_MIN_FRAMES && (
              <div
                className="mt-3 flex items-start gap-2 px-3 py-2 rounded-lg"
                style={{ background: 'var(--color-warning-soft)', border: '1px solid rgba(251, 191, 36, 0.25)' }}
              >
                <svg className="w-4 h-4 shrink-0 mt-0.5" fill="none" stroke="var(--color-warning)" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303" />
                </svg>
                <p className="text-[11px] leading-relaxed" style={{ color: 'var(--color-warning)' }}>
                  The guitar reads ~{Math.abs(Math.round(analysis.tuningCents))} cents{' '}
                  {analysis.tuningCents > 0 ? 'sharp' : 'flat'} of concert pitch. The transcriber
                  snaps notes to in-tune pitches, so tuning up and re-recording will give a
                  noticeably better tab.
                </p>
              </div>
            )}

            {changed && isVideo && (
              <div
                className="mt-3 flex items-start gap-2 px-3 py-2 rounded-lg"
                style={{ background: 'var(--color-warning-soft)', border: '1px solid rgba(251, 191, 36, 0.25)' }}
              >
                <svg className="w-4 h-4 shrink-0 mt-0.5" fill="none" stroke="var(--color-warning)" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                </svg>
                <p className="text-[11px] leading-relaxed" style={{ color: 'var(--color-warning)' }}>
                  Cleanup can&apos;t be re-embedded into the video, so the cleaned upload is
                  audio-only and fretboard tracking will be skipped. Reset the settings to keep
                  the full video pipeline.
                </p>
              </div>
            )}

            {/* Trim */}
            <div className="field-card mt-4 p-4">
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Trim</p>
                {autoTrim && (
                  <button
                    className="btn btn-ghost text-xs"
                    style={{ padding: '4px 10px' }}
                    onClick={() =>
                      updateSettings(
                        {
                          trimStart: Math.min(autoTrim.start, duration - MIN_KEEP_SECONDS),
                          trimEnd: Math.max(autoTrim.end, autoTrim.start + MIN_KEEP_SECONDS),
                        },
                        true,
                      )
                    }
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M7.848 8.25l1.536.887M7.848 8.25a3 3 0 11-5.196-3 3 3 0 015.196 3zm1.536.887a2.165 2.165 0 011.083 1.839c.005.351.054.695.14 1.024M9.384 9.137l2.077 1.199M7.848 15.75l1.536-.887m-1.536.887a3 3 0 11-5.196 3 3 3 0 015.196-3zm1.536-.887a2.165 2.165 0 001.083-1.838c.005-.352.054-.695.14-1.025m-1.223 2.863l2.077-1.199m0-3.328a4.323 4.323 0 012.068-1.379l5.325-1.628a4.5 4.5 0 012.48-.044l.803.215-7.794 4.5m-2.882-1.664A4.331 4.331 0 0010.607 12m3.736 0l7.794 4.5-.802.215a4.5 4.5 0 01-2.48-.043l-5.326-1.629a4.324 4.324 0 01-2.068-1.379M14.343 12l-2.882 1.664" />
                    </svg>
                    Auto-trim silence
                  </button>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[11px] mb-1 tabular-nums" style={{ color: 'var(--text-muted)' }}>
                    Start · {formatTime(settings.trimStart)}
                  </p>
                  <input
                    type="range"
                    className="slider"
                    min={0}
                    max={duration}
                    step={0.05}
                    value={settings.trimStart}
                    style={{ ['--fill' as string]: `${(settings.trimStart / duration) * 100}%` }}
                    onChange={(e) => {
                      const v = Math.min(parseFloat(e.target.value), settings.trimEnd - MIN_KEEP_SECONDS);
                      updateSettings({ trimStart: Math.max(0, v) }, true);
                    }}
                  />
                </div>
                <div>
                  <p className="text-[11px] mb-1 tabular-nums" style={{ color: 'var(--text-muted)' }}>
                    End · {formatTime(settings.trimEnd)}
                  </p>
                  <input
                    type="range"
                    className="slider"
                    min={0}
                    max={duration}
                    step={0.05}
                    value={settings.trimEnd}
                    style={{ ['--fill' as string]: `${(settings.trimEnd / duration) * 100}%` }}
                    onChange={(e) => {
                      const v = Math.max(parseFloat(e.target.value), settings.trimStart + MIN_KEEP_SECONDS);
                      updateSettings({ trimEnd: Math.min(duration, v) }, true);
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Level + filter */}
            <div className="field-card mt-3 p-4">
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Level &amp; clarity</p>
                <label className="flex items-center gap-2 text-xs cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
                  <input
                    type="checkbox"
                    className="toggle"
                    checked={settings.normalize}
                    onChange={(e) => updateSettings({ normalize: e.target.checked })}
                  />
                  Normalize loudness
                </label>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[11px] mb-1 tabular-nums" style={{ color: 'var(--text-muted)' }}>
                    Gain · {settings.gainDb >= 0 ? '+' : ''}{settings.gainDb.toFixed(1)} dB
                    {settings.normalize && <span> (before normalize)</span>}
                  </p>
                  <input
                    type="range"
                    className="slider"
                    min={-12}
                    max={12}
                    step={0.5}
                    value={settings.gainDb}
                    style={{ ['--fill' as string]: gainFill }}
                    onChange={(e) => updateSettings({ gainDb: parseFloat(e.target.value) })}
                  />
                </div>
                <div>
                  <p className="text-[11px] mb-1 tabular-nums" style={{ color: 'var(--text-muted)' }}>
                    Rumble filter · {settings.highPassHz > 0 ? `${settings.highPassHz} Hz high-pass` : 'off'}
                  </p>
                  <input
                    type="range"
                    className="slider"
                    min={0}
                    max={160}
                    step={20}
                    value={settings.highPassHz}
                    style={{ ['--fill' as string]: `${(settings.highPassHz / 160) * 100}%` }}
                    onChange={(e) => updateSettings({ highPassHz: parseInt(e.target.value, 10) })}
                  />
                </div>
              </div>
              <p className="mt-3 text-[11px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                The transcriber is robust to mic noise, so light touches are enough. Keep the rumble
                filter at or below 80&nbsp;Hz — low E on a guitar sits at 82&nbsp;Hz.
              </p>
            </div>
          </>
        )}

        <TranscriptionOptions showRoi={isVideo} />

        {/* Actions */}
        <div className="mt-4 flex gap-2">
          <button className="btn btn-secondary" onClick={() => { stopPlayback(); onCancel(); }} disabled={rendering}>
            {cancelLabel}
          </button>
          <button className="btn btn-primary flex-1" onClick={handleSubmit} disabled={rendering}>
            {rendering ? (
              <>
                <div
                  className="w-3.5 h-3.5 rounded-full animate-spin-slow"
                  style={{ border: '2px solid currentColor', borderTopColor: 'transparent' }}
                />
                Rendering…
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303" />
                </svg>
                {changed ? 'Transcribe cleaned audio' : 'Transcribe'}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
