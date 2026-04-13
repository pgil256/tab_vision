import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { useProcessVideo } from '../utils/useProcessVideo';

type RecorderState = 'idle' | 'preview' | 'countin' | 'recording';

const BPM_MIN = 40;
const BPM_MAX = 240;

function pickMimeType(): string {
  const candidates = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm;codecs=vp9',
    'video/webm',
    'video/mp4',
  ];
  for (const type of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return '';
}

// Schedules metronome clicks on a WebAudio timeline so timing stays precise
// even if the main thread is busy. Returns a stop function.
function startMetronome({
  audioCtx,
  bpm,
  beatsPerBar,
  muted,
  startAtBeat = 0,
  onBeat,
}: {
  audioCtx: AudioContext;
  bpm: number;
  beatsPerBar: number;
  muted: boolean;
  startAtBeat?: number;
  onBeat?: (beatIndex: number, barPosition: number) => void;
}): () => void {
  const beatInterval = 60 / bpm;
  const lookahead = 0.1; // seconds
  let nextBeatTime = audioCtx.currentTime + 0.05;
  let beatIndex = startAtBeat;
  let stopped = false;

  function scheduleClick(time: number, accent: boolean) {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.frequency.value = accent ? 1500 : 1000;
    const peak = muted ? 0 : 0.25;
    gain.gain.setValueAtTime(0, time);
    gain.gain.linearRampToValueAtTime(peak, time + 0.002);
    gain.gain.exponentialRampToValueAtTime(0.001, time + 0.06);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start(time);
    osc.stop(time + 0.08);
  }

  const timer = window.setInterval(() => {
    if (stopped) return;
    while (nextBeatTime < audioCtx.currentTime + lookahead) {
      const barPosition = beatIndex % beatsPerBar;
      scheduleClick(nextBeatTime, barPosition === 0);
      const scheduledIndex = beatIndex;
      const delayMs = Math.max(0, (nextBeatTime - audioCtx.currentTime) * 1000);
      window.setTimeout(() => onBeat?.(scheduledIndex, scheduledIndex % beatsPerBar), delayMs);
      nextBeatTime += beatInterval;
      beatIndex += 1;
    }
  }, 25);

  return () => {
    stopped = true;
    window.clearInterval(timer);
  };
}

export function RecordPanel() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const stopMetronomeRef = useRef<(() => void) | null>(null);
  const startTimeRef = useRef<number>(0);

  const [state, setState] = useState<RecorderState>('idle');
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const [bpm, setBpm] = useState(80);
  const [beatsPerBar, setBeatsPerBar] = useState(4);
  const [muted, setMuted] = useState(false);
  const [countIn, setCountIn] = useState(true);
  const [pulseBeat, setPulseBeat] = useState<number | null>(null);
  const [countInRemaining, setCountInRemaining] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const { setError } = useAppStore();
  const processVideo = useProcessVideo();

  const cleanupStream = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const stopEverything = useCallback(() => {
    stopMetronomeRef.current?.();
    stopMetronomeRef.current = null;
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop();
    }
  }, []);

  useEffect(() => {
    return () => {
      stopEverything();
      cleanupStream();
      audioCtxRef.current?.close().catch(() => {});
    };
  }, [stopEverything, cleanupStream]);

  // Elapsed timer during recording
  useEffect(() => {
    if (state !== 'recording') return;
    const id = window.setInterval(() => {
      setElapsed((performance.now() - startTimeRef.current) / 1000);
    }, 100);
    return () => window.clearInterval(id);
  }, [state]);

  const requestPreview = useCallback(async () => {
    setPermissionError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        await videoRef.current.play().catch(() => {});
      }
      setState('preview');
    } catch (err) {
      setPermissionError(err instanceof Error ? err.message : 'Unable to access camera/microphone');
    }
  }, []);

  const beginRecording = useCallback(() => {
    const stream = streamRef.current;
    if (!stream) return;

    chunksRef.current = [];
    const mimeType = pickMimeType();
    let recorder: MediaRecorder;
    try {
      recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    } catch (err) {
      setPermissionError(err instanceof Error ? err.message : 'Recorder init failed');
      return;
    }
    recorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      const type = recorder.mimeType || mimeType || 'video/webm';
      const ext = type.includes('mp4') ? 'mp4' : 'webm';
      const blob = new Blob(chunksRef.current, { type });
      const file = new File([blob], `recording-${Date.now()}.${ext}`, { type });
      cleanupStream();
      setState('idle');
      setElapsed(0);
      setPulseBeat(null);
      processVideo(file).catch((err) => {
        setError(err instanceof Error ? err.message : 'Processing failed');
      });
    };

    recorder.start(500);
    startTimeRef.current = performance.now();
    setState('recording');
    setElapsed(0);
  }, [cleanupStream, processVideo, setError]);

  const startWithMetronome = useCallback(async () => {
    if (!streamRef.current) return;
    const ctx = audioCtxRef.current ?? new AudioContext();
    audioCtxRef.current = ctx;
    if (ctx.state === 'suspended') await ctx.resume();

    stopMetronomeRef.current?.();
    stopMetronomeRef.current = null;

    if (!countIn) {
      beginRecording();
      stopMetronomeRef.current = startMetronome({
        audioCtx: ctx,
        bpm,
        beatsPerBar,
        muted,
        onBeat: (_, barPos) => setPulseBeat(barPos),
      });
      return;
    }

    // Count-in: one bar of clicks, then start recording on beat 0 of next bar
    setState('countin');
    setCountInRemaining(beatsPerBar);
    let ticks = 0;
    stopMetronomeRef.current = startMetronome({
      audioCtx: ctx,
      bpm,
      beatsPerBar,
      muted,
      onBeat: (beatIndex, barPos) => {
        setPulseBeat(barPos);
        if (beatIndex < beatsPerBar) {
          ticks = beatIndex + 1;
          setCountInRemaining(Math.max(0, beatsPerBar - ticks));
          if (ticks === beatsPerBar) {
            setCountInRemaining(null);
            // Start recording just before the next downbeat
            window.setTimeout(() => beginRecording(), 0);
          }
        }
      },
    });
  }, [beginRecording, beatsPerBar, bpm, countIn, muted]);

  const stopRecording = useCallback(() => {
    stopEverything();
    setPulseBeat(null);
    setCountInRemaining(null);
  }, [stopEverything]);

  const canStart = state === 'preview';
  const isLive = state === 'recording' || state === 'countin';

  return (
    <div className="w-full max-w-xl animate-slide-up relative">
      <div className="ambient-bg" />
      <div className="relative z-10">
        <div className="text-center mb-6">
          <div
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full mb-4"
            style={{
              background: 'rgba(251, 113, 133, 0.08)',
              border: '1px solid rgba(251, 113, 133, 0.2)',
            }}
          >
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--color-error)' }} />
            <span className="text-xs font-medium" style={{ color: 'var(--color-error)' }}>
              Record in Browser
            </span>
          </div>
          <h2 className="text-2xl font-bold mb-2 tracking-tight" style={{ color: 'var(--text-primary)' }}>
            Record a take with metronome
          </h2>
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Use headphones for the click — otherwise it leaks into the recorded audio and confuses pitch detection.
          </p>
        </div>

        {/* Preview */}
        <div
          className="rounded-xl overflow-hidden relative"
          style={{
            background: '#000',
            border: '1px solid var(--border-subtle)',
            aspectRatio: '16 / 9',
          }}
        >
          <video
            ref={videoRef}
            playsInline
            muted
            className="w-full h-full object-cover"
            style={{ transform: 'scaleX(-1)' }}
          />
          {state === 'idle' && (
            <div className="absolute inset-0 flex items-center justify-center">
              <button className="btn btn-primary px-6" onClick={requestPreview}>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                </svg>
                Enable camera &amp; mic
              </button>
            </div>
          )}
          {isLive && (
            <div className="absolute top-3 left-3 flex items-center gap-2 px-2.5 py-1 rounded-md" style={{ background: 'rgba(0,0,0,0.55)' }}>
              <div
                className="w-2 h-2 rounded-full animate-pulse"
                style={{ background: state === 'recording' ? 'var(--color-error)' : 'var(--color-warning)' }}
              />
              <span className="text-xs font-medium tabular-nums" style={{ color: 'white' }}>
                {state === 'recording' ? `REC ${elapsed.toFixed(1)}s` : `Count-in ${countInRemaining ?? ''}`}
              </span>
            </div>
          )}
          {isLive && pulseBeat !== null && (
            <div className="absolute top-3 right-3 flex gap-1">
              {Array.from({ length: beatsPerBar }).map((_, i) => (
                <div
                  key={i}
                  className="w-2 h-2 rounded-full transition-all duration-75"
                  style={{
                    background: i === pulseBeat
                      ? (i === 0 ? 'var(--accent-tertiary)' : 'var(--text-primary)')
                      : 'rgba(255,255,255,0.25)',
                    transform: i === pulseBeat ? 'scale(1.4)' : 'scale(1)',
                  }}
                />
              ))}
            </div>
          )}
        </div>

        {permissionError && (
          <p className="mt-3 text-xs" style={{ color: 'var(--color-error)' }}>{permissionError}</p>
        )}

        {/* Metronome controls */}
        <div
          className="mt-4 rounded-xl p-4"
          style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}
        >
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Metronome</p>
            <label className="flex items-center gap-2 text-xs cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
              <input
                type="checkbox"
                checked={muted}
                onChange={(e) => setMuted(e.target.checked)}
                disabled={isLive}
              />
              Mute click (visual only)
            </label>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <p className="text-[11px] mb-1" style={{ color: 'var(--text-muted)' }}>BPM</p>
              <div className="flex items-center gap-1.5">
                <button
                  className="btn btn-ghost btn-icon"
                  onClick={() => setBpm(Math.max(BPM_MIN, bpm - 1))}
                  disabled={isLive || bpm <= BPM_MIN}
                  style={{ padding: '4px' }}
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 12h-15" />
                  </svg>
                </button>
                <input
                  type="number"
                  min={BPM_MIN}
                  max={BPM_MAX}
                  value={bpm}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (!isNaN(v)) setBpm(Math.min(BPM_MAX, Math.max(BPM_MIN, v)));
                  }}
                  disabled={isLive}
                  className="w-14 text-center text-base font-bold tabular-nums bg-transparent focus:outline-none"
                  style={{ color: 'var(--accent-tertiary)' }}
                />
                <button
                  className="btn btn-ghost btn-icon"
                  onClick={() => setBpm(Math.min(BPM_MAX, bpm + 1))}
                  disabled={isLive || bpm >= BPM_MAX}
                  style={{ padding: '4px' }}
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                  </svg>
                </button>
              </div>
            </div>
            <div>
              <p className="text-[11px] mb-1" style={{ color: 'var(--text-muted)' }}>Beats / bar</p>
              <select
                value={beatsPerBar}
                onChange={(e) => setBeatsPerBar(parseInt(e.target.value, 10))}
                disabled={isLive}
                className="w-full rounded-md px-2 py-1.5 text-sm"
                style={{
                  background: 'var(--bg-base)',
                  border: '1px solid var(--border-subtle)',
                  color: 'var(--text-primary)',
                }}
              >
                {[2, 3, 4, 5, 6, 7, 8].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
            <div>
              <p className="text-[11px] mb-1" style={{ color: 'var(--text-muted)' }}>Count-in</p>
              <label className="flex items-center gap-2 text-sm cursor-pointer h-[34px]" style={{ color: 'var(--text-primary)' }}>
                <input
                  type="checkbox"
                  checked={countIn}
                  onChange={(e) => setCountIn(e.target.checked)}
                  disabled={isLive}
                />
                One bar
              </label>
            </div>
          </div>
        </div>

        {/* Action buttons */}
        <div className="mt-4 flex gap-2">
          {state !== 'recording' && state !== 'countin' && (
            <button
              className="btn btn-primary flex-1"
              onClick={startWithMetronome}
              disabled={!canStart}
              style={{ padding: '10px 16px' }}
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="6" />
              </svg>
              Start recording
            </button>
          )}
          {(state === 'recording' || state === 'countin') && (
            <button
              className="btn btn-secondary flex-1"
              onClick={stopRecording}
              style={{ padding: '10px 16px' }}
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
              Stop &amp; transcribe
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
