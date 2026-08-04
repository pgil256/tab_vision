import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { useProcessVideo } from '../utils/useProcessVideo';
import { detectPitchHz, midiFloatFromHz, noteNameForMidi } from '../utils/audioAnalysis';
import { tuningPreset } from '../utils/pitch';
import { TranscriptionOptions } from './TranscriptionOptions';
import { AudioReviewPanel } from './AudioReviewPanel';

type RecorderState = 'idle' | 'preview' | 'countin' | 'recording';
type CaptureMode = 'video' | 'audio';

const BPM_MIN = 40;
const BPM_MAX = 240;

// Only mime types whose file extension the backend accepts (mp4 / webm).
function pickMimeType(mode: CaptureMode): string {
  const video = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm;codecs=vp9',
    'video/webm',
    'video/mp4',
  ];
  const audio = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
  ];
  const candidates = mode === 'audio' ? audio : video;
  for (const type of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return '';
}

// What the live tuner is currently hearing. When the pitch is within reach of
// an open string of the selected tuning, we tune toward that string (like a
// clip-on tuner); otherwise fall back to the nearest chromatic note.
interface TunerReading {
  /** Display note name (target string or nearest chromatic note). */
  noteName: string;
  /** Signed cents from the display note, −50..50 (clamped for the needle). */
  cents: number;
  /** 1–6 when tuning toward a string of the selected preset, null when chromatic. */
  stringNumber: number | null;
}

/** Semitone distance within which a detected pitch locks onto a preset string. */
const TUNER_STRING_LOCK = 1.5;
const TUNER_POLL_MS = 150;
const TUNER_HISTORY = 4;
const TUNER_MAX_MISSES = 3;

// How the audible click behaves once recording starts. The visual beat dots
// always keep pulsing, so the metronome stays usable even when silent.
type ClickMode = 'always' | 'fade' | 'countin' | 'off';

// Fade mode: this many full-volume bars into the take, then one bar of
// fade-down to silence. Enough to lock in tempo; the clicked bars can be
// trimmed away afterwards in the review step.
const FADE_FULL_BARS = 2;

// Click volume multiplier for a given beat. Beats before `recordStartBeat`
// are the count-in, which stays audible (it ends before recording begins,
// so it can never leak into the take).
function clickPeakFactor(
  mode: ClickMode,
  beatIndex: number,
  recordStartBeat: number,
  beatsPerBar: number,
): number {
  if (mode === 'off') return 0;
  if (beatIndex < recordStartBeat) return 1;
  if (mode === 'always') return 1;
  if (mode === 'countin') return 0;
  const recBeat = beatIndex - recordStartBeat;
  const fullBeats = FADE_FULL_BARS * beatsPerBar;
  if (recBeat < fullBeats) return 1;
  const fade = (recBeat - fullBeats) / beatsPerBar;
  return fade >= 1 ? 0 : 1 - fade;
}

// Schedules metronome clicks on a WebAudio timeline so timing stays precise
// even if the main thread is busy. Returns a stop function.
function startMetronome({
  audioCtx,
  bpm,
  beatsPerBar,
  peakForBeat,
  startAtBeat = 0,
  onBeat,
}: {
  audioCtx: AudioContext;
  bpm: number;
  beatsPerBar: number;
  /** 0–1 volume multiplier per beat; 0 skips the audible click entirely. */
  peakForBeat: (beatIndex: number) => number;
  startAtBeat?: number;
  onBeat?: (beatIndex: number, barPosition: number) => void;
}): () => void {
  const beatInterval = 60 / bpm;
  const lookahead = 0.1; // seconds
  let nextBeatTime = audioCtx.currentTime + 0.05;
  let beatIndex = startAtBeat;
  let stopped = false;

  function scheduleClick(time: number, accent: boolean, peak: number) {
    if (peak <= 0) return;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.frequency.value = accent ? 1500 : 1000;
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
      scheduleClick(nextBeatTime, barPosition === 0, 0.25 * peakForBeat(beatIndex));
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
  const countInStartTimerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);
  // Audio-mode input-level meter
  const analyserRef = useRef<AnalyserNode | null>(null);
  const meterSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const levelRafRef = useRef<number | null>(null);

  const [captureMode, setCaptureMode] = useState<CaptureMode>('video');
  const [state, setState] = useState<RecorderState>('idle');
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const [bpm, setBpm] = useState(80);
  const [beatsPerBar, setBeatsPerBar] = useState(4);
  const [clickMode, setClickMode] = useState<ClickMode>('fade');
  const [countIn, setCountIn] = useState(true);
  const [pulseBeat, setPulseBeat] = useState<number | null>(null);
  const [countInRemaining, setCountInRemaining] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [inputLevel, setInputLevel] = useState(0);
  // Audio-mode takes pause here for listen-back/cleanup before upload.
  const [reviewFile, setReviewFile] = useState<File | null>(null);
  // Live tuner (preview state only)
  const [tunerReading, setTunerReading] = useState<TunerReading | null>(null);
  const tunerHistoryRef = useRef<number[]>([]);
  const tunerMissesRef = useRef(0);

  const { setError } = useAppStore();
  const tuningInput = useAppStore((s) => s.tuningInput);
  const { processVideo } = useProcessVideo();

  const stopLevelMeter = useCallback(() => {
    if (levelRafRef.current !== null) {
      cancelAnimationFrame(levelRafRef.current);
      levelRafRef.current = null;
    }
    meterSourceRef.current?.disconnect();
    meterSourceRef.current = null;
    analyserRef.current?.disconnect();
    analyserRef.current = null;
    setInputLevel(0);
  }, []);

  const startLevelMeter = useCallback((stream: MediaStream, showMeter: boolean) => {
    const ctx = audioCtxRef.current ?? new AudioContext();
    audioCtxRef.current = ctx;
    if (ctx.state === 'suspended') ctx.resume().catch(() => {});

    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    // 4096 samples ≈ 85 ms @48k — enough window for the tuner to resolve low E.
    analyser.fftSize = 4096;
    // Intentionally NOT connected to ctx.destination — monitoring would echo the mic.
    source.connect(analyser);
    meterSourceRef.current = source;
    analyserRef.current = analyser;

    // Video mode only needs the analyser (for the tuner) — skip the rAF meter.
    if (!showMeter) return;

    const data = new Uint8Array(analyser.fftSize);
    const tick = () => {
      if (!analyserRef.current) return;
      analyserRef.current.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / data.length);
      setInputLevel((prev) => {
        const next = Math.min(1, rms * 2.5);
        // Fast attack, slow decay so the meter reads naturally.
        return next > prev ? next : prev * 0.85 + next * 0.15;
      });
      levelRafRef.current = requestAnimationFrame(tick);
    };
    tick();
  }, []);

  const cleanupStream = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const stopEverything = useCallback(() => {
    if (countInStartTimerRef.current !== null) {
      window.clearTimeout(countInStartTimerRef.current);
      countInStartTimerRef.current = null;
    }
    stopMetronomeRef.current?.();
    stopMetronomeRef.current = null;
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop();
    }
  }, []);

  useEffect(() => {
    return () => {
      stopEverything();
      stopLevelMeter();
      cleanupStream();
      audioCtxRef.current?.close().catch(() => {});
    };
  }, [stopEverything, stopLevelMeter, cleanupStream]);

  // Elapsed timer during recording
  useEffect(() => {
    if (state !== 'recording') return;
    const id = window.setInterval(() => {
      setElapsed((performance.now() - startTimeRef.current) / 1000);
    }, 100);
    return () => window.clearInterval(id);
  }, [state]);

  // Live tuner: poll the analyser while the mic preview is live (before the
  // take starts). setInterval, not rAF — the beat-grid work showed rAF freezes
  // in hidden tabs.
  useEffect(() => {
    if (state !== 'preview') {
      setTunerReading(null);
      tunerHistoryRef.current = [];
      tunerMissesRef.current = 0;
      return;
    }
    const targets = tuningPreset(tuningInput).midi;
    const id = window.setInterval(() => {
      const analyser = analyserRef.current;
      const ctx = audioCtxRef.current;
      if (!analyser || !ctx) return;
      const frame = new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(frame);
      const hz = detectPitchHz(frame, ctx.sampleRate);

      if (hz === null) {
        // Hold the last reading briefly so the display doesn't flicker
        // between plucks; clear after a few consecutive silent polls.
        if (++tunerMissesRef.current >= TUNER_MAX_MISSES) {
          tunerHistoryRef.current = [];
          setTunerReading(null);
        }
        return;
      }
      tunerMissesRef.current = 0;
      const history = tunerHistoryRef.current;
      history.push(midiFloatFromHz(hz));
      if (history.length > TUNER_HISTORY) history.shift();
      const sorted = [...history].sort((a, b) => a - b);
      const midiFloat = sorted[Math.floor(sorted.length / 2)];

      let target = targets[0];
      for (const m of targets) {
        if (Math.abs(midiFloat - m) < Math.abs(midiFloat - target)) target = m;
      }
      const onString = Math.abs(midiFloat - target) <= TUNER_STRING_LOCK;
      const refMidi = onString ? target : Math.round(midiFloat);
      setTunerReading({
        noteName: noteNameForMidi(refMidi),
        cents: Math.max(-50, Math.min(50, (midiFloat - refMidi) * 100)),
        stringNumber: onString ? 6 - targets.indexOf(target) : null,
      });
    }, TUNER_POLL_MS);
    return () => window.clearInterval(id);
  }, [state, tuningInput]);

  const requestPreview = useCallback(async () => {
    setPermissionError(null);
    try {
      const audioConstraints = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      };
      const constraints: MediaStreamConstraints =
        captureMode === 'audio'
          ? { audio: audioConstraints }
          : {
              video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
              audio: audioConstraints,
            };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (captureMode === 'video' && videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        await videoRef.current.play().catch(() => {});
      }
      // Both modes: analyser drives the tuner; the bar meter is audio-only.
      startLevelMeter(stream, captureMode === 'audio');
      setState('preview');
    } catch (err) {
      const what = captureMode === 'audio' ? 'microphone' : 'camera/microphone';
      setPermissionError(err instanceof Error ? err.message : `Unable to access ${what}`);
    }
  }, [captureMode, startLevelMeter]);

  const beginRecording = useCallback(() => {
    const stream = streamRef.current;
    if (!stream) return;

    chunksRef.current = [];
    const mimeType = pickMimeType(captureMode);
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
      const type = recorder.mimeType || mimeType || (captureMode === 'audio' ? 'audio/webm' : 'video/webm');
      const ext = type.includes('mp4') ? 'mp4' : 'webm';
      const prefix = captureMode === 'audio' ? 'audio' : 'recording';
      const blob = new Blob(chunksRef.current, { type });
      const file = new File([blob], `${prefix}-${Date.now()}.${ext}`, { type });
      stopLevelMeter();
      cleanupStream();
      setState('idle');
      setElapsed(0);
      setPulseBeat(null);
      // Every take pauses at the review stage. Video takes preview their
      // audio track; cleanup there falls back to an audio-only upload.
      setReviewFile(file);
    };

    recorder.start(500);
    startTimeRef.current = performance.now();
    setState('recording');
    setElapsed(0);
  }, [captureMode, cleanupStream, stopLevelMeter]);

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
        // No count-in: recording starts at beat 0.
        peakForBeat: (beat) => clickPeakFactor(clickMode, beat, 0, beatsPerBar),
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
      // One count-in bar precedes the take: recording starts at beat `beatsPerBar`.
      peakForBeat: (beat) => clickPeakFactor(clickMode, beat, beatsPerBar, beatsPerBar),
      onBeat: (beatIndex, barPos) => {
        setPulseBeat(barPos);
        if (beatIndex < beatsPerBar) {
          ticks = beatIndex + 1;
          setCountInRemaining(Math.max(0, beatsPerBar - ticks));
          if (ticks === beatsPerBar) {
            setCountInRemaining(null);
            // Start recording just before the next downbeat
            countInStartTimerRef.current = window.setTimeout(() => {
              countInStartTimerRef.current = null;
              beginRecording();
            }, 0);
          }
        }
      },
    });
  }, [beginRecording, beatsPerBar, bpm, countIn, clickMode]);

  const stopRecording = useCallback(() => {
    const wasCountingIn = state === 'countin';
    stopEverything();
    setPulseBeat(null);
    setCountInRemaining(null);
    // A count-in has no MediaRecorder to fire onstop and restore the UI.
    // Keep the live preview available so Space can start another take.
    if (wasCountingIn) setState('preview');
  }, [state, stopEverything]);

  useEffect(() => {
    const handleRecordingShortcut = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented
        || event.repeat
        || event.code !== 'Space'
        || event.ctrlKey
        || event.metaKey
        || event.altKey
      ) {
        return;
      }

      const target = event.target as HTMLElement | null;
      if (
        target instanceof HTMLInputElement
        || target instanceof HTMLTextAreaElement
        || target instanceof HTMLSelectElement
        || target instanceof HTMLButtonElement
        || target instanceof HTMLAnchorElement
        || (target && target.isContentEditable)
      ) {
        return;
      }

      if (state === 'preview') {
        event.preventDefault();
        void startWithMetronome();
      } else if (state === 'recording' || state === 'countin') {
        event.preventDefault();
        stopRecording();
      }
    };

    window.addEventListener('keydown', handleRecordingShortcut);
    return () => window.removeEventListener('keydown', handleRecordingShortcut);
  }, [startWithMetronome, state, stopRecording]);

  const isLive = state === 'recording' || state === 'countin';

  // Switching capture mode tears down any live preview so constraints re-acquire.
  const handleModeChange = useCallback((next: CaptureMode) => {
    if (next === captureMode || isLive) return;
    stopEverything();
    stopLevelMeter();
    cleanupStream();
    if (videoRef.current) videoRef.current.srcObject = null;
    setState('idle');
    setElapsed(0);
    setPulseBeat(null);
    setCountInRemaining(null);
    setPermissionError(null);
    setCaptureMode(next);
  }, [captureMode, isLive, stopEverything, stopLevelMeter, cleanupStream]);

  const canStart = state === 'preview';
  const isAudio = captureMode === 'audio';

  if (reviewFile) {
    return (
      <AudioReviewPanel
        file={reviewFile}
        onSubmit={(file) => {
          setReviewFile(null);
          processVideo(file).catch((err) => {
            setError(err instanceof Error ? err.message : 'Processing failed');
          });
        }}
        onCancel={() => setReviewFile(null)}
        cancelLabel="Discard take"
        isVideo={reviewFile.type.startsWith('video/')}
      />
    );
  }

  return (
    <div className="capture-flow capture-flow--record animate-slide-up relative">
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
            {clickMode === 'always' &&
              'Use headphones for the click — otherwise it leaks into the recorded audio and confuses pitch detection.'}
            {clickMode === 'fade' &&
              `The click fades out after ${FADE_FULL_BARS} bars, so most of the take stays clean — trim the clicked intro in the review step, or wear headphones.`}
            {clickMode === 'countin' &&
              'The click only plays during the count-in, so it never overlaps the recording — no headphones needed.'}
            {clickMode === 'off' &&
              'Silent metronome — follow the pulsing beat dots to keep time.'}
          </p>
        </div>

        {/* Capture-mode toggle: audio-only or video + audio */}
        <div className="capture-mode-row">
          <div className="segmented" role="radiogroup" aria-label="Capture mode">
            {([
              { id: 'video', label: 'Video + audio' },
              { id: 'audio', label: 'Audio only' },
            ] as const).map((opt) => (
              <button
                key={opt.id}
                role="radio"
                aria-checked={captureMode === opt.id}
                onClick={() => handleModeChange(opt.id)}
                disabled={isLive}
                className={`segmented-btn ${captureMode === opt.id ? 'active' : ''}`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Preview */}
        <div
          className="record-preview rounded-xl overflow-hidden relative"
          style={{
            background: '#000',
            border: '1px solid var(--border-subtle)',
            aspectRatio: '16 / 9',
          }}
        >
          {/* Video preview (video mode only) */}
          {!isAudio && (
            <video
              ref={videoRef}
              playsInline
              muted
              className="w-full h-full object-cover"
              style={{ transform: 'scaleX(-1)' }}
            />
          )}

          {/* Audio-mode visualizer */}
          {isAudio && state !== 'idle' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
              <div className="flex items-end gap-1.5" style={{ height: '64px' }}>
                {Array.from({ length: 9 }).map((_, i) => {
                  // Center bars react most strongly to the input level.
                  const weight = 1 - Math.abs(i - 4) / 5;
                  const h = 8 + inputLevel * 56 * (0.4 + weight);
                  return (
                    <div
                      key={i}
                      className="rounded-full"
                      style={{
                        width: '6px',
                        height: `${Math.min(64, h)}px`,
                        background: 'linear-gradient(180deg, var(--accent-primary), var(--accent-secondary))',
                        transition: 'height 60ms linear',
                        opacity: 0.5 + weight * 0.5,
                      }}
                    />
                  );
                })}
              </div>
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="var(--text-muted)" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                </svg>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Microphone live — no camera
                </span>
              </div>
            </div>
          )}

          {/* Idle overlay: enable button */}
          {state === 'idle' && (
            <div className="absolute inset-0 flex items-center justify-center">
              <button className="btn btn-primary px-6" onClick={requestPreview}>
                {isAudio ? (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                  </svg>
                )}
                {isAudio ? 'Enable microphone' : 'Enable camera & mic'}
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

        {/* Live tuner — mic is hot but the take hasn't started */}
        {state === 'preview' && (() => {
          const inTune = tunerReading !== null && Math.abs(tunerReading.cents) <= 5;
          const accent = inTune ? 'var(--color-success)' : 'var(--color-warning)';
          return (
            <div className="field-card mt-4 p-4">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Tuner</p>
                <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                  {tuningPreset(tuningInput).name} · {tuningPreset(tuningInput).label}
                </p>
              </div>
              <div className="flex items-center gap-4">
                <div className="w-20 text-center shrink-0">
                  {tunerReading ? (
                    <>
                      <p className="text-2xl font-bold tabular-nums leading-none" style={{ color: accent }}>
                        {tunerReading.noteName}
                      </p>
                      <p className="text-[11px] mt-1" style={{ color: 'var(--text-muted)' }}>
                        {tunerReading.stringNumber !== null
                          ? `string ${tunerReading.stringNumber}`
                          : 'off scale'}
                      </p>
                    </>
                  ) : (
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Play a<br />string…</p>
                  )}
                </div>
                <div className="flex-1">
                  <div
                    className="relative h-2 rounded-full"
                    style={{ background: 'rgba(255,255,255,0.08)' }}
                  >
                    {/* Center (in-tune) mark */}
                    <div
                      className="absolute top-[-3px] bottom-[-3px] w-px"
                      style={{ left: '50%', background: 'var(--text-muted)' }}
                    />
                    {tunerReading && (
                      <div
                        className="absolute top-[-4px] w-4 h-4 rounded-full transition-all duration-150"
                        style={{
                          left: `calc(${50 + tunerReading.cents}% - 8px)`,
                          background: accent,
                          boxShadow: `0 0 8px ${accent}`,
                        }}
                      />
                    )}
                  </div>
                  <div className="flex justify-between mt-1.5 text-[10px] tabular-nums" style={{ color: 'var(--text-muted)' }}>
                    <span>−50¢</span>
                    <span style={{ color: tunerReading ? accent : 'var(--text-muted)' }}>
                      {tunerReading
                        ? `${tunerReading.cents > 0 ? '+' : ''}${Math.round(tunerReading.cents)}¢${inTune ? ' — in tune' : ''}`
                        : 'listening'}
                    </span>
                    <span>+50¢</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })()}

        {/* Metronome controls */}
        <div className="field-card metronome-card mt-4 p-4">
          <div className="flex items-center justify-between mb-3 gap-3">
            <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Metronome</p>
            <div className="flex items-center gap-2">
              <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Click</span>
              <select
                value={clickMode}
                onChange={(e) => setClickMode(e.target.value as ClickMode)}
                disabled={isLive}
                className="select"
                aria-label="Click behavior"
                style={{ width: 'auto' }}
              >
                <option value="fade">Fade out after {FADE_FULL_BARS} bars</option>
                <option value="countin">Count-in only</option>
                <option value="always">Whole take</option>
                <option value="off">Off (visual only)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <p className="text-[11px] mb-1" style={{ color: 'var(--text-muted)' }}>BPM</p>
              <div className="flex items-center gap-1.5">
                <button
                  className="btn btn-ghost btn-icon"
                  onClick={() => setBpm(Math.max(BPM_MIN, bpm - 1))}
                  disabled={isLive || bpm <= BPM_MIN}
                  aria-label="Decrease metronome tempo"
                  data-tooltip="Decrease tempo"
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
                  aria-label="Increase metronome tempo"
                  data-tooltip="Increase tempo"
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
                className="select"
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
                  className="toggle"
                  checked={countIn}
                  onChange={(e) => setCountIn(e.target.checked)}
                  disabled={isLive}
                />
                One bar
              </label>
            </div>
          </div>
        </div>

        {/* Transcription settings — same options as file upload; the fretboard
            area only applies when a camera is involved. */}
        <TranscriptionOptions showRoi={!isAudio} />

        {/* Action buttons */}
        <div className="record-actions">
          {state !== 'recording' && state !== 'countin' && (
            <button
              className="btn btn-primary flex-1 record-primary-action"
              onClick={startWithMetronome}
              disabled={!canStart}
              style={{ padding: '10px 16px' }}
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="6" />
              </svg>
              Start recording
              <span className="record-action-hint">Space</span>
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
