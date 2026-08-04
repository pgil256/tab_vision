// Client-side take analysis: frame pitch detection (YIN), clipping detection,
// concert-pitch offset estimation, and silence-trim suggestion. Used by the
// pre-transcription review step and the live tuner in the record panel.

const MIN_HZ = 60;
const MAX_HZ = 1000;
const YIN_THRESHOLD = 0.15;
const YIN_FALLBACK_MAX = 0.35;
const RMS_GATE = 0.005;

/**
 * YIN pitch detector (cumulative mean normalized difference) on a single
 * time-domain frame. Returns the fundamental in Hz, or null when the frame
 * is too quiet or aperiodic.
 */
export function detectPitchHz(frame: Float32Array, sampleRate: number): number | null {
  const n = frame.length;
  const maxLag = Math.min(Math.floor(sampleRate / MIN_HZ), Math.floor(n / 2));
  const minLag = Math.max(2, Math.floor(sampleRate / MAX_HZ));
  const w = n - maxLag;
  if (w < maxLag / 2) return null;

  let energy = 0;
  for (let i = 0; i < w; i++) energy += frame[i] * frame[i];
  if (Math.sqrt(energy / w) < RMS_GATE) return null;

  // Difference function
  const d = new Float32Array(maxLag + 1);
  for (let lag = 1; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < w; i++) {
      const diff = frame[i] - frame[i + lag];
      sum += diff * diff;
    }
    d[lag] = sum;
  }

  // Cumulative mean normalized difference
  const cmnd = new Float32Array(maxLag + 1);
  cmnd[0] = 1;
  let running = 0;
  for (let lag = 1; lag <= maxLag; lag++) {
    running += d[lag];
    cmnd[lag] = running > 0 ? (d[lag] * lag) / running : 1;
  }

  // First dip below threshold (descend to its local minimum), else global min.
  let lag = -1;
  for (let l = minLag; l <= maxLag; l++) {
    if (cmnd[l] < YIN_THRESHOLD) {
      while (l + 1 <= maxLag && cmnd[l + 1] < cmnd[l]) l++;
      lag = l;
      break;
    }
  }
  if (lag === -1) {
    let best = minLag;
    for (let l = minLag; l <= maxLag; l++) if (cmnd[l] < cmnd[best]) best = l;
    if (cmnd[best] > YIN_FALLBACK_MAX) return null;
    lag = best;
  }

  // Parabolic interpolation for sub-sample lag
  let refined = lag;
  if (lag > minLag && lag < maxLag) {
    const a = cmnd[lag - 1];
    const b = cmnd[lag];
    const c = cmnd[lag + 1];
    const denom = a - 2 * b + c;
    if (Math.abs(denom) > 1e-9) refined = lag + (a - c) / (2 * denom) / 1;
  }
  return sampleRate / refined;
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export function midiFloatFromHz(hz: number): number {
  return 69 + 12 * Math.log2(hz / 440);
}

export function noteNameForMidi(midi: number): string {
  return `${NOTE_NAMES[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

export interface TakeAnalysis {
  /** Samples inside flat-topped full-scale runs (true digital clipping). */
  clippedSamples: number;
  /** Number of distinct clipped runs. */
  clippedRuns: number;
  /** Circular-mean offset from concert pitch in cents (−50..50), null if too few voiced frames. */
  tuningCents: number | null;
  /** Resultant length of the circular mean, 0–1: how consistent the offset is. */
  tuningConfidence: number;
  voicedFrames: number;
}

const CLIP_LEVEL = 0.97;
const CLIP_MIN_RUN = 4;
const ANALYSIS_FRAME = 2048;
const MAX_ANALYSIS_FRAMES = 80;

export function analyzeTake(buffer: AudioBuffer): TakeAnalysis {
  // --- Clipping: flat runs at (or beyond) full scale, worst channel wins.
  let clippedSamples = 0;
  let clippedRuns = 0;
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    let run = 0;
    let chSamples = 0;
    let chRuns = 0;
    for (let i = 0; i < data.length; i++) {
      if (Math.abs(data[i]) >= CLIP_LEVEL) {
        run++;
      } else {
        if (run >= CLIP_MIN_RUN) {
          chSamples += run;
          chRuns++;
        }
        run = 0;
      }
    }
    if (run >= CLIP_MIN_RUN) {
      chSamples += run;
      chRuns++;
    }
    if (chSamples > clippedSamples) {
      clippedSamples = chSamples;
      clippedRuns = chRuns;
    }
  }

  // --- Concert-pitch offset: YIN over sampled frames, circular mean of the
  // per-frame deviation from the nearest semitone (deviation lives mod 100
  // cents, so a plain average would cancel ±50 wraparounds).
  const mono = buffer.getChannelData(0);
  const hop = Math.max(ANALYSIS_FRAME, Math.floor(mono.length / MAX_ANALYSIS_FRAMES));
  let sumSin = 0;
  let sumCos = 0;
  let voiced = 0;
  for (let start = 0; start + ANALYSIS_FRAME <= mono.length; start += hop) {
    const hz = detectPitchHz(mono.subarray(start, start + ANALYSIS_FRAME), buffer.sampleRate);
    if (hz === null) continue;
    const midiFloat = midiFloatFromHz(hz);
    const dev = (midiFloat - Math.round(midiFloat)) * 100; // −50..50 cents
    const angle = (dev / 100) * 2 * Math.PI;
    sumSin += Math.sin(angle);
    sumCos += Math.cos(angle);
    voiced++;
  }
  let tuningCents: number | null = null;
  let tuningConfidence = 0;
  if (voiced > 0) {
    tuningCents = (Math.atan2(sumSin, sumCos) / (2 * Math.PI)) * 100;
    tuningConfidence = Math.sqrt(sumSin * sumSin + sumCos * sumCos) / voiced;
  }

  return { clippedSamples, clippedRuns, tuningCents, tuningConfidence, voicedFrames: voiced };
}

// Thresholds the review UI applies to TakeAnalysis.
export const CLIPPING_WARN_SAMPLES = 256;
export const TUNING_WARN_CENTS = 20;
export const TUNING_MIN_CONFIDENCE = 0.45;
export const TUNING_MIN_FRAMES = 8;

const TRIM_WINDOW = 1024;
const TRIM_PAD_START = 0.15;
const TRIM_PAD_END = 0.25;

/**
 * Suggested trim bounds: first/last moment the signal rises above a noise
 * floor, padded slightly outward. Null when the take is essentially silent
 * or has no silence worth trimming.
 */
export function findAutoTrim(buffer: AudioBuffer): { start: number; end: number } | null {
  const data = buffer.getChannelData(0);
  const windows = Math.floor(data.length / TRIM_WINDOW);
  if (windows < 2) return null;

  const rms = new Float32Array(windows);
  let peak = 0;
  for (let wi = 0; wi < windows; wi++) {
    let e = 0;
    const base = wi * TRIM_WINDOW;
    for (let i = 0; i < TRIM_WINDOW; i++) e += data[base + i] * data[base + i];
    rms[wi] = Math.sqrt(e / TRIM_WINDOW);
    if (rms[wi] > peak) peak = rms[wi];
  }

  const threshold = Math.max(peak * 0.05, 0.003);
  let first = -1;
  let last = -1;
  for (let wi = 0; wi < windows; wi++) {
    if (rms[wi] >= threshold) {
      if (first === -1) first = wi;
      last = wi;
    }
  }
  if (first === -1) return null;

  const secPerWindow = TRIM_WINDOW / buffer.sampleRate;
  const start = Math.max(0, first * secPerWindow - TRIM_PAD_START);
  const end = Math.min(buffer.duration, (last + 1) * secPerWindow + TRIM_PAD_END);
  // Nothing meaningful to trim → treat as no suggestion.
  if (start < 0.05 && end > buffer.duration - 0.05) return null;
  return { start, end };
}
