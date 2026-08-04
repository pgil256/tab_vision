// Client-side audio cleanup for the pre-transcription review step.
// Decodes an uploaded/recorded audio file, lets the UI preview edits
// (trim / gain / normalize / high-pass), and renders the final result
// to a 16-bit PCM WAV the backend already accepts.

export interface CleanupSettings {
  /** Seconds into the take where the kept region starts. */
  trimStart: number;
  /** Seconds into the take where the kept region ends. */
  trimEnd: number;
  /** Manual gain in dB, applied before normalization. */
  gainDb: number;
  /** Scale the trimmed result so its peak sits at NORMALIZE_PEAK. */
  normalize: boolean;
  /** High-pass cutoff in Hz to cut rumble/hum; 0 disables the filter. */
  highPassHz: number;
}

export const NORMALIZE_PEAK = 0.95;

export function defaultSettings(duration: number): CleanupSettings {
  return { trimStart: 0, trimEnd: duration, gainDb: 0, normalize: false, highPassHz: 0 };
}

/** True when the settings would leave the audio bit-identical — upload the original instead. */
export function isPassthrough(s: CleanupSettings, duration: number): boolean {
  return (
    s.trimStart <= 0 &&
    s.trimEnd >= duration - 1e-3 &&
    s.gainDb === 0 &&
    !s.normalize &&
    s.highPassHz === 0
  );
}

export function dbToGain(db: number): number {
  return Math.pow(10, db / 20);
}

export async function decodeAudioFile(ctx: BaseAudioContext, file: File): Promise<AudioBuffer> {
  const bytes = await file.arrayBuffer();
  return ctx.decodeAudioData(bytes);
}

/** Min/max amplitude per bin across all channels, for waveform drawing. */
export function computePeaks(buffer: AudioBuffer, bins: number): { min: Float32Array; max: Float32Array } {
  const min = new Float32Array(bins);
  const max = new Float32Array(bins);
  const samplesPerBin = buffer.length / bins;
  const channels: Float32Array[] = [];
  for (let c = 0; c < buffer.numberOfChannels; c++) channels.push(buffer.getChannelData(c));

  for (let i = 0; i < bins; i++) {
    const start = Math.floor(i * samplesPerBin);
    const end = Math.min(buffer.length, Math.ceil((i + 1) * samplesPerBin));
    let lo = 0;
    let hi = 0;
    for (const data of channels) {
      for (let j = start; j < end; j++) {
        const v = data[j];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    min[i] = lo;
    max[i] = hi;
  }
  return { min, max };
}

/** Absolute peak of the [start, end] window in seconds, across channels. */
export function windowPeak(buffer: AudioBuffer, start: number, end: number): number {
  const from = Math.max(0, Math.floor(start * buffer.sampleRate));
  const to = Math.min(buffer.length, Math.ceil(end * buffer.sampleRate));
  let peak = 0;
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = from; i < to; i++) {
      const v = Math.abs(data[i]);
      if (v > peak) peak = v;
    }
  }
  return peak;
}

/**
 * Renders the cleaned take (trim → high-pass → gain → optional normalize)
 * to a mono/stereo 16-bit WAV File named after the source file.
 */
export async function renderCleanedWav(
  buffer: AudioBuffer,
  settings: CleanupSettings,
  sourceName: string,
): Promise<File> {
  const duration = Math.max(0.05, settings.trimEnd - settings.trimStart);
  const channels = Math.min(2, buffer.numberOfChannels);
  const length = Math.ceil(duration * buffer.sampleRate);
  const offline = new OfflineAudioContext(channels, length, buffer.sampleRate);

  const source = offline.createBufferSource();
  source.buffer = buffer;
  let node: AudioNode = source;

  if (settings.highPassHz > 0) {
    const hp = offline.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.value = settings.highPassHz;
    hp.Q.value = 0.707;
    node.connect(hp);
    node = hp;
  }

  const gain = offline.createGain();
  gain.gain.value = dbToGain(settings.gainDb);
  node.connect(gain);
  gain.connect(offline.destination);

  source.start(0, settings.trimStart, duration);
  const rendered = await offline.startRendering();

  if (settings.normalize) {
    let peak = 0;
    for (let c = 0; c < rendered.numberOfChannels; c++) {
      const data = rendered.getChannelData(c);
      for (let i = 0; i < data.length; i++) {
        const v = Math.abs(data[i]);
        if (v > peak) peak = v;
      }
    }
    if (peak > 0) {
      const scale = NORMALIZE_PEAK / peak;
      for (let c = 0; c < rendered.numberOfChannels; c++) {
        const data = rendered.getChannelData(c);
        for (let i = 0; i < data.length; i++) data[i] *= scale;
      }
    }
  }

  const wav = encodeWav(rendered);
  const base = sourceName.replace(/\.[^.]+$/, '') || 'audio';
  return new File([wav], `${base}-clean.wav`, { type: 'audio/wav' });
}

// Interleaved 16-bit PCM WAV.
function encodeWav(buffer: AudioBuffer): ArrayBuffer {
  const channels = buffer.numberOfChannels;
  const frames = buffer.length;
  const sampleRate = buffer.sampleRate;
  const bytesPerFrame = channels * 2;
  const dataSize = frames * bytesPerFrame;
  const out = new ArrayBuffer(44 + dataSize);
  const view = new DataView(out);

  const writeString = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerFrame, true);
  view.setUint16(32, bytesPerFrame, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  const channelData: Float32Array[] = [];
  for (let c = 0; c < channels; c++) channelData.push(buffer.getChannelData(c));

  let offset = 44;
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < channels; c++) {
      const clamped = Math.max(-1, Math.min(1, channelData[c][i]));
      view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
      offset += 2;
    }
  }
  return out;
}
