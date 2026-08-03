// Karplus–Strong pluck synthesis for note audition. Buffers are precomputed
// per MIDI pitch in plain JS (no AudioWorklet) and cached — the guitar range
// needs at most a few dozen — so each scheduled note is a one-shot
// BufferSource -> Gain graph that is trivially cancellable.

const cache = new Map<string, AudioBuffer>();

const BUFFER_SECONDS = 1.2;
const DAMPING = 0.996;

export function getPluckBuffer(ctx: AudioContext, midi: number): AudioBuffer {
  const sr = ctx.sampleRate;
  const key = `${sr}:${midi}`;
  const cached = cache.get(key);
  if (cached) return cached;

  const f0 = 440 * Math.pow(2, (midi - 69) / 12);
  const period = Math.max(2, Math.round(sr / f0));
  const length = Math.round(sr * BUFFER_SECONDS);
  const data = new Float32Array(length);

  // Noise-filled delay line; averaging filter with damping on each pass.
  const delay = new Float32Array(period);
  for (let i = 0; i < period; i++) delay[i] = Math.random() * 2 - 1;
  let idx = 0;
  for (let i = 0; i < length; i++) {
    const current = delay[idx];
    const next = delay[(idx + 1) % period];
    data[i] = current;
    delay[idx] = DAMPING * 0.5 * (current + next);
    idx = (idx + 1) % period;
  }

  const buffer = ctx.createBuffer(1, length, sr);
  buffer.copyToChannel(data, 0);
  cache.set(key, buffer);
  return buffer;
}

/** Schedule one pluck at `when` (AudioContext time). Returns the source so
 * the caller can cancel it on seek/pause. */
export function schedulePluck(
  ctx: AudioContext,
  destination: AudioNode,
  midi: number,
  when: number,
  velocity: number,
  durationS: number
): AudioBufferSourceNode {
  const source = ctx.createBufferSource();
  source.buffer = getPluckBuffer(ctx, midi);
  const gain = ctx.createGain();
  const level = Math.max(0.05, Math.min(1, velocity));
  const end = when + Math.max(0.15, Math.min(durationS, BUFFER_SECONDS));
  gain.gain.setValueAtTime(level, when);
  gain.gain.exponentialRampToValueAtTime(0.001, end);
  source.connect(gain);
  gain.connect(destination);
  source.start(when);
  source.stop(end + 0.05);
  return source;
}
