// Client-side Standard MIDI File writer for the (edited) TabDocument.
//
// Mirrors the server renderer (tabvision/tabvision/render/midi.py): format 1,
// 480 ticks per quarter, one channel per string (channel 0 = low E), velocity
// from confidence. Tempo comes from the document's detected beat grid
// (fallback 120 BPM). Quantization happens on a copy — the document itself is
// never mutated.

import { TabDocument } from '../types/tab';
import { buildQuantizeGrid, getBeatGrid, snapToGrid } from './beatGrid';
import { midiPitchForNote } from './pitch';

export interface MidiExportOptions {
  quantize?: boolean;
  /** Grid subdivision when quantizing: 16 = sixteenth notes. */
  subdivision?: number;
}

const TPQ = 480;
const MIN_NOTE_SECONDS = 0.05;
const DEFAULT_NOTE_SECONDS = 0.25;

/** Variable-length quantity encoding (big-endian 7-bit groups). */
function vlq(value: number): number[] {
  let v = Math.max(0, Math.floor(value));
  const bytes = [v & 0x7f];
  while ((v >>= 7) > 0) bytes.unshift((v & 0x7f) | 0x80);
  return bytes;
}

function trackChunk(body: number[]): number[] {
  const len = body.length;
  return [
    0x4d, 0x54, 0x72, 0x6b, // 'MTrk'
    (len >>> 24) & 0xff, (len >>> 16) & 0xff, (len >>> 8) & 0xff, len & 0xff,
    ...body,
  ];
}

export function exportToMidi(doc: TabDocument, opts: MidiExportOptions = {}): Uint8Array {
  const grid = getBeatGrid(doc);
  const bpm = grid?.tempoBpm ?? 120;
  const beatsPerBar = grid?.beatsPerBar ?? 4;
  const secondsToTicks = (s: number) => Math.round(((s * bpm) / 60) * TPQ);

  let notes = doc.notes.filter(n => typeof n.fret === 'number');
  if (opts.quantize && grid) {
    const qGrid = buildQuantizeGrid(grid.beatTimes, opts.subdivision ?? 16);
    notes = notes.map(n => {
      const snapped = snapToGrid(n.timestamp, qGrid);
      const delta = snapped - n.timestamp;
      return {
        ...n,
        timestamp: snapped,
        endTime: typeof n.endTime === 'number' ? n.endTime + delta : n.endTime,
      };
    });
  }

  // Track 0: tempo + time signature metas.
  const usPerQuarter = Math.round(60_000_000 / bpm);
  const track0 = [
    ...vlq(0), 0xff, 0x51, 0x03,
    (usPerQuarter >>> 16) & 0xff, (usPerQuarter >>> 8) & 0xff, usPerQuarter & 0xff,
    ...vlq(0), 0xff, 0x58, 0x04, beatsPerBar & 0xff, 2 /* denominator 2^2=4 */, 24, 8,
    ...vlq(0), 0xff, 0x2f, 0x00,
  ];

  // Track 1: note events. Channel = 6 - string (channel 0 = low E).
  interface Ev { tick: number; off: boolean; channel: number; pitch: number; velocity: number }
  const events: Ev[] = [];
  for (const n of notes) {
    const pitch = Math.min(127, Math.max(0, midiPitchForNote(n.string, n.fret as number, doc.capoFret)));
    const channel = 6 - n.string;
    const velocity = Math.min(127, Math.max(1, Math.round((n.confidence ?? 1) * 127)));
    const start = Math.max(0, n.timestamp);
    const durS = Math.max(
      MIN_NOTE_SECONDS,
      (typeof n.endTime === 'number' ? n.endTime : n.timestamp + DEFAULT_NOTE_SECONDS) - n.timestamp
    );
    const onTick = secondsToTicks(start);
    const offTick = Math.max(onTick + 1, secondsToTicks(start + durS));
    events.push({ tick: onTick, off: false, channel, pitch, velocity });
    events.push({ tick: offTick, off: true, channel, pitch, velocity: 0 });
  }
  // note_off first at equal ticks so retriggered pitches never get stuck.
  events.sort((a, b) => a.tick - b.tick || Number(b.off) - Number(a.off));

  const track1: number[] = [];
  let lastTick = 0;
  for (const ev of events) {
    track1.push(...vlq(ev.tick - lastTick));
    lastTick = ev.tick;
    track1.push((ev.off ? 0x80 : 0x90) | ev.channel, ev.pitch, ev.velocity);
  }
  track1.push(...vlq(0), 0xff, 0x2f, 0x00);

  const header = [
    0x4d, 0x54, 0x68, 0x64, // 'MThd'
    0, 0, 0, 6,
    0, 1, // format 1
    0, 2, // two tracks
    (TPQ >> 8) & 0xff, TPQ & 0xff,
  ];
  return new Uint8Array([...header, ...trackChunk(track0), ...trackChunk(track1)]);
}
