/**
 * M3 headless verification — parses the client-side SMF writer's output
 * byte-for-byte (the project has no test runner; this mirrors the existing
 * standalone verification-script pattern).
 *
 * Run:  npx tsx scripts/verify-midi-export.mts
 */
import { exportToMidi } from '../src/utils/exportMidi';
import { buildQuantizeGrid, snapToGrid } from '../src/utils/beatGrid';
import { midiPitchForNote } from '../src/utils/pitch';
import type { TabDocument, TabNote } from '../src/types/tab';

let failures = 0;
function check(name: string, cond: boolean) {
  if (cond) {
    console.log(`  ok  ${name}`);
  } else {
    failures++;
    console.error(`FAIL  ${name}`);
  }
}

function note(
  id: string,
  string: TabNote['string'],
  fret: number | 'X',
  t: number,
  end?: number,
  confidence = 0.9
): TabNote {
  return {
    id,
    string,
    fret,
    timestamp: t,
    endTime: end,
    confidence,
    confidenceLevel: 'high',
    isEdited: false,
  };
}

const BPM = 90;
const beatTimes: number[] = [];
for (let i = 0; i < 16; i++) beatTimes.push(0.5 + (i * 60) / BPM);

function makeDoc(capo = 0): TabDocument {
  return {
    id: 'doc',
    createdAt: '',
    duration: 12,
    capoFret: capo,
    tuning: ['E', 'B', 'G', 'D', 'A', 'E'],
    notes: [
      note('n1', 6, 3, 1.0, 1.5), // low E fret 3 -> MIDI 43, channel 0
      note('n2', 1, 0, 2.0, 4.0), // high e open -> MIDI 64, channel 5
      note('n3', 3, 7, 2.0), // no endTime -> default duration
      note('n4', 4, 'X', 3.0, 3.2), // muted -> skipped
    ],
    metadata: { tempoBpm: BPM, beatTimes, beatsPerBar: 4 },
  };
}

// --- Minimal SMF reader ---
interface MidiEvent { tick: number; type: 'on' | 'off'; channel: number; pitch: number; velocity: number }
function parse(bytes: Uint8Array) {
  const u16 = (o: number) => (bytes[o] << 8) | bytes[o + 1];
  const u32 = (o: number) =>
    ((bytes[o] << 24) | (bytes[o + 1] << 16) | (bytes[o + 2] << 8) | bytes[o + 3]) >>> 0;
  const magic = (o: number, s: string) =>
    String.fromCharCode(...bytes.slice(o, o + s.length)) === s;

  const header = {
    mthd: magic(0, 'MThd'),
    format: u16(8),
    ntrks: u16(10),
    tpq: u16(12),
  };

  let offset = 14;
  const tracks: { tempoUsPerQuarter: number | null; timeSig: number[] | null; events: MidiEvent[] }[] = [];
  for (let tr = 0; tr < header.ntrks; tr++) {
    if (!magic(offset, 'MTrk')) throw new Error(`track ${tr}: bad MTrk magic at ${offset}`);
    const len = u32(offset + 4);
    let p = offset + 8;
    const end = p + len;
    let tick = 0;
    let tempoUsPerQuarter: number | null = null;
    let timeSig: number[] | null = null;
    const events: MidiEvent[] = [];
    while (p < end) {
      let delta = 0;
      while (bytes[p] & 0x80) { delta = (delta << 7) | (bytes[p] & 0x7f); p++; }
      delta = (delta << 7) | bytes[p]; p++;
      tick += delta;
      const status = bytes[p];
      if (status === 0xff) {
        const metaType = bytes[p + 1];
        const metaLen = bytes[p + 2];
        if (metaType === 0x51) {
          tempoUsPerQuarter = (bytes[p + 3] << 16) | (bytes[p + 4] << 8) | bytes[p + 5];
        } else if (metaType === 0x58) {
          timeSig = [bytes[p + 3], bytes[p + 4], bytes[p + 5], bytes[p + 6]];
        }
        p += 3 + metaLen;
      } else if ((status & 0xf0) === 0x90 || (status & 0xf0) === 0x80) {
        events.push({
          tick,
          type: (status & 0xf0) === 0x90 ? 'on' : 'off',
          channel: status & 0x0f,
          pitch: bytes[p + 1],
          velocity: bytes[p + 2],
        });
        p += 3;
      } else {
        throw new Error(`unexpected status 0x${status.toString(16)} at ${p}`);
      }
    }
    tracks.push({ tempoUsPerQuarter, timeSig, events });
    offset = end;
  }
  return { header, tracks };
}

// --- Header + metas ---
const doc = makeDoc();
const bytes = exportToMidi(doc);
const parsed = parse(bytes);
check('MThd magic', parsed.header.mthd);
check('format 1', parsed.header.format === 1);
check('two tracks', parsed.header.ntrks === 2);
check('480 TPQ', parsed.header.tpq === 480);
check(
  `tempo meta = ${BPM} BPM`,
  parsed.tracks[0].tempoUsPerQuarter === Math.round(60_000_000 / BPM)
);
check('time signature 4/4', JSON.stringify(parsed.tracks[0].timeSig) === '[4,2,24,8]');

// --- Note events ---
const evs = parsed.tracks[1].events;
const ons = evs.filter(e => e.type === 'on');
const offs = evs.filter(e => e.type === 'off');
check('3 playable notes -> 3 note_on (muted skipped)', ons.length === 3);
check('3 note_off', offs.length === 3);

const ticksPerSecond = (BPM / 60) * 480; // 720 at 90 BPM
const n1 = ons.find(e => e.pitch === 43);
check('low E fret 3 -> pitch 43 on channel 0', !!n1 && n1.channel === 0);
check('n1 onset tick = 1.0s * 720', !!n1 && n1.tick === Math.round(1.0 * ticksPerSecond));
const n1off = offs.find(e => e.pitch === 43);
check('n1 off tick = 1.5s * 720', !!n1off && n1off.tick === Math.round(1.5 * ticksPerSecond));
const n2 = ons.find(e => e.pitch === 64);
check('high e open -> pitch 64 on channel 5', !!n2 && n2.channel === 5);
const n3 = ons.find(e => e.pitch === midiPitchForNote(3, 7, 0));
check('G string fret 7 -> pitch 62 on channel 3', !!n3 && n3.channel === 3 && n3.pitch === 62);
const n3off = offs.find(e => e.pitch === 62);
check(
  'missing endTime -> 0.25s default duration',
  !!n3 && !!n3off && n3off.tick - n3.tick === Math.round(0.25 * ticksPerSecond)
);
check('velocity = confidence * 127', ons.every(e => e.velocity === Math.round(0.9 * 127)));
check('ticks monotonically ordered', evs.every((e, i) => i === 0 || evs[i - 1].tick <= e.tick));

// --- Capo shifts pitch ---
const capoBytes = exportToMidi(makeDoc(2));
const capoOns = parse(capoBytes).tracks[1].events.filter(e => e.type === 'on');
check(
  'capo 2 raises every pitch by 2',
  JSON.stringify(capoOns.map(e => e.pitch).sort((a, b) => a - b)) ===
    JSON.stringify(ons.map(e => e.pitch + 2).sort((a, b) => a - b))
);

// --- Quantization ---
// Beat spacing at 90 BPM = 0.667s; 16ths = 0.1667s. beatTimes start at 0.5.
const qDoc = makeDoc();
qDoc.notes = [note('q1', 6, 3, 1.21, 1.71)]; // nearest 16th: 0.5 + 4*(1/6) ≈ 1.1667
const grid16 = buildQuantizeGrid(beatTimes, 16);
const expectedSnap = snapToGrid(1.21, grid16);
check('snap target is a real grid point off the raw time', Math.abs(expectedSnap - 1.21) > 1e-6);
const qBytes = exportToMidi(qDoc, { quantize: true });
const qOns = parse(qBytes).tracks[1].events.filter(e => e.type === 'on');
const qOffs = parse(qBytes).tracks[1].events.filter(e => e.type === 'off');
check('quantized onset lands on the grid', qOns[0].tick === Math.round(expectedSnap * ticksPerSecond));
check(
  'quantize shifts endTime by the same delta (duration preserved)',
  qOffs[0].tick - qOns[0].tick === Math.round(0.5 * ticksPerSecond)
);
check('quantize does not mutate the document', qDoc.notes[0].timestamp === 1.21 && qDoc.notes[0].endTime === 1.71);

// --- Empty/edge ---
const emptyDoc: TabDocument = { ...makeDoc(), notes: [] };
const emptyParsed = parse(exportToMidi(emptyDoc));
check('empty doc still yields a valid 2-track file', emptyParsed.header.ntrks === 2);
const noGridDoc: TabDocument = { ...makeDoc(), metadata: {} };
noGridDoc.notes = [note('n1', 6, 3, 1.0, 1.5)];
const ngParsed = parse(exportToMidi(noGridDoc, { quantize: true }));
check('no beat grid -> 120 BPM fallback, quantize is a no-op', ngParsed.tracks[0].tempoUsPerQuarter === 500_000);

if (failures) {
  console.error(`\n${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('\nAll MIDI export checks passed.');
