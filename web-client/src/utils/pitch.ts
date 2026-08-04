// Shared pitch table for the store (pitch-preserving moves), MIDI export,
// and the audition synth.

// Tuning presets offered by the client. `midi` is low E to high E (index 0 =
// string 6), matching the server registry in tabvision-server/app/models.py —
// the two lists must stay in sync. Only presets the server accepts appear here.
export type TuningId =
  | 'standard'
  | 'drop-d'
  | 'eb-standard'
  | 'd-standard'
  | 'drop-c'
  | 'dadgad'
  | 'open-g';

export interface TuningPreset {
  id: TuningId;
  name: string;
  /** Open-string note names, low to high (display). */
  label: string;
  /** Open-string MIDI, low E-side to high E-side. */
  midi: number[];
}

export const TUNING_PRESETS: TuningPreset[] = [
  { id: 'standard', name: 'Standard', label: 'E A D G B E', midi: [40, 45, 50, 55, 59, 64] },
  { id: 'drop-d', name: 'Drop D', label: 'D A D G B E', midi: [38, 45, 50, 55, 59, 64] },
  { id: 'eb-standard', name: 'E♭ standard', label: 'Eb Ab Db Gb Bb Eb', midi: [39, 44, 49, 54, 58, 63] },
  { id: 'd-standard', name: 'D standard', label: 'D G C F A D', midi: [38, 43, 48, 53, 57, 62] },
  { id: 'drop-c', name: 'Drop C', label: 'C G C F A D', midi: [36, 43, 48, 53, 57, 62] },
  { id: 'dadgad', name: 'DADGAD', label: 'D A D G A D', midi: [38, 45, 50, 55, 57, 62] },
  { id: 'open-g', name: 'Open G', label: 'D G D G B D', midi: [38, 43, 50, 55, 59, 62] },
];

export function tuningPreset(id: TuningId): TuningPreset {
  return TUNING_PRESETS.find(p => p.id === id) ?? TUNING_PRESETS[0];
}

// Standard-tuning open-string MIDI, keyed by the client's string number
// (1 = high E … 6 = low E — see tab_events_to_tab_document `6 - string_idx`).
// Fallback for documents that predate per-document tuning.
export const STRING_OPEN_MIDI: Record<number, number> = {
  1: 64,
  2: 59,
  3: 55,
  4: 50,
  5: 45,
  6: 40,
};

export const MIN_STRING = 1;
export const MAX_STRING = 6;
export const MAX_FRET = 24;

/** Open-string MIDI for a client string number (1 = high E … 6 = low E).
 * `tuningMidi` is the document's low-to-high list; absent means standard. */
export function openStringMidi(string: number, tuningMidi?: number[]): number {
  if (tuningMidi && tuningMidi.length === 6) return tuningMidi[6 - string];
  return STRING_OPEN_MIDI[string];
}

/** Sounding MIDI pitch of a fretted note. The capo raises every string, so
 * displayed fret numbers are capo-relative and the capo adds on top. */
export function midiPitchForNote(
  string: number,
  fret: number,
  capoFret: number,
  tuningMidi?: number[],
): number {
  return openStringMidi(string, tuningMidi) + capoFret + fret;
}
