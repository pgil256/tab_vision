// Shared pitch table for the store (pitch-preserving moves), MIDI export,
// and the audition synth.

// Standard-tuning open-string MIDI, keyed by the client's string number
// (1 = high E … 6 = low E — see tab_events_to_tab_document `6 - string_idx`).
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

/** Sounding MIDI pitch of a fretted note. The capo raises every string, so
 * displayed fret numbers are capo-relative and the capo adds on top. */
export function midiPitchForNote(string: number, fret: number, capoFret: number): number {
  return STRING_OPEN_MIDI[string] + capoFret + fret;
}
