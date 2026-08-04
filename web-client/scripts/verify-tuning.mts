// Tuning-aware pitch math checks: presets, fallback-to-standard, and the
// drag/export/synth helpers under drop D and DADGAD.
//
// Run: npx tsx scripts/verify-tuning.mts

import { pitchPreservingFret } from '../src/store/appStore';
import {
  midiPitchForNote,
  openStringMidi,
  TUNING_PRESETS,
  tuningPreset,
} from '../src/utils/pitch';

let failed = 0;
function check(name: string, ok: boolean) {
  console.log(`  ${ok ? 'ok ' : 'FAIL'} ${name}`);
  if (!ok) failed = 1;
}

const dropD = tuningPreset('drop-d').midi; // [38,45,50,55,59,64]
const dadgad = tuningPreset('dadgad').midi; // [38,45,50,55,57,62]

// Fallback (no tuning) is still standard
check('standard: open string 6 = 40', openStringMidi(6) === 40);
check('drop-d: open string 6 = 38', openStringMidi(6, dropD) === 38);
check('drop-d: open string 5 unchanged = 45', openStringMidi(5, dropD) === 45);

// midiPitchForNote under drop D: string 6 fret 0 = D2 (38); capo adds on top
check('drop-d pitch: s6 f0 = 38', midiPitchForNote(6, 0, 0, dropD) === 38);
check('drop-d pitch: s6 f0 capo2 = 40', midiPitchForNote(6, 0, 2, dropD) === 40);

// pitchPreservingFret under drop D: s6 f7 (=45) -> s5 open
check('drop-d preserve: s6 f7 -> s5 f0', pitchPreservingFret(6, 5, 7, dropD) === 0);
// s6 f0 (=38) -> s5 would need fret -7 -> null
check('drop-d preserve: s6 f0 -> s5 null', pitchPreservingFret(6, 5, 0, dropD) === null);
// standard behaviour unchanged when the tuning argument is omitted
check('standard preserve: s6 f5 -> s5 f0', pitchPreservingFret(6, 5, 5) === 0);

// DADGAD: string 1 open = D4 (62), string 2 open = A3 (57); s1 f0 -> s2 f5
check('dadgad preserve: s1 f0 -> s2 f5', pitchPreservingFret(1, 2, 0, dadgad) === 5);

// Registry sanity
check('presets: all 6-string', TUNING_PRESETS.every(p => p.midi.length === 6));
check('presets: standard first/default', tuningPreset('standard').midi[0] === 40);
check(
  'presets: ids unique',
  new Set(TUNING_PRESETS.map(p => p.id)).size === TUNING_PRESETS.length,
);

if (failed) {
  console.log('TUNING CHECKS FAILED');
  process.exit(1);
}
console.log('ALL TUNING CHECKS PASSED');
