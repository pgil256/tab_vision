# S0 SynthTab acquisition audit

## all_jams_midi_V2_60000_tracks.zip

- bytes: 1,113,087,972
- SHA-256: `da678dba303a45984b8944ef644e7a5fef33301db49d53137c0f5ce52dc0576d`
- members: 453,315 (CRC ok: True)
- by extension: `{'.mid': 271415, '<dir>': 60634, '.jams': 60633, '.txt': 60633}`

## SynthTab_Dev.zip

- bytes: 822,158,849
- SHA-256: `0501d1404a4dfe6abfed87a391ea278a43da23b3c56f10c7d5e8ec50a76cda4f`
- members: 2,729 (CRC ok: True)
- by extension: `{'.pkl': 826, '.mid': 821, '<dir>': 573, '.flac': 171, '.jams': 169, '.txt': 169}`

## JAMS spot-parse

- `outall/ - Isolated - gp4__1 - Distortion Guitar__midi/1 - Distortion Guitar.jams`: namespaces `{'note_tab': 6, 'tempo': 1}`, note_tab × 6, strings `[{'string_index': 1, 'open_tuning': 64, 'notes': 379}, {'string_index': 2, 'open_tuning': 59, 'notes': 334}, {'string_index': 3, 'open_tuning': 55, 'notes': 215}, {'string_index': 4, 'open_tuning': 50, 'notes': 206}, {'string_index': 5, 'open_tuning': 45, 'notes': 152}, {'string_index': 6, 'open_tuning': 40, 'notes': 168}]`, first note `{'time': 1920.0, 'duration': 480.0, 'value': {'fret': 12, 'velocity': 95}}`, track sandbox `{'instrument': 30, 'fret_count': 24}`
- `outall/Fugazi - Waiting Room - gp3__2 - Distortion Guitar__midi/2 - Distortion Guitar.jams`: namespaces `{'note_tab': 6, 'tempo': 1}`, note_tab × 6, strings `[{'string_index': 1, 'open_tuning': 64, 'notes': 11}, {'string_index': 2, 'open_tuning': 59, 'notes': 0}, {'string_index': 3, 'open_tuning': 55, 'notes': 0}, {'string_index': 4, 'open_tuning': 50, 'notes': 0}, {'string_index': 5, 'open_tuning': 45, 'notes': 0}, {'string_index': 6, 'open_tuning': 40, 'notes': 0}]`, first note `{'time': 125280.0, 'duration': 120.0, 'value': {'fret': 14, 'velocity': 95}}`, track sandbox `{'instrument': 30, 'fret_count': 24}`
- `outall/Oldfield, Mike - Tubular bells (2) - gp3__1 - Acoustic Nylon Guitar__midi/1 - Acoustic Nylon Guitar.jams`: namespaces `{'note_tab': 6, 'tempo': 1}`, note_tab × 6, strings `[{'string_index': 1, 'open_tuning': 64, 'notes': 256}, {'string_index': 2, 'open_tuning': 59, 'notes': 190}, {'string_index': 3, 'open_tuning': 55, 'notes': 142}, {'string_index': 4, 'open_tuning': 50, 'notes': 180}, {'string_index': 5, 'open_tuning': 45, 'notes': 86}, {'string_index': 6, 'open_tuning': 38, 'notes': 140}]`, first note `{'time': 960.0, 'duration': 480.0, 'value': {'fret': 1, 'velocity': 95}}`, track sandbox `{'instrument': 24, 'fret_count': 24}`

## Gate verdict — S0 PASS

- **60,633 JAMS tracks** (≥ 50k required), all-member CRC clean on both
  archives, SHA-256 recorded above.
- Per-note **string+fret+onset is derivable**: one `note_tab` annotation per
  string with `sandbox.string_index` (1–6) and `sandbox.open_tuning` (MIDI);
  observations carry tick `time`/`duration` and `value.{fret, velocity}`; a
  `tempo` annotation provides the tick→seconds map (S1a count priors need
  only event order + string/fret/tuning).
- **Tuning is per-string explicit** — the third sample is drop-D
  (`open_tuning` 38 on string 6), so the standard-tuning filter for
  `synthtab-v1` is a trivial six-value equality check
  (64/59/55/50/45/40 → matches the registered priors' validated domain).
- **Instrument is per-track explicit** (GM program in `sandbox.instrument`;
  also embedded in member paths), so acoustic/clean subsets can be selected
  symbolically. Companion `.mid` files (271,415) include per-string MIDI;
  `.txt` members carry track metadata.
- Dev set contents (2,729 members: `.flac` audio + `.pkl` per-string F0 +
  JAMS/MIDI for 169 tracks) match the S2 bring-up expectation.

Branch: proceed to S1a (SynthTab-scale count priors) on the next `proceed`.

## Provenance

- Command: `python scripts/eval/s0_synthtab_audit.py --output
  ../docs/EVAL_REPORTS/s0_synthtab_audit_2026-07-20.md --json
  ../docs/EVAL_REPORTS/s0_synthtab_audit_2026-07-20.json` (tabvision `.venv`,
  `TABVISION_DATA_ROOT=~/.tabvision/data`, `PYTHONUTF8=1`, 2026-07-20).
- Acquisition: official UR Box shares (`/v/SynthTab-Full` file
  `f_1486897975091`; `/v/SynthTab-Dev` file `f_1344018777482`) via the
  `rm=box_download_shared_file&vanity_name=…` URL form; user approved the
  download in chat (2026-07-20). License CC-BY-NC-4.0 — rows added to
  LICENSES.md (NC program acquisition addendum). Archives live under
  `$TABVISION_DATA_ROOT/datasets/synthtab/`, never committed.
