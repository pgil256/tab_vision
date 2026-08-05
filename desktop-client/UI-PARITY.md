# Desktop / browser capability parity

The WPF desktop client now covers the browser UI's capture, transcription,
refinement, playback, export, and personal-learning workflows. It remains
local-first: media processing and transcription run through the Python sidecar.

Status values:

- **Done**: implemented in the desktop client and covered by current evidence.
- **Done+**: parity is complete and the desktop adds a useful local capability.

## Capture and preparation

| Browser capability | Desktop status | Desktop implementation |
| --- | --- | --- |
| Upload video or audio | **Done** | Native picker accepts supported audio and video formats. |
| Drag-and-drop upload | **Done** | Media can be dropped directly onto the capture workspace. |
| Camera + microphone recording | **Done** | Embedded preview, record, retake, and use-take flow. |
| Audio-only microphone recording | **Done** | Native AudioGraph capture with a live input meter. |
| Live tuner | **Done** | Pitch and tuning offset update during microphone capture. |
| Metronome, count-in, BPM, meter, and click behavior | **Done** | Tap tempo, configurable tempo/meter, count-in, and recording click are integrated. |
| Listen-back before transcription | **Done** | Review playback uses the native Windows audio graph. |
| Waveform, trim, auto-trim, gain, normalize, and high-pass | **Done** | Dedicated review window previews the waveform and renders a cleaned WAV. |
| Take-health warnings | **Done** | Review reports clipping, peak level, and tuning-offset warnings. |

## Transcription setup and processing

| Browser capability | Desktop status | Desktop implementation |
| --- | --- | --- |
| Instrument, tone, and playing style | **Done** | Native controls pass exact values to the sidecar. |
| Capo 0-12 | **Done** | Desktop model, command builder, and CLI accept the full browser range. |
| Seven tuning presets | **Done** | All browser tunings map to the transcription pipeline. |
| Five speed/accuracy presets | **Done** | The five choices map to the browser's fast-to-accurate modes. |
| Optional normalized fretboard ROI | **Done** | Validated coordinates crop decoded video frames before analysis. |
| Automatic and explicit audio backends | **Done+** | Automatic behavior is retained and diagnostic backends are exposed. |
| Audio-only / video-analysis toggle | **Done** | The native option maps to the local no-video contract. |
| Stage-by-stage progress | **Done** | Sidecar progress events drive native status and progress UI. |
| Recoverable failure state | **Done** | Error details remain visible and the user can adjust settings and retry. |
| First-run setup and repair | **Done+** | Desktop owns verified offline bootstrap and repair. |

## Refine and playback

| Browser capability | Desktop status | Desktop implementation |
| --- | --- | --- |
| Timeline tablature editor | **Done** | Structured notes render on an interactive six-string timeline. |
| Original recording playback and seek | **Done** | Scrubbing, play/pause, and playback-follow are wired. |
| Skip +/-5 seconds and 0.25x-2x rates | **Done** | Dedicated skip controls and five playback rates are available. |
| Collapse source video | **Done** | The source pane can be shown or hidden. |
| Synth audition and original/synth/both modes | **Done** | A rendered plucked-string WAV plays through native AudioGraph. |
| Score view with tempo, meter, tuning, and capo | **Done** | A sheet-style tab view supports selection, seeking, and a playback cursor. |
| Rename document and title-based exports | **Done** | The title is editable and reused by exported files. |
| Confidence summary and click-to-jump | **Done** | Confidence counts and medium/low jump actions navigate the timeline. |
| Lowest-confidence review queue | **Done** | Ranked candidates and previous/next review navigation are available. |
| Candidate cycle, fret edit, mute, delete, and insert | **Done** | Buttons and keyboard shortcuts cover each operation. |
| Pitch-preserving string movement | **Done** | Moving a note to another string recalculates its fret for the same pitch. |
| Undo and redo | **Done** | Snapshot history is persisted with the editor session. |
| Zoom 25%-400% and reset | **Done** | Slider, keyboard controls, and one-click reset are available. |
| Drag retiming and vertical string drag | **Done** | Pointer edits are grouped into undoable operations. |
| Multi-selection, range selection, and batch edits | **Done** | Ctrl/Shift selection supports grouped movement and deletion. |
| Directional note navigation | **Done** | Previous/next and string-aware up/down navigation are implemented. |
| Autosave, restore, and discard | **Done** | Sessions restore automatically and can be explicitly discarded. |
| Keyboard shortcut reference | **Done** | The editor exposes its shortcut map and global dispatcher. |

## Export and local learning

| Browser capability | Desktop status | Desktop implementation |
| --- | --- | --- |
| Copy text tab | **Done** | The editor copies the current tab to the clipboard. |
| Download text tab and MIDI | **Done** | Both formats export locally. |
| MusicXML and GP5 export | **Done+** | The desktop also exposes both sidecar export formats. |
| Quantized MIDI | **Done** | MIDI export can quantize notes against the advisory beat grid. |
| Print / save PDF | **Done** | The score view uses the native print dialog and PDF printer path. |
| Bank corrected take as personal gold | **Done** | Corrected documents are validated and stored through a dedicated sidecar command. |

## Visual system and accessibility

| Requirement | Status | Evidence |
| --- | --- | --- |
| Cohesive studio identity across windows | **Done** | Shared charcoal, warm-neutral, coral, serif-display, and state-color resources mirror the current browser design across capture, review, and editor windows. |
| Clear capture -> refine -> export hierarchy | **Done** | Workflow rail, section cards, and page-level hierarchy are consistent. |
| Desktop scaling and keyboard access | **Done** | Main, review, timeline, and score surfaces were checked in the live Windows app. |
| Accessible automation names | **Done** | Core controls retain stable automation identifiers and explicit names. |

## Verification

- Desktop solution builds without warnings or errors.
- Desktop unit tests cover capture paths, command mapping, editor behavior,
  MIDI export, synth rendering, and persistence.
- Python unit tests and Ruff cover the sidecar additions and existing pipeline.
- Live Windows checks cover the main workspace, settings, microphone capture,
  audio review and cleanup, timeline editing, score view, and synth playback.
