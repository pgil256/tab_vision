# TabVision WPF desktop shell progress

Source plan: `docs/plans/2026-07-22-wpf-desktop-shell-plan.md`.

This shell is disposable by design. Keep all transcription and ranking logic
in Python. D2 is out of scope until the web editor stabilizes.

## D0 - Sidecar contract

- [x] Add additive `tabvision transcribe --json` success envelope with
  `{status, output_path, low_confidence_flags, timings}` and unit tests. Result:
  JSON mode reserves stdout by requiring `--output`; default behavior is unchanged.
- [ ] Add additive `tabvision transcribe --progress` stage lines on stderr and
  unit tests without changing default CLI output.
- [ ] Create `bootstrap/requirements.lock` with the pipeline commit and the
  planned audio-highres, vision, and render extras pinned.
- [ ] Create `bootstrap/weights.manifest.json` with URL, revision, SHA-256, and
  app-local destination for every external model/prior artifact in the plan.
- [ ] Create the .NET 8 WPF solution and test project under `desktop-client/`.
- [ ] Implement the per-job sidecar process runner with stdout/stderr capture.
- [ ] Implement JSON-envelope and progress-line parsers in C#.
- [ ] Add `desktop-client/README.md` linking the plan and stating the rebuild
  caveat and frozen-directory rule.
- [ ] **D0 gate:** a C# integration test runs the fixture sidecar and parses
  both its result envelope and progress lines.

## D1 - Viewer MVP

- [ ] Add a video file picker and selected-input summary.
- [ ] Add options for instrument, tone, style, capo, audio backend (`auto` by
  default), and the `--no-video` toggle.
- [ ] Run one sidecar process per transcription job and show stage progress.
- [ ] Show completed ASCII output in a monospace tab viewer.
- [ ] Export ASCII, GP5, MusicXML, and MIDI through the CLI `--format` option.
- [ ] Surface `TabVisionError` stderr text verbatim for exit code 2.
- [ ] Surface every low-confidence flag from the JSON envelope.
- [ ] **D1 correctness gate:** output is byte-identical to direct CLI output
  on three fixture clips; record clip names and hashes here.
- [ ] **D1 overhead gate:** a 60 s clip completes within direct CLI time +10%;
  record both wall-clock measurements and the ratio here.

## D1.5 - Bootstrapper and installer

- [ ] Bundle the self-contained WPF publish, CPython 3.11 embeddable package,
  `pip.pyz`, requirements lock, and weights manifest with Inno Setup.
- [ ] Create the app-local Python environment and install the locked pipeline
  dependencies with visible, resumable progress.
- [ ] Download every manifest artifact with resume support, verify SHA-256,
  and keep `HF_HOME` inside the app data directory.
- [ ] Run the bundled 5 s fixture smoke transcription and compare its output
  to the checked-in golden before declaring bootstrap healthy.
- [ ] Make failed/interrupted bootstrap resumable without discarding verified
  downloads.
- [ ] Add Settings > Repair / Re-download using the same bootstrap workflow.
- [ ] Verify normal transcription performs no network access after bootstrap.
- [ ] **D1.5 gate:** on a clean Windows 11 VM with no Python installed, install,
  complete first-run download, disable networking, and successfully transcribe;
  record VM version and measured result here.

## Run log

- 2026-07-22: Checklist initialized and D0.1 completed. Verification: focused
  CLI suite 11 passed; full suite 858 passed / 12 skipped; Ruff and mypy passed.
