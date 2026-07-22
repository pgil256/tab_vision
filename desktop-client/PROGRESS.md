# TabVision WPF desktop shell progress

Source plan: `docs/plans/2026-07-22-wpf-desktop-shell-plan.md`.

This shell is disposable by design. Keep all transcription and ranking logic
in Python. D2 is out of scope until the web editor stabilizes.

## D0 - Sidecar contract

- [x] Add additive `tabvision transcribe --json` success envelope with
  `{status, output_path, low_confidence_flags, timings}` and unit tests. Result:
  JSON mode reserves stdout by requiring `--output`; default behavior is unchanged.
- [x] Add additive `tabvision transcribe --progress` stage lines on stderr and
  unit tests without changing default CLI output. Result: opt-in lines cover
  preflight, pipeline stages, render, and completion at 0-100%; default stays silent.
- [x] Create `bootstrap/requirements.lock` with the pipeline commit and the
  planned audio-highres, vision, and render extras pinned. Result: CPython 3.11
  lock has 93 exact registry pins plus 3 full-SHA VCS pins, including TabVision
  `a26d61c` with both sidecar flags.
- [x] Create `bootstrap/weights.manifest.json` with URL, revision, SHA-256, and
  app-local destination for every external model/prior artifact in the plan.
  Result: 9 public/bundled assets are revision-pinned and hash-verified; the
  approved YOLO checkpoint ships from GitHub prerelease `desktop-shell-assets-v0`.
- [x] Create the .NET 8 WPF solution and test project under `desktop-client/`.
  Result: `net8.0-windows` app + xUnit test project build with 0 warnings/errors;
  1 scaffold test passes. NuGet justification: Test SDK and xUnit packages are
  test-only infrastructure required by `dotnet test`; no runtime package added.
- [x] Implement the per-job sidecar process runner with stdout/stderr capture.
  Result: each call starts an isolated process with shell-free argument passing,
  concurrently captures both streams and exit code, and kills the process tree
  on cancellation; 2 runner tests pass.
- [x] Implement JSON-envelope and progress-line parsers in C#. Result: the
  typed envelope parser preserves unknown flag details, while the stderr parser
  extracts valid 0-100 progress lines and rejects malformed machine lines; 7
  parser test cases pass.
- [x] Add `desktop-client/README.md` linking the plan and stating the rebuild
  caveat and frozen-directory rule. Result: the README links the plan and
  progress, documents the thin/disposable shell boundary, and names all three
  frozen v0 directories.
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
- 2026-07-22: D0.2 completed. `--progress` emits stable stage percentages on
  stderr only; focused suite 47 passed; full suite 861 passed / 12 skipped;
  Ruff and mypy passed.
- 2026-07-22: D0.3 completed. Python 3.11 `pip-compile` resolved the highres,
  vision, and render extras; all pins validated and the pinned TabVision wheel
  built. The architecture draft's combined Basic Pitch extra is excluded from
  this lock because its `resampy<0.4.3` conflicts with highres's `>=0.4.3`.
  Full suite: 861 passed / 12 skipped; Ruff and mypy passed.
- 2026-07-22: D0.4 completed after explicit user approval to publish the
  AGPL/CC-BY-derived YOLO checkpoint. GitHub prerelease
  `desktop-shell-assets-v0` hosts the 5,813,315-byte asset; a fresh download
  matched SHA-256 `c579b6af...`. All 9 manifest URLs, sizes, hashes, revisions,
  and app-local destinations validated.
- 2026-07-22: D0.5 completed. Installed Microsoft .NET SDK 8.0.423, scaffolded
  the WPF app and xUnit project, and linked the test project to the desktop
  assembly. `dotnet build`: 0 warnings/errors; `dotnet test`: 1 passed.
- 2026-07-22: D0.6 completed. Added the per-job sidecar process runner with
  stdout/stderr and exit-code capture, optional per-job environment/working
  directory, and cancellation cleanup. `dotnet build`: 0 warnings/errors;
  `dotnet test`: 3 passed.
- 2026-07-22: D0.7 completed. Added typed JSON result-envelope and progress
  parsers using the .NET runtime only; unknown low-confidence flag details are
  retained for cheap wire-format evolution. `dotnet build`: 0 warnings/errors;
  `dotnet test`: 10 passed.
- 2026-07-22: D0.8 completed. Added the desktop-client README with the rebuild
  warning, Python/C# responsibility boundary, deferred D2 scope, and frozen v0
  directories. Verification: both Markdown links resolve and all required
  boundary statements are present.
