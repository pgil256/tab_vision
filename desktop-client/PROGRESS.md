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
- [x] **D0 gate:** a C# integration test runs the fixture sidecar and parses
  both its result envelope and progress lines. Result: the C# runner launched
  the real Python CLI with a deterministic fixture pipeline, parsed `status=ok`
  plus one low-confidence flag and all 7 progress stages, and verified the
  rendered ASCII output.

## D1 - Viewer MVP

- [x] Add a video file picker and selected-input summary. Result: the WPF shell
  uses the built-in Windows file dialog and shows the selected file name, type,
  exact byte size, and full path; 2 metadata tests pass with no new dependency.
- [x] Add options for instrument, tone, style, capo, audio backend (`auto` by
  default), and the `--no-video` toggle. Result: the dependency-free WPF panel
  mirrors the pinned CLI choices (including capo 0-7) and initializes every
  control from a tested default options record.
- [x] Run one sidecar process per transcription job and show stage progress.
  Result: each Transcribe click creates a unique job output, launches one
  `tabvision` process with the selected pinned CLI options, and streams parsed
  stderr stages into a progress bar while retaining full stderr capture.
- [x] Show completed ASCII output in a monospace tab viewer. Result: successful
  jobs load the envelope's UTF-8 output path without newline normalization and
  reveal a read-only, no-wrap Consolas viewer with both scrollbars.
- [x] Export ASCII, GP5, MusicXML, and MIDI through the CLI `--format` option.
  Result: the completed viewer offers four save formats and reruns one sidecar
  with the displayed result's input/options snapshot, chosen destination, and
  exact pinned CLI format; C# performs no format conversion.
- [x] Surface `TabVisionError` stderr text verbatim for exit code 2. Result:
  transcription and export share an exit-2 handler that assigns the runner's
  complete stderr string directly to a read-only error pane without trimming,
  prefixing, filtering, or newline changes.
- [x] Surface every low-confidence flag from the JSON envelope. Result: the
  completed viewer lists every flag in envelope order with all required fields
  and retained future JSON details; an empty flag array hides the warning panel.
- [x] **D1 correctness gate:** output is byte-identical to direct CLI output
  on three fixture clips. Result: PASS; every direct/desktop pair was 144 bytes
  with output SHA-256 `57229c70081f22a230185373a2455dcbe2de7b09c927fec50f74cbd98f4234e0`.
  Fixture input SHA-256 values: `027_Zpswc.mp4`
  `b952fd2c455dce7bfd55ad9fd9137e8ffd6813c1ec04122291fe9912479d3b5c`;
  `031_vpswc.mp4`
  `949e8de8096f2ead64dae3247aa3e6a57e51e9cba935934b95f72210703b27e9`;
  `043_bc1wc.mp4`
  `c1c48027831f262e650dcd1658fc143f0db84c6de4a5553d2caaef2c5dd4eea1`.
- [ ] **D1 overhead gate:** a 60 s clip completes within direct CLI time +10%;
  record both wall-clock measurements and the ratio here. BLOCKED 2026-07-22:
  the development host has no `ffmpeg`/`ffprobe`, so the real pipeline cannot
  execute. A non-qualifying deterministic-fixture diagnostic measured direct
  CLI 340.485 ms versus desktop 429.012 ms (ratio 1.260001, FAIL); because that
  fixture ignores media duration, it is not accepted as the required 60 s
  end-to-end result.

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
- 2026-07-22: D0 gate passed. The cross-process C# test ran the actual Python
  CLI against `test_a440.mp4` with deterministic inference injected, then
  parsed stdout/stderr and verified the rendered file. `dotnet build`: 0
  warnings/errors; `dotnet test`: 11 passed. Python: 861 passed / 12 skipped;
  Ruff (package + fixture) and mypy passed.
- 2026-07-22: D1.1 completed. Replaced the empty window with a video picker and
  selected-input card backed by a testable file-metadata summary. The built-in
  WPF dialog avoids a runtime package. `dotnet build`: 0 warnings/errors;
  `dotnet test`: 13 passed.
- 2026-07-22: D1.2 completed. Added instrument, tone, style, capo, backend, and
  audio-only controls backed by exact pinned-CLI choice lists and defaults. No
  dependency or process wiring was added. `dotnet build`: 0 warnings/errors;
  `dotnet test`: 15 passed.
- 2026-07-22: D1.3 completed. Wired the Transcribe button to one isolated
  sidecar process per unique local-app-data job and mapped live `PROGRESS`
  stderr lines to the UI without losing captured error text. Added exact command
  and streaming-capture tests. `dotnet build`: 0 warnings/errors; `dotnet test`:
  18 passed.
- 2026-07-22: D1.4 completed. Added a completed-result pane that strictly
  decodes the CLI's UTF-8 ASCII file, preserves fixed-width content and line
  endings, and hides stale output when a new job starts. `dotnet build`: 0
  warnings/errors; `dotnet test`: 20 passed.
- 2026-07-22: D1.5 completed. Added post-result export for `.tab`, `.gp5`,
  `.musicxml`, and `.mid`; each export goes through a fresh CLI process with
  `--format ascii|gp5|musicxml|midi` and verifies the reported output exists.
  `dotnet build`: 0 warnings/errors; `dotnet test`: 26 passed.
- 2026-07-22: D1.6 completed. Added a dedicated exit-2 error pane and direct
  stderr pass-through for both transcription and export. An actual fixture CLI
  invocation produced the expected `TabVisionError`, including its final
  newline, unchanged. `dotnet build`: 0 warnings/errors; `dotnet test`: 29 passed.
- 2026-07-22: D1.7 completed. Added a scrollable low-confidence panel to the
  completed viewer and deterministic presentation of every flag field,
  including unknown extension data. Successful transcription and export
  envelopes refresh it; new jobs clear it. `dotnet build`: 0 warnings/errors;
  `dotnet test`: 31 passed.
- 2026-07-22: D1 correctness gate passed on public cached GAPS fixtures
  `027_Zpswc`, `031_vpswc`, and `043_bc1wc`. The deterministic fixture pipeline
  ran each clip through direct CLI mode and desktop machine mode; both used
  `--no-preflight` to isolate output transport after `027_Zpswc` correctly
  failed the guitar-presence preflight. All three raw byte/hash comparisons
  matched. The pinned CLI remains unchanged from `a26d61c`. `dotnet build`: 0
  warnings/errors; `dotnet test`: 31 passed.
- 2026-07-22: D1 overhead gate remains unchecked. An exact 60.000 s,
  120-frame/2 FPS fixture derived from public cached `031_vpswc` had SHA-256
  `cceb145b1f8ed5252bffe61791b3c69df4376c00f547ee301c6ed0d62f11aa15`.
  The production measurement is blocked because this host has neither
  `ffmpeg` nor `ffprobe`. A temporary deterministic-sidecar diagnostic failed
  the relative limit at 340.485 ms direct / 429.012 ms desktop = 1.260001;
  it was not promoted to gate evidence because its mocked inference ignores
  the 60 s workload. The temporary harness was removed; the clean build had 0
  warnings/errors and all 31 standing tests passed. Per the loop stop rule, no
  D1.5 work started.
