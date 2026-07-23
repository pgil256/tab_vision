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
  lock has 94 exact registry pins, two hash-verified upstream commit archives,
  and the published fixed-commit TabVision wheel; no Git client is required.
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
- [x] **D1 overhead gate:** a 60 s clip completes within direct CLI time +10%;
  record both wall-clock measurements and the ratio here. Result: PASS; direct
  CLI 356.821334 s versus desktop sidecar 305.082714 s = 0.855001 ratio, below
  the 1.100000 limit (392.503467 s). Both runs produced the same 2,580-byte
  output with SHA-256 `e588b041a4e54be4d95fdbc24a5b585f763cb04cf87da4cebec22c4ecb251b4c`.

## D1.5 - Bootstrapper and installer

- [x] Bundle the self-contained WPF publish, CPython 3.11 embeddable package,
  `pip.pyz`, requirements lock, and weights manifest with Inno Setup. Result:
  pinned/hash-verified inputs produced a 63,920,983-byte installer (SHA-256
  `f3506ef65f9c191b3e1d3041265db591ecb389e06496973047aae532a80f6616`);
  a silent-install audit verified the self-contained runtime and all payloads.
- [x] Create the app-local Python environment and install the locked pipeline
  dependencies with visible, resumable progress. Result: first launch expands
  embedded CPython under local app data, streams 0-100% pip status through the
  existing UI, and installs the 97-package `--no-deps` closure. A payload-hash
  ready marker skips completed work, while an app-local pip cache resumes failed
  runs; a real install and `pip check` passed without Git.
- [x] Download every manifest artifact with resume support, verify SHA-256,
  and keep `HF_HOME` inside the app data directory. Result: all 9 artifacts
  (211,853,452 bytes) passed size/hash verification; a real 1 MiB partial
  resumed, offline Hugging Face lookup resolved both checkpoints, and a repeat
  launch reused every verified file without rewriting it.
- [x] Run the bundled 5 s fixture smoke transcription and compare its output
  to the checked-in golden before declaring bootstrap healthy. Result: the
  installed pinned CLI ran real high-resolution inference on the 5.000 s
  synthetic fixture and matched all 222 golden bytes; only then did it write
  a fingerprinted health marker, which the next launch reused.
- [x] Make failed/interrupted bootstrap resumable without discarding verified
  downloads. Result: app-close cancellation now stops active setup work while
  atomic stage markers, the pip cache, verified artifacts, and digest-keyed
  partials survive; retries re-extract an unmarked partial runtime, skip every
  verified file, and range-resume only the interrupted artifact.
- [x] Add Settings > Repair / Re-download using the same bootstrap workflow.
  Result: the menu action invalidates only Python/smoke completion markers,
  then reuses the first-run progress path to reinstall the lock, hash-check all
  artifacts, download only missing/corrupt files, and rerun the smoke test.
- [x] Verify normal transcription performs no network access after bootstrap.
  Result: the exact installed sidecar completed a normal acoustic/clean/mixed,
  `auto` audio+video job under a process-level outbound socket guard with 0 DNS,
  TCP, or UDP attempts; 487 output bytes matched SHA-256 `22c337d7...0370`.
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
- 2026-07-22: After user approval, acquired BtbN LGPL shared build
  `ffmpeg-N-125716-g1b1f602699-20260722` from pinned release
  `autobuild-2026-07-22-13-36` into the app-local bootstrap cache. The
  67,053,838-byte archive matched release SHA-256
  `04d3def4406324e479ab7cb9abe8e9472c103e5be26298aa876ed93074b39386`;
  `ffmpeg.exe` / `ffprobe.exe` hashes are `b8ba7ca0...05294` /
  `076ee1bc...a16bf`. A real audio+video fixture derived from public
  `031_vpswc` was exactly 60.000 s, passed preflight on all preview frames,
  and had SHA-256
  `351206c53d1c4ed572299cce70ca405f86c071322866217d39d6d7636009084c`.
  Direct production CLI completed in 132.973164 s. The desktop process exited
  0, but dependency diagnostics preceded its JSON stdout, causing the C#
  envelope parser to fail; no desktop measurement or ratio was accepted. The
  temporary harness was removed and D1.5 remained untouched.
- 2026-07-22: After explicit user approval, repaired the additive `--json`
  contract at the CLI dispatch boundary: dependency stdout is redirected to
  stderr only in machine mode, while the retained original stdout receives the
  final JSON envelope. Regression tests prove JSON stdout stays parseable and
  default stdout behavior is unchanged. Full Python verification: 863 passed / 12
  skipped, Ruff passed, mypy passed. Clean desktop verification: build 0
  warnings/errors, 31 tests passed. The real 60.000 s fixture then passed the
  D1 overhead gate at 356.821334 s direct / 305.082714 s desktop = 0.855001;
  both outputs were byte-identical. The timing harness was removed after use,
  and D1.5 remains the next phase.
- 2026-07-22: D1.5 installer bundle completed. Added a reproducible build that
  pins CPython 3.11.9 embed, immutable pip 26.1.2 zipapp, and signed Inno Setup
  7.0.2 with URL/size/SHA-256 verification; publishes WPF self-contained for
  `win-x64`; and packages those inputs with the existing requirements lock and
  weights manifest. The unsigned personal-use installer was 63,911,774 bytes
  with SHA-256 `363c38dc...c4d9`. A fresh silent install yielded 472 files;
  `hostfxr.dll`/`coreclr.dll` proved the app-local .NET runtime was present,
  and every bootstrap input matched its source hash. The generated uninstaller
  removed the smoke installation. Clean verification: build 0 warnings/errors,
  31 tests passed. No runtime or NuGet dependency was added.
- 2026-07-22: D1.5 environment installation stopped at the required pin-drift
  check. `requirements.lock` pins TabVision `a26d61c`; direct source inspection
  proved it has the original machine flags but not `redirect_stdout`, while the
  production fix is in local-only `b2368c1`. No remote branch contains that
  commit. The lock's TabVision and two high-resolution dependencies are also
  `git+https` pins, so the planned clean Windows install would depend on an
  unbundled Git client. No environment, dependency, or C# change was made.
  Preferred resolution: publish immutable wheels/source archives for the three
  VCS pins (including the fixed TabVision commit), record their hashes, and
  repin the lock without adding Git to the installer.
- 2026-07-22: After explicit user approval, published the 627,782-byte
  `tabvision-1.0.0-py3-none-any.whl` built from fixed commit `b2368c1` to the
  existing desktop asset prerelease; GitHub and a fresh download both matched
  SHA-256 `fe250480...46918`. Replaced both external VCS requirements with
  full-commit codeload archives pinned to SHA-256 `aecb4185...fda89` and
  `f6ac16e2...f1aaf`; each archive built successfully. A resolver probe proved
  upstream metadata otherwise reclones Git and conflicts with the archive pin,
  so the complete exact-pinned lock is explicitly installed with `--no-deps`.
  An isolated install with Git absent from `PATH` succeeded for all three direct
  artifacts. Clean desktop build: 0 warnings/errors; all 31 tests passed.
  Environment/bootstrap UI work remains in the unchecked item.
- 2026-07-22: D1.5 app-local Python environment completed. First launch now
  expands CPython 3.11.9 to `%LOCALAPPDATA%\\TabVision\\python`, preserves the
  bundled `_pth` file as a backup, mirrors extension modules into standard
  `DLLs` for pip build isolation, and installs the exact lock with `--no-deps`,
  `--upgrade`, visible 0-100% status, and a persistent pip cache. Two real
  interrupted attempts retained their runtime/cache and resumed successfully.
  The initial 96-package install completed in 423.4 s; `pip check` then exposed
  pip-compile's omitted unsafe `setuptools` dependency for Torch, so the final
  97-package lock pins `setuptools==83.0.0` and the cached repair took 54.2 s.
  Final `pip check` reported no broken requirements; imports verified TabVision
  1.0.0, Torch 2.13.0+cpu, OpenCV 5.0.0, and MediaPipe 0.10.35. The 43,517-file
  environment is 1,753,790,877 bytes with a reusable 534,909,462-byte cache;
  the next invocation used the ready-marker fast path. Clean desktop build:
  0 warnings/errors; 36 tests passed. The rebuilt 63,920,983-byte installer
  matched the final lock in a 471-file silent install, and the installed app
  stayed running through its startup smoke before clean uninstall. No NuGet
  dependency was added.
- 2026-07-22: D1.5 manifest artifact installation completed. The WPF first-run
  path now reads and validates the bundled manifest, confines `HF_HOME`, the
  TabVision data root, MediaPipe, and YOLO paths under local app data, and uses
  BCL HTTP Range requests plus persistent `.part` files for resumption. Files
  are promoted only after exact byte-size and SHA-256 verification; valid
  destinations are reused. A real installed-app run resumed a seeded 1 MiB
  checkpoint partial and verified all 9 artifacts / 211,853,452 bytes. Both
  high-resolution checkpoints resolved with `HF_HUB_OFFLINE=1`; a repeat run
  rewrote no artifacts and left no partials. Clean desktop build: 0
  warnings/errors; all 43 tests passed. The final rebuilt 63,929,289-byte
  installer has SHA-256 `10ad9ade...468ee`. No NuGet dependency was added.
- 2026-07-22: D1.5 bootstrap smoke verification completed. Added a bundled
  5.000 s / 122,080-byte synthetic A440 fixture and 222-byte ASCII golden, plus
  the previously user-approved BtbN `N-125716-g1b1f602699` LGPL shared FFmpeg
  runtime required by the pinned demuxer. The 67,053,838-byte archive and all
  10 redistributed files are size/hash-pinned; its license ships beside the
  replaceable DLLs. First launch now runs the real `highres` CPU pipeline with
  machine progress, compares output bytes, logs failures, and writes a marker
  fingerprinting the lock, weights manifest, tools, fixture, and golden only
  after success. Two direct pinned-CLI runs were byte-identical at SHA-256
  `6f310389...259ce3` (57.263 s cold / 21.018 s warm). A real installed-app run
  passed in 24.218 s and the next launch skipped it without rewriting the
  marker. Clean desktop build: 0 warnings/errors; all 46 tests passed. The
  105,207,001-byte installer has SHA-256 `2335ef9f...745ca`; silent install,
  17-file bundle audit, startup, and clean uninstall passed. No NuGet or Python
  dependency was added.
- 2026-07-22: D1.5 interruption recovery completed. CPython extraction now
  writes an atomic archive-fingerprint marker only after the runtime is fully
  expanded; an unmarked partial runtime is safely re-extracted on retry. Window
  close passes cancellation through pip, HTTP downloads, and smoke inference,
  while setup failures tell the user to relaunch. A two-artifact failure test
  proved the completed destination was reused without network access and only
  the seven-byte partial was range-resumed. Clean Release build: 0
  warnings/errors; all 48 tests passed. No dependency was added.
- 2026-07-22: D1.5 Settings repair completed. Added a dependency-free Settings
  menu with Repair / Re-download, refactored startup and repair through the same
  bootstrap method, and disabled repair during setup or a transcription job.
  Repair removes only the Python-environment and smoke completion markers;
  runtime extraction state, pip cache, verified models, and partial downloads
  remain reusable while every artifact is hash-checked. A focused idempotency
  test proved those preservation boundaries. Clean Release build: 0
  warnings/errors; all 49 tests passed. No dependency was added.
- 2026-07-22: D1.5 offline-transcription verification passed after the guard
  exposed and this increment fixed one real gap. The first normal run caught
  Ultralytics resolving `one.one.one.one` and `dns.google` at import; the pinned
  package documents `YOLO_OFFLINE=1`, now set beside `HF_HUB_OFFLINE=1` for
  every post-bootstrap sidecar. The repeatable PowerShell audit injected a
  fail-closed socket guard into the exact installed `tabvision.exe` and ran the
  normal acoustic/clean/mixed, `auto` audio+video command on a 5.015510 s public
  GAPS-derived clip (396,988 bytes, SHA-256 `39afaaa2...b58`). It completed in
  47.181 s with 8 progress lines, 0 outbound DNS/TCP/UDP attempts, and a
  487-byte output at SHA-256 `22c337d7...0370`. Clean Release build: 0
  warnings/errors; all 49 tests passed. No dependency was added.
