# WPF Desktop Shell — Plan (2026-07-22)

> ## ⚠️ REBUILD EXPECTED — THIS SHELL IS DISPOSABLE BY DESIGN
>
> The v1 pipeline is under active accuracy development (backends, priors,
> CLI flags, and the web editor's review features are all moving). This
> desktop shell is built against that moving target. **The finished product
> is not "done" — plan on rebuilding or heavily reworking it as TabVision
> continues to develop.** Every design choice below optimizes for cheap
> rebuild over polish: keep the shell thin, keep all transcription/ranking
> logic in Python, and treat each shell release as disposable.
> See DECISIONS.md 2026-07-22.

## 1. Goal

A native Windows desktop app (WPF, .NET 8) that wraps the **unmodified**
Python v1 pipeline as a local sidecar. Fully offline after first run.
Small installer (~100 MB); first launch downloads the Python environment,
wheels, and model weights (~2–3 GB total).

**Non-goals:** no port of any ML to .NET; no accuracy changes; no §8
contract changes; frozen v0 clients (`tabvision-client/`, `web-client/`,
`tabvision-server/`) stay frozen; the web editor is unaffected.

## 2. Architecture

```
desktop-client/                  ← new top-level dir (WPF solution)
├── TabVision.Desktop/           ← WPF app (.NET 8, self-contained publish)
│   ├── Sidecar/                 ← process spawn, JSON envelope parsing
│   ├── Bootstrap/               ← first-run env + weight downloader
│   └── Views/                   ← file picker, options, progress, tab viewer
├── bootstrap/
│   ├── requirements.lock        ← pinned pipeline + extras (pip-compile)
│   └── weights.manifest.json    ← model artifacts: URL, sha256, dest path
└── README.md                    ← points back to this plan + rebuild caveat
```

**Sidecar model.** The app ships the CPython 3.11 embeddable package. First
run creates an app-local venv and installs `tabvision` (pinned commit) with
extras `[audio-baseline,audio-highres,vision,render]` from
`requirements.lock`. Transcription = spawn
`tabvision transcribe <input> -o <out> --format <fmt> [flags]`, stream
stderr for progress, read the result file. No persistent server in D1 —
a plain process per job is simpler and crash-isolated. D2 (editor) may
switch to a persistent local HTTP service; defer that decision.

**Pipeline-side prerequisite (additive only).** The CLI currently has no
machine-readable output or progress. Add to `tabvision/cli.py`:

- `--json`: final result envelope on stdout (`{status, output_path,
  low_confidence_flags, timings}`).
- `--progress`: one `PROGRESS <stage> <pct>` line per stage on stderr.

Both are additive flags; no §8 contract or default-behavior changes, so no
SPEC edit is required. Ship with unit tests.

## 3. First-run bootstrapper

Installer (Inno Setup — simpler than MSIX, no store/signing friction)
contains: self-contained WPF app, Python 3.11 embeddable, `pip.pyz`,
`requirements.lock`, `weights.manifest.json`.

First launch, with progress UI and resume-on-failure:

1. Create venv; `pip install -r requirements.lock` (torch/TF are the bulk).
2. Download weights per manifest, verify sha256:
   - basic-pitch — ships inside the wheel; nothing to fetch.
   - highres / highres-ensemble / GAPS checkpoints — via `huggingface-hub`
     into an app-local `HF_HOME` (keeps the cache inside the app dir,
     uninstall removes everything).
   - `gaps-v1` / `gaps-seq-v1` prior artifacts (hash-verified, same hashes
     the pipeline registry expects).
   - MediaPipe hand landmark model; YOLO fretboard weights.
3. Smoke test: `tabvision transcribe` on a bundled 5 s fixture clip;
   compare against a golden output before declaring the install healthy.

After bootstrap the app never needs the network. Settings page gets a
**Repair / Re-download** action that re-runs the bootstrapper.

## 4. Phases

**D0 — Sidecar contract.** Add `--json`/`--progress` to the Python CLI
(additive, tested). Write `requirements.lock` + `weights.manifest.json`.
*Gate:* C# integration test parses envelope + progress from a fixture run.

**D1 — Viewer MVP.** Open video → options panel (instrument/tone/style,
capo, audio backend `auto`, `--no-video` toggle) → run with progress →
monospace ASCII tab view → export gp5/musicxml/midi via `--format`.
Surface `TabVisionError` text (exit code 2) verbatim; show low-confidence
flags from the envelope (SPEC rule: flag, don't hallucinate).
*Gate:* byte-identical tab output vs. direct CLI on 3 fixture clips;
60 s clip end-to-end ≤ CLI time + 10 % overhead.

**D1.5 — Bootstrapper + installer.** As §3.
*Gate:* clean Windows 11 VM with no Python installed → install → first-run
download → offline (network disabled) transcription succeeds.

**D2 — Editor (deferred).** Port the assisted review queue (R) and
pitch-preserving candidate cycling (C). Ranking runs **locally** in the
sidecar — the server-ranked flow from the web editor becomes a local call;
the ranking logic is already ours, so this is wiring, not new modeling.
**Do not start D2 until the web editor's review feature set stabilizes** —
building it now guarantees an immediate rebuild.

## 5. Rebuild triggers (revisit this plan when any fire)

- CLI flags/backends/priors change (they have, three times since 2026-06).
- Editor features evolve on the web client (D2 target moves).
- v2 electric scope lands (new checkpoints → manifest changes).
- A persistent-service architecture becomes necessary for D2 latency.
- Pinned pipeline commit falls far enough behind `main` that repinning is
  a migration, not a bump.

Mitigation is structural: the shell contains **zero** transcription logic,
so a rebuild is UI + bootstrap manifest work only.

## 6. Risks / notes

- **First run on slow connections:** ~2–3 GB; manifest supports resume.
- **HF Hub availability:** manifest pins exact revisions; mirror the
  checkpoint files if Hub flakiness becomes a problem.
- **AV/Defender:** embedded Python spawning processes can trip SmartScreen;
  unsigned Inno installer will warn. Acceptable for a personal app.
- **Licensing:** `vision` extras pull ultralytics (AGPL) and the shipped
  posture includes NC-labeled artifacts — fine under the personal,
  non-commercial posture (CLAUDE.md 2026-07-20; LICENSES.md). Do not
  distribute the installer beyond personal use without revisiting this.
- **Frozen-code discipline:** nothing in this plan touches
  `tabvision-server/`, `tabvision-client/`, or `web-client/`.
