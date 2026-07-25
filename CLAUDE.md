# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Posture update (2026-07-20)

TabVision is a **personal, non-commercial application** (SPEC §1.5 amended;
DECISIONS.md 2026-07-20). NC-licensed datasets/weights (CC-BY-NC[-SA]) are
acceptable in the shipping default and as training substrate — each NC-derived
artifact is labeled in LICENSES.md. Private/user recordings remain banned from
all training/eval roles. Shipped under this posture: `highres-ensemble` is the
clean-acoustic `auto` audio backend (+0.021 player-05 aggregate Tab F1);
classical sessions route to the GAPS-trained `gaps-v1`/`gaps-seq-v1` priors;
the web editor has an assisted review queue (R key) with server-ranked
pitch-preserving candidate cycling (C key), shipped at the measured Phase 6
level (38.76% wrong-position reduction @60s), reported separately from
automatic Tab F1.

## Project status (2026-07-25)

**v1.0.0 is released and `main` has moved well past it.** Work happens on
`main`; cut new branches from it. SPEC §7's ten phases are complete and the
`v1.0.0` tag records the 2026-06-03 acceptance run (aggregate Tab F1 0.600).

**Current default pipeline** — `highres-ensemble` audio + `guitarset-v1`
position prior + `acoustic-physics-v1` string evidence with partial-aware
isolation. Held-out player-05, 60 clips: **single-line 0.7257, strummed
0.7435, aggregate 0.7346** (+0.1006 [+0.0615, +0.1416] over the pre-physics
0.6340). Report: `docs/EVAL_REPORTS/player05_batched_confirm_2026-07-24.md`.

Routing that `auto` performs per session: classical/nylon → `gaps-v1` /
`gaps-seq-v1`; `--capo N` > 0 → capo-covariant position prior; anything
outside clean steel-string acoustic standard tuning → physics channel
abstains by construction.

**Read these before any non-trivial change:**
- `SPEC.md` — canonical spec (10-phase plan, §8 immutable contracts). §1.4 /
  §1.4.1 remain the source of truth for scope and acceptance targets.
- `docs/NARRATIVE.md` — what was tried, what worked, what was refuted, and
  the two published claims that turned out to be wrong.
- `docs/DECISIONS.md` — non-obvious branches taken (per SPEC §0.5). Append
  only; ~139 entries.
- `LICENSES.md` — per-artifact license map. Default *deps* are permissive;
  default *artifacts* include CC-BY-NC-SA classical priors.
- `AUDIT.md` — Phase 0 audit: inventory, what works, reusable artifacts.

**Two live caveats when reading state files.** `docs/accuracy-loop-state.md`
went stale before its program ended — for the final iterations the commit
messages are authoritative. And this repo is often worked by more than one
session at once: check `git worktree list` and re-read `git log` before
assuming a branch tip is where you left it.

## Desktop shell + FretCam (2026-07-23/24)

`desktop-client/` is a WPF (.NET 8) shell over the Python CLI: pinned
installer, resumable first-run env/weight bootstrap, audited
offline-after-bootstrap transcription, in-app camera + mic recording.
**It is disposable by design: the pipeline is a moving target and the shell
is expected to be rebuilt.** Keep all transcription/ranking logic in Python;
the shell must stay thin. Plan: `docs/plans/2026-07-22-wpf-desktop-shell-plan.md`.

`fretcam/` is a live fretboard/position HUD and the source of the **opt-in**
`--video-backend fretcam` bridge. `legacy` remains the default: end-to-end
against real audio the bridge measured +0.000836 on ten source-disjoint clips
(95% CI lower bound exactly 0) and −0.000155 on the development set. Do not
promote it without the L2 controlled-live gate plus a larger frozen result.
The exact-string video path stays quarantined.

## Layout

```
tab_vision/
├── tabvision/              ← v1 (active) — Python package + CLI
│   ├── tabvision/          ← importable package
│   │   ├── types.py        ← SPEC §8 contracts (immutable)
│   │   ├── audio/, video/, fusion/, render/, preflight/, demux/, cli.py
│   ├── pyproject.toml
│   ├── tests/{unit,integration,eval}/
│   ├── scripts/{acquire,train,eval,augment,annotate}/
│   └── data/{fixtures,eval,augmented}/
├── desktop-client/         ← WPF (.NET 8) shell over the CLI. Thin, disposable.
├── fretcam/                ← live position HUD + opt-in video bridge
├── tabvision-server/       ← FROZEN v0 backend (Flask). No further dev.
├── tabvision-client/       ← FROZEN v0 desktop UI (Electron). No further dev.
├── web-client/             ← FROZEN v0 web client (Vite + Vercel).
├── docs/
│   ├── plans/              ← design docs (current + historical)
│   └── DECISIONS.md        ← record of non-obvious choices
├── AUDIT.md
├── LICENSES.md
├── SPEC.md                 ← canonical specification
└── CLAUDE.md               ← this file
```

## Operating rules (per SPEC §0)

1. **Audit before refactor.** Phase 0 audit (`AUDIT.md`) is non-negotiable.
2. **One phase at a time.** Phase N+1 starts only after Phase N's acceptance
   gate (SPEC §9.3) passes AND user says "proceed."
3. **§8 contracts are immutable** within a phase. Implementations may change;
   signatures may not, except by explicit user approval and a SPEC update.
4. **Tests over commits.** Every phase ships with new tests. A phase is
   "done" when its acceptance criterion is met on the eval set.
5. **Track decisions.** Append to `docs/DECISIONS.md` per the format there.
6. **Free tools first.** Pretrained > fine-tuning > training from scratch.
   Local > Colab > Kaggle. CPU-runnable > GPU-required.
7. **Flag, don't hallucinate.** Borderline metrics → low-confidence flag in
   the result, not a guess.
8. **Stop and ask** when the spec is ambiguous, when a phase test fails in a
   way the decision tree doesn't cover, or when an action would add a
   dependency / training run that costs money.

## v1 dev commands

```bash
# Install (dev)
cd tabvision
pip install -e '.[dev]'

# Run tests
pytest -v

# Lint + types
ruff check .
ruff format --check .
mypy tabvision

# CLI
tabvision --version
tabvision transcribe input.mov --format ascii -o out.tab
```

FretCam bridge tests (CI installs the package and gates the four bridge files):

```bash
cd fretcam
pip install --no-deps -e .    # into the tabvision venv
pytest -q
```

Desktop shell:

```bash
cd desktop-client
dotnet test
```

## v0 (frozen) reference

The v0 backend at `tabvision-server/` is preserved as a working desktop demo
and as porting source for Phases 1, 4, 5. Do not develop new features in
v0; reference its modules during port work:

| v0 module | v1 destination |
|---|---|
| `tabvision-server/app/audio_pipeline.py` | `tabvision.audio.basicpitch` (Phase 1) |
| `tabvision-server/app/video_pipeline.py` | `tabvision.video.hand.mediapipe_backend` (Phase 4) |
| `tabvision-server/app/fretboard_detection.py` | `tabvision.video.fretboard.geometric` (Phase 3) |
| `tabvision-server/app/fusion_engine.py` | `tabvision.fusion.{viterbi,playability,chord}` (Phase 5) |
| `tabvision-server/app/guitar_mapping.py` | `tabvision.fusion.candidates` (Phase 5) |
| `tabvision-server/app/chord_shapes.py` | `tabvision.fusion.chord` (Phase 5) |

If v0 needs to stay runnable for the demo:

```bash
cd tabvision-server
source venv/bin/activate
python run.py    # Flask dev server, port 5000
pytest tests/    # 17 v0 tests
```

## Acceptance targets (SPEC §1.4)

**v1 scope (2026-06-02): acoustic, audio-first.** Honest targets on
GuitarSet (see SPEC §1.4.1): single-line Tab F1 ≥ 0.45, strummed ≥ 0.60,
aggregate ≥ 0.55, + onset ≥ 0.92 / pitch ≥ 0.90 / chord ≥ 0.85 / latency ≤ 5 min.
These are the **v1.0.0 acceptance gates** and remain the contractual targets;
the current default clears all of them with substantial margin (0.7257 /
0.7435 / 0.7346).

⚠️ **The "single-line is information-limited" framing in older docs is
superseded.** It was used to argue that only video could resolve strings.
Inharmonicity — string stiffness stretching a note's partials by an amount that
depends on the string — carries string identity in the audio, and reading it
moved single-line 0.5503 → 0.7257. Treat "audio cannot resolve X" claims in
pre-2026-07-22 documents as unverified. The retired 0.94 single-line figure was
a *video-assisted* aspiration and is not a live target.

**Electric tiers → v2** (clean-electric measured **0.12**; acoustic-trained
backbone, no in-repo training code — `cross_dataset_prior_2026-06-02.md`). v1
ships the **tone toggle** (electric → separate `highres-electric` checkpoint).
**SPEC §1.4 + §1.4.1 are the single source of truth**; don't change
scope/targets without a SPEC edit + user approval.

| Metric | Target (v1, audio-only acoustic) | Definition |
|---|---|---|
| Onset F1 (50 ms) | ≥ 0.92 | mir_eval onset_f_measure |
| Pitch F1 (50 ms, no offset) | ≥ 0.90 | mir_eval note_f_measure |
| Tab F1 (string + fret + onset), aggregate | ≥ 0.55 | TP iff string + fret + onset all match |
| Chord-instance accuracy | ≥ 0.85 | Full fingering set per chord |
| End-to-end latency for 60 s clip on laptop CPU | ≤ 5 min | Wall-clock |

Per-tier acoustic targets (single-line ≥ 0.45 / strummed ≥ 0.60) + the v1.1
video stretch (0.94 / 0.86): see SPEC §1.4.1.

## Glossary (selective)

- **§8 contracts** — the dataclasses and protocols in `SPEC.md` §8, mirrored
  in `tabvision/tabvision/types.py`. Immutable within v1.
- **Phase** — a section of SPEC §7. Each has Goal / Deliverables / Acceptance
  test / Decision tree.
- **Port** — wrap existing v0 logic to fit a §8 contract (Phases 1, 4, 5
  per design doc §3).
- **Build** — net-new work (Phases 0, 1.5, 2, 6, 9 per design doc §3).
