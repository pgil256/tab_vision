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

## Project status (2026-05-05)

**TabVision is mid-spec-adoption.** A new canonical specification at
`SPEC.md` (formerly `TAB_SPEC_UPDATE.md`) reframes the project as a Python
CLI with strict module boundaries. v0 (Electron + Flask, ~91.6% F1 on
11-clip set) is **frozen**; v1 (`tabvision/` package) is being built in
parallel under `refactor/v1`.

**Read these before any non-trivial change:**
- `SPEC.md` — canonical spec (10-phase plan, §8 immutable contracts).
- `docs/plans/2026-05-05-tabvision-spec-adoption-design.md` — adoption design
  (hybrid approach, phase mapping, sequencing, eval set strategy).
- `AUDIT.md` — Phase 0 audit: inventory, what works, reusable artifacts.
- `LICENSES.md` — dependency license map; ⚠️ items gate respective phase entry.
- `docs/DECISIONS.md` — non-obvious branches taken (per SPEC §0.5).

**Active branch (2026-05-13):** `main`. The Modal production deploy
(`936a5cc`) and v1 CI hardening landed on `main`; `refactor/v1` is now
**23 commits behind `main`** and should be treated as historical. Cut new
work branches off `main`. Older design docs (and earlier paragraphs in
this file) may reference paths that exist on `main` but not on
`refactor/v1` — verify with `git cat-file -e origin/main:<path>` before
relying on them. The full pipeline (`tabvision/tabvision/pipeline.py`),
the Modal production adapter (`tabvision-server/modal_app.py`,
`tabvision-server/app/v1_adapter.py`), and the highres audio backend all
live on `main`. Phase 5 fusion has shipped. See
`docs/2026-05-12-session-handoff.md` for the production state and
`docs/plans/2026-05-12-tab-f1-to-spec-design.md` (+ companion Phase 0
implementation plan) for current accuracy work.

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

# CLI (Phase 0 stub)
tabvision --version
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

**v1 scope (2026-06-02): acoustic, audio-only.** Honest audio-only targets on
GuitarSet (see SPEC §1.4.1): single-line Tab F1 ≥ 0.45, strummed ≥ 0.60,
aggregate ≥ 0.55, + onset ≥ 0.92 / pitch ≥ 0.90 / chord ≥ 0.85 / latency ≤ 5 min.
**Single-line is information-limited** — audio can't resolve which string a pitch
is on; 0.94 is a **v1.1 video** target (`docs/EVAL_REPORTS/acoustic_single_line_2026-06-02.md`).
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
