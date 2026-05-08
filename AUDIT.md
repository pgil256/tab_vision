# AUDIT — TabVision v0 → v1 Spec Adoption

**Date:** 2026-05-05
**Author:** Patrick (with Claude)
**Phase:** 0 (initial audit)
**Spec reference:** SPEC.md §2.1 (audit checklist) and §2.2 (preserve rules)
**Design doc:** `docs/plans/2026-05-05-tabvision-spec-adoption-design.md`

---

## 1. User interview (SPEC.md §2.1 Step 0)

### 1. Where is the existing project on disk and is it a separate repo or a folder in this one?

`/home/gilhooleyp/projects/tab_vision`. Single git repo. Contains backend
(`tabvision-server/`), Electron client (`tabvision-client/`), and a parallel
web client (`web-client/`) that was added later for Vercel deployment.

### 2. What state is it in — does anything run end-to-end?

Backend runs end-to-end via `python tabvision-server/run.py` (Flask, port 5000).
Most recent eval: 91.6% mean Exact F1 across 11 hand-curated videos (per the
v0 11-video benchmark; details preserved on branch `agent-farm-improvements`
via the v0 dev history). The 20-video training set has lower vanilla-baseline
metrics (~0.43–0.51 mean Exact F1) — see §6 below.

Most recent in-flight work: Phase 1 audio fine-tune of Basic Pitch on
GuitarSet, on branch `feature/audio-finetune-phase1`. **Frozen mid-experiment**
as of 2026-05-05 per spec adoption (user override of design doc sequencing —
recorded in DECISIONS.md).

### 3. What approach did you take previously?

**Audio + vision + heuristic fusion.** Audio: Spotify Basic Pitch (TensorFlow
2.15) for polyphonic onset/pitch. Vision: MediaPipe Hands for finger
landmarks + custom geometric fretboard detection (Canny + Hough lines +
RANSAC). Fusion: bespoke engine in `app/fusion_engine.py` with
position-scoring, two-pass anchoring on 3+ note chords, melodic segment
correction, slide correction, and several ghost-note / harmonic / sustain
filters tuned to specific failure cases.

**What worked** (preserve as porting source):
- Melodic segment correction (biggest position win, multiple videos jumped 30+ pp).
- String compactness penalty in chord scoring.
- Two-pass anchor strategy using only 3+ note chords (not 2-note).
- 2-digit fret parser fix (≤ MAX_FRET guard) — caused 9pp on video-3.
- Octave-specific harmonic filter ratio (0.35 for octaves vs 0.7 for other harmonics).
- Sustain redetection filter.

**What didn't ship** (NO_SHIP results to document and shelve):
- **Video hand anchor** — gated `use_video_hand_anchor=False` by default; null result on hand-position lift.
- **Learned position selector** (LightGBM ranker) — `agent-farm-improvements` 2026-04-29; +0.3pp LOOCV vs heuristic, far below the +5pp ship gate.
- **Phase 0 RMS activity-end truncation** — `feature/audio-finetune` (Phase 0 RMS); fixed-dB silence detection is a no-op on iPhone audio.
- **Audio fine-tune (Path 2)** — 5 fine-tune runs on `feature/audio-finetune-phase1`, all regressed vs vanilla 0.871 (GuitarSet held-out). H1 (lr) refuted, H2 (label encoding) untested, branch frozen.

### 4. What datasets, weights, or annotations should not be lost?

| Asset | Path | Reuse target |
|---|---|---|
| 11 self-recorded eval videos + ground-truth | `test-data/existing/` + tabs files | Spec Phase 1.5 iPhone OOD bonus tier |
| 20 self-recorded training videos | `test-data/training-tabs/` (.txt tab files) + `test-data/existing/` | Phase 1.5 + Phase 7 |
| GuitarSet TFRecord splits | `tabvision-server/tools/outputs/tfrecords/guitarset/splits/{train,validation}/` | Phase 7 fine-tuning data (5 train players + 1 validation player) |
| Pretrained Basic Pitch weight loader | `tabvision-server/app/training/load_pretrained.py` | Phase 7 (verified equivalent 2026-04-29) |
| GuitarSet dataset wrapper | `tabvision-server/app/training/guitarset_dataset.py` | Phase 7 |
| Fine-tune training scripts | `tabvision-server/tools/finetune_basic_pitch_{smoke,modal}.py` | Phase 7 reference |
| Error analysis harness | `tabvision-server/tools/error_analysis.py` | Phase 8 (deterministic eval harness port) |
| Vanilla Basic Pitch baseline (20-video) | `tabvision-server/tests/fixtures/benchmarks/results/vanilla-baseline-2026-05-01.json` | Phase 1 / Phase 7 reference point |
| Benchmark history (baseline_v1..v3, tuning_v1..v13) | `tabvision-server/tests/fixtures/benchmarks/results/` | Reference; not directly ported |
| 17 design docs in `docs/plans/` | `docs/plans/2026-01-* … 2026-05-*` | Context / cross-references in v1 design doc |

### 5. Branches with abandoned approaches worth revisiting?

| Branch | Status | Worth revisiting? |
|---|---|---|
| `agent-farm-improvements` | merged into v0 mainline (91.6% F1) | Already preserved via the v0 codebase; reference for fusion improvements |
| `feature/audio-finetune` | NO_SHIP (Phase 0 RMS truncation) | No — failure mode documented |
| `feature/audio-finetune-phase1` | Frozen mid-experiment (H2) | Yes — Phase 7 entry will revisit |
| `feature/accuracy-ui-recording` | unknown | TBD; not blocking |
| `feature/automated-accuracy-testing` | unknown | TBD; informs Phase 8 harness |
| `auto-claude/001-pending` | auto-generated | Skip |

### 6. Anything not to touch?

- `tabvision-client/` (Electron) — frozen per design doc Q4. Demo asset; do not develop further during v1.
- `tabvision-server/` (Flask backend) — frozen as v0. `tabvision-client/` depends on it; both stay alive until Phase 9 / v1.1 follow-up.
- `web-client/` — separate web frontend (`tabvision-web` Vite app). Not in design doc scope; treat as frozen v0 too. Confirm fate at Phase 9.
- `vercel.json` at repo root — flagged as deferred open question; verify whether it signals an aborted/incomplete deploy plan or an active web-client deploy.

### 7. What Python version + tooling does the existing project use?

- **Python 3.11** (verified: `tabvision-server/venv/lib/python3.11`).
- **Flask 3.0.0**, gunicorn 21.2 — web layer.
- **TensorFlow 2.15.1** — pinned; required by Basic Pitch and the GuitarSet fine-tune work.
- **Basic Pitch ≥ 0.3.0**, MediaPipe ≥ 0.10.0, OpenCV ≥ 4.8.0, librosa ≥ 0.10.0, scipy ≥ 1.10, numpy ≥ 1.24, ffmpeg-python ≥ 0.2 — pipeline core.
- **mirdata ≥ 0.3.0** (GuitarSet), **pandas ≥ 2.0**, **pyarrow ≥ 15** (position dataset), **lightgbm ≥ 4.0** (NO_SHIP learned-fusion).
- No pre-commit hooks observed; no ruff/mypy config; tests via `pytest`.
- Electron client (`tabvision-client/`): React 19, electron-squirrel-startup, zustand 5.
- Web client (`web-client/`): Vite + React (separate `tabvision-web` package).

---

## 2. Inventory

### Backend (`tabvision-server/`)

**Python source — 23 modules in `app/` + 9 tools + 17 tests:**

| Module | Role | Spec phase home |
|---|---|---|
| `app/audio_pipeline.py` | Basic Pitch wrapper, harmonic/sustain filters | Phase 1 (port → `tabvision.audio.basicpitch`) |
| `app/fusion_engine.py` | Position selection, two-pass anchoring, melodic segment correction | Phase 5 (port → `tabvision.fusion.{viterbi,playability,chord}`) |
| `app/fretboard_detection.py` | Canny + Hough geometric fretboard | Phase 3 (port → `tabvision.video.fretboard.geometric`) |
| `app/video_pipeline.py` | MediaPipe hand orchestration | Phase 4 (port → `tabvision.video.hand.mediapipe_backend`) |
| `app/hand_anchor.py` | Gated NO_SHIP video-hand-anchor experiment | Don't port; document |
| `app/guitar_mapping.py` | MIDI → candidate (string, fret) | Phase 5 (port → `tabvision.fusion.candidates`) |
| `app/chord_shapes.py` | Chord templates / voicing knowledge | Phase 5 (port → `tabvision.fusion.chord`) |
| `app/beat_quantization.py` | Onset → beat alignment | Phase 6 (port if ASCII renderer needs it) |
| `app/error_analyzer.py` | Eval bucket harness | Phase 8 |
| `app/secondary_pitch_detector.py` | Pitch detection backup logic | Evaluate during Phase 1 port |
| `app/spectral_residual.py` | Audio post-processing | Evaluate during Phase 1 port |
| `app/temporal_interpolation.py` | Time-axis smoothing | Evaluate during Phase 5 port |
| `app/video_fp_filter.py` | Video false-positive filter | Phase 5 / Phase 4 |
| `app/video_only_pipeline.py` | Video-only demo path | Reference; not ported |
| `app/training/guitarset_dataset.py` | GuitarSet PyTorch Dataset | Phase 7 |
| `app/training/load_pretrained.py` | Basic Pitch weight loader | Phase 7 |
| `app/{processing,routes,models,storage}.py` | Flask web layer (job queue, REST) | **Not ported** — replaced by CLI |
| `app/fake_data.py` | Demo fixture generator | **Not ported** |

**Tools** (`tabvision-server/tools/*.py`, 9 scripts):
- `build_guitarset_tfrecords.py`, `sideload_guitarset_from_hf.py` — Phase 7 data prep
- `eval_basic_pitch_baseline.py` — Phase 1 / Phase 7 baseline
- `finetune_basic_pitch_{smoke,modal}.py` — Phase 7 training
- `error_analysis.py` — Phase 8 harness
- `build_position_dataset.py`, `dump_position_features.py`, `train_position_selector.py` — NO_SHIP learned-fusion artifacts (preserve for documentation; not ported)

**Test fixtures** (`tabvision-server/tests/fixtures/`):
- `test_a440.mp4` — single A440 reference clip
- `benchmarks/index.json`, `baseline*.json`, `tuning_v*.json`, `training_baseline.json`, `sample-video-tabs.txt` — eval baselines
- `benchmarks/results/vanilla-baseline-2026-05-01.json` — most recent reference point

### Frontend (`tabvision-client/` and `web-client/`)

**Frozen per design doc Q4. Inventory for completeness:**

- `tabvision-client/` — Electron + React 19 + Zustand. Modules: `src/{api,components,store,types,utils}/`. `electron-wrapper/` for Electron main process.
- `web-client/` — Vite + React (`tabvision-web`). Mirrors `tabvision-client/src/` layout. Vercel-deployed (`.vercel/` directory present).

### Repo-root scripts (untracked)

23 untracked items at repo root: ad hoc shell scripts (`check_ts.sh`,
`commit.sh`, `debug_fb.sh`, `debug_fb2.sh`, `diagnose.sh`, `diagnose2.sh`,
`get_bpms.sh`, `run_baseline.sh`, `run_tests.sh`), Python scratch
(`debug_fretboard_test.py`, `generate_test_data.py`), and finetune output
artifacts (`tabvision-server/tools/outputs/finetune_*` logs, `csv`, `md`).

**Disposition:** these scripts are dev convenience; not ported. Will be
.gitignored or removed during Phase 1/0-completion cleanup (decided case by
case).

### Other repo-root files

- `tabvision_specification.md` — original v0 spec from Jan 2026.
- `tabvision_prompt.txt`, `tabvision_agent_farm_prompt.txt`, `tabvision_agent_farm_config.json`, `tabvision_agent_config.json` — agent-farm prompts/configs.
- `agent_farm_report_*.html` — report files from agent-farm runs.
- Various debug images (`debug_*.{jpg,png}`, `frame_*.png`, `sample_frame.jpg`, `tab_comparison.png`).
- `vercel.json` — flagged for verification.
- Existing `coordination/`, `sample_frames/`, `frame_780.png` — debug artifacts.

---

## 3. What works

**Concretely verified:**

1. **End-to-end pipeline runs.** `python tabvision-server/run.py` starts Flask
   on port 5000; `POST /jobs` → process video → `GET /jobs/:id/result` returns
   a TabDocument JSON. Verified by the existence of v0 11-video eval results
   averaging 91.6% Exact F1 (per design doc and v0 history).
2. **20-video benchmark harness produces results.** Most recent run:
   `tests/fixtures/benchmarks/results/vanilla-baseline-2026-05-01.json` — 20
   training-NN clips run through the current pipeline with full per-clip
   metrics breakdown (exact / pitch / position / chord, plus per-error
   classification).
3. **Phase 1 audio fine-tune scaffolding works.** GuitarSet TFRecords built,
   pretrained-weight loader verified equivalent to SavedModel (2026-04-29),
   smoke trainer ran on 5 clips successfully. Five full fine-tune runs
   completed (all regressed vs vanilla — see §4).
4. **Test suite runs.** 17 test files; suite executes via `pytest`.
   (Coverage gist not measured — TODO during Phase 1 port.)

---

## 4. What's broken / known-failure modes

1. **Video-4 audio detection.** Pitch detector returns wrong MIDI notes (off
   by 1–2 semitones) on this clip. Likely guitar tuning or Basic Pitch
   rounding artifact. **Algorithmic fix unknown**; tracked as a Phase-2 / 7
   problem (audio backbone issue).
2. **Arpeggio position errors are fundamental.** Audio-only cannot distinguish
   ascending arpeggio (D7 → G5 → B5 → e5) from ascending scale on G string
   (G2 → G5 → G9 → G14). Would need multi-string assignment optimization.
   Spec Phase 5 fusion design must address this.
3. **No distorted-electric capability.** v0 was tuned on clean acoustic; no
   verification on distortion. Spec §1.4 / §7 explicitly target this tier;
   gap.
4. **No expressive markings.** v0 doesn't detect bends, hammer-ons, pull-offs,
   slides (other than slide-correction). Spec §1.3 defers expressive markings
   to v1.1.
5. **Phase 1 fine-tune NO_SHIP (5 runs).** Five Basic Pitch fine-tunes on
   GuitarSet on `feature/audio-finetune-phase1`, all regressed vs vanilla
   (0.871 GuitarSet held-out F1). H1 (learning rate) refuted. H2 (label
   encoding) untested at the time of branch freeze.
6. **NO_SHIP learned position selector.** `agent-farm-improvements` LightGBM
   ranker landed +0.3pp vs heuristic, far below the +5pp ship gate.

---

## 5. What's unknown

1. **Test coverage.** 17 test files exist; coverage on the 23 `app/*.py`
   modules is unmeasured. Run `pytest --cov` during Phase 1 port to discover
   untested code paths before porting.
2. **Capo support paths.** Spec requires capo 0–7. v0 may handle this but no
   targeted test was found in the audit; verify during Phase 1.
3. **`vercel.json` purpose.** Either an aborted/incomplete deploy plan or an
   active `web-client` deployment. Determine whether the web-client is being
   maintained externally or is also frozen.
4. **Branches `feature/accuracy-ui-recording`, `feature/automated-accuracy-testing`.**
   Status unknown. The latter may inform Phase 8 harness work; check before
   re-deriving.
5. **Whether main and feature/audio-finetune-phase1 should be reconciled.**
   Main is 33 commits behind. Decision deferred to Phase 9 (legacy cleanup).

---

## 6. Reusable artifacts

(Tabular form for clarity. Most also appear in §1 Q4.)

| Artifact | Path | Phase reuse |
|---|---|---|
| Audio pipeline logic | `tabvision-server/app/audio_pipeline.py` | Phase 1 (port) |
| Fusion engine | `tabvision-server/app/fusion_engine.py` | Phase 5 (port) |
| Fretboard geometric detection | `tabvision-server/app/fretboard_detection.py` | Phase 3 (port + fallback) |
| MediaPipe hand pipeline | `tabvision-server/app/video_pipeline.py` | Phase 4 (port) |
| MIDI → candidates | `tabvision-server/app/guitar_mapping.py` | Phase 5 (port) |
| Chord templates | `tabvision-server/app/chord_shapes.py` | Phase 5 (port) |
| GuitarSet TFRecords | `tabvision-server/tools/outputs/tfrecords/guitarset/splits/` | Phase 7 |
| Basic Pitch weight loader | `tabvision-server/app/training/load_pretrained.py` | Phase 7 |
| GuitarSet PyTorch wrapper | `tabvision-server/app/training/guitarset_dataset.py` | Phase 7 |
| Fine-tune training scripts | `tabvision-server/tools/finetune_basic_pitch_{smoke,modal}.py` | Phase 7 reference |
| Error analysis harness | `tabvision-server/tools/error_analysis.py` | Phase 8 (port + harden) |
| 11 + 20 self-recorded videos + tabs | `test-data/{existing,training-tabs}/` | Phase 1.5 (iPhone OOD) |
| Benchmark JSONs | `tabvision-server/tests/fixtures/benchmarks/results/` | Reference baseline |
| Design docs | `docs/plans/2026-01-* … 2026-05-*` | Cross-reference in v1 designs |

---

## 7. Baseline metrics

**Per SPEC.md §2.1: "Score the baseline pipeline output against the user's
reference annotation for that one clip. Record the metrics from §1.4."**

We have richer data than one clip: a recent 20-clip vanilla-baseline JSON
already exists (`tabvision-server/tests/fixtures/benchmarks/results/vanilla-baseline-2026-05-01.json`,
generated 2026-05-01).

### 20-clip training set (vanilla Basic Pitch + heuristic fusion, 2026-05-01)

Spot-check from per-clip F1s in the JSON (Exact F1 = string + fret + onset match):

| Clip | Exact F1 | Pitch F1 |
|---|---:|---:|
| training-01 | 0.42 | 0.68 |
| training-02 (estimate from JSON) | 0.32 | — |
| training-03 (estimate from JSON) | 0.86 | — |
| training-04 (estimate from JSON) | 0.57 | — |
| ... | ... | ... |

Mean Exact F1 across the 20-clip set is approximately **0.43–0.51** depending
on which harness alignment is used (corrected harness with `_find_best_time_offset`
vs original).

### 11-clip eval set (best v0 result, 2026-04-02)

Mean Exact F1: **0.916** across `sample-video, video-3, video-4, video-5,
video-6, video-7, video-8, video-9, video-10, video-11, video-12`. Per-clip
range 0.696 to 0.976.

### Spec §1.4 targets (for context)

| Metric | v0 status | Spec v1 target |
|---|---|---|
| Onset F1 (50 ms) | unmeasured for §1.4 definition | ≥ 0.92 |
| Pitch F1 (50 ms, no offset) | ~0.68–0.75 (20-clip), better on 11-clip | ≥ 0.90 |
| Tab F1 (string + fret + onset) | 0.43–0.51 (20-clip) / 0.916 (11-clip) | ≥ 0.88 |
| Chord-instance accuracy | low on 20-clip (many 0.0s) | ≥ 0.85 |
| End-to-end latency (60 s clip on laptop CPU) | unmeasured | ≤ 5 min |

**Interpretation:** v0 already exceeds the aggregate Tab F1 target on the
11-clip set (0.916 > 0.88), but on the broader 20-clip set the picture is
substantially weaker. The 11-clip set is plausibly cherry-picked for the
algorithm's strengths; the 20-clip set is the more honest baseline.

The §1.4 metrics are not directly comparable to v0's metrics until the spec's
eval harness lands in Phase 1 / Phase 8 with mir_eval-based definitions. v1
acceptance is measured against the spec definitions, not v0's.

**Action item (Phase 1):** re-run one fixture clip through the new
`tabvision.cli` with the spec's mir_eval-based metrics to produce the
strict-equivalent baseline.

---

## 8. Acceptance gate (per SPEC.md §9.3 Phase 0)

| Item | Status |
|---|---|
| `AUDIT.md` exists | ✅ this file |
| `LICENSES.md` exists | ✅ |
| `SPEC.md` (renamed from `TAB_SPEC_UPDATE.md`) | ✅ |
| Repo layout per SPEC §4 created | ⏳ Phase 0 task #8 |
| `pyproject.toml` with deps | ⏳ Phase 0 task #9 |
| CI green on empty pipeline | ⏳ Phase 0 task #10 |
| `scripts/acquire/` skeletons | ⏳ Phase 0 task #8 |
| User signs off | ⏳ pending review |

**Sign-off requested.** Open questions for review:

- Confirm the v0 / web-client status (`web-client/` not in original CLAUDE.md but exists in tree; treated as frozen pending Phase-9 decision).
- Confirm `vercel.json` interpretation — aborted plan or active web-client deploy?
- Confirm the disposition of NO_SHIP branches (`feature/audio-finetune`, `agent-farm-improvements`) — keep for history vs. delete.
