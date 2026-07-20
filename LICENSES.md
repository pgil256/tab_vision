# LICENSES — TabVision Dependency & Asset Map

**Last updated:** 2026-07-20 (personal non-commercial posture amendment)
**Spec reference:** SPEC.md §1.5 (personal non-commercial licensing) and §6 (resource acquisition)

## Posture

**Amended 2026-07-20 (user-approved; DECISIONS.md same date).** TabVision is a
**personal, non-commercial application**. The **shipping default** end-to-end
pipeline must use only weights, models, and dependencies whose licenses permit
personal non-commercial use with public source display (public repo, README,
demo). Default preference remains permissive (Apache-2.0, MIT, BSD, CC-BY);
**AGPL-3.0 is accepted by deliberate deviation for the YOLO guitar detector**
(DECISIONS.md 2026-05-05 "Phase 3 detector path").

**Non-commercial licenses (CC-BY-NC, CC-BY-NC-SA) are now acceptable in the
default pipeline and as training substrate for shipped derived artifacts**,
under three standing conditions: (1) use stays non-commercial; (2) attribution
and ShareAlike obligations are honored per artifact below; (3) every NC-derived
artifact is labeled **NC** in this file so a future commercialization knows
exactly what to replace. The pre-2026-07-20 permissive-only rows below are kept
with their history; rows re-opened by this amendment say so explicitly.

**Private/user recordings remain banned** from all training/eval/label roles —
the 2026-06-11 cleanup stands and was explicitly retained by the user on
2026-07-20.

**Effect of AGPL acceptance:** the entire TabVision pipeline becomes a
"work based on" ultralytics under AGPL §1, so distribution requires we
also distribute the source under AGPL. That is fine for a public-portfolio
repo. It does mean any future commercial / proprietary-SaaS use of
TabVision would either need an ultralytics enterprise license or a swap
to a permissive detector path (e.g., fine-tune YOLOS / DETR / RT-DETR).

**Phase 9 acceptance** includes a CI check verifying default-pipeline artifacts
match this list and that AGPL contagion is documented prominently in the
public README. The current scaffold is
`tabvision/scripts/check_default_licenses.py`, which verifies that
`[project].dependencies` remains free of opt-in render/model/AGPL packages.
Phase 0 (this document) produces the initial map; Phase 9 verifies.

## Status legend

- ✅ — verified permissive, OK for default pipeline
- ⚠️ — needs verification before its phase commits
- ❌ — non-permissive, default-pipeline-blocked (experiments only)

## Models & weights

| Component | Phase | License | Status | Notes |
|---|---|---|---|---|
| Spotify Basic Pitch | 1 | Apache-2.0 | ✅ | `pip install basic-pitch` — current v0 default backend |
| Riley/Edwards High-Res Guitar | 2 | **MIT** | ✅ | **Corrected 2026-05-05.** The actual implementation is at `xavriley/hf_midi_transcription` (multi-instrument: sax/bass/guitar/piano, original repo description "solo saxophone" was misleading). Pip-installable as `hf-midi-transcription`. Pretrained weights on HF: https://huggingface.co/xavriley/midi-transcription-models. License declared MIT in three places: pyproject.toml `License :: OSI Approved :: MIT License` classifier, HF model card YAML frontmatter `license: mit`, README. The earlier ❌ for `xavriley/HighResolutionGuitarTranscription` (paper-companion website only) stands — that repo is *not* the implementation. |
| GAPS benchmark model | 2 (alt) | **MIT — same package** | ✅ | **Corrected 2026-05-05.** GAPS-trained checkpoint is `guitar-gaps.pth` inside the same `hf-midi-transcription` package. `instruments.json` exposes it as `guitar` (default) or `guitar_gaps`. No separate repo needed. |
| trimplexx tab CRNN | 2/5 | **README claims MIT, no LICENSE file** | ❌ | Verified 2026-05-05: `trimplexx/music-transcription` README badge says MIT but the repo has no `LICENSE` file. Under copyright default ("no license = all rights reserved"), README claims aren't binding. Cannot ship as default. Could open issue requesting LICENSE addition; deferred for now. |
| Cwitkowitz FretNet | 2 (alt) | MIT | ⚠️ pretrained weights | `cwitkowitz/guitar-transcription-continuous`, 34★, MIT-licensed code. **No pretrained weights or releases** — training from scratch on GuitarSet is required. Same author as GAPS / amt-tools / lhvqt (all MIT). |
| Cwitkowitz amt-tools | 2 (framework) | MIT | ✅ | `cwitkowitz/amt-tools`, 38★, MIT. Underlying framework for FretNet and other Cwitkowitz transcription work. |
| Cwitkowitz lhvqt | 2 (filterbank) | MIT | ✅ | `cwitkowitz/lhvqt`, 21★, MIT. HVQT-initialized filterbank used by FretNet input. |
| Cwitkowitz with-inhibition (TabCNN+inhibition) | 2/5 (alt) | MIT | ⚠️ pretrained weights | `cwitkowitz/guitar-transcription-with-inhibition`, 19★, MIT. Same caveat as FretNet — code is MIT but pretrained weights aren't published; training is required. |
| MediaPipe Hands | 4 | Apache-2.0 | ✅ | `pip install mediapipe` — current v0 hand tracker |
| YOLO (ultralytics) — incl. YOLOv8/v10/11 detect + OBB + pose | 3 | AGPL-3.0 | ⚠️ accepted | **Verified 2026-05-05.** ultralytics LICENSE is the full GNU AGPL v3 (no permissive carve-out). Using it taints the whole TabVision pipeline as AGPL. Accepted by user 2026-05-05 ("do A") because no permissive pretrained guitar detector exists; see DECISIONS.md 2026-05-05 entry. Phase 3 trains YOLO-OBB on Roboflow `b101/guitar-3` for guitar bbox + rotation. |
| YOLO-OBB pretrained backbone (`yolov8n-obb.pt` / `yolo11n-obb.pt`) | 3 | weights distributed under AGPL alongside the code | ⚠️ accepted | Pretrained on DOTA (aerial); fine-tuned on Roboflow guitar dataset. Inherits AGPL contagion. |

## Datasets

| Dataset | Phase | License | Status | Notes |
|---|---|---|---|---|
| GuitarSet | 1.5 / 7 / string assignment | CC-BY-4.0 | ✅ | https://guitarset.weebly.com — JAMS annotations, hexaphonic. Audio/annotations are not redistributed. Checked-in position/sequence counts use players 00–04 only, exclude player 05, and carry hash-verified manifests with construction commands. Solo/comp candidates failed their OOF gates and remain unregistered. The 2026-07-14 16 kHz timbral probe and 2026-07-16 native-rate adjacent-string probe also used players 00–04 but failed; no learned timbral weights were exported or shipped. |
| Guitar-TECHS | Phase 0 (eval) / 1.5 / 7 | CC-BY-4.0 (Zenodo record 14963133) | ✅ eval-only | arXiv:2501.03720 — 3 electric guitarists, 5h12m multi-mic + DI; per-string 6-track MIDI. **Acquirer landed** (`scripts.acquire.datasets guitar-techs`, Zenodo API). **Scanner landed** (`manifest_builder.scan_guitar_techs` → `clean_electric` tier) — layout *inferred*, verify against first real download. Not redistributed here; required attribution must appear in the public README. |
| IDMT-SMT-Guitar | 1.5 / 7 | research-use, registration | ⚠️ | Training-only; not redistributed in our repo. Verified 2026-05-13 research pass; superseded by Guitar-TECHS for v1 acceptance — kept for potential future training augmentation. |
| EGDB | 1.5 / 7 / Phase 0 (eval) | **author-granted use (2026-06-01)** | ✅ eval-only | https://ss12f32v.github.io/Guitar-Transcription/ — 240 tracks, ~12h with multi-amp electric variants, GuitarPro tabs + aligned MIDI. **Access is open** — the audio is a public Google Drive folder linked from the project page; the *license* was the only gate (the repo has no LICENSE file → default all-rights-reserved). Author (`f08946011@ntu.edu.tw`) granted portfolio use 2026-06-01. **ACTION REQUIRED: save the grant email under `docs/` (e.g. `docs/licenses/egdb-grant-2026-06-01.eml`) and log it in `docs/DECISIONS.md` — the written grant is the only evidence the gate cleared (SPEC §1.4 hard rule).** Treated like GuitarSet: held-out distorted-electric eval source, **not redistributed** here and **not a shipped-weight substrate** unless the grant explicitly permits portfolio distribution. If the grant is research-only, it remains an eval gate only. |
| GAPS | v1.1 eval + **classical-domain training substrate (re-opened 2026-07-20)** | CC-BY-NC-SA-4.0 | ✅ **NC** (personal posture) | Zenodo 10.5281/zenodo.13962272. 14h of real classical guitar audio-score aligned pairs with high-resolution MIDI alignments and performance-video links. Do not commit or redistribute media. **2026-07-20 posture amendment:** the train split may now feed shipped derived artifacts (e.g. `gaps-v1` / `gaps-seq-v1` count-statistics priors). Conditions: attribution here + README; derived artifacts are labeled **NC-SA** (they inherit CC-BY-NC-SA-4.0 and must be replaced before any commercial use); the eval test split (test-22 / clean-12) never enters training manifests. |
| ~~GOAT~~ | DROPPED from default pipeline; candidate only if access/license changes | request-only, license pending | ❌ | arXiv:2509.22655 / GOAT-Dataset. The paper describes DI electric guitar audio plus amp-rendered variants annotated with string/fret tablatures, but dataset access is by request and must be rechecked before any use. Not portfolio-compatible until explicit access and dataset license terms are saved. |
| SynthTab | **re-opened 2026-07-20** as candidate training substrate (not yet used) | dataset CC-BY-NC-4.0 (code CC-BY-4.0) | ⚠️ **NC** (personal posture) | github.com/yongyizang/SynthTab. Was DROPPED 2026-05-13 because the NC clause taints derived weights under the old portfolio rule. **2026-07-20 posture amendment:** NC-derived weights are now acceptable (labeled NC). No SynthTab data has been downloaded or used yet; any future weight trained on it must be recorded here as **NC**. The repo code (Apache/CC-BY) remains usable for our own renderers. |
| DadaGP | research/dev only — **not in default pipeline** | access-by-email; underlying GP tabs derive from copyrighted songs | ⚠️ | https://github.com/dada-bots/dadaGP. Per 2026-05-13 design plan §4.2, acceptable as internal training augmentation only. Synthetic-source clips are blocked from non-train manifest splits by `tabvision.eval.manifest.validate_manifest` (the `SYNTHETIC_IN_EVAL_SPLIT` guard). |
| ~~User clips (the private eval + training corpus)~~ | BANNED | self-owned | ⛔ | Banned from all roles per 2026-06-11 cleanup: not as accuracy gate, dev set, label source, or historical benchmark source. The tracked tabs and stale result artifacts were removed; replace with GuitarSet / Kaggle UT-Austin / GAPS-style offline public corpora depending on the eval tier. |
| PDMX (`no_license_conflict` subset, 222,856 songs) | v1.1 A15 (sequence-prior corpus; committed artifacts `pdmx-seq-v1`, `guitarset-pdmx-seq-v1`) | **CC-BY (dataset) over PD-Mark/CC0 scores** | ✅ cleared 2026-07-02; conditions honoured 2026-07-05 | Long et al., "PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing", ICASSP 2025; arXiv 2409.10831 / Zenodo 10.5281/zenodo.15571083 / github.com/pnlong/PDMX. **License review 2026-07-02 (read-only, no downloads): CLEAR-WITH-CONDITIONS** for a shipped derived n-gram prior — same artifact class as the GuitarSet-derived `guitarset-v1`/`guitarset-seq-v1` priors. Conditions honoured 2026-07-05: attribution here + `tabvision/README.md` ("Dataset attribution"); `no_license_conflict` subset only (CSV filter in `scripts/acquire/pdmx_extract_transitions.py`); count tables only — no score content committed (archive + CSV stay in the local data root). Yield resolved: 3,435 guitar songs → 734 TAB-bearing → 554 standard-tuning used → 71,527 singleton transitions. MXL files targeted (MuseScore 3.6.2 exports preserve `<technical><string>/<fret>`), NOT MusicRender JSON. |
| Roboflow `b101/guitar-3` | 3 (training) | **CC BY 4.0** | ✅ | **Verified 2026-05-05.** Source: https://universe.roboflow.com/b101/guitar-3. Forked into Patrick's workspace as `patricks-workspace-vozcg/guitar-3-4efcd` v2; YOLOv8-OBB export downloaded (926 images, 710/144/72 split, classes: fret / neck / nut). License declared in the dataset's README.dataset.txt: "License: CC BY 4.0". Attribution: "guitar 3" by b101 on Roboflow Universe (https://universe.roboflow.com/b101/guitar-3), CC BY 4.0; export downloaded May 5, 2026 via the Roboflow SDK. **Required attribution must appear in the public README and any blog post.** |

### Phase 5 data/license addendum (2026-07-16)

The original direct per-string experiment used only the CC-BY-4.0 GuitarSet
microphone/JAMS material for players 00-04. Player 05 remained sealed. The OOF
gold-pitch gate failed, so no learned weight was exported, registered, or
shipped. Guitar-TECHS remains a separate CC-BY-4.0 electric track and was not
mixed into the acoustic acceptance model. GOAT was rechecked against its
official paper/project material: no dataset download or dataset-license grant
suitable for shipped derived weights was found; the arXiv paper license is not
a dataset license. GOAT, SynthTab, GAPS, EGDB, and private data were excluded
from training. See the frozen Phase 5A design and Phase 5 evaluation report.

### Personal-posture amendment addendum (2026-07-20)

User directive (chat, 2026-07-20): TabVision is a personal, non-commercial
application; NC licenses are accepted per the amended Posture above, and two
previously governance-blocked items are promoted:

1. **`highres-ensemble` becomes the clean-acoustic `auto` audio backend.** Both
   checkpoints it loads (`guitar-gaps.pth`, `guitar-fl.pth`) ship in the MIT
   `hf-midi-transcription` package — the earlier `guitar-fl.pth` block in the
   default-artifact CI was a *scope* rule ("not the accepted acoustic default"),
   not a license finding, and is lifted now that the ensemble is the accepted
   default. The 1,111-byte calibration artifact `ensemble_v1.json` is original
   TabVision work (repo license).
2. **GAPS train split may produce shipped derived priors** (`gaps-v1`,
   `gaps-seq-v1`) for the classical route. These derived count tables inherit
   **CC-BY-NC-SA-4.0** and are labeled NC-SA in this file; media is never
   committed; the GAPS eval splits stay out of training manifests.

Private/user recordings remain banned from all roles (unchanged).

### Phase 7 integrated audit addendum (2026-07-16)

Phase 7 introduced no dataset, runtime dependency, trained weight, or
redistributed media. Its clean replay used only the already mapped GuitarSet
(CC-BY-4.0), GAPS, and Guitar-TECHS (CC-BY-4.0) evaluation inputs. The Phase 3
scalar ensemble JSON remains an explicit-only artifact; its committed LF bytes
and SHA-256 are now recorded consistently in the Phase 3 and Phase 7 reports.
Failed Phase 2, 4, 5, and 6 artifacts remain unregistered or absent from the
runtime. `scripts/check_default_licenses.py --pyproject pyproject.toml` passed
both the default dependency and default artifact policies from a clean
checkout and from a fresh `.[dev]` install.

## Library dependencies (default pipeline)

| Library | License | Status | Notes |
|---|---|---|---|
| Python 3.11 | PSF | ✅ | TBD: confirm v0 actually uses 3.11 (Phase 0 inventory task) |
| numpy | BSD-3-Clause | ✅ | |
| ffmpeg-python | Apache-2.0 | ✅ | Wrapper; the underlying `ffmpeg` binary is LGPL/GPL depending on build, used as a shell-out only |
| opencv-python | Apache-2.0 | ✅ | |
| structlog | Apache-2.0 / MIT | ✅ | |
| pydantic | MIT | ✅ | |
| ruff | MIT | ✅ | Dev dep |
| mypy | MIT | ✅ | Dev dep |
| pytest | MIT | ✅ | Dev dep |
| PyGuitarPro (Phase 6, `render` extra) | **LGPL-3.0-only** | ✅ cleared 2026-07-09 (D3) | Verified via PyPI `license_expression: LGPL-3.0-only`. Portfolio-safe as an **opt-in library import** from the pip-installable CLI: LGPL permits an application under *any* license (incl. permissive) that merely *uses* — dynamically links / imports — the library, provided the library stays LGPL and is replaceable; Python `import guitarpro` + pip satisfy that, so TabVision does **not** become copyleft. Conditions: (1) keep it in the `render` extra, never `[project].dependencies` (already CI-enforced by `scripts/check_default_licenses.py`); (2) use unmodified — don't fork-and-bundle without releasing those mods under LGPL; (3) NOTICE/README attribution. Revisit only if TabVision is ever shipped as a frozen/static binary (relinking clause). Strictly *less* restrictive than the already-accepted AGPL detector. See DECISIONS 2026-07-09 (D3). |
| music21 (Phase 6, `render` extra) | BSD-3-Clause | ✅ verified 2026-07-09 (D3) | PyPI `license_expression: BSD-3-Clause` (classifier "OSI Approved :: BSD License"), latest v10.5.0. Permissive; portfolio-clear. Only obligation: retain the BSD LICENSE + copyright notice in NOTICES/README. (music21 was LGPL pre-v5; current versions are BSD.) |
| mido (Phase 6) | MIT | ✅ | |
| mir_eval | MIT | ✅ | Eval-only |

## Frozen v0 dependencies

The v0 desktop app (`tabvision-server/` Flask + `tabvision-client/` Electron) is
frozen per design doc §2 Q4. Its dependencies are not part of this LICENSES.md
audit because v0 is not the v1 shipping artifact. v0 retains its existing
license posture; if revived as a Phase-9 / v1.1 deliverable, a separate audit
will be needed.

## Phase-by-phase verification gates

- **Phase 0 (this document):** initial map exists. ⚠️ items flagged.
- **Phase 2 (verified 2026-05-05, then corrected same day):** initial pass
  found only the Nerfies-template companion website (`xavriley/HighResolutionGuitarTranscription`)
  and assumed Phase 2 was blocked. **Correction:** the real implementation
  is `xavriley/hf_midi_transcription` (pip-installable as `hf-midi-transcription`),
  hosting MIT-licensed pretrained guitar checkpoints (gaps + fl) on HF. Both
  Riley/Edwards High-Res and GAPS ship in this single package. Phase 2 path
  is open. See DECISIONS.md 2026-05-05 reversal entry.
- **Phase 3:** YOLOv8 AGPL question resolved. If unresolvable for portfolio,
  pick alternative detector path (Faster R-CNN, DETR, or self-trained from
  permissive base).
- **Phase 6 (resolved 2026-07-09, D3):** PyGuitarPro (LGPL-3.0-only) + music21
  (BSD-3-Clause) both cleared for portfolio TAB / Guitar Pro export; MIDI export
  uses `mido` (MIT). **Gate CLOSED — export work unblocked.** PyGuitarPro is
  portfolio-safe as an opt-in library import (LGPL "use, don't modify" case; not
  copyleft-contagious for a pip-installable CLI). Standing conditions: the LGPL
  dep stays in the `render` extra (CI-enforced), used unmodified, with NOTICE /
  README attribution; revisit only for a frozen-binary distribution. See
  DECISIONS 2026-07-09.
- **Phase 9:** CI check verifies all loaded model identifiers in default
  pipeline match the ✅ list above. Any ⚠️ that bled through fails the check.
  Current scaffold: `cd tabvision && python scripts/check_default_licenses.py`.

## Action items (resolve before respective phase)

- [x] **Phase 2:** Verified Riley/Edwards weights license (2026-05-05, corrected same day) — `hf-midi-transcription` is MIT-licensed and bundles pretrained guitar weights. ✅
- [x] **Phase 2:** Verified trimplexx CRNN license (2026-05-05) — README claims MIT, no LICENSE file in repo; ❌ as default. Not needed since hf-midi-transcription supersedes.
- [ ] **Phase 2 (open):** Add `hf-midi-transcription` to dependencies and verify it runs on Python 3.11 / our platform.
- [ ] **Phase 2 (open):** Confirm the `guitar-gaps.pth` checkpoint covers our acoustic + electric clean tier (per the GAPS paper, GAPS = "Classical Guitar Dataset" so it's mostly classical). May need `guitar-fl.pth` (Francois Leduc, electric/jazz) as a complementary backbone for some clips.
- [ ] **Phase 3:** Resolve ultralytics AGPL applicability to weights-only consumption.
- [x] **EGDB license — author-granted use 2026-06-01** (eval-only; save grant email under `docs/` + log in `docs/DECISIONS.md`; not a shipped-weight substrate unless the grant permits portfolio distribution).
- [ ] **Phase 7:** Verify DadaGP license for synthetic-data rendering.
- [x] **Phase 6:** PyGuitarPro LGPL implications verified for portfolio distribution
  (2026-07-09, D3) — cleared as an opt-in `render`-extra library import (not
  copyleft-contagious); `music21` BSD-3-Clause + `mido` MIT confirmed. See
  DECISIONS 2026-07-09 and the Phase 6 verification gate above.
- [x] **Phase 9:** Expand the license-check scaffold to compare loaded model artifacts
  against the ✅ list (2026-07-09) — `scripts/check_default_licenses.py` now enforces a
  second "default artifact policy" alongside the dependency policy. It resolves what the
  default pipeline actually loads from the real CLI defaults (`highres` →
  `xavriley/midi-transcription-models:guitar-gaps.pth`, MIT; `guitarset-v1` /
  `guitarset-seq-v1` via clean-acoustic `auto` routing → derived count statistics,
  CC-BY-4.0) and fails CI if any resolved
  artifact is off the ✅ list or on the blocked list (ultralytics weights,
  `guitar-fl.pth`, the electric checkpoint). Closes the SPEC §11 "NC weights leaked into
  the default" risk row.
