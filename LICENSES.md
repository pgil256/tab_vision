# LICENSES — TabVision Dependency & Asset Map

**Last updated:** 2026-05-05 (Phase 0 initial map)
**Spec reference:** SPEC.md §1.5 (portfolio-friendly licensing) and §6 (resource acquisition)

## Posture

TabVision is a **portfolio project** (SPEC §1.5). The **shipping default**
end-to-end pipeline must use only weights, models, and dependencies with
permissive licenses (Apache-2.0, MIT, BSD, CC-BY, or compatible). Non-commercial
or research-only artifacts may be used in optional fine-tune *experiments* but
**must not be required by the default backend**.

**Phase 9 acceptance** includes a CI check verifying default-pipeline artifacts
match this list. Phase 0 (this document) produces the initial map; Phase 9
verifies.

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
| YOLOv8n / ultralytics | 3 | AGPL-3.0 (code) + CC weights | ⚠️ | **AGPL is contagious for derivative code.** Verify whether weights-only usage avoids AGPL obligation. If not, find alternative detector (e.g., Faster R-CNN, or fine-tune our own from a permissive base). |
| YOLOv8-pose (fretboard keypoint fallback) | 3 | same as above | ⚠️ | Same AGPL question as YOLOv8n. |

## Datasets

| Dataset | Phase | License | Status | Notes |
|---|---|---|---|---|
| GuitarSet | 1.5 / 7 | CC-BY-4.0 | ✅ | https://guitarset.weebly.com — JAMS annotations, hexaphonic. Already used in v0 finetune work. Re-distribution requires attribution; not committed to repo. |
| IDMT-SMT-Guitar | 1.5 / 7 | research-use, registration | ⚠️ | Training-only; not redistributed in our repo. Verify scope of "research use" for portfolio context. |
| EGDB | 1.5 / 7 | TBD | ⚠️ | https://github.com/ss12f32v/GuitarTranscription — multi-amp distorted electric. Verify before relying on it for distorted-electric tier eval. |
| DadaGP | 7 | TBD | ⚠️ | https://github.com/dada-bots/dadaGP — GuitarPro tabs as synthetic-data substrate. |
| User clips (existing 11/20 self-recorded) | 1.5 (bonus) | self-owned | ✅ | iPhone OOD bonus tier per design doc §6. Owned by Patrick. |

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
| PyGuitarPro (Phase 6) | LGPL-3.0 | ⚠️ | LGPL is OK for use, but distribution implications need verification for the portfolio claim. |
| music21 (Phase 6) | BSD-3-Clause | ✅ | |
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
- **Phase 6:** PyGuitarPro LGPL implications resolved.
- **Phase 9:** CI check verifies all loaded model identifiers in default
  pipeline match the ✅ list above. Any ⚠️ that bled through fails the check.

## Action items (resolve before respective phase)

- [x] **Phase 2:** Verified Riley/Edwards weights license (2026-05-05, corrected same day) — `hf-midi-transcription` is MIT-licensed and bundles pretrained guitar weights. ✅
- [x] **Phase 2:** Verified trimplexx CRNN license (2026-05-05) — README claims MIT, no LICENSE file in repo; ❌ as default. Not needed since hf-midi-transcription supersedes.
- [ ] **Phase 2 (open):** Add `hf-midi-transcription` to dependencies and verify it runs on Python 3.11 / our platform.
- [ ] **Phase 2 (open):** Confirm the `guitar-gaps.pth` checkpoint covers our acoustic + electric clean tier (per the GAPS paper, GAPS = "Classical Guitar Dataset" so it's mostly classical). May need `guitar-fl.pth` (Francois Leduc, electric/jazz) as a complementary backbone for some clips.
- [ ] **Phase 3:** Resolve ultralytics AGPL applicability to weights-only consumption.
- [ ] **Phase 7:** Verify EGDB license for distorted-electric eval/training.
- [ ] **Phase 7:** Verify DadaGP license for synthetic-data rendering.
- [ ] **Phase 6:** Verify PyGuitarPro LGPL implications for portfolio distribution.
- [ ] **Phase 9:** Implement license-check CI step that compares loaded model artifacts against the ✅ list.
