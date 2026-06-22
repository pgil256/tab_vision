# v1.1 chunk-5 — GAPS video real-chain (video half)

**Date:** 2026-06-22
**Branch:** `v1.1/oracle-string-resolution`
**Status:** **Video half DONE (measured; negative result).** The chunk-3
confidence-gated CV chain (YOLO-OBB neck → homography → MediaPipe → fretting-hand
selection → fusion) now runs end-to-end over the GAPS clean-12: source video is
acquired via yt-dlp and aligned to the GAPS audio crop by onset-envelope
cross-correlation (the net-new piece). **Headline: video does not lift Tab F1 on
GAPS clean-12 and does not reach the 0.94 single-line video target.** The
string-axis *lever* is confirmed large (gold-audio oracle 0.973), but the
Kaggle-tuned CV chain does not transfer to diverse YouTube classical-guitar
footage, so the no-regression gate (chunk-3 §5.3) correctly suppresses it.
**Depends on:** `v1_1_gaps_chunk5_2026-06-19.md` (audio half),
`docs/plans/2026-06-03-v1.1-video-string-resolution-design.md` (§6 eval gate).
**Raw auto-report:** `v1_1_gaps_video_chain_auto_2026-06-22.md` (regenerate via §7).

## 1. What this chunk delivered

| Deliverable | Where |
|---|---|
| yt-dlp acquisition + video↔audio offset aligner | `tabvision/scripts/acquire/gaps_video.py` |
| Aligner unit tests (10) | `tabvision/tests/unit/test_gaps_video_align.py` |
| GAPS video real-chain probe (align → CV → fuse → score) | `tabvision/scripts/eval/v1_1_gaps_video_chain_probe.py` |
| Raw auto-report (this run) | `docs/EVAL_REPORTS/v1_1_gaps_video_chain_auto_2026-06-22.md` |

The acquisition + alignment is net-new; the fusion / vision-evidence / scoring is
reused verbatim from the v1 package and the chunk-2/3 probes. GAPS is
non-commercial, offline-eval-only: source media stays local and is never
committed or redistributed (clean-12 mp4s live under `~/.tabvision/cache/`).

## 2. Video↔audio crop-offset alignment (net-new — works)

GAPS ships no video; the audio is a crop of the YouTube upload, so frame time ≠
gold time. The offset is recovered per clip by cross-correlating the
**onset-strength envelopes** (librosa) of the GAPS WAV and the video's decoded
audio at 100 fps (`video_time = gold_onset + offset`); the peak ratio is the
correlation peak over the best competitor outside a ±5-frame guard band. Onset
envelopes give far sharper peaks than raw RMS (ratio 2.3–11.2 vs ~1.4).

| Clip | Gold | offset (s) | xcorr peak ratio | wav dur | vid dur |
|---|---:|---:|---:|---:|---:|
| 027_Zpswc | 1607 | +0.040 | 7.91 | 404.9 | 405.0 |
| 031_vpswc | 887 | +0.040 | 5.59 | 186.5 | 186.5 |
| 043_bc1wc | 1401 | +0.050 | 11.24 | 328.0 | 328.0 |
| 063_bV1wc | 841 | +0.050 | 10.03 | 181.5 | 181.4 |
| 104_xf1wc | 422 | +0.040 | 9.57 | 184.4 | 184.4 |
| 118_VD1wc | 678 | +0.040 | 2.32 | 120.7 | 120.7 |
| 142_GD1wc | 701 | +0.040 | 4.97 | 238.9 | 238.9 |
| 179_pM1wc | 515 | +0.020 | 4.71 | 189.1 | 189.0 |
| 212_y41wc | 946 | +0.040 | 2.49 | 322.5 | 322.4 |
| 235_Ny1wc | 1582 | +0.040 | 8.58 | 415.8 | 415.8 |
| 294_BSswc | 474 | +0.040 | 3.24 | 125.0 | 125.0 |
| 341_1M1wc | 691 | +0.010 | 3.94 | 127.4 | 127.4 |

**Finding:** all offsets are sub-frame (+0.01 to +0.05 s, < one 24 fps frame
≈ 42 ms), and the wav/video durations match to < 0.1 s — for these clips the
uploaded video *is* essentially the GAPS crop (`cropped_duration`; the CSV's
shorter `duration` is the score-aligned span). The alignment is confirmed three
independent ways (duration match, RMS xcorr, onset xcorr) and is not the
bottleneck. The aligner is the reusable deliverable; the feared "large crop
offset" did not materialise on the clean-12.

## 3. Video-assisted Tab F1 (clean-12)

Four conditions per clip: `audio-only` (strings from the playability prior),
`+real (auto)` (CV chain resolves the string, orientation auto-selected per clip
— the honest number), `+real (best-orient)` (best over the 4 fixed orientations
— a diagnostic ceiling on orientation *selection*), `+oracle` (gold strings — the
absolute ceiling). Gate: vote 1 frame / ±60 ms, `min_clip_coverage = 0.71`
(chunk-3 §5.3 no-regression value).

| Audio source | audio-only | +real (auto) | +real (best-orient) | +oracle | vs 0.94 |
|---|---:|---:|---:|---:|---|
| **gold** (string axis isolated) | 0.8148 | **0.8148** (lo-95 0.768) | 0.8148 | **0.9726** | **below 0.94** |
| **highres** (honest end-to-end) | 0.7612 | **0.7612** (lo-95 0.706) | 0.7612 | 0.9099 | below 0.94 |

(Per-clip table in the auto-report.) `+real (auto) == +real (best-orient) ==
audio-only` on **every** clip: per-clip coverage is < 0.71 for all 12 (max 0.58),
so the no-regression gate drops the video evidence everywhere and the clip falls
back to audio-only. **Video contributes nothing on the clean-12, and the gate
prevents it from doing harm.**

The 0.94 target lives in the gold-audio frame (chunk-2/3 convention: gold pitch,
string/fret stripped → video resolves the string). The highres frame is the
honest end-to-end number and is additionally capped by audio pitch/onset
(`+oracle` 0.910 < gold's 0.973).

## 4. Why video doesn't help here — gate/orientation sweep (gold audio)

A cache-only sweep (frames computed once, fusion re-run) isolates whether the
failure is gating, orientation selection, or the CV evidence itself
(audio-only 0.8148, oracle 0.9726):

| Config | mean Tab F1 | vs audio-only |
|---|---:|---|
| gate 0.71 / auto **(default)** | 0.8148 | no-op — gated out on all 12 |
| gate 0.50 / auto | 0.7674 | **regresses** (hurts 3 clips, −0.047) |
| no-gate / auto | 0.6597 | hurts 11/12 (−0.155) |
| no-gate / fixed "none" | 0.6334 | hurts 11/12 (−0.181) |
| no-gate / **best fixed orient per clip** | 0.7632 | hurts 10/12 (−0.052) |
| oracle strings | 0.9726 | — |

**Under no setting does video beat the audio-only prior.** Even ungated, with the
per-clip *oracle* orientation, video hurts 10/12 clips — so the bottleneck is the
**CV-derived string evidence itself**, not gating or orientation selection. The
correct-orientation evidence is frequently incompatible with the audio pitch
(per-event gate drops it → coverage < 0.71 → fallback); the orientation that
*does* produce dense audio-compatible evidence assigns the wrong string. This is
consistent with the chunk-2/3 caveat that the chain was tuned to the
non-mirrored Kaggle UT-Austin rig; GAPS is in-the-wild classical-guitar footage
with diverse camera geometry, neck angle, and framing.

## 5. Error decomposition + the lever

Gold-audio `+real (auto)` (= audio-only) over 10,745 gold notes:
`correct 8494`, **`wrong_position_same_pitch 2213` (98.3% of all loss)**,
missed 34, pitch_off 2, timing 2, extra 0. With gold pitch, essentially the
*only* error is wrong-string — and the gold-audio **oracle is 0.973**, nearly
clearing 0.94. So the lever is real and large; the audio-only playability prior
already captures most of it (0.815), and the remaining ~0.16 to oracle is exactly
the string headroom the video chain was meant to take — but currently cannot.

## 6. Honesty bounds + next levers

- **vote_frames = 1.** A single frame per onset (the offset is sub-frame; the
  homography is 0.99-stable). More frames would only help if the residual were
  MediaPipe *jitter*; the no-gate sweep shows the evidence is systematically
  *wrong* (hurts even with oracle orientation), which voting cannot fix. The
  frame cache is incremental, so a `--vote-frames 3` re-run is cheap if desired.
- **The result is a clean negative, not a pipeline bug:** alignment is verified,
  `+oracle` = 0.973 confirms the gold/fusion path and the lever, and the smoke
  reproduced the chunk-5 audio-only baseline exactly (179 = 0.857).
- **Next levers (a chunk-6, not audio tuning):** the CV chain must transfer to
  in-the-wild footage — per-clip fretboard/fret-cell calibration, a string-axis
  homography that doesn't assume the Kaggle orientation, and a fingertip→string
  model robust to perspective. The string axis is worth ~0.16 (gold) here, so
  this is the right place to invest.

## 7. Reproduce

```bash
cd tabvision
export TABVISION_DATA_ROOT=~/.tabvision/data
export PATH=~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin:$PATH
# acquire clean-12 source video + per-clip offsets (NC, local-only)
python -m scripts.acquire.gaps_video --download --offsets --clips clean12 \
  --ffmpeg-location ~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin
# run the video real-chain (writes the *_auto_* raw report)
python -m scripts.eval.v1_1_gaps_video_chain_probe \
  --checkpoint ~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt
```

Caches (resumable): offsets / highres events / raw per-frame fingerings under
`~/.tabvision/cache/gaps_video_chain/`; source mp4s under
`~/.tabvision/cache/gaps_video/`.
