# FretCam side quest — live webcam fretboard + hand-position HUD (design + resource survey)

**Date:** 2026-07-22
**Status:** DESIGN — side quest, separate application. No SPEC scope change; no
TabVision pipeline code is touched until the M4 bridge gate passes.
**Relation to the closed video lever:** video fusion was closed 2026-07-02/06
(DECISIONS 2026-06-29, A14; ROI deep-dive §5.1) *for in-the-wild footage*. The
recorded reopen condition is a **changed capture contract — a user-owned, fixed,
live camera**. This side quest *is* that experiment, run as a standalone tool so
the main pipeline's measured state stays untouched.

---

## 1. Goal and hypothesis

A separate app: open a browser page, grant webcam access, play. The app

1. detects the guitar and fretting hand, and draws the **fretboard outline**
   (rectified neck polygon + fret ticks) live on the video;
2. shows a live **playing-position readout** — "Position V (index ≈ fret 5,
   window 5–8)" — with a confidence indicator.

**Hypothesis (Pat, 2026-07-22):** if the live position readout is accurate, it
bounds the candidate set for every pitch — we've eliminated the frets it
*can't* be — which is exactly the wrong-string error (57.3% of all Tab F1
loss; 98.3% of loss at gold pitch on GAPS; oracle strings 0.815 → 0.973).

**Why coarse position is enough:** same-pitch candidates on adjacent strings
sit 5 frets apart (4 across G→B). Middle C at Position V → 5th fret G string,
not 1st fret B. A ±2-fret position estimate discriminates almost every
ambiguous pair. This is also why the bar for the CV is low: we need the *fret
axis only*, not per-string fingertip assignment.

**One hard correction to "eliminated the frets it CAN'T be":** open strings
are playable from *any* hand position. The filter is "fret ∈ window ∪ {0}",
never a hard kill of fret 0. (Hand parked at V while an open G rings is the
adversarial case — the HUD should display "open strings always possible.")

### Why this can work where the GAPS chain failed

| GAPS chain (closed) | FretCam (this design) |
|---|---|
| In-the-wild YouTube footage, arbitrary rigs | One webcam, one guitar, user-controlled framing |
| No feedback loop — footage is what it is | Live overlay: user adjusts camera until it locks |
| Per-note **string** resolution (string axis) | **Fret-axis only** — position, never string |
| String-axis mirror/orientation ambiguity (per-clip best orientation varied — A14) | Fret-axis direction self-resolves: fret spacing shrinks geometrically toward the bridge |
| Offline eval, no user in the loop | User sees wrong readouts instantly |

The failure evidence (WS1 0.574 vs audio 0.778; learned −0.117; A14
anti-enrichment 0.285) was all about string identity from uncontrolled footage.
None of it measured a fret-window signal from a controlled live camera.

---

## 2. Output spec

- **Position label:** Roman numeral I–XII+, defined classically: Position N =
  index finger at fret N; span [N, N+3] (+1 stretch tolerance).
- **Fret window:** [N−1, N+4] soft window for downstream use (±1 slack for
  estimator noise + stretch), plus fret 0 always.
- **Confidence:** product of fretboard-lock confidence (homography residual),
  hand-presence score, and temporal agreement (last ~10 frames).
- **Smoothing:** EMA on the index-fret coordinate + hysteresis — the label
  switches only after k≈5 consecutive frames agree; during fast shifts show
  "shifting…" instead of flapping.
- **Status line:** framing guidance ("neck partly out of frame", "raise
  camera", "glare on frets") from preflight-style checks.
- **Latency target:** readout ≤ 150 ms behind reality; ≥ 10 FPS end-to-end.

---

## 3. Architecture

### P0 — local server prototype (recommended first; days, zero training)

Browser does `getUserMedia` capture + canvas overlay only; frames go over a
localhost WebSocket to a small FastAPI server that reuses the **existing v1
vision chain on `main`**:

| Existing asset | Role in FretCam |
|---|---|
| `~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt` (yolo11n-obb, 50 ep; neck 21/21, homography 21/21 on the rig probe) | Neck detection |
| `tabvision/video/fretboard/{geometric,calibrate,tracker}.py` | Homography + WS1 nonlinear fret map + temporal smoothing |
| `tabvision/video/hand/{mediapipe_backend,neck_anchor}.py` | Fretting-hand landmarks + **coarse neck anchor — this module already computes center-fret/span** (DECISIONS 2026-05-07 design) |

`neck_anchor.py` is the whole point: the coarse-anchor signal was designed and
built but never had a live, controlled camera to run against. FretCam is its
test rig. Server emits `{neck_quad, fret_ticks, hand_landmarks, index_fret,
position, confidence}` per frame; browser renders.

Practical notes: browser-side capture means the server never opens the camera
— so it runs identically from Windows-native Python or WSL (no WSL webcam
passthrough pain). JPEG frames at 480–640 px over loopback add ~10–30 ms;
the budget is dominated by detector+MediaPipe inference (~50–100 ms CPU).
Run the YOLO detector at 1–2 Hz + track in between (the homography is
quasi-static — measured 0.99-stable per chunk-5); run hands every frame.

### P1 — pure-browser build (optional port, after P0 proves the signal)

- **Hands:** `@mediapipe/tasks-vision` **HandLandmarker** (Apache-2.0,
  v0.10.35, `runningMode:"VIDEO"`, `numHands:1`, GPU delegate, run in a Web
  Worker — `detectForVideo()` blocks the main thread). Realistic 30+ FPS on a
  laptop.
- **Fretboard:** two routes —
  (a) export our YOLO to ONNX (`opset=12`) → **ONNX Runtime Web, WebGPU EP**
  with WASM fallback. Measured reality: WASM alone is ~4–5 FPS at 640 px
  (YOLOv8n, M3 Pro — PyImageSearch 2025-07); WebGPU is the viable path
  (est. 15–30 FPS, unverified). MIT scaffold with custom-model upload:
  `nomi30701/yolo-object-detection-onnxruntime-web`.
  (b) **MediaPipe Model Maker** (MobileNetV2 family — *not* EfficientDet, per
  2026-01 docs) → `.tflite` → tasks-vision `ObjectDetector` next to the hand
  task. Least code, fully Apache-2.0 (no AGPL), but 320 px input may cost
  fret-tick precision.
- **Mirroring gotchas:** browsers usually mirror the selfie preview; MediaPipe
  handedness labels assume a mirrored feed — on an unmirrored stream
  Left/Right swap. Don't trust handedness; pick the fretting hand by
  neck-relative x-position. Fret-axis direction (nut vs bridge) comes from the
  fret-spacing gradient (spacing ∝ 2^(−n/12)) and/or headstock side.

### Position estimation (both variants)

1. Project index-fingertip (landmark 8; fallback: hand centroid when
   fingertips are occluded) through the homography into fret coordinates via
   the calibrated fret map.
2. Position = floor of the index-fret estimate; window per §2.
3. **Fallback when MediaPipe drops out** (palm hidden behind the neck —
   mediapipe issue #5667; motion blur during shifts — arXiv:2303.04566):
   TapToTab's mechanism, hand-*bbox* × fret-zone IoU, needs no landmarks at
   all. Our YOLO can grow a `hand` class for this (the ghaleb dataset below
   labels exactly `Hand` + 12 fret zones).

---

## 4. Dataset survey (deep dive 2026-07-22; all links fetched unless flagged)

| # | Dataset | Size / annotations | License | Role for FretCam |
|---|---|---|---|---|
| 1 | [ghaleb/guitar-fretboard](https://universe.roboflow.com/ghaleb/guitar-fretboard) (Roboflow) | 384 imgs; bbox `Hand` + `Zone1`–`zone12` | CC BY 4.0 | **Primary.** Real YouTube playing frames; almost certainly the TapToTab training set (author match). Hand+zone labels = the IoU fallback signal for free |
| 2 | [bandsucore/guitar-neck-detection](https://universe.roboflow.com/bandsucore/guitar-neck-detection-suhgk) | 1,001 imgs; 1-class `neck` bbox | CC BY 4.0 | Neck-detector fine-tune bulk (low diversity — consecutive frames) |
| 3 | [Kaggle UT-Austin guitar-transcription-dataset](https://www.kaggle.com/datasets/jacksonlightfoot/guitar-transcription-dataset) | 355 frames, polygon fretboard masks (COCO/VGG) + 1,995 tab-labeled | CC-BY-NC-SA per our 2026-06-03 doc (Kaggle page JS-hidden — reconfirm on page) | Already our YOLO's fine-tune source; webcam-like viewpoints |
| 4 | [my-workspace-xslxf/guitar-neck-chords](https://universe.roboflow.com/my-workspace-xslxf/guitar-neck-chords-yrnmt) | 6,281 imgs; polygon seg, 28 noisy classes | CC BY 4.0 | Bulk seg data *after* remap-to-`neck` + heavy pruning |
| 5 | [joaomarcoscrs/guitar-chords](https://universe.roboflow.com/joaomarcoscrs/guitar-chords-daewp) | ~343 imgs; **keypoints** (guitar-neck/hand/strings) | CC BY 4.0 | Only public keypoint-format source; nut/fret keypoint head if geometric fret ticks prove noisy |
| 6 | [soen357/fretboard](https://universe.roboflow.com/soen357/fretboard) | small; per-fret bboxes E1–E10 | CC BY 4.0 | Auxiliary fret localization |
| 7 | [GAPS video](https://zenodo.org/records/13962272) (YouTube IDs; mp4 cache already local) | ~14 h classical; no visual labels | CC BY-NC-SA | Offline sanity/pseudo-label harvest only — *not* the FretCam contract |
| 8 | [Guitar-TECHS](https://zenodo.org/records/14963133) | 4.1 GB audio + per-string MIDI; **video NOT released** | CC BY 4.0 | ⚠️ Correction to prior assumption: unusable for vision (audio-only release) |
| 9 | test-l4egp guitars-strings-frets (Roboflow) | 626 imgs? fret/nut/string | unknown | UNVERIFIED — page failed to render 3× |

Negative results: **zero** pretrained fretboard/fret-keypoint checkpoints with
downloadable weights on Hugging Face or GitHub (HF "fretboard" = 0 models);
Zenodo has no guitar image-annotation sets. Everything real routes through
Roboflow + Kaggle above, or our own checkpoint.

**Training plan (only if the existing OBB detector disappoints on webcam
frames):** merge #1+#2+#3 (+cleaned #4) → resume-fine-tune the yolo11n-obb or
train fresh YOLOv8n; free Colab scale; then ONNX-export for P1a. All-permissive
alternative for P1b: same data through Model Maker. New LICENSES.md rows:
each Roboflow set (CC BY 4.0), Kaggle set already/ tracked, **Ultralytics
AGPL-3.0 — fine-tuned weights are AGPL by default per ultralytics.com/license;
fine for a personal non-distributed tool, but the label matters if this repo
is ever published.**

---

## 5. Prior art (what's proven, what's unclaimed)

- **TapToTab** ([arXiv:2409.08618](https://arxiv.org/abs/2409.08618), NILES
  2024): YOLOv8-OBB fret zones + hand-as-a-class; played fret = max IoU(hand
  box, zone box); optional 5-fret zones ≈ our position output. Detection mAP50
  0.993 — but **no end-to-end tab/position accuracy reported**, no code; data
  ≈ dataset #1. Validates the coarse mechanism and the OBB choice; explicitly
  rejected Canny/Hough as lighting-fragile (matches our v0 experience).
- **Duke & Salgian, ISVC 2019** ([Springer](https://link.springer.com/chapter/10.1007/978-3-030-33723-0_20)):
  markerless webcam strings/frets + skin-color fingers, real-time tab; no
  public numbers or code. (The ROI doc's "ISVC 2023" cite appears to be this
  2019 paper; no 2023 ISVC guitar paper exists.)
- **Kerdvibulvech & Saito 2008**: ARTag on the neck + colored fingertip
  markers + particle filter — worked, but marker UX is unacceptable; the
  ARTag insight survives as "give the tracker a strong neck coordinate frame,"
  which our homography provides markerlessly.
- **AlbertMitjans/chord-detection** ([GitHub](https://github.com/AlbertMitjans/chord-detection),
  no license): stacked-hourglass heatmaps for frets/strings/fingertips,
  weights downloadable; fret recall 96.7 — a reference architecture if we
  ever want learned fret ticks (license-blocked from shipping; read-only
  reference).
- **Unclaimed territory:** no prior system outputs live position labels, none
  runs vision in-browser, none reports end-to-end position accuracy. We define
  the metric ourselves (§6).

Hand tracking: MediaPipe HandLandmarker is the only maintained browser option
(TF.js hand-pose-detection wraps the deprecated legacy solution, last release
2023-07). Known guitar-relevant failure modes, all with sources: palm-hidden
detection loss (#5667), 4-joint occlusion degradation + ≥50% miss under
diagonal motion blur (arXiv:2303.04566), handedness flips on unmirrored feeds.
Mitigations in §3.

---

## 6. Acceptance protocol (side quest)

Live self-test — this is a personal tool; private recordings stay banned from
TabVision training/eval roles, and nothing here creates one (live inference,
nothing persisted).

- **A1 — lock:** fretboard overlay visibly locked (ticks on frets) within 3 s
  of framing, holds through normal playing motion, two lighting conditions.
- **A2 — position accuracy:** 5-s holds at positions I, III, V, VII, IX ×
  single notes / chords / barre: readout correct (±0 position after
  hysteresis) in ≥ 90% of holds; never off by > 1 position.
- **A3 — shifts:** I→V→IX runs: "shifting…" during motion, correct label
  ≤ 500 ms after arrival; recovery from full hand occlusion ≤ 1 s.
- **A4 — throughput:** ≥ 10 FPS end-to-end, readout latency ≤ 150 ms.

Fail A2/A3 → try the IoU fallback before more training; fail after both →
record the negative honestly (house rule 7) and the side quest ends there.

---

## 7. M4 — bridge back to TabVision (only after §6 passes)

The integration path already exists and is contract-safe: timed anchors →
`AudioEvent.fret_prior` (the DECISIONS 2026-05-07 design; Phase 5 fusion
accepts it as emission evidence — no §8 change). Design constraints learned
from the closed lever:

1. **Soft prior, never a hard filter.** Bounded bonus for frets in
   [N−1, N+4] ∪ {0}; confidence-weighted; zero-weight below a lock threshold.
   The GAPS no-regression gate pattern (absent/low-conf → audio-only) carries
   over verbatim.
2. **Timestamping:** hands move *before* onsets — sample the anchor in a
   window ending at onset −30 ms, not at the onset frame.
3. **Evidence before pipeline code:** two cheap probes, in order —
   (a) cache-only GAPS probe: hand-centroid → fret-window from the cached
   per-frame fingerings/homographies; score **P(gold fret ∈ window | audio
   wrong)** on the banked ambiguous lattice. A positive is strong evidence;
   a negative does *not* kill FretCam (wrong capture contract) but calibrates
   expectations. ~A day, $0.
   (b) live A/B on personal sessions via the **assisted metric**: recording
   session + FretCam running → anchors re-rank the review queue / C-key
   candidate ordering; measure wrong-position reduction @60 s vs the shipped
   38.76%. Reported separately from automatic Tab F1, per the 2026-07-20
   posture — no eval-role violation.
4. Automatic Tab F1 claims, if ever, go through the standard entry-gate →
   OOF → player-05 lo-95 discipline on public data only.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Open strings defeat position filtering | Fret 0 always in the candidate window; HUD says so (§1) |
| MediaPipe drops the fretting hand (palm behind neck, blur) | Hand-bbox × zone IoU fallback needs no landmarks (§3); hysteresis carries short gaps |
| Neck leaves the webcam FOV at high positions | Framing-guidance status line; wide FOV / camera-left placement |
| Readout accurate but fusion still doesn't lift Tab F1 | The A14 lesson — anti-enrichment — could recur even with good anchors; that's what probe 7.3a/b measure before any pipeline work |
| AGPL (ultralytics) if repo goes public | LICENSES.md row now; Model Maker (Apache) path exists for P1 |
| Scope creep into the frozen pipeline | FretCam lives in `fretcam/` at repo top level; imports `tabvision` as a library; zero edits inside the package until M4 gates pass |

## 9. Milestones

| M | Deliverable | Est. |
|---|---|---|
| M0 | `fretcam/` scaffold: FastAPI + WS + static page (getUserMedia, canvas) | ½ day |
| M1 | P0 chain wired (existing OBB ckpt + fret map + MediaPipe + `neck_anchor`), overlay + position HUD | 2–4 days |
| M2 | §6 acceptance run; tune smoothing/hysteresis; IoU fallback if needed | 1–2 days |
| M3 (opt) | Pure-browser port (HandLandmarker + WebGPU ONNX or Model Maker) | ~1 wk |
| M4 | Bridge probes 7.3a/b; DECISIONS entry either way | 2–3 days |

## 10. Sources

In-repo: `docs/2026-07-21-tab-accuracy-roi-deep-dive.md` (§2, §5.1),
`docs/EVAL_REPORTS/a14_video_complementarity_2026-07-06.md`,
`docs/EVAL_REPORTS/v1_1_gaps_video_chain_2026-06-22.md`,
`docs/plans/2026-06-03-v1.1-video-string-resolution-design.md`,
`docs/DECISIONS.md` (2026-05-07 hand-neck anchors; 2026-06-29 ×2),
`tabvision/tabvision/video/` modules.
External (fetched 2026-07-22): Roboflow/Kaggle/Zenodo dataset pages tabled in
§4; [arXiv:2409.08618](https://arxiv.org/abs/2409.08618);
[arXiv:2303.04566](https://arxiv.org/abs/2303.04566);
[MediaPipe HandLandmarker web guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js);
[MediaPipe ObjectDetector web](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/web_js);
[Model Maker customization](https://ai.google.dev/edge/mediapipe/solutions/customization/object_detector);
[mediapipe#5667](https://github.com/google-ai-edge/mediapipe/issues/5667);
[ONNX Runtime Web WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html);
[nomi30701 YOLO ORT-web demo](https://github.com/nomi30701/yolo-object-detection-onnxruntime-web);
[Hyuto/yolov8-onnxruntime-web](https://github.com/Hyuto/yolov8-onnxruntime-web);
[PyImageSearch ORT-WASM FPS](https://pyimagesearch.com/2025/07/28/run-yolo-model-in-the-browser-with-onnx-webassembly-and-next-js/);
[Ultralytics license](https://www.ultralytics.com/license);
[AlbertMitjans/chord-detection](https://github.com/AlbertMitjans/chord-detection);
[Duke & Salgian ISVC 2019](https://link.springer.com/chapter/10.1007/978-3-030-33723-0_20).
Flagged unverified: WebGPU YOLO FPS estimate; test-l4egp Roboflow set;
Kaggle on-page license.
