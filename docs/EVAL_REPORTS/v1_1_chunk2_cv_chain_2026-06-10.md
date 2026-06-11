# v1.1 chunk-2 — MediaPipe CV chain on the Kaggle rig — 2026-06-10

**Context.** Chunk-1 locked the eval-DATA pipeline and re-confirmed the resolver
with an *oracle* `FrameFingering` (real-video oracle 0.42 → 1.00,
`v1_1_kaggle_oracle_probe.py`). Chunk-2 is the design's named "open unknown"
(`docs/plans/2026-06-03-v1.1-video-string-resolution-design.md` §7): does the real
CV chain — PNG frame → MediaPipe `HandSample` → per-frame fretboard homography →
`fingertip_to_fret` → `FrameFingering` — actually run on the **Kaggle UT-Austin**
footage, a different rig than the iPhone angle the detector was built for?

**Setup.** Installed the `[vision]` runtime (`opencv-python` 4.13, `mediapipe`
0.10.35 on Python 3.12 / numpy 2.4; ultralytics deliberately NOT installed — see
below) and the `hand_landmarker.task` model. Frames are 1920×1080, seated
front-angle acoustic, guitar body at image-left / neck extending up-right.

Probe: `scripts/eval/v1_1_mediapipe_sanity.py`.

## Finding 1 — MediaPipe hand detection works on this rig (✅)

Both hands detected on every sampled frame (7/7 clip-0..24 first-frames; 6/6 across
clip-0 stride-20), handedness confidence ≈ 1.0. Detection is **not** the problem.

## Finding 2 — hand *selection* inverts on this (non-mirrored) rig (❌, fixable)

`mediapipe_backend._select_fretting_hand` picks the hand MediaPipe labels **"Right"**,
on v0's premise that the iPhone *front-camera* image is mirrored (so the player's
fretting/left hand reads as "Right"). This footage is **not** mirrored, so the labels
flip: the fretting hand on the neck is consistently labeled **"Left"** and the
*strumming* hand over the soundhole is **"Right"**. Result: v0's rule selects the
**wrong (strumming) hand 100% of the time** here.

| clip | strumming hand (low wrist-x) | fretting hand (high wrist-x, on neck) |
|---|---|---|
| 0 | Right @ x=0.19 | **Left** @ x=0.68 |
| 5 | Right @ x=0.16 | **Left** @ x=0.52 |
| 20 | Right @ x=0.23 | **Left** @ x=0.71 |
| 24 | Right @ x=0.22 | **Left** @ x=0.75 |

(7/7 sampled clips: fretting = the higher-wrist-x hand = MediaPipe-"Left".)

The label is mirror-dependent and therefore unreliable as a selector. The robust
fix is **geometric**: of the detected hands, the fretting hand is the one whose
wrist projects onto the fretboard (canonical `[0,1]×[0,1]` via the homography), or
— absent a homography — nearest the neck region / furthest from the guitar body.
This couples hand-selection to the homography (Finding 3), so both resolve together.

## Finding 3 — neither fretboard-homography backend works on this rig (❌, the blocker)

The homography is the pixel→(string,fret) grid; without it `fingertip_to_fret`
cannot place fingers. Both §8 `FretboardBackend` impls fail here:

- **Keypoint (production path, `KeypointFretboardBackend` → `YoloOBBBackend`).**
  Needs a fine-tuned YOLO11-OBB neck/nut/fret checkpoint
  (`~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt`) — **absent locally** — and
  `ultralytics` (**AGPL-3.0**, taints the distribution; LICENSES.md). Even installed,
  the checkpoint was trained on Roboflow `b101/guitar-3`, a different framing; no
  evidence it generalizes to this rig. Not pursued (operating rules 6 + 8: free tools
  first; stop-and-ask before a new dep / training run).
- **Geometric fallback (`GeometricFretboardBackend` → v0 Canny+Hough+RANSAC).**
  Runs model-free (cv2 only) but **mis-fires**: it returns a near-full-frame
  axis-aligned quad (e.g. TL=(72,9)→BR=(1840,1019) — **92%×94% of the frame**) with a
  *false-high* confidence 0.85, locking onto room/frame edges rather than the diagonal
  neck. Unusable, and dangerously confident. (Imported directly from
  `tabvision-server/app/fretboard_detection.py`, bypassing the Flask `app/__init__`.)

## Conclusion

The CV chain's **only** open blocker on this rig is the **per-frame fretboard
homography**. Everything downstream is already validated: MediaPipe detection works
(Finding 1), `fingertip_to_fret` + `marginal_string_fret` are unit-tested, and the
resolver turns a correct `(string,fret)` signal into near-perfect tab (chunk-1
oracle 0.42→1.00; `v1_1_oracle_string_probe.py` 0.57→0.995). Hand-selection
(Finding 2) is a small fix that rides on whatever homography solution we choose.

This is a design fork with cost/license implications → **stop and ask** (operating
rule 8). Candidate paths, cheapest first:

1. **Synthetic-from-GuitarSet (design §6.1).** Render a clean neck + fretting hand
   from GuitarSet's own string/fret labels with a *known* homography → validates the
   chain's ceiling end-to-end with zero new deps. Doesn't exercise real-MediaPipe
   noise (but Finding 1 shows detection is solid).
2. **Manual neck-quad annotation of the Kaggle eval clips (one-time).** Frame-perfect
   homography for the offline eval corpus without any model — unblocks a *real-video*
   number. Tedious; per-clip (the camera is static within a clip, so ~4 corners/clip).
3. **New lightweight homography estimator for this rig** — net-new CV code (the design
   scoped chunk-2 as "wire, don't build"); e.g. MediaPipe-landmark-anchored neck axis,
   or a geometric estimator retuned for this framing.
4. **Train/obtain a YOLO-OBB neck model that generalizes here** — adds AGPL
   `ultralytics` + a training run + a fretboard-annotated dataset. Highest cost;
   license-tainting; explicitly a stop-and-ask per rule 8.

**Recommendation:** (2) for a genuine real-video Tab F1 on the chosen eval set
(static camera makes per-clip annotation cheap and it's the acceptance gate), with
(1) as the zero-dep ceiling check if a clean *headline* number is wanted first.
Avoid (4) unless (1)–(3) prove insufficient.

---

## Update (2026-06-11): Option 4 executed — the chain runs and the lever holds on REAL video

User chose **Option 4 (train the YOLO-OBB detector)**, local CPU.

**Dataset.** The wired Roboflow `b101/guitar-3-4efcd` is **dead via API** — export zips
404 (GCS `NoSuchKey`) on every version+format, even after forced regeneration; other
public datasets download fine with the same private key, so it is project-specific data
loss on Roboflow's side, not the key. (The publishable key 401s — only the *private* key
downloads datasets.) **Salvaged via the browser "Download Dataset" button** (YOLOv8-OBB),
extracted to `~/.tabvision/data/datasets/roboflow-b101-guitar-3-4efcd-v2/`: real OBB labels,
classes `0:fret 1:neck 2:nut` (matches the hardcoded schema — no code change), 710/144/72.

**Training.** `yolo11n-obb`, CPU. Batch 16 hit a transient memory spike (died epoch 1);
batch 8 ran clean but the machine **slept overnight** and froze the process at epoch 5.
**Epoch-4 `best.pt` is strong enough** — val **mAP50 0.868 / mAP50-95 0.549** — and was
promoted to `~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt`. (4 epochs = ~32 min;
no need for all 50.)

**Generalization (the original worry) — PASS.** On the Kaggle rig the epoch-4 detector
gives **neck 21/21 (100%)** and **localized homography 21/21 (100%)** (quad height
0.07–0.21 of frame — a real neck strip), vs the geometric detector's 92%×94% garbage.
Probe: `scripts/eval/v1_1_yolo_rig_probe.py`.

**Homography orientation is inverted on this rig (Finding 3b).** Feeding the real
`FrameFingering` *un*corrected made tab *worse* (clip 0: 0.96→0.17): gold frets 1–4 (near
nut) predicted as 19–24. `keypoint.py`'s no-nut fallback assumes nut = low-X and a specific
lap framing, but here the nut is high-X and strings read reversed. Flipping **both** canonical
axes recovers it (clip 0 → 1.000 = oracle). This is the known "preflight orientation check"
the keypoint docstring + §9 anticipate — a chunk-3 robustness item, not a chain defect.

**Real-video result (orientation-corrected, geometric hand-selection, 24 clips / 527 notes)** —
`scripts/eval/v1_1_real_chain_probe.py --flip-fret --flip-string`:

| Condition | Tab F1 |
|---|---|
| audio-only | 0.424 |
| **+ real video** | **0.540** (+0.116; up to **+0.58**/clip) |
| + oracle (ceiling) | 1.000 |

**Conclusion.** The full CV chain runs end-to-end on real footage and the **string lever is
real on real video** (+0.12 aggregate, +0.58 best-clip), not just under oracle. The gap to the
oracle ceiling (1.0) is real-CV noise — homography precision, MediaPipe jitter, *per-clip*
orientation, fret-cell calibration, and the lack of confidence-gating (a few clips where audio
was already perfect regress, e.g. clip 11 1.0→0.53). **Chunk-2 is done; chunk-3 is exactly:**
robust per-frame orientation (so the manual `--flip` becomes automatic), confidence-gating so
video never drags a strong audio result down (the §5.3 no-regression guarantee), multi-frame
voting, fret-span calibration, then real highres audio for the headline acceptance number.
