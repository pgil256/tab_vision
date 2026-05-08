# Video-Driven Hand Anchor — Design

**Date:** 2026-04-23
**Branch:** `agent-farm-improvements`
**Status:** Design approved, implementation starting

## Motivation

Training benchmark (20 videos):

| Config | exact F1 | pitch F1 | position acc |
|---|---|---|---|
| Baseline (audio only) | 0.385 | 0.676 | 0.607 |
| Current audio+video (v13) | 0.446 | 0.750 | 0.557 |

The 30-point gap between pitch F1 (0.75) and exact F1 (0.446) is almost entirely position error. Position accuracy actually *dropped* when video was added. Video is helping pitch filtering but hurting position selection — it's contributing noise to the position layer.

Root cause (confirmed by reading `fusion_engine.assemble_tab_document` around line 1635): video is used in an **override-or-ignore** mode. `match_video_to_candidates_enhanced` requires an exact (string, fret) match to an audio candidate; below the 0.3 threshold the video is discarded and the anchor-based `_select_best_position` runs as if video didn't exist. The `hand_position_fret` anchor itself is derived from *past audio picks*, which is circular — when early picks go wrong, the anchor locks into the wrong zone and drags subsequent picks with it.

The dominant scoring term in position selection is `hand_position_fret` (weight `-0.35` per fret of distance, vs `-0.05` lower-fret bias and `-0.15` previous-position). Fixing this anchor is the biggest single lever.

## Goal

Compute the hand anchor directly from video (palm centroid projected onto the detected fretboard) instead of deriving it from retrospective audio picks. Keep the existing anchor code as a fallback when video is unavailable.

Success criteria on training benchmark:
- Position accuracy ≥ 0.70 (from 0.557)
- Exact F1 ≥ 0.55 (from 0.446)
- No regression on training videos that already work (e.g., training-19 at 0.81)

## Architecture

```
video_observations + fretboard_geometry
        │
        ▼
build_hand_position_timeline()
        │
        ▼
list[HandAnchorPoint]  (sorted by timestamp, gap-tolerant)
        │
        ▼
get_hand_anchor_at(timestamp)  →  (anchor_fret, confidence) | (None, 0.0)
        │
        ▼
effective_hand_pos = video_anchor  (primary, when v_conf ≥ 0.6)
                   | v·video + (1-v)·audio_anchor  (blend, 0.4 ≤ v < 0.6)
                   | existing audio-derived anchor  (fallback)
        │
        ▼
_select_best_position(... hand_position_fret=effective_hand_pos)
```

New module: `app/hand_anchor.py`
Feature flag: `FusionConfig.use_video_hand_anchor` (default `False` until proven)

## Landmark → fret projection

**Landmark choice: palm centroid = mean of MCP joints (MediaPipe landmarks 5, 9, 13, 17).**

- Wrist (landmark 0) sits behind the neck and shifts with wrist rotation.
- Single MCP gives a hand edge, not a center, and jitters more per-frame.
- Averaging the four MCP landmarks gives a robust centroid directly over the fretboard, naturally corresponding to the center fret of the current hand pose.

**Projection:** reuses the existing neck-axis projection in `fretboard_detection.py:964-993`. Computes `rel_x ∈ [0, 1]` along the neck, then interpolates through `fret_positions` / `actual_fret_numbers` to produce a **fractional fret number in absolute fret space** (e.g. `5.4` means hand centered between fret 5 and 6).

Existing `_find_nearest_fret_smart` snaps to the nearest integer — fine for a fingertip, wrong for an anchor. New helper `_interpolate_fret_from_rel_x` returns a float.

**Per-frame confidence:**
```
anchor_confidence = geometry.detection_confidence
                  * hand_observation.hand_confidence
                  * bounds_penalty        # rel_x in [-0.15, 1.15]
```
Frames with confidence < 0.4 drop from the timeline.

## Temporal smoothing

Raw per-frame anchors are noisy. The timeline is built via:

1. **Outlier rejection (median window).** For each raw sample, compare to median of ±0.3s neighbors. If current sample is > 3 frets from median **and** confidence < 0.7, drop it. Kills single-frame fretboard-detection glitches without erasing real position shifts.

2. **Confidence-weighted EMA:**
   ```
   α = clamp(dt * 2.0, 0.1, 0.6) * confidence
   smoothed[i] = (1-α) * smoothed[i-1] + α * raw[i]
   ```
   Larger `dt` → more weight on new sample (allows real shifts during pauses). Lower confidence → less weight. Clamp prevents teleportation.

3. **Gap handling.** If `dt > 0.5s`, restart from the new raw value (don't extrapolate across gaps). Queries that fall inside a gap > `max_gap` return `(None, 0)` → fusion falls back to audio anchor.

**Not a Kalman filter:** hand dynamics are non-Gaussian (real playing has sudden jumps during position shifts). Median+EMA is simpler, tunable, and preserves genuine jumps when consecutive high-confidence samples agree.

## Sampling density

**Phase A (ship first):** use existing onset-sampled `video_observations` as-is. Sparse (~5 frames per onset) but sufficient for queries at chord timestamps.

**Phase B (later, if A isn't enough):** add a uniform-interval (10 fps) sampling pass across the whole video for a dense continuous timeline.

## Fusion integration

Three touch points in `fusion_engine.assemble_tab_document` (around line 1635):

**1. Build timeline once, up front** (after the `chord_anchors` first pass):
```python
hand_timeline = build_hand_position_timeline(video_observations, fretboard) \
    if config.use_video_hand_anchor else []
```

**2. Replace `effective_hand_pos` computation** (currently lines 1736-1742):
```python
video_anchor, v_conf = get_hand_anchor_at(hand_timeline, chord_timestamp)
audio_anchor = _get_nearest_anchor(i)

if video_anchor is not None and v_conf >= 0.6:
    effective_hand_pos = video_anchor
    anchor_source = "video"
elif video_anchor is not None:       # 0.4 ≤ v_conf < 0.6
    fallback = hand_position_fret if hand_position_fret is not None else audio_anchor
    effective_hand_pos = (v_conf * video_anchor + (1 - v_conf) * fallback
                          if fallback is not None else video_anchor)
    anchor_source = "blend"
else:
    # existing behavior: running EMA + audio chord anchor
    effective_hand_pos = hand_position_fret
    if audio_anchor is not None:
        effective_hand_pos = (audio_anchor if effective_hand_pos is None
                              else effective_hand_pos * 0.3 + audio_anchor * 0.7)
    anchor_source = "audio"
```

**3. Damp running EMA update when video is authoritative** (lines 1824-1827, 1876-1879):
```python
if anchor_source == "video":
    hand_position_fret = effective_hand_pos   # track video directly
else:
    # existing EMA from picked positions
    ...
```

**Why hard cutover at 0.6 and not a continuous blend:**
When video is clearly reliable, blending with a wrong audio anchor weakens the win. The blend band 0.4–0.6 handles marginal cases; below 0.4 we discard.

**Why damp the EMA under authoritative video:**
Without damping, a single wrong audio pick drags the anchor toward the wrong zone for the next chord, undoing video's influence. Tracking video directly breaks the positive-feedback loop.

**Capo:** `hand_position_fret` and video anchor are both absolute-fret-space. No capo-relative conversion needed.

## Testing

**Layer 1 — unit tests** (no MediaPipe/ffmpeg):
- `project_palm_to_fret`: synthetic observation + geometry → expected fractional fret. Edges: palm behind nut, past body, rotated neck, partial fretboard starting at fret 5.
- `build_hand_position_timeline`: spike rejection (bad frame at fret 15 among fret-5 neighbors), genuine shift preservation (5 samples 3→7).
- `get_hand_anchor_at`: gap > max_gap → None; between points → linear interp.

**Layer 2 — anchor quality diagnostic** (new script `debug_hand_anchor.py`):
For each training video, compute per-GT-note:
- `|anchor_at(gt.timestamp) - gt.fret|` (MAE)
- fraction of notes with `|err| ≤ 2`
- gap coverage (fraction of GT timestamps with a valid anchor)

If anchor MAE > 3 frets on most videos, projection is too noisy and projection/smoothing needs debugging before touching fusion.

**Layer 3 — end-to-end benchmark:**
Run the existing training benchmark with flag OFF and ON, compare exact_f1 / pitch_f1 / position_acc per video.

**Observability — debug fields on TabNote:**
```python
_debug_anchor_fret: Optional[float]
_debug_anchor_source: Optional[str]   # "video" | "blend" | "audio"
```
Lets us dump `(gt_fret, picked_fret, anchor_fret, source)` on per-video regressions.

## Risk controls

- **Feature-flagged.** `use_video_hand_anchor=False` means identical behavior to today.
- **Graceful fallback is the default path.** Empty timeline or None-query falls into the existing audio branch. No regression on audio-only cases.
- **Watch training-04 specifically** — pitch detection returns wrong MIDI notes on that video. A correct anchor could make it *worse* (picks wrong strings at right frets with high confidence on bad pitches). Inspect in the comparison.

## Rollout

1. Build `hand_anchor.py` + Layer 1 unit tests → green.
2. Run Layer 2 diagnostic on training set → anchor MAE acceptable?
3. Integrate into fusion with flag OFF → existing tests still pass.
4. Flip flag ON for benchmark run → compare per-video.
5. If wins: flip default to ON. If not: analyze and decide whether Phase B (dense sampling) or a projection fix is needed.

## Decisions deferred

- **Phase B dense sampling.** Only if Phase A doesn't clear the 0.70 position-accuracy bar.
- **Viterbi / joint sequence optimization.** Considered as a later refinement once emission signals (audio confidence × video anchor) are clean.
- **Using video to rule out candidates entirely.** If post-A we still see position picks > 4 frets from the anchor, add a hard gate in `_select_best_position`.

## Result (2026-04-23)

**Feature is NOT shipped. Kept in tree with `use_video_hand_anchor=False` (default).**

Full training-benchmark comparison against the existing audio+video baseline (tuning_v13_video):

| Metric | v13 (baseline) | v14 (anchor on) | Δ |
|---|---|---|---|
| Exact F1 | 0.446 | 0.370 | **−0.075** |
| Pitch F1 | 0.750 | 0.742 | −0.008 |
| Position acc | 0.557 | 0.453 | **−0.104** |

20 videos: 8 regressed (position drops up to −31), 12 unchanged (sanity gate rejected timeline, fell back to baseline), **0 improved**. When the gate rejects, behavior matches v13 exactly. When the gate passes, the anchor consistently biases picks the wrong way.

**Root cause** (per the training-13 diagnostic): even on videos where fretboard detection looks healthy (starting_fret=0, 26 frets detected, confidence 0.85), the palm projects consistently to fret ~20 while ground-truth notes are at frets 0–3. **Anchor MAE of 17 frets.** The core hypothesis — "palm centroid projected onto neck axis = reliable hand position" — is empirically wrong for this dataset.

Likely causes (not yet verified):
1. `_select_fretting_hand` may pick the picking/strumming hand in some videos — that hand sits over the sound hole, which projects to "fret ~18–22" on a detected neck.
2. Palm-centroid-of-fingertips falls back to wrist often when fewer than 2 non-thumb fingers are extended. Wrist screen position does not correspond to where fingers actually fret when the hand is angled.
3. Fretboard quad corners may be plausible-looking but still wrong enough to break the projection.

## Retained

- `app/hand_anchor.py` (210 lines) + 26 unit tests — isolated, no runtime impact when flag is off.
- `FusionConfig.use_video_hand_anchor` flag — off by default.
- `debug_hand_anchor.py` diagnostic — ready for future iteration.
- `run_benchmarks.py --with-video --use-video-hand-anchor` / `--save` — ready for future runs.

Future work should start from the diagnostic: pick 2–3 videos with large MAE, dump palm pixel coordinates + fretboard corners + selected hand index, and figure out which of the three likely causes dominates before changing the algorithm.
