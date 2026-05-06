# Video-stack-into-pipeline integration — Design

**Date:** 2026-05-06
**Author:** Patrick (brainstormed with Claude)
**Status:** Proposed — pending sign-off
**Spec source:** `SPEC.md` §3.1 (pipeline diagram), §8 (public entrypoints).
**Branch:** `claude/pipeline-integration` off `refactor/v1`.

## 0. Status snapshot — what's missing

The Phase 5 fusion code accepts `lambda_vision > 0` and does the right thing
when it gets non-empty `FrameFingering` evidence. Today it never does, because:

| Gap | Location | What's there |
|---|---|---|
| Frame iterator is a no-op generator | [tabvision/demux/__init__.py:118](tabvision/tabvision/demux/__init__.py:118) | Stub: `if False: yield ...; return`. Comment says "Phase 3 will replace this." |
| No public `detect_guitar / track_fretboard / track_hand` orchestrators | per SPEC §8 | Per-frame `*Backend.detect()` methods exist; no function that walks frames and produces a `GuitarTrack` / `list[Homography]` / `list[FrameFingering]`. |
| CLI stubs `fingerings: list = []` | [tabvision/cli.py:159](tabvision/tabvision/cli.py:159) | Wired only when this design ships. |
| Phase 5 eval `_run_pipeline` raises | [tabvision/tests/eval/test_phase5_eval.py](tabvision/tests/eval/test_phase5_eval.py) | Will call the new `run_pipeline` once it lands. |

The components themselves exist:
[YoloGuitarBackend](tabvision/tabvision/video/guitar/yolo_backend.py),
[KeypointFretboardBackend](tabvision/tabvision/video/fretboard/keypoint.py),
[MediaPipeHandBackend](tabvision/tabvision/video/hand/mediapipe_backend.py),
plus the smoothers in `video/{guitar,fretboard}/tracker.py`. This design is
about assembly, not capability.

## 1. Goal & acceptance bars

- A single callable `tabvision.pipeline.run_pipeline(video_path, *, lambda_vision, ...)` that takes a video file and returns `list[TabEvent]`, exercising the full demux → audio → guitar → fretboard → hand → fuse chain.
- `_cmd_transcribe` in `cli.py` calls it instead of stubbing fingerings.
- The Phase 5 eval test (`test_phase5_audio_plus_vision_beats_audio_only`) calls it and the +8 pp Tab F1 delta gate becomes runnable (subject to the eval set being labelled).
- One integration test lands a real fixture video through the whole pipeline and asserts non-empty fingerings + non-trivial tab output. Skips cleanly when YOLO checkpoint, mediapipe, or cv2 are unavailable.

## 2. Decisions

### 2.1 Frame iterator implementation: opencv

`cv2.VideoCapture` over `subprocess.run("ffmpeg ...")` pipe.

- Already in `[vision]` extras (mediapipe transitively requires it).
- Already used elsewhere — `scripts/annotate/frames.read_frame` opens videos with `cv2.VideoCapture`.
- Streaming behaviour fits the lazy-iterator contract demux already advertises.
- Returns BGR `np.uint8` arrays — what the backend `detect()` methods expect.

The iterator yields `(timestamp_s, frame_bgr)`. Timestamps are computed from
`frame_index / fps` rather than `cap.get(cv2.CAP_PROP_POS_MSEC)`, which is
inconsistent across codecs. ffprobe-reported fps is the source of truth (already done in `_probe_metadata`).

Failure modes: bad codec / corrupt file → `BackendError` from the iterator's
first `__next__`, propagated to the caller.

### 2.2 Single-pass frame loop in the orchestrator

One walk through `demuxed.frame_iterator`, running all three video backends
per frame:

```python
for t, frame in demuxed.frame_iterator:
    if not _should_sample(frame_idx, stride):
        frame_idx += 1
        continue
    bbox = guitar_backend.detect(frame)
    if bbox is None:
        per_frame_detections.append(None)
        continue
    H = fretboard_backend.detect(frame, bbox)
    fingering = hand_backend.detect(frame, H, cfg)
    fingering = replace(fingering, t=t)  # FrameFingering carries its own timestamp
    per_frame_fingerings.append(fingering)
    per_frame_detections.append(bbox)
    frame_idx += 1
```

Three separate passes would 3× I/O and decode cost — non-starter for ~5-min iPhone clips at 30 fps.

### 2.3 Subsampling: stride, default 3 (10 fps effective)

At 30 fps native, both YOLO-OBB and MediaPipe Hands are O(20–50 ms) per
frame on CPU. Running every frame on a 5-min clip is 5–15 minutes of
inference — too slow for the eval loop.

- **Pick: stride sampling.** Default `video_stride=3` → ~10 fps effective rate. Configurable via `--video-stride N` CLI flag.
- The fingerings produced thereby form a regular grid; `playability.find_fingering_at` already does nearest-neighbour lookup so onset-misalignment within ~100 ms is fine.
- *Rejected: onset-aligned sampling* (only run video at audio onsets). Tighter coupling between audio and video timing, harder to test, doesn't help when audio onsets are clustered (sustains).

### 2.4 Module layout

- New file: `tabvision/tabvision/pipeline.py` — exports `run_pipeline(video_path, *, audio_backend_name, lambda_vision, video_stride, video_enabled, cfg, session) -> list[TabEvent]`.
- Public entrypoints from SPEC §8 land in their respective package `__init__.py`:
  - `tabvision/video/guitar/__init__.py` exposes `detect_guitar(frames, backend, *, fps) -> GuitarTrack`.
  - `tabvision/video/fretboard/__init__.py` exposes `track_fretboard(frames, guitar_track, backend) -> list[Homography]`.
  - `tabvision/video/hand/__init__.py` exposes `track_hand(frames, homographies, backend, cfg) -> list[FrameFingering]`.
- These three orchestrators each iterate frames *given to them*, so `run_pipeline` is the single owner of the frame-walk and dispatches per-frame backend calls inline (rather than three iterator-based public functions, which would each need to re-decode the video).

Net effect: the SPEC §8 functions are kept as the documented surface (`run_pipeline` is the integrated convenience), but the iteration is a single loop in `pipeline.py`.

### 2.5 Graceful degradation

`run_pipeline` accepts `video_enabled: bool = True`. If `False`, OR if any of {mediapipe, cv2, YOLO checkpoint} fails to load, the pipeline emits an audio-only result (empty `fingerings`) with a `logger.warning` instead of erroring. This keeps `tabvision transcribe` usable on machines without the `[vision]` extras installed.

`--no-video` CLI flag exposes the explicit-disable path.

### 2.6 YOLO checkpoint — fail explicitly when missing

If the user has `[vision]` extras installed but no YOLO weights checkpoint
(needs an `acquire` step), the pipeline raises `BackendError` with the exact
acquire command. Don't silently fall back to audio-only here — the user
explicitly asked for video, the failure is fixable, and silent degradation
hides the regression.

## 3. Phasing within this work

Each step independently mergeable.

| Step | What | Tests | Effort |
|---|---|---|---|
| **1** | Implement `demux._frame_iterator` with opencv `VideoCapture`. | Unit: open `data/fixtures/test_a440.mp4`, count frames, check timestamps monotonic. Skip when cv2 not installed. | ½ day |
| **2** | Add `detect_guitar / track_fretboard / track_hand` public functions in their `__init__.py`s. Each takes an iterable of `(t, frame)`-or-similar. | Unit: pass a fake backend that records its inputs; verify the orchestrator iterates and assembles correctly. No real backend needed. | ½ day |
| **3** | Build `tabvision/pipeline.py` with `run_pipeline()`. Unit-test with mock backends (no real model weights). | Unit: 5+ tests covering audio-only fallback, video-disabled flag, stride subsampling, mock-backend orchestration, error path on missing checkpoint. | ½ day |
| **4** | Update `_cmd_transcribe` to call `run_pipeline`. Add `--no-video` and `--video-stride N` CLI flags. | Unit: parser tests for the new flags. | ¼ day |
| **5** | Update `tabvision/tests/eval/test_phase5_eval.py::_run_pipeline` to call `pipeline.run_pipeline`. Drop the `NotImplementedError`. | Eval test now skips only on missing deps (not on stub). | ¼ day |
| **6** | Integration test in `tabvision/tests/integration/test_pipeline_e2e.py`: run on `data/fixtures/test_a440.mp4` (or a 1-sec sample). Skip when deps missing. Asserts non-zero fingerings produced and `fuse(...)` returns a non-empty list. | One real test. | ½ day |

Total ~2 days. Each step ships its own commit; the whole thing is one PR.

## 4. Risks & open questions

- **YOLO checkpoint location**: today the YoloGuitarBackend looks for weights under a configurable path (env var). Need to document the acquire path or ship a "lite" fallback. *Mitigation: detect missing checkpoint at backend construction, emit precise instructions.*
- **MediaPipe + Python 3.11 + macOS arm64**: MediaPipe wheels can be flaky on some platforms. *Mitigation: ImportError caught at `run_pipeline` entry, falls back to audio-only with warning.*
- **Frame timestamp drift**: opencv's frame index doesn't always start at 0 for some codecs (variable-frame-rate iPhone HEVC). *Mitigation: use ffprobe duration / frame count, fall back to `cap.get(CAP_PROP_FRAME_COUNT) / fps`. Add a sanity check that `timestamp ≤ duration_s`.*
- **YOLO + mediapipe on the same frame**: both modify or transit the frame. *Mitigation: pass `frame.copy()` between backends; the per-frame-copy cost is negligible vs. the inference cost.*
- **Memory pressure for long clips**: storing per-frame fingerings for a 5-min clip @ 10 fps = 3000 fingerings × `(4, 6, 25)` floats = ~7 MB. Fine.

## 5. Out of scope

- Re-running the legacy fusion engine on the same input — that's a separate experiment.
- Fine-tuning any of the video backends — see Phase 7 plan.
- Onset-aligned video sampling — punt unless stride sampling proves insufficient empirically.
- Per-clip preflight gate inside `run_pipeline` — `cli.py` already runs preflight; the pipeline assumes it's been gated upstream when needed.

---

**For sign-off:** confirm (a) opencv for the frame iterator (§2.1), (b) `pipeline.py` as the orchestration home (§2.4), (c) stride-sampling at default 3 (§2.3), (d) graceful audio-only fallback when video deps missing (§2.5). If those look right I'll start on Step 1 (demux frame iterator) — the smallest, highest-confidence first chunk.
