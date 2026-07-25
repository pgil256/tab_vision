# FretCam → audio fusion bridge: fixed-policy integration check

**Status:** IMPLEMENTED, EXPLICIT OPT-IN; **NOT PROMOTED TO DEFAULT**.

This check answers a narrow question: can the stabilized FretCam playing
position be joined to the audio timeline and conservatively re-rank physically valid
same-pitch string/fret candidates? Yes. The new route is functional and the
cache-only proxy is directionally positive. A later source-disjoint paired
real-prediction evaluation is also positive but small; the controlled-live L2
gate and the statistical/default-promotion gate still do not pass.

## Follow-up: current-solver paired end-to-end evaluation

The live current FretCam solver was subsequently run through
`run_pipeline_with_artifacts(..., video_backend="fretcam")` against shared real
highres audio predictions. The primary population was ten GAPS test clips that
were source-disjoint from FretCam development; clean-12 is reported separately
as a development confirmation.

| Population | Audio baseline | + FretCam | Macro delta (paired 95% CI) | Wrong-position/same-pitch |
|---|---:|---:|---:|---:|
| Source-disjoint 10 (primary) | 0.623750 | 0.624586 | +0.000836 [0.000000, 0.001994] | 1,021 -> 1,014 (-7) |
| Clean 12 (development) | 0.772970 | 0.772815 | -0.000155 [-0.000623, 0.000159] | 1,930 -> 1,931 (+1) |
| Combined test 22 | 0.705143 | 0.705438 | +0.000296 [-0.000219, 0.000935] | 2,951 -> 2,945 (-6) |

The primary held-out result has two improved clips, eight unchanged, and no
regressions. The clean development confirmation has one improvement, ten
unchanged, and one regression (`118_VD1wc`). Thus the bridge does reduce the
target error on held-out data, but the effect is too small and heterogeneous
to make FretCam the default. Full paired reports:

- `docs/EVAL_REPORTS/fretcam_e2e_source_disjoint10_2026-07-24.md`
- `docs/EVAL_REPORTS/fretcam_e2e_clean12_2026-07-24.md`
- `docs/EVAL_REPORTS/fretcam_e2e_test22_2026-07-24.md`

## Frozen bridge policy

- Video route: `--video-backend fretcam`; `legacy` remains the default rollback.
- Clock: per-frame ffprobe best-effort PTS normalized to decoded-audio stream
  start, with an offset-aware monotonic CFR fallback for missing PTS.
- Runtime: synchronous `DetectionChain` plus `PositionEstimator`; no browser or
  wall-clock timestamps.
- Accepted states: `locked` and `holding`, confidence at least `0.20`.
- Onset join: latest valid observation in the 150 ms lookback ending at
  `onset - 30 ms`; post-target evidence is never used.
- Candidate support: exactly `{open/capo} ∪ [N-1, N+4]`, clipped to the
  configured neck. The open or capoed-open candidate is always supported.
- Strength: confidence-weighted likelihood with a one-nat input-odds cap at
  the validated default decoder weights.
- Scope: ambiguous playable pitches only. Missing, weak, stale, malformed, or
  non-discriminating evidence returns the original `AudioEvent` unchanged.
- Anti-double-counting: the FretCam route does not run the legacy per-string
  `FrameFingering` posterior.

The policy was fixed before the checks below. No window, threshold, weight,
clip, or orientation tuning was performed.

## Same-pitch regression

The deterministic fusion regression uses physically valid MIDI 69 candidates.
The audio-only decoder chooses high-E fret 5; a locked Position X window changes
the choice to B-string fret 10 while onset, offset, and MIDI pitch remain
unchanged. Additional regressions prove:

- open strings receive the same support as frets inside the hand window;
- a strong conflicting audio prior is not overturned by the capped bonus;
- `lambda_vision=0`, no observation, and low/unstable evidence are exact
  object-preserving no-ops;
- custom capo and `max_fret` configurations emit only playable positions.

## Corrected-cache proxy

This is a gold-pitch isolation check over the public GAPS clean-12 bank. It
reuses the existing rich CV cache and F2b corrected coarse anchors, converts
them to the frozen bridge records on the media clock, and mirrors the current
clean-classical automatic string policy: `gaps-v1`, `gaps-seq-v1` at weight
`4.0`, and the baseline assignment decoder. It does **not** claim current
stabilized-solver or real-audio end-to-end accuracy.

| clip | assignment-scored notes | baseline correct | bridge correct | net | events enriched |
|---|---:|---:|---:|---:|---:|
| 027_Zpswc | 1,587 | 1,205 | 1,209 | +4 | 422 |
| 031_vpswc | 887 | 743 | 743 | 0 | 342 |
| 043_bc1wc | 1,401 | 1,129 | 1,129 | 0 | 206 |
| 063_bV1wc | 841 | 598 | 598 | 0 | 4 |
| 104_xf1wc | 419 | 344 | 344 | 0 | 127 |
| 118_VD1wc | 788 | 735 | 733 | -2 | 140 |
| 142_GD1wc | 701 | 596 | 598 | +2 | 91 |
| 179_pM1wc | 515 | 442 | 442 | 0 | 73 |
| 212_y41wc | 946 | 685 | 685 | 0 | 2 |
| 235_Ny1wc | 1,571 | 1,121 | 1,122 | +1 | 161 |
| 294_BSswc | 474 | 434 | 434 | 0 | 21 |
| 341_1M1wc | 691 | 626 | 627 | +1 | 211 |
| **Total** | **10,821** | **8,658** | **8,664** | **+6** | **1,800** |

There are 10,855 individually playable gold notes. The decoder emits 10,821
assignment-scored notes in both arms; 34 duplicated/unison chord notes cannot
survive its per-string monophony constraint and are excluded symmetrically.
The metric below is therefore conditional on matched decoder assignments, not
coverage over every individually playable gold note.

Aggregate exact string/fret assignment accuracy moved `0.800111 → 0.800665`
(`+0.000554` absolute). Relative error reduction was `6 / 2,163 = 0.28%`.
That is a small aggregate net reduction in the target
right-pitch/wrong-position error under the production-aligned classical
policy. The `118_VD1wc` regression and absent current-solver/real-audio lift
prevent any default-promotion claim.

## Reproduction

Prerequisites are the repository checkout plus the existing public GAPS
MusicXML under `~/.tabvision/data/gaps/musicxml`, public MP4 cache under
`~/.tabvision/cache/gaps_video`, and trusted rich CV/offset cache under
`~/.tabvision/cache/gaps_video_chain`. From `tabvision/`, with the sibling
FretCam package installed in the same environment:

```powershell
.\.venv\Scripts\python -m fretcam.tabvision_bridge_probe --output-report
```

The command writes deterministic JSON and an optional Markdown reproduction
under `~/.tabvision/reports`. The inputs are pre-existing public-data caches;
the probe performs no inference, download, training, or policy tuning. It
fails closed when a `TABVISION_*` fusion-sweep override is present rather than
silently producing a different “fixed” result.

## Current-solver smoke

The new non-UI adapter was manually run directly on the public `031_vpswc`
MP4:

- first 6.0 s, source 25 fps, stride 3;
- 24 accepted observations, all `locked`/`holding`, Position I;
- first accepted observation at media time 2.4 s, confidence `0.717`;
- cold runtime `10.63 s` on this Windows CPU.

A longer stateful replay through 81.9 s of `118_VD1wc` completed in `56.50 s`.
The solver abstained heavily and emitted seven accepted observations overall;
three target-window notes were enriched and the ten-note gold-pitch decode was
unchanged (`7/10 → 7/10`). This is honest abstention, not evidence of lift.
Neither smoke opens the reserved held-out benchmark split.
These timing/smoke observations are machine-local development evidence; they
do not yet have an independent checked-in runner or hardware-normalized claim.
The demux clock assumes the ordinary single-audio-stream input used by the
project; exotic multi-audio/edit-list files remain a promotion-test case.

## Verification

- TabVision: `932 passed, 12 skipped`.
- FretCam: `240 passed, 1 skipped, 5 subtests passed`.
- Focused bridge/CLI/pipeline suite: `156 passed`.
- Demux clock/probe suite: `11 passed` locally, including VFR PTS and
  nonzero stream offsets. Seven real-frame checks are gated on
  OpenCV, FFmpeg/ffprobe, and the fixture (`18` total when available).
- Ruff: bridge source and tests pass.
- Mypy: all 81 TabVision modules plus both FretCam bridge modules pass.

## Verdict

The bounded integration mechanism is complete. Its deterministic regression
fixes the intended same-pitch/wrong-position case, and live current-solver
inference over the source-disjoint ten reduces that target error by seven
while increasing macro Tab F1 by 0.000836. Confidence in the direction is
moderate, not promotion-grade: only two held-out clips improve, the lower CI
touches zero, clean-12 regresses by one net target error, and combined test-22
has a CI spanning zero. Keep the route explicit opt-in. Promotion still
requires the controlled-live gate plus a larger frozen evaluation showing a
material effect, no important subgroup/full-clip regression, and latency
within the project target.
