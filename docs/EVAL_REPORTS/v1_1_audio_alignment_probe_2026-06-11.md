# v1.1 audio alignment probe

**Date:** 2026-06-11T20:29:32+00:00
**Dataset:** KaggleUTAustin
**Root:** `C:\Users\patri\.tabvision\data\datasets\guitar-transcription-utaustin\tablature_dataset\tablature_dataset`
**Cache:** `C:\Users\patri\.tabvision\cache\v1_1_audio_alignment`

## Summary

| Backend | Status | Clips | Raw events | Global shift | Global time | Per-clip onset | Per-clip pitch | Per-clip Tab | Per-clip oracle | Global oracle | Pitch mode | Large-time clips | Zero-oracle clips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| highres | ok | 24 | 380 | -1 | +0.14 | 0.2537 | 0.2019 | 0.0603 | 0.1979 | 0.3535 | -1 | 2 | 7 |
| basicpitch | error | 0 | - | - | - | - | - | - | - | - | - | 0 | 0 |
| highres-fl | ok | 24 | 319 | -1 | +0.16 | 0.1983 | 0.1372 | 0.0355 | 0.1372 | 0.1966 | -1 | 2 | 7 |

## Diagnosis

- Best per-clip oracle-video ceiling is highres at 0.1979.
- Best global-calibration oracle-video ceiling is highres at 0.3535.
- highres: Most clips prefer a -1 semitone correction; treat corpus tuning/reference pitch as a first-class suspect.
- highres: Clips 0 and 1 both need large time-origin shifts; inspect timestamp or label origin handling separately from the rest of the corpus.
- highres: 7 clips still have zero oracle-video Tab F1 after per-clip calibration (9, 10, 11, 12, 15, 22, 23); suspect thresholds, grouping, or backend mismatch.
- highres: Skipped clips with no parsed gold events: 24.
- basicpitch: Backend did not score any clips: basic-pitch is not installed. Install with: pip install basic-pitch
- highres-fl: Most clips prefer a -1 semitone correction; treat corpus tuning/reference pitch as a first-class suspect.
- highres-fl: Clips 0 and 1 both need large time-origin shifts; inspect timestamp or label origin handling separately from the rest of the corpus.
- highres-fl: 7 clips still have zero oracle-video Tab F1 after per-clip calibration (1, 9, 10, 11, 15, 18, 22); suspect thresholds, grouping, or backend mismatch.
- highres-fl: Skipped clips with no parsed gold events: 24.

## Per-Clip Results

### highres

| Clip | Gold | Raw events | Pitch shift | Time shift | Align matches | Onset | Pitch | Tab | Oracle | Raw pitch | Raw oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24 | 18 | +0 | +1.24 | 10 | 0.3333 | 0.1905 | 0.1905 | 0.1905 | 0.0000 | 0.0000 |
| 1 | 24 | 22 | +0 | +1.26 | 12 | 0.0435 | 0.0435 | 0.0435 | 0.0435 | 0.0000 | 0.0000 |
| 2 | 24 | 19 | -1 | +0.12 | 9 | 0.5789 | 0.4211 | 0.1579 | 0.3684 | 0.0000 | 0.0000 |
| 3 | 24 | 19 | -1 | +0.10 | 16 | 0.6667 | 0.5714 | 0.2381 | 0.5714 | 0.0000 | 0.0000 |
| 4 | 24 | 17 | -1 | +0.08 | 13 | 0.3902 | 0.2927 | 0.0488 | 0.2927 | 0.0000 | 0.0000 |
| 5 | 24 | 29 | -1 | +0.06 | 17 | 0.4400 | 0.3600 | 0.1200 | 0.3600 | 0.0000 | 0.0000 |
| 6 | 24 | 20 | -1 | +0.10 | 17 | 0.6364 | 0.5909 | 0.0909 | 0.5909 | 0.0000 | 0.0000 |
| 7 | 24 | 21 | -1 | +0.04 | 19 | 0.1333 | 0.1333 | 0.0444 | 0.1333 | 0.0000 | 0.0000 |
| 8 | 24 | 22 | -1 | +0.16 | 16 | 0.6512 | 0.6047 | 0.1395 | 0.6047 | 0.0000 | 0.0000 |
| 9 | 17 | 22 | -1 | +0.02 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10 | 17 | 12 | -1 | +0.02 | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 11 | 17 | 13 | -1 | +0.04 | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | 16 | 11 | -1 | +0.08 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 13 | 17 | 8 | -1 | +0.02 | 6 | 0.2500 | 0.2500 | 0.0833 | 0.2500 | 0.0000 | 0.0000 |
| 14 | 17 | 11 | -1 | +0.00 | 8 | 0.3200 | 0.3200 | 0.0800 | 0.3200 | 0.0000 | 0.0000 |
| 15 | 17 | 11 | -1 | +0.02 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 16 | 17 | 15 | -1 | +0.00 | 7 | 0.2143 | 0.2143 | 0.0000 | 0.2143 | 0.0000 | 0.0000 |
| 17 | 17 | 14 | -1 | +0.06 | 11 | 0.3333 | 0.3333 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
| 18 | 16 | 13 | -1 | +0.08 | 3 | 0.1538 | 0.0769 | 0.0769 | 0.0769 | 0.0000 | 0.0000 |
| 19 | 33 | 16 | -1 | +0.12 | 9 | 0.5106 | 0.3404 | 0.0851 | 0.2979 | 0.0000 | 0.0000 |
| 20 | 33 | 10 | -1 | +0.14 | 2 | 0.0976 | 0.0488 | 0.0488 | 0.0488 | 0.0000 | 0.0000 |
| 21 | 33 | 6 | -1 | +0.08 | 2 | 0.1081 | 0.0541 | 0.0000 | 0.0541 | 0.0000 | 0.0000 |
| 22 | 22 | 23 | -1 | +0.04 | 4 | 0.2273 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 23 | 22 | 8 | -1 | +0.06 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | 0 | skipped | - | - | - | - | - | - | - | - | - |

### basicpitch

| Clip | Gold | Raw events | Pitch shift | Time shift | Align matches | Onset | Pitch | Tab | Oracle | Raw pitch | Raw oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24 | error | - | - | - | - | - | - | - | - | - |
| 1 | 24 | error | - | - | - | - | - | - | - | - | - |
| 2 | 24 | error | - | - | - | - | - | - | - | - | - |
| 3 | 24 | error | - | - | - | - | - | - | - | - | - |
| 4 | 24 | error | - | - | - | - | - | - | - | - | - |
| 5 | 24 | error | - | - | - | - | - | - | - | - | - |
| 6 | 24 | error | - | - | - | - | - | - | - | - | - |
| 7 | 24 | error | - | - | - | - | - | - | - | - | - |
| 8 | 24 | error | - | - | - | - | - | - | - | - | - |
| 9 | 17 | error | - | - | - | - | - | - | - | - | - |
| 10 | 17 | error | - | - | - | - | - | - | - | - | - |
| 11 | 17 | error | - | - | - | - | - | - | - | - | - |
| 12 | 16 | error | - | - | - | - | - | - | - | - | - |
| 13 | 17 | error | - | - | - | - | - | - | - | - | - |
| 14 | 17 | error | - | - | - | - | - | - | - | - | - |
| 15 | 17 | error | - | - | - | - | - | - | - | - | - |
| 16 | 17 | error | - | - | - | - | - | - | - | - | - |
| 17 | 17 | error | - | - | - | - | - | - | - | - | - |
| 18 | 16 | error | - | - | - | - | - | - | - | - | - |
| 19 | 33 | error | - | - | - | - | - | - | - | - | - |
| 20 | 33 | error | - | - | - | - | - | - | - | - | - |
| 21 | 33 | error | - | - | - | - | - | - | - | - | - |
| 22 | 22 | error | - | - | - | - | - | - | - | - | - |
| 23 | 22 | error | - | - | - | - | - | - | - | - | - |
| 24 | 0 | skipped | - | - | - | - | - | - | - | - | - |

### highres-fl

| Clip | Gold | Raw events | Pitch shift | Time shift | Align matches | Onset | Pitch | Tab | Oracle | Raw pitch | Raw oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24 | 12 | +0 | +1.22 | 7 | 0.2222 | 0.1111 | 0.1111 | 0.1111 | 0.0000 | 0.0000 |
| 1 | 24 | 16 | +0 | +1.26 | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 24 | 16 | -1 | +0.10 | 10 | 0.6000 | 0.4500 | 0.2500 | 0.4500 | 0.0000 | 0.0000 |
| 3 | 24 | 14 | -1 | +0.04 | 10 | 0.1053 | 0.0526 | 0.0526 | 0.0526 | 0.0000 | 0.0000 |
| 4 | 24 | 14 | -1 | +0.08 | 7 | 0.3684 | 0.2105 | 0.0000 | 0.2105 | 0.0000 | 0.0000 |
| 5 | 24 | 22 | -1 | +0.04 | 15 | 0.2609 | 0.2174 | 0.0870 | 0.2174 | 0.0000 | 0.0000 |
| 6 | 24 | 15 | -1 | +0.06 | 12 | 0.1538 | 0.1538 | 0.0000 | 0.1538 | 0.0000 | 0.0000 |
| 7 | 24 | 19 | -1 | +0.04 | 14 | 0.1395 | 0.0465 | 0.0000 | 0.0465 | 0.0000 | 0.0000 |
| 8 | 24 | 20 | -1 | +0.14 | 14 | 0.3636 | 0.3182 | 0.0455 | 0.3182 | 0.0000 | 0.0000 |
| 9 | 17 | 9 | -1 | +0.02 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10 | 17 | 11 | -1 | +0.04 | 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 11 | 17 | 16 | -1 | +0.04 | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | 16 | 10 | -1 | +0.08 | 6 | 0.1667 | 0.1667 | 0.1667 | 0.1667 | 0.0000 | 0.0000 |
| 13 | 17 | 8 | -1 | +0.04 | 4 | 0.1600 | 0.1600 | 0.0000 | 0.1600 | 0.0000 | 0.0000 |
| 14 | 17 | 10 | -1 | +0.00 | 7 | 0.3704 | 0.2963 | 0.0000 | 0.2963 | 0.0000 | 0.0000 |
| 15 | 17 | 9 | -1 | +0.00 | 7 | 0.0769 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 16 | 17 | 9 | -1 | +0.00 | 6 | 0.3077 | 0.3077 | 0.0000 | 0.3077 | 0.0000 | 0.0000 |
| 17 | 17 | 7 | -1 | +0.06 | 6 | 0.4167 | 0.3333 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
| 18 | 16 | 3 | -2 | +0.06 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 19 | 33 | 19 | -1 | +0.10 | 11 | 0.1923 | 0.1923 | 0.0000 | 0.1923 | 0.0000 | 0.0000 |
| 20 | 33 | 9 | -1 | +0.24 | 6 | 0.1429 | 0.0952 | 0.0476 | 0.0952 | 0.0000 | 0.0000 |
| 21 | 33 | 12 | -1 | +0.08 | 5 | 0.1333 | 0.0889 | 0.0444 | 0.0889 | 0.0000 | 0.0000 |
| 22 | 22 | 18 | -2 | +0.04 | 4 | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 23 | 22 | 21 | -1 | +0.18 | 5 | 0.2791 | 0.0930 | 0.0465 | 0.0930 | 0.0000 | 0.0000 |
| 24 | 0 | skipped | - | - | - | - | - | - | - | - | - |

## Notes

- Per-clip metrics apply each clip's best whole-semitone and time-origin correction.
- Global metrics apply one correction shared across all clips for that backend.
- Oracle-video Tab F1 uses gold string/fret evidence, so low oracle scores point back to audio event quality/alignment.

## Real-Video Rerun

The audio probe found the best highres global correction at `pitch_shift=-1`
and `time_shift_s=+0.14`. Because that moved the oracle-video ceiling above
the prior `0.1959`, the real-video chain was rerun with:

```powershell
cd tabvision
.\.venv\Scripts\python.exe -u scripts\eval\v1_1_real_chain_probe.py `
  --checkpoint $env:USERPROFILE\.tabvision\data\models\guitar-yolo-obb-finetuned.pt `
  --audio-source highres `
  --pitch-shift -1 `
  --time-shift-s 0.14
```

The run log is local-only at
`~/.tabvision/logs/v1_1_real_chain_highres_global_audio_20260611-160749.out.log`.

| Condition | Prior highres run | Global-corrected run |
|---|---:|---:|
| highres audio-only | 0.0583 | 0.1415 |
| highres audio + real video | 0.0739 | 0.1656 |
| highres audio + oracle video | 0.1959 | 0.3535 |

Real video now adds `+0.0241` over audio-only, but it recovers only part of the
new oracle headroom. The next implementation target is therefore split:
promote/test the global audio correction path, then inspect why the real-video
gate keeps usable evidence on only 96 events in this run.

## Basic Pitch Attempt

The probe attempted `basicpitch`, but the backend was not installed. Installing
the repo extra with `python -m pip install -e .[audio-baseline]` failed in this
Windows Python 3.12 environment because all compatible `basic-pitch` releases
depend on `tensorflow`, and no matching TensorFlow distribution was available.
The install log is local-only at
`~/.tabvision/logs/install_basicpitch_audio_baseline_20260611-161922.out.log`.
