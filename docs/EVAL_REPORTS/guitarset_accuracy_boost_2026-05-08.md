# GuitarSet Accuracy Boost Gate

Date: 2026-05-08

## Decision

Use GuitarSet held-out validation as the production accuracy gate for this
iteration. The 20 personal training videos are no longer used as the primary
acceptance set because their labels are not reliable enough for tuning.

Production candidate:

- Pipeline: `v1`
- Audio backend: `highres`
- Position prior: `guitarset-v1`
- Video: disabled
- Melodic prior: disabled

## Fresh Modal Evidence

Both runs used the full GuitarSet validation split: held-out player `05`, 60
tracks, 8715 gold notes. Modal app runs:

- Candidate: `ap-djmHrsMIXObLuYtMBlyWKR`
- No-prior baseline: `ap-9YmO9TlQldDE7dwXdPTUDd`

| Condition | Onset F1 | Pitch F1 | Tab F1 |
| --- | ---: | ---: | ---: |
| Highres, no prior | 0.9218 | 0.9022 | 0.3878 |
| Highres, `guitarset-v1` | 0.9218 | 0.9022 | 0.6104 |

Delta: `+22.26 pp` Tab F1 with no pitch/onset regression.

Reports:

- `tabvision-server/tools/outputs/guitarset_audio_eval-highres-validation-none-2026-05-08.md`
- `tabvision-server/tools/outputs/guitarset_audio_eval-highres-validation-guitarset-v1-2026-05-08.md`

## Production Smoke

The deployed Modal API was exercised with a GuitarSet-derived MP4 fixture, not a
personal training clip:

- Source audio: `05_Rock1-90-C#_comp_mic.wav`
- Fixture: `/tmp/tabvision-guitarset-05_Rock1-90-Csharp_comp_12s.mp4`
- Endpoint: `https://pgil256--tabvision-api-flask-app.modal.run`
- Job: `5e0b8da3-fc3b-48e7-a0ab-5505524d7ac5`
- Result: completed, `94` notes, `pipeline=v1`, `backend=highres`,
  `prior=guitarset-v1`, `video=false`, `fallbackUsed=false`

## Notes

The experimental melodic-segment prior improved one personal fast-scale clip,
but it regressed GuitarSet aggregate Tab F1 from `0.6104` to `0.5989`, so it is
kept behind an explicit flag and is not part of the default production path.
