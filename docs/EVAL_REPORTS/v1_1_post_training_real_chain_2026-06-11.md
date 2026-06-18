# v1.1 post-training real-chain rerun

**Date:** 2026-06-11
**Dataset:** Kaggle UT-Austin tablature dataset, local copy under
`~/.tabvision/data/datasets/guitar-transcription-utaustin`
**Detector:** completed YOLO-OBB checkpoint at
`~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt`

## Why this rerun happened

Chunk 3 completed while the YOLO-OBB fine-tune had only reached the interrupted
epoch-4 checkpoint. The 50-epoch CPU resume has now completed, so the real-video
probes were rerun against the final exported checkpoint.

Training finished at:

| Metric | Value |
|---|---:|
| Precision | 0.9400 |
| Recall | 0.9040 |
| mAP50 | 0.9437 |
| mAP50-95 | 0.6882 |

## Commands

```powershell
cd tabvision
.\.venv\Scripts\python.exe scripts\eval\v1_1_yolo_rig_probe.py --checkpoint $env:USERPROFILE\.tabvision\data\models\guitar-yolo-obb-finetuned.pt
.\.venv\Scripts\python.exe scripts\eval\v1_1_real_chain_probe.py --checkpoint $env:USERPROFILE\.tabvision\data\models\guitar-yolo-obb-finetuned.pt
.\.venv\Scripts\python.exe -u scripts\eval\v1_1_real_chain_probe.py --checkpoint $env:USERPROFILE\.tabvision\data\models\guitar-yolo-obb-finetuned.pt --audio-source highres
```

The highres run was captured in:
`~/.tabvision/logs/v1_1_real_chain_highres_20260611-131459.out.log`.

## YOLO rig probe

The completed detector localizes the neck on the sampled UT-Austin frames:

| Check | Result |
|---|---:|
| Neck detected | 21/21 (100%) |
| Localized non-full-frame homography | 21/21 (100%) |

## Gold-pitch real-video probe

This isolates string/fret selection by using gold-pitch audio events.

| Condition | Mean Tab F1 |
|---|---:|
| audio-only | 0.4243 |
| audio + real video | 0.5373 |
| audio + oracle video | 1.0000 |

Delta: **+0.1130** across 24 scored clips / 527 notes. The final checkpoint
still proves the video lever, but it is slightly below the chunk-3 gold-pitch
run (**0.5453**) because clip 9 regressed under the current gate.

## Real highres audio probe

| Condition | Mean Tab F1 |
|---|---:|
| highres audio-only | 0.0583 |
| highres audio + real video | 0.0739 |
| highres audio + oracle video | 0.1959 |

Delta: **+0.0156** across 24 scored clips / 527 notes. The completed detector
improves the real-video lift versus chunk 3 (**+0.0074**), but the oracle ceiling
is unchanged. This confirms the next chunk should focus on highres audio
transcription/alignment before further video weighting.

## Conclusion

The completed YOLO checkpoint is good enough for the real-video harness. The
remaining blocker is upstream audio on the UT-Austin WAVs: pitch/time alignment,
note grouping, and possibly backend choice. Do not tune more video fusion against
the highres numbers until the highres+oracle ceiling moves materially above
0.1959.
