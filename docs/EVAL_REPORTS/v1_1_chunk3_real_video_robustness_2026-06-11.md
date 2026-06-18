# v1.1 chunk 3: real-video robustness + real-audio check

**Date:** 2026-06-11  
**Dataset:** Kaggle UT-Austin tablature dataset, local copy under
`~/.tabvision/data/datasets/guitar-transcription-utaustin`  
**Detector:** trained YOLO-OBB checkpoint at
`~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt`  
**Probe:** `tabvision/scripts/eval/v1_1_real_chain_probe.py`

## What changed

Chunk 2 proved the real MediaPipe -> homography -> fingering chain can lift Tab
F1 when the rig-specific orientation is supplied manually. Chunk 3 makes that
path robust enough to run as an eval harness:

- automatic per-clip orientation selection from audio-compatible candidate mass
  (`none`, `flip-fret`, `flip-string`, `flip-both`);
- multi-frame voting around each audio event onset;
- zero-confidence homographies are ignored by fusion;
- `homography_confidence` now scales the vision emission weight;
- audio-compatible per-event gating plus a clip-coverage fallback
  (`--min-clip-coverage 0.71`) so sparse video evidence collapses to audio-only;
- `--audio-source highres` runs the real WAVs, with automatic pitch and time-shift
  calibration before fusion.

## Commands

```powershell
cd tabvision
.\.venv\Scripts\python.exe -m ruff check tabvision\audio\highres.py tabvision\fusion\vision_evidence.py tabvision\fusion\playability.py scripts\eval\v1_1_real_chain_probe.py tests\unit\test_vision_evidence.py tests\unit\test_playability.py
.\.venv\Scripts\python.exe -m pytest tests\unit\test_vision_evidence.py tests\unit\test_playability.py tests\unit\test_video_string_resolution.py -q
.\.venv\Scripts\python.exe scripts\eval\v1_1_real_chain_probe.py
.\.venv\Scripts\python.exe scripts\eval\v1_1_real_chain_probe.py --audio-source highres
```

## Gold-pitch real-video robustness

This run keeps the chunk-2 isolation setup: audio events are gold-pitch
`AudioEvent`s, so the metric isolates whether real video helps string/fret
choice. The default robust gate produced no per-clip regressions.

| Condition | Mean Tab F1 |
|---|---:|
| audio-only | 0.4243 |
| audio + real video | 0.5453 |
| audio + oracle video | 1.0000 |

Delta: **+0.1211** over audio-only across 24 scored clips / 527 notes.  
The gate kept 239 event-aligned real-video fingerings and fell back to audio-only
on clips where evidence was too sparse or fragile.

## Real highres audio

The highres path now runs end-to-end on the real WAVs. The wrapper needed a
Windows stdio encoding guard because `hf_midi_transcription` prints a Unicode
status glyph while downloading its checkpoint.

| Condition | Mean Tab F1 |
|---|---:|
| highres audio-only | 0.0583 |
| highres audio + real video | 0.0657 |
| highres audio + oracle video | 0.1959 |

Delta: **+0.0074** over highres audio-only across 24 scored clips / 527 notes.
The small lift is expected from the oracle ceiling: even perfect video reaches
only 0.1959 when fused with these highres events, so the current bottleneck is
upstream audio transcription / alignment on this corpus, not the real-video
string-resolution chain.

## Conclusion

Chunk 3 is complete as a robustness/eval chunk: the manual chunk-2 orientation
flag is automatic, video evidence is voted and gated, fusion degrades to
audio-only when video evidence is weak, and the real highres-audio path has a
measured headline number. The next accuracy work should not keep tuning the
real-video gate against this result; it should address the highres audio failure
mode on the UT-Austin WAVs or evaluate a better audio backbone for this corpus.
