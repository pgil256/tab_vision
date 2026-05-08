# Data

Real data lives at `$TABVISION_DATA_ROOT` (env var; defaults to `~/.tabvision/data`).

The paths inside this directory are placeholders for tooling layout:

- `fixtures/` — tiny clips checked into the repo for unit/integration tests (≤ 20 MB total).
- `eval/` — manifest + metadata only. Source clips are external.
- `augmented/` — generated data; `.gitignored` (do not commit).

See SPEC.md §6 for resource acquisition and §9.1 for eval datasets.

Model/dependency readiness:

```bash
cd tabvision
python -m scripts.acquire.models list
python -m scripts.acquire.models status
python -m scripts.acquire.models prepare-yolo-dir
```

`prepare-yolo-dir` creates the expected model directory and prints the
checkpoint path used by the video backend. Place a trained
`guitar-yolo-obb-finetuned.pt` there, or set `TABVISION_GUITAR_YOLO_CHECKPOINT`
to an explicit checkpoint path.
