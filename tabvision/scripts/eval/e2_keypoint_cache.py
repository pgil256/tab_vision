"""Phase E2: cache learned fret keypoints over the Phase A frames.

Runs the fine-tuned ``yolo11n-pose`` fret/nut keypoint model (see
``scripts/train/yolo_fret_keypoints_modal.py``) over **exactly the frame indices
already present in the Phase A rich CV cache**, so the keypoint arm and the OBB
arm of the E2 A/B see the same frames of the same clips. Nothing else about the
chain changes.

Each detected instance carries 6 keypoints — the wire's intersections with the
six strings — so a ``fret`` detection supplies the wire's position *and* its
extent across the neck, which is what ``calibrate.py`` currently has to
reconstruct by RANSAC-fitting rule-of-18 to noisy OBB centres.

Output: ``<cache-dir>/<stem>.kpt.pkl`` = ``dict[int, FretKeypointFrame | None]``
keyed by the same frame index as ``<stem>.rawcv.c0.25.crop.pkl``.

Reproduce::

    cd tabvision
    python -m scripts.eval.e2_keypoint_cache
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.gaps_cv_cache import rawcv_cache_path
from tabvision.demux import _frame_iterator, _probe_metadata

DEFAULT_VIDEO_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video_720"
DEFAULT_CV_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video_chain_720"
DEFAULT_OUT = Path.home() / ".tabvision" / "cache" / "gaps_fret_keypoints_720"
DEFAULT_CHECKPOINT = Path.home() / ".tabvision" / "data" / "models" / "guitar-yolo-pose-fret6pt.pt"


def kpt_cache_path(cache_dir: Path, stem: str, det_conf: float) -> Path:
    """Cache path, keyed by detection floor.

    The floor is in the filename for the same reason ``rawcv_cache_path`` keys on
    the YOLO conf: caches built at different floors are not interchangeable, and
    silently reusing one for the other would corrupt an A/B without erroring.
    """
    return cache_dir / f"{stem}.kpt.c{det_conf:.2f}.pkl"


@dataclass
class FretKeypointFrame:
    """Learned fret/nut keypoints for one frame, in full-frame image pixels.

    ``fret_kpts`` is ``(n_fret, 6, 3)`` — six ``(x, y, visibility)`` rows per
    detected wire, ordered as the model emits them (the annotation orders them
    along the wire, i.e. across the six strings). ``nut_kpts`` is the same for
    the ``nut`` class. Confidences are the per-instance box scores.
    """

    fret_kpts: np.ndarray
    fret_conf: np.ndarray
    nut_kpts: np.ndarray
    nut_conf: np.ndarray

    @property
    def n_frets(self) -> int:
        return int(self.fret_kpts.shape[0])


def _empty_frame() -> FretKeypointFrame:
    return FretKeypointFrame(
        fret_kpts=np.zeros((0, 6, 3), dtype=np.float32),
        fret_conf=np.zeros((0,), dtype=np.float32),
        nut_kpts=np.zeros((0, 6, 3), dtype=np.float32),
        nut_conf=np.zeros((0,), dtype=np.float32),
    )


def _split_result(result, fret_class: int, nut_class: int) -> FretKeypointFrame:  # noqa: ANN001
    """Split one ultralytics pose Result into fret/nut keypoint arrays."""
    boxes = result.boxes
    kpts = result.keypoints
    if boxes is None or kpts is None or len(boxes) == 0:
        return _empty_frame()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy().astype(np.float32)
    data = kpts.data.cpu().numpy().astype(np.float32)  # (n, 6, 3)
    is_fret = cls == fret_class
    is_nut = cls == nut_class
    return FretKeypointFrame(
        fret_kpts=data[is_fret],
        fret_conf=conf[is_fret],
        nut_kpts=data[is_nut],
        nut_conf=conf[is_nut],
    )


def build_clip_cache(
    stem: str,
    *,
    video_cache: Path,
    cv_cache: Path,
    out_dir: Path,
    model,  # noqa: ANN001 - ultralytics YOLO
    conf: float,
    cache_suffix: str,
    det_conf: float,
    fret_class: int,
    nut_class: int,
    overwrite: bool = False,
) -> tuple[int, int] | None:
    """Cache keypoints for one clip. Returns ``(n_frames, n_with_frets)``."""
    out_path = kpt_cache_path(out_dir, stem, det_conf)
    if out_path.exists() and not overwrite:
        with open(out_path, "rb") as fh:
            done: dict[int, FretKeypointFrame | None] = pickle.load(fh)
        hits = sum(1 for v in done.values() if v is not None and v.n_frets > 0)
        print(f"  [skip] {stem}: cached ({len(done)} frames, {hits} with frets)")
        return len(done), hits

    rich = rawcv_cache_path(cv_cache, stem, conf, suffix=cache_suffix)
    if not rich.exists():
        print(f"  [skip] {stem}: no Phase A cache at {rich.name}")
        return None
    with open(rich, "rb") as fh:
        raw = pickle.load(fh)
    wanted = set(raw)
    if not wanted:
        print(f"  [skip] {stem}: empty Phase A cache")
        return None

    vid = video_cache / f"{stem}.mp4"
    if not vid.exists():
        print(f"  [skip] {stem}: no video at {vid.name}")
        return None
    _dur, fps = _probe_metadata(vid)

    max_fi = max(wanted)
    out: dict[int, FretKeypointFrame | None] = {}
    for fi, (_t, frame) in enumerate(_frame_iterator(vid, fps)):
        if fi > max_fi:
            break
        if fi not in wanted:
            continue
        result = model.predict(frame, conf=det_conf, verbose=False)[0]
        out[fi] = _split_result(result, fret_class, nut_class)

    # A frame index the decoder never reached stays explicitly None rather than
    # silently absent, so the A/B can tell "no detection" from "no frame".
    for fi in wanted - set(out):
        out[fi] = None

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(out, fh)
    hits = sum(1 for v in out.values() if v is not None and v.n_frets > 0)
    print(f"  [ok] {stem}: {len(out)} frames, {hits} with >=1 fret keypoint set")
    return len(out), hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-cache", type=Path, default=DEFAULT_VIDEO_CACHE)
    ap.add_argument("--cv-cache", type=Path, default=DEFAULT_CV_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--clips", default="clean12", help="'clean12' or comma-separated stems")
    ap.add_argument("--conf", type=float, default=0.25, help="Phase A cache key")
    ap.add_argument("--cache-suffix", default=".crop", help="Phase A cache suffix")
    ap.add_argument(
        "--det-conf",
        type=float,
        default=0.10,
        help="keypoint-model detection floor. Defaults to 0.10 to MATCH the "
        "floor Phase A's crop-then-detect fret pass used, so the E2 A/B varies "
        "the wire source and not the detection threshold.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if not args.checkpoint.exists():
        print(
            f"error: keypoint checkpoint not found at {args.checkpoint}.\n"
            "Train it first: modal run scripts/train/yolo_fret_keypoints_modal.py",
        )
        return 2

    stems = (
        CLEAN_12
        if args.clips == "clean12"
        else tuple(s.strip() for s in args.clips.split(",") if s.strip())
    )

    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint))
    names = {v: k for k, v in model.names.items()} if isinstance(model.names, dict) else {}
    fret_class = int(names.get("fret", 0))
    nut_class = int(names.get("nut", 1))
    print(f"model classes: {model.names} -> fret={fret_class} nut={nut_class}")
    print(f"clips={len(stems)} out={args.out_dir}")

    total_frames = 0
    total_hits = 0
    for stem in stems:
        res = build_clip_cache(
            stem,
            video_cache=args.video_cache,
            cv_cache=args.cv_cache,
            out_dir=args.out_dir,
            model=model,
            conf=args.conf,
            cache_suffix=args.cache_suffix,
            det_conf=args.det_conf,
            fret_class=fret_class,
            nut_class=nut_class,
            overwrite=args.overwrite,
        )
        if res is not None:
            total_frames += res[0]
            total_hits += res[1]

    share = total_hits / total_frames if total_frames else 0.0
    print(f"\ndone: {total_frames} frames, {total_hits} with fret keypoints ({share:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
