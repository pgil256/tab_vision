"""Phase A guard rail: render full-frame vs crop-then-detect fret detections.

The F2b coordinate bug (DECISIONS 2026-07-22) is the reason this exists: the
largest video error on record was a coordinate-system mistake that unit tests
on real media would have caught immediately. Before any cache is built with the
crop pass, this renders one frame per clip with the full-frame fret centers in
red and the crop-pass additions in green, so wire alignment is eyeballed rather
than assumed.

Reproduce::

    cd tabvision
    python -m scripts.viz.overlay_crop_detect \\
        --video-cache ~/.tabvision/cache/gaps_video_720 \\
        --clips 031_vpswc,104_xf1wc,142_GD1wc --out-dir /tmp/cropcheck
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from scripts.eval.gaps_cv_cache import crop_rect_for_neck, merge_crop_predictions
from tabvision.demux import _frame_iterator, _probe_metadata


def _draw(frame: np.ndarray, preds, color, radius: int, label: str) -> None:  # noqa: ANN001
    import cv2

    for det in preds:
        cv2.circle(frame, (int(round(det.cx)), int(round(det.cy))), radius, color, 2)
    cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main(argv: list[str] | None = None) -> int:
    import cv2

    from tabvision.video.fretboard.keypoint import predictions_to_homography
    from tabvision.video.guitar.yolo_backend import YoloOBBBackend

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--video-cache", type=Path, default=Path.home() / ".tabvision/cache/gaps_video_720"
    )
    ap.add_argument("--clips", default="031_vpswc,104_xf1wc,142_GD1wc")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/cropcheck"))
    ap.add_argument("--frame", type=int, default=900, help="frame index to render")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--crop-conf", type=float, default=0.10)
    ap.add_argument("--crop-pad", type=float, default=0.12)
    ap.add_argument("--crop-min-long-edge", type=int, default=1280)
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args(argv)

    ckpt = args.checkpoint or os.environ.get("TABVISION_GUITAR_YOLO_CHECKPOINT")
    yolo = YoloOBBBackend(checkpoint_path=ckpt, conf=args.conf, device="cpu")
    yolo_crop = YoloOBBBackend(checkpoint_path=ckpt, conf=args.crop_conf, device="cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for stem in (s.strip() for s in args.clips.split(",") if s.strip()):
        vid = args.video_cache / f"{stem}.mp4"
        if not vid.exists():
            print(f"[skip] {stem}: no video")
            continue
        _dur, fps = _probe_metadata(vid)
        frame = None
        for fi, (_t, f) in enumerate(_frame_iterator(vid, fps)):
            if fi == args.frame:
                frame = f.copy()
                break
        if frame is None:
            print(f"[skip] {stem}: frame {args.frame} out of range")
            continue

        full = yolo.predict_all(frame)
        fh, fw = frame.shape[:2]
        rect = crop_rect_for_neck(full, fw, fh, pad_frac=args.crop_pad)
        merged = full
        if rect is not None:
            x0, y0, x1, y1 = rect
            crop = frame[y0:y1, x0:x1]
            ch, cw = crop.shape[:2]
            scale = max(1.0, args.crop_min_long_edge / max(ch, cw))
            if scale > 1.0:
                crop = cv2.resize(
                    crop,
                    (int(round(cw * scale)), int(round(ch * scale))),
                    interpolation=cv2.INTER_CUBIC,
                )
            merged = merge_crop_predictions(
                full,
                yolo_crop.predict_all(crop),
                x0=float(x0),
                y0=float(y0),
                sx=crop.shape[1] / cw,
                sy=crop.shape[0] / ch,
            )
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 200, 0), 2)

        full_centers = {(round(d.cx, 3), round(d.cy, 3)) for d in full.frets}
        added = [d for d in merged.frets if (round(d.cx, 3), round(d.cy, 3)) not in full_centers]
        _draw(frame, full.frets, (0, 0, 255), 7, f"{stem}  full={len(full.frets)}")
        _draw(frame, added, (0, 255, 0), 11, "")
        h_full = predictions_to_homography(full)
        h_merged = predictions_to_homography(merged)
        cv2.putText(
            frame,
            f"crop-added={len(added)}  merged={len(merged.frets)}  "
            f"Hconf {h_full.confidence:.2f}->{h_merged.confidence:.2f}",
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        out = args.out_dir / f"{stem}_f{args.frame}.png"
        cv2.imwrite(str(out), frame)
        print(
            f"{stem}: full={len(full.frets)} crop_added={len(added)} "
            f"merged={len(merged.frets)} nut={len(merged.nut)} "
            f"Hconf {h_full.confidence:.3f}->{h_merged.confidence:.3f} -> {out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
