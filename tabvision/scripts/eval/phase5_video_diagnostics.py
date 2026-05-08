"""Phase 5 video-evidence diagnostic.

Usage:
    python -m scripts.eval.phase5_video_diagnostics --clip-id training-01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_INDEX = (
    REPO_ROOT / "tabvision-server" / "tests" / "fixtures" / "benchmarks" / "index.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Phase 5 video evidence for one benchmark clip."
    )
    parser.add_argument("--clip-id", default="training-01")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--sample-frames", type=int, default=10)
    args = parser.parse_args(argv)

    video = args.video if args.video is not None else _video_for_clip(args.clip_id)
    report = diagnose_video_evidence(video, sample_frames=args.sample_frames)
    print(_format_report(args.clip_id, video, report))
    return 0


def diagnose_video_evidence(video: Path, *, sample_frames: int = 10) -> dict:
    from tabvision.demux import demux
    from tabvision.pipeline import (
        _make_fretboard_backend,
        _make_guitar_backend,
        _make_hand_backend,
        _run_video_stack,
    )
    from tabvision.types import GuitarConfig

    cfg = GuitarConfig()
    demuxed = demux(video)
    hand_backend = _make_hand_backend()
    try:
        result = _run_video_stack(
            demuxed.frame_iterator,
            stride=3,
            cfg=cfg,
            guitar_backend=_make_guitar_backend(),
            fretboard_backend=_make_fretboard_backend(),
            hand_backend=hand_backend,
        )
    finally:
        close = getattr(hand_backend, "close", None)
        if close is not None:
            close()

    fingerings = result.fingerings
    anchors = result.neck_anchors
    homography_conf = [float(ff.homography_confidence) for ff in fingerings]
    logits = [np.asarray(ff.finger_pos_logits, dtype=np.float64) for ff in fingerings]
    sums = [float(arr.sum()) for arr in logits]
    stds = [float(arr.std()) for arr in logits]
    maxes = [float(arr.max()) for arr in logits]

    report = {
        "fingerings": len(fingerings),
        "anchors": len(anchors),
        "homography_conf": _stats(homography_conf),
        "homography_positive": sum(c > 0.0 for c in homography_conf),
        "logits_nonzero": sum(s != 0.0 for s in sums),
        "logits_nonuniform": sum(s > 1e-9 for s in stds),
        "logits_std": _stats(stds),
        "logits_max": _stats(maxes),
        "samples": [],
        "anchor_center": {},
        "anchor_conf": {},
        "anchor_samples": [],
    }

    for ff, arr, std in zip(fingerings[:sample_frames], logits, stds, strict=False):
        report["samples"].append(
            {
                "t": float(ff.t),
                "homography_confidence": float(ff.homography_confidence),
                "logit_sum": float(arr.sum()),
                "logit_std": std,
                "logit_max": float(arr.max()),
            }
        )

    if anchors:
        centers = [float(anchor.center_fret) for _t, anchor in anchors]
        confs = [float(anchor.confidence) for _t, anchor in anchors]
        report["anchor_center"] = _stats(centers)
        report["anchor_conf"] = _stats(confs)
        for t, anchor in anchors[:sample_frames]:
            report["anchor_samples"].append(
                {
                    "t": float(t),
                    "center_fret": float(anchor.center_fret),
                    "min_fret": float(anchor.min_fret),
                    "max_fret": float(anchor.max_fret),
                    "confidence": float(anchor.confidence),
                }
            )

    return report


def _video_for_clip(clip_id: str) -> Path:
    if not BENCHMARK_INDEX.exists():
        raise FileNotFoundError(f"benchmark index not found: {BENCHMARK_INDEX}")
    benchmarks = json.loads(BENCHMARK_INDEX.read_text()).get("benchmarks", [])
    for bench in benchmarks:
        if bench.get("id") == clip_id:
            video = REPO_ROOT / bench["video_path"]
            if not video.exists():
                raise FileNotFoundError(f"benchmark video not found: {video}")
            return video
    raise KeyError(f"benchmark clip id not found: {clip_id}")


def _stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": fmean(values), "max": max(values)}


def _format_report(clip_id: str, video: Path, report: dict) -> str:
    lines = [
        f"clip={clip_id}",
        f"video={video}",
        f"fingerings={report['fingerings']} anchors={report['anchors']}",
        _stat_line(
            "homography_conf",
            report["homography_conf"],
            suffix=f" positive={report['homography_positive']}/{report['fingerings']}",
        ),
        (
            f"logits_nonzero={report['logits_nonzero']}/{report['fingerings']} "
            f"logits_nonuniform={report['logits_nonuniform']}/{report['fingerings']}"
        ),
        _stat_line("logits_std", report["logits_std"]),
        _stat_line("logits_max", report["logits_max"]),
    ]
    if report["anchors"]:
        lines.extend(
            [
                _stat_line("anchor_center", report["anchor_center"]),
                _stat_line("anchor_conf", report["anchor_conf"]),
            ]
        )
    lines.append("sample_frames:")
    for row in report["samples"]:
        lines.append(
            "  "
            f"t={row['t']:.3f} H={row['homography_confidence']:.3f} "
            f"sum={row['logit_sum']:.6f} std={row['logit_std']:.6f} "
            f"max={row['logit_max']:.6f}"
        )
    if report["anchor_samples"]:
        lines.append("sample_anchors:")
        for row in report["anchor_samples"]:
            lines.append(
                "  "
                f"t={row['t']:.3f} center={row['center_fret']:.3f} "
                f"span=({row['min_fret']:.3f},{row['max_fret']:.3f}) "
                f"conf={row['confidence']:.3f}"
            )
    return "\n".join(lines)


def _stat_line(label: str, stats: dict, *, suffix: str = "") -> str:
    if stats["min"] is None:
        return f"{label}: none{suffix}"
    return (
        f"{label}: min={stats['min']:.6f} mean={stats['mean']:.6f} "
        f"max={stats['max']:.6f}{suffix}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
