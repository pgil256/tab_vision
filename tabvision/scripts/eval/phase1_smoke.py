"""Phase 1 smoke eval — run the new pipeline on a few user clips, report counts.

Per SPEC §7 Phase 1 acceptance: "Eval harness reports any numbers on at
least 3 user clips." We report detected event counts vs. ground-truth
fret-count. Real mir_eval-based metrics (Onset F1, Tab F1) come in
Phase 8 harness hardening.

Usage:
    python -m scripts.eval.phase1_smoke --videos PATH ... --gt-dir PATH
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

# Regex finds fret numbers (1–2 digits) in a tab line, ignoring barlines/dashes.
FRET_RE = re.compile(r"\d+")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 smoke eval")
    parser.add_argument("--videos", nargs="+", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from tabvision.audio.basicpitch import BasicPitchBackend
    from tabvision.demux import demux
    from tabvision.fusion import fuse
    from tabvision.types import GuitarConfig, SessionConfig

    cfg = GuitarConfig()
    session = SessionConfig()
    backend = BasicPitchBackend()

    rows: list[dict] = []
    for video in args.videos:
        gt_path = _find_gt(args.gt_dir, video)
        gt_count = _count_gt_notes(gt_path) if gt_path else None

        try:
            demuxed = demux(video)
            audio_events = backend.transcribe(demuxed.wav, demuxed.sample_rate, session)
            tab_events = fuse(audio_events, [], cfg, session)
            row = {
                "clip": video.stem,
                "duration_s": round(demuxed.duration_s, 2),
                "audio_events": len(audio_events),
                "tab_events": len(tab_events),
                "gt_notes": gt_count,
                "ratio_detected_to_gt": (
                    round(len(tab_events) / gt_count, 3) if gt_count else None
                ),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 — Phase 1 smoke eval, surface anything
            row = {
                "clip": video.stem,
                "audio_events": None,
                "tab_events": None,
                "gt_notes": gt_count,
                "ratio_detected_to_gt": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(json.dumps(row, indent=2))

    summary = {
        "n_clips": len(rows),
        "n_succeeded": sum(1 for r in rows if r["error"] is None),
        "rows": rows,
    }

    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


def _find_gt(gt_dir: Path, video: Path) -> Path | None:
    """Look for a ``<stem>-tabs.txt`` next to the video or in gt_dir."""
    candidates = [
        gt_dir / f"{video.stem}-tabs.txt",
        video.with_name(f"{video.stem}-tabs.txt"),
        video.parent.parent / "training-tabs" / f"{video.stem}-tabs.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _count_gt_notes(path: Path) -> int:
    """Count fret-positions across all tab lines (each digit-run = 1 note)."""
    text = path.read_text()
    count = 0
    for line in text.splitlines():
        # Tab lines start with a string label like "e|", "B|", "G|", etc.
        if "|" not in line[:3]:
            continue
        for _ in FRET_RE.finditer(line):
            count += 1
    return count


if __name__ == "__main__":
    raise SystemExit(main())
