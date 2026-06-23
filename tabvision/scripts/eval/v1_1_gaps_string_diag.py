"""v1.1 chunk-6 (WS0): cache-only string-resolution diagnostic for the GAPS chain.

Reproduces — and makes reusable — the chunk-5 ``_diag`` analysis: per *ambiguous*
gold note, the string the CV chain predicts (gold pitch, **best fixed orientation
per clip**) vs the gold string. Emits per-clip and aggregate **ambiguous-note
string accuracy** plus the ``pred − gold`` string/fret offset histograms.

This is the chunk-6 *leading* indicator. The headline probe
(``v1_1_gaps_video_chain_probe``) reports the lagging, gated Tab F1; this isolates
the raw string-resolution quality of the CV evidence itself — the thing the
geometry workstreams (WS1/WS2/WS4) must move. The chunk-5 baseline is
**0.543** (4178/7697); the chunk-6 target is **≥ 0.75**.

"Best orientation per clip" here is the orientation maximizing *string
correctness* against gold — a diagnostic ceiling on orientation selection, and
deliberately distinct from the probe's gated ``real_best`` (which maxes Tab F1
and is dragged to ``none`` by the coverage gate). It uses gold only to *pick* the
flip and to score; it never feeds gold into the prediction.

Runs entirely from cache (no re-download / re-transcribe / MediaPipe / YOLO):
per-frame fingerings come from the rich v2 cache when present
(:mod:`scripts.eval.gaps_cv_cache`), else the legacy ``FrameFingering`` cache.
The per-clip offset is read from ``{stem}.offset.pkl``; fps from ffprobe on the
local mp4. Seconds to run over the clean-12.

Reproduce::

    cd tabvision
    export TABVISION_DATA_ROOT=~/.tabvision/data
    export PATH=~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin:$PATH
    python -m scripts.eval.v1_1_gaps_string_diag
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.gaps_cv_cache import load_frame_fingerings, needed_frames
from scripts.eval.v1_1_kaggle_oracle_probe import _events_from_gold
from tabvision.demux import _probe_metadata
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.vision_evidence import (
    ORIENTATIONS,
    combine_fingerings,
    orient_fingering,
)
from tabvision.types import FrameFingering, GuitarConfig, TabEvent


@dataclass
class ClipStringDiag:
    """Per-clip ambiguous-note string-resolution diagnosis (best orientation)."""

    stem: str
    n_gold: int  # total gold notes
    n_ambiguous: int  # gold notes playable on >= 2 strings
    have_cv: int  # ambiguous notes with CV evidence near onset (orientation-invariant)
    correct: int  # correct-string count under the best orientation
    best_orient: str
    str_hist: dict[int, int] = field(default_factory=dict)  # pred − gold string offsets
    fret_hist: dict[int, int] = field(default_factory=dict)  # pred − gold fret offsets

    @property
    def str_acc(self) -> float:
        return self.correct / self.have_cv if self.have_cv else 0.0


def diagnose_clip_strings(
    gold: list[TabEvent],
    per_frame: dict[int, FrameFingering | None],
    offset_s: float,
    fps: float,
    cfg: GuitarConfig,
    *,
    window_s: float = 0.06,
    max_frames: int = 1,
    stem: str = "",
) -> ClipStringDiag:
    """Best-orientation ambiguous-note string accuracy + offset histograms.

    For each gold note that is *ambiguous* (playable on ≥ 2 strings), votes the
    cached fingerings near its onset, restricts the marginal to the gold pitch's
    candidate cells, and takes the arg-max cell as the predicted ``(string,
    fret)``. ``have_cv`` (ambiguous notes with usable CV evidence) is
    orientation-invariant; the reported orientation is the one maximizing the
    correct-string count. Mirrors the chunk-5 ``_diag`` exactly (no gate, gold
    pitch), so it reproduces the 0.543 baseline.
    """
    events = _events_from_gold(gold)
    _, per_onset = needed_frames(
        [e.onset_s for e in events], offset_s, fps, window_s=window_s, max_frames=max_frames
    )
    ambiguous = [i for i, g in enumerate(gold) if len(candidate_positions(g.pitch_midi, cfg)) >= 2]
    raw_by_note = {
        i: [per_frame[fi] for fi in per_onset[i] if per_frame.get(fi) is not None]
        for i in ambiguous
    }

    best: ClipStringDiag | None = None
    for orientation in ORIENTATIONS:
        correct = 0
        have = 0
        str_hist: Counter[int] = Counter()
        fret_hist: Counter[int] = Counter()
        for i in ambiguous:
            g = gold[i]
            oriented = [orient_fingering(f, orientation) for f in raw_by_note[i]]
            voted = combine_fingerings(oriented, cfg, t=g.onset_s)
            if voted.homography_confidence <= 0.0:
                continue  # no CV evidence near this onset
            have += 1
            marginal = voted.marginal_string_fret()
            cands = candidate_positions(g.pitch_midi, cfg)
            pred = max(cands, key=lambda c: marginal[c.string_idx, c.fret])
            str_hist[pred.string_idx - g.string_idx] += 1
            fret_hist[pred.fret - g.fret] += 1
            if pred.string_idx == g.string_idx:
                correct += 1
        candidate = ClipStringDiag(
            stem=stem,
            n_gold=len(gold),
            n_ambiguous=len(ambiguous),
            have_cv=have,
            correct=correct,
            best_orient=orientation.name,
            str_hist=dict(str_hist),
            fret_hist=dict(fret_hist),
        )
        if best is None or candidate.correct > best.correct:
            best = candidate
    assert best is not None  # ORIENTATIONS is non-empty
    return best


# --------------------------------------------------------------------------- #
# Cache-only file wiring
# --------------------------------------------------------------------------- #
def _diagnose_clip(
    stem: str,
    data_root: Path,
    video_cache: Path,
    cache_dir: Path,
    cfg: GuitarConfig,
    *,
    conf: float,
    window_s: float,
    max_frames: int,
) -> ClipStringDiag | None:
    gaps = data_root / "gaps"
    xml = gaps / "musicxml" / f"{stem}.xml"
    vid = video_cache / f"{stem}.mp4"
    offset_pkl = cache_dir / f"{stem}.offset.pkl"
    if not xml.exists():
        print(f"  [skip] {stem}: no musicxml")
        return None
    gold = parse_gaps(xml)
    if not gold:
        print(f"  [skip] {stem}: empty gold")
        return None
    if not offset_pkl.exists():
        print(f"  [skip] {stem}: no cached offset ({offset_pkl.name})")
        return None
    with open(offset_pkl, "rb") as fh:
        offset_s = float(pickle.load(fh).offset_s)
    if not vid.exists():
        print(f"  [skip] {stem}: no video (need fps)")
        return None
    _dur, fps = _probe_metadata(vid)
    try:
        per_frame = load_frame_fingerings(cache_dir, stem, conf=conf, cfg=cfg, fps=fps)
    except FileNotFoundError as exc:
        print(f"  [skip] {stem}: {exc}")
        return None
    return diagnose_clip_strings(
        gold, per_frame, offset_s, fps, cfg, window_s=window_s, max_frames=max_frames, stem=stem
    )


def _fmt_hist(hist: dict[int, int]) -> str:
    return " ".join(f"{k:+d}:{hist[k]}" for k in sorted(hist))


def _resolve_clips(spec: str) -> tuple[str, ...]:
    if spec == "clean12":
        return CLEAN_12
    return tuple(s.strip() for s in spec.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision" / "data")
    ap.add_argument("--video-cache", type=Path, default=Path.home() / ".tabvision/cache/gaps_video")
    ap.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".tabvision/cache/gaps_video_chain"
    )
    ap.add_argument("--clips", default="clean12", help="'clean12' or comma-separated stems")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO conf (cache key)")
    ap.add_argument("--vote-frames", type=int, default=1, help="frames per onset (chunk-5 used 1)")
    ap.add_argument("--vote-window-s", type=float, default=0.06)
    ap.add_argument("--limit", type=int, default=None, help="first N clips (smoke)")
    args = ap.parse_args(argv)

    cfg = GuitarConfig()
    clips = _resolve_clips(args.clips)
    if args.limit is not None:
        clips = clips[: args.limit]

    print(
        f"{'clip':>12} {'notes':>6} {'ambig':>6} {'haveCV':>7} {'str_acc':>7}  "
        "best_orient  hist(pred-gold)"
    )
    results: list[ClipStringDiag] = []
    for stem in clips:
        d = _diagnose_clip(
            stem,
            args.data_root,
            args.video_cache,
            args.cache_dir,
            cfg,
            conf=args.conf,
            window_s=args.vote_window_s,
            max_frames=args.vote_frames,
        )
        if d is None:
            continue
        results.append(d)
        print(
            f"{d.stem:>12} {d.n_gold:>6} {d.n_ambiguous:>6} {d.have_cv:>7} {d.str_acc:>7.3f}  "
            f"{d.best_orient:<11}  {_fmt_hist(d.str_hist)}"
        )

    if not results:
        print("no clips diagnosed")
        return 1

    total_correct = sum(d.correct for d in results)
    total_have = sum(d.have_cv for d in results)
    agg_str: Counter[int] = Counter()
    agg_fret: Counter[int] = Counter()
    for d in results:
        agg_str.update(d.str_hist)
        agg_fret.update(d.fret_hist)
    acc = total_correct / total_have if total_have else 0.0
    print(
        f"\nAGG ambiguous-note string accuracy (best-orient, gold pitch): "
        f"{total_correct}/{total_have} = {acc:.3f}  (target >= 0.75; chunk-5 baseline 0.543)"
    )
    print(f"string-offset histogram (pred-gold), all clips best-orient: {_fmt_hist(dict(agg_str))}")
    print(f"fret-offset histogram (pred-gold): {_fmt_hist(dict(agg_fret))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
