"""Why does ``choose_orientation`` pick wrong? — Phase A follow-up diagnostic.

The Phase A ungated A/B measured auto-orientation at Tab F1 **0.6142** against a
best-fixed-orientation ceiling of **0.7635** — a 0.149 gap from orientation
selection alone (`294_BSswc`: auto ``none`` 0.2658 vs ``flip-both`` 0.8080).

Hypothesis under test: :func:`choose_orientation` scores an orientation by
``sum(log(candidate_support) * homography_confidence)`` over events, where
``candidate_support`` is the video posterior mass on *any* audio-plausible
(string, fret) cell for that pitch. But the string-axis mirror maps a candidate
to **another candidate of the same pitch** — that is precisely the
mirrored-cluster structure behind GAPS's graded −1..−4 string bias. If so, total
support is near-invariant to the flip, the four scores nearly tie, and ``max``
resolves the tie by ``ORIENTATIONS`` order (``none`` first) rather than by
evidence.

This prints, per clip: the four selector scores, their spread, which orientation
the selector picks, which one maximises *gold* string accuracy, and whether they
agree. Cache-only.

Reproduce::

    cd tabvision
    python -m scripts.eval.phasea_orientation_diag \\
        --video-cache ~/.tabvision/cache/gaps_video_720 \\
        --cache-dir ~/.tabvision/cache/gaps_video_chain_720 --cache-suffix .crop
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.gaps_cv_cache import (
    load_frame_fingerings,
    make_fret_xs_calibrator,
    needed_frames,
)
from scripts.eval.v1_1_kaggle_oracle_probe import _events_from_gold
from tabvision.demux import _probe_metadata
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.vision_evidence import (
    ORIENTATIONS,
    choose_orientation,
    combine_fingerings,
    orient_fingering,
)
from tabvision.types import GuitarConfig


def _clip_row(
    stem: str,
    data_root: Path,
    video_cache: Path,
    cache_dir: Path,
    cfg: GuitarConfig,
    *,
    conf: float,
    suffix: str,
) -> dict | None:
    gaps = data_root / "gaps"
    xml = gaps / "musicxml" / f"{stem}.xml"
    vid = video_cache / f"{stem}.mp4"
    offset_pkl = cache_dir / f"{stem}.offset.pkl"
    if not (xml.exists() and vid.exists() and offset_pkl.exists()):
        return None
    gold = parse_gaps(xml)
    with open(offset_pkl, "rb") as fh:
        offset_s = float(pickle.load(fh).offset_s)
    _dur, fps = _probe_metadata(vid)
    per_frame = load_frame_fingerings(
        cache_dir,
        stem,
        conf=conf,
        cfg=cfg,
        fps=fps,
        calibrate=make_fret_xs_calibrator(cfg),
        cache_suffix=suffix,
    )
    events = _events_from_gold(gold)
    _, per_onset = needed_frames(
        [e.onset_s for e in events], offset_s, fps, window_s=0.06, max_frames=1
    )
    raw_by_event = [
        [per_frame[fi] for fi in per_onset[i] if per_frame.get(fi) is not None]
        for i in range(len(events))
    ]

    picked, scores = choose_orientation(raw_by_event, events, cfg)

    # Gold-truth best orientation, by ambiguous-note string accuracy.
    ambiguous = [i for i, g in enumerate(gold) if len(candidate_positions(g.pitch_midi, cfg)) >= 2]
    acc: dict[str, float] = {}
    for orientation in ORIENTATIONS:
        correct = have = 0
        for i in ambiguous:
            g = gold[i]
            voted = combine_fingerings(
                [orient_fingering(f, orientation) for f in raw_by_event[i]], cfg, t=g.onset_s
            )
            if voted.homography_confidence <= 0.0:
                continue
            have += 1
            marginal = voted.marginal_string_fret()
            cands = candidate_positions(g.pitch_midi, cfg)
            pred = max(cands, key=lambda c: marginal[c.string_idx, c.fret])
            correct += int(pred.string_idx == g.string_idx)
        acc[orientation.name] = correct / have if have else 0.0
    gold_best = max(acc, key=lambda k: acc[k])

    vals = np.array([scores[o.name] for o in ORIENTATIONS], dtype=float)
    finite = vals[np.isfinite(vals)]
    spread = float(finite.max() - finite.min()) if finite.size else float("nan")
    rel = abs(spread / finite.max()) if finite.size and finite.max() != 0 else float("nan")
    return {
        "stem": stem,
        "picked": picked.name,
        "gold_best": gold_best,
        "agree": picked.name == gold_best,
        "acc_picked": acc[picked.name],
        "acc_best": acc[gold_best],
        "spread": spread,
        "rel_spread": rel,
        "scores": {o.name: scores[o.name] for o in ORIENTATIONS},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision" / "data")
    ap.add_argument(
        "--video-cache", type=Path, default=Path.home() / ".tabvision/cache/gaps_video_720"
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".tabvision/cache/gaps_video_chain_720"
    )
    ap.add_argument("--cache-suffix", default=".crop")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--clips", default="clean12")
    args = ap.parse_args(argv)

    stems = (
        CLEAN_12
        if args.clips == "clean12"
        else tuple(s.strip() for s in args.clips.split(",") if s.strip())
    )
    cfg = GuitarConfig()
    rows = []
    print(
        f"{'clip':>12} {'picked':>11} {'goldbest':>11} {'ok':>3} "
        f"{'acc_pick':>9} {'acc_best':>9} {'lost':>7} {'relspread':>10}"
    )
    for stem in stems:
        row = _clip_row(
            stem,
            args.data_root,
            args.video_cache,
            args.cache_dir,
            cfg,
            conf=args.conf,
            suffix=args.cache_suffix,
        )
        if row is None:
            continue
        rows.append(row)
        print(
            f"{row['stem']:>12} {row['picked']:>11} {row['gold_best']:>11} "
            f"{'Y' if row['agree'] else 'N':>3} {row['acc_picked']:>9.3f} "
            f"{row['acc_best']:>9.3f} {row['acc_picked'] - row['acc_best']:>+7.3f} "
            f"{row['rel_spread']:>10.4f}"
        )
    if not rows:
        print("no clips")
        return 1
    n_ok = sum(r["agree"] for r in rows)
    lost = float(np.mean([r["acc_best"] - r["acc_picked"] for r in rows]))
    print(f"\nselector agrees with gold-best on {n_ok}/{len(rows)} clips")
    print(f"mean string accuracy lost to orientation choice: {lost:.4f}")
    print(
        f"median relative score spread across the 4 orientations: "
        f"{np.nanmedian([r['rel_spread'] for r in rows]):.4f}"
    )
    print(
        "\n(a small relative spread means the four scores nearly tie, so the "
        "argmax is resolved by ORIENTATIONS order rather than by evidence)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
