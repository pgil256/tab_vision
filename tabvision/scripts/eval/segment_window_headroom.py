"""Headroom diagnostic for the Stage 1 segment reranker.

Stage 1 asks whether position windows can *choose* better among the segment
decoder's retained paths. This asks the prior question: are those paths
different tabs at all, and if a perfect oracle picked among them, how much
Tab F1 would it win?

The decoder's K-best runs over the product space of latent hand state x chord
state, but only the chord-state half reaches the emitted ``TabEvent``. Two
paths can therefore differ in cost and in hand-state label while assigning
every note identically — in which case no reranker, however good its
evidence, can move a single note.

Diagnostic only: no gate, no promotion, reported alongside the Stage 1
verdict so a null delta is attributable. Run from ``tabvision/``::

    TABVISION_DATA_ROOT=~/.tabvision/data python -m scripts.eval.segment_window_headroom \
        --k 3 10 25 --json ../docs/EVAL_REPORTS/segment_window_headroom_2026-07-29.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.segment_window_stage1 import _event_from_json
from tabvision.eval.metrics import tab_f1
from tabvision.eval.parsers.registry import get_parser
from tabvision.fusion.position_prior import apply_pitch_position_prior, load_pitch_position_prior
from tabvision.fusion.viterbi import decode_segment_v1_with_analysis
from tabvision.pipeline import sequence_decode_context
from tabvision.types import GuitarConfig, SessionConfig


def _assignment_key(events) -> tuple[tuple[int, int], ...]:
    return tuple((int(e.string_idx), int(e.fret)) for e in events)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 10, 25])
    parser.add_argument("--gaps-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args(argv)

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", os.path.expanduser("~/.tabvision/data")))
    gaps = args.gaps_root or (data_root / "gaps")
    cache_dir = args.cache_dir or (data_root / "models" / "q6_gaps_cache")

    cfg = GuitarConfig()
    session = SessionConfig(style="fingerstyle")
    prior = load_pitch_position_prior("gaps-v1")
    parse = get_parser("gaps_musicxml_tab")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for clip in CLEAN_12:
        xml_path = gaps / "musicxml" / f"{clip}.xml"
        cache = cache_dir / f"{clip}.ensemble.json"
        if not xml_path.is_file() or not cache.is_file():
            continue
        gold = list(parse(xml_path, cfg))
        events = [_event_from_json(item) for item in json.loads(cache.read_text("utf-8"))]
        prepared = apply_pitch_position_prior(events, prior)

        row: dict[str, Any] = {"clip": clip}
        for k in args.k:
            with sequence_decode_context("gaps-seq-v1"):
                decoded = decode_segment_v1_with_analysis(
                    prepared, cfg=cfg, session=session, k_paths=k
                )
            if not decoded.paths:
                continue
            keys = {_assignment_key(path.events) for path in decoded.paths}
            scores = [float(tab_f1(path.events, gold).f1) for path in decoded.paths]
            base = scores[0]
            row[f"k{k}"] = {
                "paths_returned": len(decoded.paths),
                "distinct_assignments": len(keys),
                "base_tab_f1": base,
                "oracle_best_tab_f1": max(scores),
                "oracle_gain": max(scores) - base,
                "max_notes_differing": max(
                    (
                        sum(
                            1
                            for b, a in zip(decoded.paths[0].events, path.events, strict=True)
                            if (b.string_idx, b.fret) != (a.string_idx, a.fret)
                        )
                        for path in decoded.paths[1:]
                    ),
                    default=0,
                ),
            }
            print(
                f"  {clip} k={k:<3d} paths={len(decoded.paths):<3d} "
                f"distinct={len(keys):<3d} oracle_gain={max(scores) - base:+.4f} "
                f"notes_diff={row[f'k{k}']['max_notes_differing']}",
                flush=True,
            )
        rows.append(row)

    if not rows:
        raise SystemExit("no clips scored — build the audio-event cache first")

    summary: dict[str, Any] = {
        "clips": len(rows),
        "k_values": args.k,
        "wall_seconds": time.perf_counter() - started,
        "per_clip": rows,
    }
    for k in args.k:
        cells = [row[f"k{k}"] for row in rows if f"k{k}" in row]
        if not cells:
            continue
        summary[f"aggregate_k{k}"] = {
            "mean_distinct_assignments": sum(c["distinct_assignments"] for c in cells) / len(cells),
            "clips_with_any_alternative": sum(1 for c in cells if c["distinct_assignments"] > 1),
            "mean_base_tab_f1": sum(c["base_tab_f1"] for c in cells) / len(cells),
            "mean_oracle_tab_f1": sum(c["oracle_best_tab_f1"] for c in cells) / len(cells),
            "mean_oracle_gain": sum(c["oracle_gain"] for c in cells) / len(cells),
        }

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== aggregate ===")
    for k in args.k:
        agg = summary.get(f"aggregate_k{k}")
        if agg:
            print(
                f"k={k:<3d} distinct/clip {agg['mean_distinct_assignments']:.2f}  "
                f"clips with an alternative {agg['clips_with_any_alternative']}/{len(rows)}  "
                f"oracle best-of-k gain {agg['mean_oracle_gain']:+.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
