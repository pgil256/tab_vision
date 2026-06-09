"""v1.1 oracle string-resolution probe.

Isolates the v1.1 lever. Given PERFECT pitch + onset (from GuitarSet gold) and an
ORACLE fretting-hand signal (a ``FrameFingering`` peaked on the true string/fret),
does the *existing* fusion resolve the string that audio alone cannot?

Per tier on the GuitarSet player-05 validation manifest, compares:

- ``audio``    -- ``fuse(events, [])``: string from the audio prior + playability only.
- ``+oracle``  -- ``fuse(events, oracle_fingerings)``: add the oracle hand signal.

No audio model, no video, no rendering, no inference: pure fusion over the gold
labels. Runs in seconds. This validates the resolver's *ceiling* under a perfect
hand signal -- if ``+oracle`` reaches ~0.94+ single-line, the resolver + wiring
are correct and v1.1 reduces to an eval-data problem (real/synthetic video);
if it does not, the bug is in fuse/playability, not the data (design doc §9).
"""

from __future__ import annotations

import argparse
import os
import tomllib
from pathlib import Path

import numpy as np

from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.eval.metrics import tab_f1
from tabvision.fusion import fuse
from tabvision.fusion.chord import CHORD_MAX_GAP_S
from tabvision.fusion.position_prior import (
    apply_pitch_position_prior,
    load_pitch_position_prior,
)
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig, TabEvent

N_FINGERS = 4  # matches video.hand.fingertip_to_fret.FRETTING_FINGERS
_PEAK_LOGIT = 5.0
_FLOOR_LOGIT = -10.0


def _resolve(path_str: str, data_root: str) -> Path:
    if "$TABVISION_DATA_ROOT" in path_str:
        if not data_root:
            raise ValueError("manifest uses $TABVISION_DATA_ROOT but --data-root is unset")
        path_str = path_str.replace("$TABVISION_DATA_ROOT", data_root)
    return Path(path_str).expanduser()


def _events_from_gold(gold: list[TabEvent]) -> list[AudioEvent]:
    """Perfect audio: right pitch + timing, no string/fret (that's the audio limit)."""
    return [
        AudioEvent(
            onset_s=g.onset_s,
            offset_s=g.onset_s + g.duration_s,
            pitch_midi=g.pitch_midi,
            velocity=1.0,
            confidence=1.0,
        )
        for g in gold
    ]


def _oracle_fingerings(gold: list[TabEvent], cfg: GuitarConfig) -> list[FrameFingering]:
    """One FrameFingering per gold note, peaked on that note's (string, fret) plus
    any chord-mates within ``CHORD_MAX_GAP_S`` (so a cluster's fingering carries every
    cell played at that instant, regardless of which note ``find_fingering_at`` picks).
    """
    fingerings: list[FrameFingering] = []
    for g in gold:
        logits = np.full((N_FINGERS, cfg.n_strings, cfg.max_fret + 1), _FLOOR_LOGIT)
        for h in gold:
            if abs(h.onset_s - g.onset_s) > CHORD_MAX_GAP_S:
                continue
            if 0 <= h.string_idx < cfg.n_strings and 0 <= h.fret <= cfg.max_fret:
                logits[0, h.string_idx, h.fret] = _PEAK_LOGIT
        fingerings.append(
            FrameFingering(t=g.onset_s, finger_pos_logits=logits, homography_confidence=1.0)
        )
    return fingerings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", default=os.environ.get("TABVISION_DATA_ROOT", ""))
    ap.add_argument(
        "--position-prior",
        default="guitarset-v1",
        help="audio position prior applied to BOTH conditions ('none' to disable)",
    )
    args = ap.parse_args(argv)

    cfg = GuitarConfig()

    prior = None
    if args.position_prior and args.position_prior.lower() != "none":
        try:
            prior = load_pitch_position_prior(args.position_prior, cfg=cfg)
        except Exception as exc:  # noqa: BLE001 -- probe: degrade to prior-less
            print(f"warning: could not load prior {args.position_prior!r} ({exc}); continuing")

    payload = tomllib.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_tier: dict[str, list[tuple[float, float]]] = {}
    for clip in payload.get("clips", []):
        if clip.get("split") not in ("validation", "test"):
            continue
        if clip.get("annotation_format") != "guitarset_jams":
            continue
        gold = parse_guitarset_jams(_resolve(clip["annotation_path"], args.data_root), cfg)
        if not gold:
            continue
        events = _events_from_gold(gold)
        if prior is not None:
            events = apply_pitch_position_prior(events, prior)
        pred_audio = fuse(events, [], cfg)
        pred_oracle = fuse(events, _oracle_fingerings(gold, cfg), cfg)
        by_tier.setdefault(clip["tier"], []).append(
            (tab_f1(pred_audio, gold).f1, tab_f1(pred_oracle, gold).f1)
        )

    print(f"prior: {args.position_prior}")
    print(f"{'tier':32} {'clips':>5} {'audio':>8} {'+oracle':>8} {'delta':>7}")
    all_rows: list[tuple[float, float]] = []
    for tier in sorted(by_tier):
        rows = by_tier[tier]
        all_rows.extend(rows)
        ma = sum(a for a, _ in rows) / len(rows)
        mo = sum(o for _, o in rows) / len(rows)
        print(f"{tier:32} {len(rows):>5} {ma:>8.4f} {mo:>8.4f} {mo - ma:>+7.4f}")
    if all_rows:
        ma = sum(a for a, _ in all_rows) / len(all_rows)
        mo = sum(o for _, o in all_rows) / len(all_rows)
        print(f"{'AGGREGATE':32} {len(all_rows):>5} {ma:>8.4f} {mo:>8.4f} {mo - ma:>+7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
