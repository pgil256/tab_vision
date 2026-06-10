"""v1.1 chunk-1: oracle string-resolution probe on the Kaggle UT-Austin video dataset.

Locks the real-video DATA pipeline end-to-end *except* the MediaPipe CV chain: parse the
per-frame finger labels into per-note gold ``TabEvent``s, then (exactly like the GuitarSet
oracle probe, ``v1_1_oracle_string_probe.py``) feed the gold ``(string, fret)`` back as an
oracle ``FrameFingering`` and confirm the existing resolver lifts string accuracy on these
REAL clips.

Gold derivation. The label array is ``(n_frames, 4_fingers, 3)`` =
``[active, fret, their_string]``. A note onset is a **new** ``(fret, their_string)`` finger
placement vs the previous frame (each pick in these chromatic/positional exercises).
Convention (audio-verified, see docs/EVAL_REPORTS): ``our_string_idx = 6 - their_string``
(their 6 = low E); fret as-labelled. Onsets are timed via ``timestamps.csv``.

Pure fusion over the labels — no audio model, no video, no MediaPipe. Runs in seconds.
The tuning offset between the labels and the real audio does NOT matter here (the audio
events are built from the gold pitch, as in the oracle probe); it becomes a chunk-3
concern only when the real highres audio is introduced.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tabvision.eval.metrics import tab_f1
from tabvision.fusion import fuse
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig, TabEvent

N_FINGERS = 4
_PEAK_LOGIT = 5.0
_FLOOR_LOGIT = -10.0
_DEFAULT_ROOT = (
    Path.home()
    / ".tabvision/data/datasets/guitar-transcription-utaustin"
    / "tablature_dataset/tablature_dataset"
)


def _load_timestamps(root: Path) -> dict[str, float]:
    ts: dict[str, float] = {}
    with open(root / "timestamps.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ts[row["frame"]] = float(row["timestamp"])
    return ts


def parse_clip(
    clip_id: str, root: Path, ts: dict[str, float], cfg: GuitarConfig, default_dur: float = 0.3
) -> list[TabEvent]:
    """Per-frame finger labels -> per-note gold TabEvents (new-placement = onset)."""
    arr = np.load(root / "tablature_labels" / f"{clip_id}.npy")
    gold: list[TabEvent] = []
    prev: set[tuple[int, int]] = set()
    for fi in range(arr.shape[0]):
        cur = {
            (int(arr[fi, k, 1]), int(arr[fi, k, 2]))
            for k in range(arr.shape[1])
            if arr[fi, k].any()
        }
        # Only the highest fretted position on a string sounds when picked; collapse
        # simultaneous same-string new placements (resting fingers) to that one note.
        highest: dict[int, int] = {}
        for fret, their in cur - prev:
            highest[their] = max(fret, highest.get(their, -1))
        for their, fret in sorted(highest.items()):
            our = 6 - their
            t = ts.get(f"{clip_id}_{fi}.png")
            if t is None or not (0 <= our < cfg.n_strings) or not (0 <= fret <= cfg.max_fret):
                continue
            gold.append(
                TabEvent(
                    onset_s=t,
                    duration_s=default_dur,
                    string_idx=our,
                    fret=fret,
                    pitch_midi=cfg.tuning_midi[our] + fret,
                    confidence=1.0,
                )
            )
        prev = cur
    gold.sort(key=lambda e: (e.onset_s, e.string_idx, e.fret))
    return gold


def _events_from_gold(gold: list[TabEvent]) -> list[AudioEvent]:
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


def _oracle_fingerings(
    gold: list[TabEvent], cfg: GuitarConfig, gap_s: float = 0.12
) -> list[FrameFingering]:
    out: list[FrameFingering] = []
    for g in gold:
        logits = np.full((N_FINGERS, cfg.n_strings, cfg.max_fret + 1), _FLOOR_LOGIT)
        for h in gold:
            if abs(h.onset_s - g.onset_s) <= gap_s:
                logits[0, h.string_idx, h.fret] = _PEAK_LOGIT
        out.append(FrameFingering(t=g.onset_s, finger_pos_logits=logits, homography_confidence=1.0))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    args = ap.parse_args(argv)

    cfg = GuitarConfig()
    ts = _load_timestamps(args.root)
    clip_ids = sorted((p.stem for p in (args.root / "tablature_labels").glob("*.npy")), key=int)

    rows: list[tuple[str, int, float, float]] = []
    total_notes = 0
    for cid in clip_ids:
        gold = parse_clip(cid, args.root, ts, cfg)
        if not gold:
            continue
        total_notes += len(gold)
        ev = _events_from_gold(gold)
        fa = tab_f1(fuse(ev, [], cfg), gold).f1
        fo = tab_f1(fuse(ev, _oracle_fingerings(gold, cfg), cfg), gold).f1
        rows.append((cid, len(gold), fa, fo))

    print(f"{'clip':>5} {'notes':>6} {'audio':>8} {'+oracle':>8} {'delta':>8}")
    for cid, n, fa, fo in rows:
        print(f"{cid:>5} {n:>6} {fa:>8.4f} {fo:>8.4f} {fo - fa:>+8.4f}")
    if rows:
        ma = sum(r[2] for r in rows) / len(rows)
        mo = sum(r[3] for r in rows) / len(rows)
        print(
            f"{'ALL':>5} {total_notes:>6} {ma:>8.4f} {mo:>8.4f} {mo - ma:>+8.4f}"
            f"  ({len(rows)} clips)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
