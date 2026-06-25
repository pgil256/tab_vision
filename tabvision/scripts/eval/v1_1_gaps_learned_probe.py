"""WS4 go/no-go probe: learned string-resolution vs the geometric chain.

Eval the trained string-resolution model (``learned_string``) on the held-out
GAPS **test** clips (clean-12), in the gold-audio frame (string axis isolated).
Per clip: ``audio-only`` / ``+learned`` / ``+oracle`` Tab F1. The bar is the
geometric leading 0.574 (string acc) and the gated audio-only 0.8148; oracle is
0.9726.

Per note the YOLO neck-crop at the onset frame → a 6-way string posterior; the
known pitch restricts it to its candidate strings; that becomes a
``FrameFingering`` (same construction as ``_oracle_fingerings``) fed through the
existing fusion — no §8 change. ``--checkpoint`` is optional (a random model
gives a wiring-only smoke).

Usage::

    cd tabvision
    export TABVISION_DATA_ROOT=~/.tabvision/data
    export PATH=~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin:$PATH
    python -m scripts.eval.v1_1_gaps_learned_probe \
        --checkpoint ~/.tabvision/data/models/string-resolver.pt
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from scripts.acquire.gaps_video import CLEAN_12, estimate_offset
from scripts.eval.v1_1_kaggle_oracle_probe import _events_from_gold, _oracle_fingerings
from scripts.train.extract_string_dataset import _sample_neck_rect
from tabvision.demux import _frame_iterator, _probe_metadata
from tabvision.eval.metrics import tab_f1
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.fusion import fuse
from tabvision.fusion.candidates import candidate_positions
from tabvision.types import FrameFingering, GuitarConfig, TabEvent

_FLOOR = -10.0


def fingering_from_proba(
    proba: np.ndarray, pitch_midi: int, cfg: GuitarConfig, *, t: float
) -> FrameFingering:
    """A FrameFingering encoding the learned string posterior over candidates.

    Places ``log p(string)`` on each candidate ``(string, fret)`` of the pitch;
    :meth:`FrameFingering.marginal_string_fret` then softmaxes to the candidate
    distribution. Mirrors ``_oracle_fingerings`` (which spikes the gold cell).
    """
    logits = np.full((4, cfg.n_strings, cfg.max_fret + 1), _FLOOR, dtype=np.float64)
    cands = candidate_positions(pitch_midi, cfg)
    if not cands:
        return FrameFingering(t=t, finger_pos_logits=logits, homography_confidence=0.0)
    for c in cands:
        logits[0, c.string_idx, c.fret] = float(np.log(max(float(proba[c.string_idx]), 1e-6)))
    return FrameFingering(t=t, finger_pos_logits=logits, homography_confidence=1.0)


def _learned_fingerings(
    gold: list[TabEvent],
    vid: Path,
    offset: float,
    fps: float,
    rect: tuple[int, int, int, int],
    model,  # noqa: ANN001 - StringResolverNet
    cfg: GuitarConfig,
    *,
    crop_size: int,
) -> list[FrameFingering]:
    """One learned FrameFingering per gold note (string posterior from the crop)."""
    import cv2

    from tabvision.video.hand.learned_string import predict_string_proba

    x0, y0, x1, y1 = rect
    want: dict[int, list[int]] = {}
    for i, g in enumerate(gold):
        fi = int(round((g.onset_s + offset) * fps))
        if fi >= 0:
            want.setdefault(fi, []).append(i)
    crops: dict[int, np.ndarray] = {}
    max_fi = max(want) if want else -1
    for fi, (_t, frame) in enumerate(_frame_iterator(vid, fps)):
        if fi > max_fi:
            break
        if fi not in want:
            continue
        crop = frame[y0:y1, x0:x1]
        if crop.size:
            crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
            for i in want[fi]:
                crops[i] = crop

    out: list[FrameFingering] = []
    idxs = [i for i in range(len(gold)) if i in crops]
    proba_by_note: dict[int, np.ndarray] = {}
    chunk = 128  # mini-batch so a long clip doesn't OOM the backbone forward
    for start in range(0, len(idxs), chunk):
        block = idxs[start : start + chunk]
        probs = predict_string_proba(model, np.stack([crops[i] for i in block], axis=0))
        for j, i in enumerate(block):
            proba_by_note[i] = probs[j]
    for i, g in enumerate(gold):
        if i in proba_by_note:
            out.append(fingering_from_proba(proba_by_note[i], g.pitch_midi, cfg, t=g.onset_s))
        else:
            out.append(
                FrameFingering(
                    t=g.onset_s,
                    finger_pos_logits=np.full(
                        (4, cfg.n_strings, cfg.max_fret + 1), _FLOOR, dtype=np.float64
                    ),
                    homography_confidence=0.0,
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision" / "data")
    ap.add_argument("--video-cache", type=Path, default=Path.home() / ".tabvision/cache/gaps_video")
    ap.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".tabvision/cache/gaps_video_chain"
    )
    ap.add_argument(
        "--checkpoint", type=Path, default=None, help="string-resolver .pt (random if omitted)"
    )
    ap.add_argument("--yolo-checkpoint", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--clips", default="clean12")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = GuitarConfig()
    from tabvision.video.guitar.yolo_backend import YoloOBBBackend
    from tabvision.video.hand.learned_string import StringResolverNet, load_string_resolver

    if args.checkpoint is not None:
        model = load_string_resolver(args.checkpoint)
        print(f"loaded checkpoint {args.checkpoint}", flush=True)
    else:
        model = StringResolverNet(n_strings=cfg.n_strings, pretrained=False)
        print("WARNING: no --checkpoint; random model (wiring smoke only)", flush=True)

    yolo_ckpt = args.yolo_checkpoint or os.environ.get("TABVISION_GUITAR_YOLO_CHECKPOINT")
    yolo = YoloOBBBackend(checkpoint_path=yolo_ckpt, conf=args.conf, device="cpu")

    stems = CLEAN_12 if args.clips == "clean12" else tuple(args.clips.split(","))
    if args.limit is not None:
        stems = stems[: args.limit]

    gaps = args.data_root / "gaps"
    print(f"{'clip':>12} {'gold':>5}  audio/+learned/+oracle", flush=True)
    ao_l, le_l, or_l = [], [], []
    for stem in stems:
        xml = gaps / "musicxml" / f"{stem}.xml"
        wav = gaps / "audio" / f"{stem}.wav"
        vid = args.video_cache / f"{stem}.mp4"
        if not (xml.exists() and wav.exists() and vid.exists()):
            print(f"  [skip] {stem}: missing media", flush=True)
            continue
        gold = parse_gaps(xml)
        if not gold:
            continue
        off_pkl = args.cache_dir / f"{stem}.offset.pkl"
        if off_pkl.exists():
            with open(off_pkl, "rb") as fh:
                offset = float(pickle.load(fh).offset_s)
        else:
            offset = float(estimate_offset(wav, vid).offset_s)
        _dur, fps = _probe_metadata(vid)
        rect = _sample_neck_rect(vid, yolo, n_samples=20, pad_frac=0.35)
        if rect is None:
            print(f"  [skip] {stem}: no neck box", flush=True)
            continue

        events = _events_from_gold(gold)
        audio_only = tab_f1(fuse(events, [], cfg), gold).f1
        oracle = tab_f1(fuse(events, _oracle_fingerings(gold, cfg), cfg), gold).f1
        fings = _learned_fingerings(
            gold, vid, offset, fps, rect, model, cfg, crop_size=args.crop_size
        )
        learned = tab_f1(fuse(events, fings, cfg), gold).f1
        ao_l.append(audio_only)
        le_l.append(learned)
        or_l.append(oracle)
        print(f"{stem:>12} {len(gold):>5}  {audio_only:.3f}/{learned:.3f}/{oracle:.3f}", flush=True)

    if ao_l:
        print(
            f"\nMEAN  audio-only {np.mean(ao_l):.4f} -> +learned {np.mean(le_l):.4f}  "
            f"oracle {np.mean(or_l):.4f}   (geometric leading bar 0.574)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
