"""Phase E2: do learned fret keypoints beat calibrate.py's consensus fit?

The pre-registered E2 go bar (design 2026-07-27 §8) is *"keypoint-derived fret
registration beats ``calibrate.py``'s consensus fit on wire-sparse clips"*. This
is that comparison, and it is deliberately a **one-variable swap**:

* the cached homography is reused as-is, exactly as ``make_fret_xs_calibrator``
  does — the keypoint arm never re-fits it;
* the rule-of-18 fit (:func:`fit_fret_map`), the nut anchoring, the canonical-x
  window and ``_MIN_WIRES`` are the *same code* in both arms;
* only the **source of the fret-wire positions** changes: YOLO-OBB box centres
  versus the pose model's predicted wire/string intersections.

So a difference here is attributable to the wire evidence, not to a different
geometry pipeline.

Three arms, all on the Phase A 720p crop cache over clean-12:

===========  ==================================================================
``uniform``  no calibration — the pre-WS1 uniform fret partition (the control)
``obb``      ``calibrate_fret_xs`` on the cached OBB detections (current default)
``keypoint`` the same fit, sourced from the E2 pose model's keypoints
===========  ==================================================================

**Primary metric: ambiguous-note string accuracy** (best orientation) — the
banked WS1/Phase A leading indicator (0.543 uniform -> 0.574 -> 0.720). Phase A's
whole lesson is that fit *rate* can move without the downstream metric moving,
so registration quality alone is reported as a diagnostic, never as the verdict.

**Pre-registered reading of the bar** (fixed before the numbers were seen):

1. wire-sparse clips are those where the ``obb`` arm's calibration fires on
   **< 0.50** of usable frames — i.e. the current fit fails more often than it
   succeeds. Chosen as a threshold, not a quantile, so it cannot be tuned to the
   result. If no clip qualifies, that is reported and the three lowest-firing
   clips are shown as an explicitly **post-hoc** secondary view.
2. E2 **passes** iff ``keypoint`` > ``obb`` on ambiguous-note string accuracy
   over the wire-sparse subset, **and** does not regress on clean-12 overall.
3. Beating ``uniform`` is necessary but not sufficient — that only repeats WS1.

Reproduce::

    cd tabvision
    python -m scripts.eval.e2_keypoint_cache        # build the keypoint cache
    python -m scripts.eval.e2_fret_registration_ab
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.e2_keypoint_cache import FretKeypointFrame, kpt_cache_path
from scripts.eval.gaps_cv_cache import RawFrameCV, fingering_from_raw, rawcv_cache_path
from scripts.eval.v1_1_gaps_string_diag import ClipStringDiag, diagnose_clip_strings
from tabvision.demux import _probe_metadata
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.types import GuitarConfig, Homography
from tabvision.video.fretboard.calibrate import (
    _MIN_WIRES,
    calibrate_fret_xs,
    fit_fret_map,
    nut_at_high_canonical_x,
    project_to_canonical,
)

DEFAULT_VIDEO_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video_720"
DEFAULT_CV_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video_chain_720"
DEFAULT_KPT_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_fret_keypoints_720"
WIRE_SPARSE_FIRE_SHARE = 0.50  # pre-registered; see the module docstring
ARMS = ("uniform", "obb", "keypoint")


def _instance_centers(kpts: np.ndarray, *, min_vis: float) -> np.ndarray:
    """Mean (x, y) of each instance's visible keypoints; shape ``(n, 2)``.

    Keypoints below ``min_vis`` are dropped before averaging, so a wire that is
    half-occluded is centred on the visible half rather than dragged toward a
    hallucinated point. Instances with no visible keypoint are dropped.
    """
    if kpts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    xy = kpts[:, :, :2].astype(np.float64)
    vis = kpts[:, :, 2] >= min_vis
    out = []
    for i in range(xy.shape[0]):
        if not vis[i].any():
            continue
        out.append(xy[i][vis[i]].mean(axis=0))
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, 2), dtype=np.float64)


def _dedupe_wires(wire_x: np.ndarray, *, frac: float = 0.5) -> np.ndarray:
    """Merge wires closer than ``frac`` x the median adjacent gap.

    The OBB arm's wires arrive already deduped — Phase A's crop-then-detect pass
    merges full-frame and crop detections "by center distance < half the local
    fret pitch" before they ever reach ``calibrate_fret_xs``. The keypoint arm
    has no such upstream step, and measurement showed the asymmetry was real: the
    median *minimum* adjacent gap was ~0.003 canonical for keypoints against
    ~0.024 for OBB, with a median of one clustered pair per frame versus zero.
    Near-duplicate wires break ``fit_fret_map``'s geometric-sequence consensus,
    so without this the arms are not comparable.

    The median gap is used as the scale because it is robust to the duplicates
    being removed.
    """
    if wire_x.size < 2:
        return wire_x
    gaps = np.diff(wire_x)
    med = float(np.median(gaps))
    if not np.isfinite(med) or med <= 0.0:
        return wire_x
    tol = frac * med
    clusters: list[list[float]] = [[float(wire_x[0])]]
    for x in wire_x[1:]:
        if float(x) - clusters[-1][-1] < tol:
            clusters[-1].append(float(x))
        else:
            clusters.append([float(x)])
    return np.array([float(np.mean(c)) for c in clusters], dtype=np.float64)


def calibrate_fret_xs_from_keypoints(
    kpt: FretKeypointFrame | None,
    homography: Homography,
    cfg: GuitarConfig,
    *,
    min_vis: float = 0.5,
    dedupe: bool = True,
) -> np.ndarray | None:
    """``calibrate_fret_xs``, but with wire positions from learned keypoints.

    Mirrors :func:`calibrate_fret_xs` step for step — same canonical window, same
    ``_MIN_WIRES`` floor, same nut-side decision, same :func:`fit_fret_map`. The
    only change is that a wire's position is the centroid of its six predicted
    string intersections instead of an OBB box centre.
    """
    if kpt is None or homography.confidence <= 0.0:
        return None
    wire_centers = _instance_centers(kpt.fret_kpts, min_vis=min_vis)
    if wire_centers.shape[0] == 0:
        return None

    wire_canon = project_to_canonical(homography, wire_centers)
    wire_x = wire_canon[:, 0]
    keep = (wire_x > -0.15) & (wire_x < 1.25)
    wire_x = np.sort(wire_x[keep])
    if dedupe:
        wire_x = _dedupe_wires(wire_x)
    if wire_x.size < _MIN_WIRES:
        return None

    nut_high = nut_at_high_canonical_x(wire_x)
    x0 = 1.0 if nut_high else 0.0
    nut_centers = _instance_centers(kpt.nut_kpts, min_vis=min_vis)
    if nut_centers.shape[0] > 0:
        # Highest-confidence nut instance, matching OBBPredictions.best_nut().
        best = int(np.argmax(kpt.nut_conf[: nut_centers.shape[0]]))
        nut_canon = float(project_to_canonical(homography, nut_centers[best : best + 1])[0, 0])
        if nut_high and nut_canon >= wire_x[-1]:
            x0 = nut_canon
        elif not nut_high and nut_canon <= wire_x[0]:
            x0 = nut_canon

    wires_from_nut = wire_x[::-1] if nut_high else wire_x
    return fit_fret_map(wires_from_nut, x0, cfg.max_fret)


@dataclass
class ArmResult:
    diag: ClipStringDiag
    fire: int  # frames where this arm produced a fret map
    usable: int  # frames with a usable CV record


def _run_clip(
    stem: str,
    *,
    data_root: Path,
    video_cache: Path,
    cv_cache: Path,
    kpt_cache: Path,
    cfg: GuitarConfig,
    conf: float,
    det_conf: float,
    cache_suffix: str,
    window_s: float,
    max_frames: int,
    arms: tuple[str, ...] = ARMS,
) -> dict[str, ArmResult] | None:
    """Run the requested arms over one clip's cached frames.

    ``arms`` lets a caller that only needs the geometry arms skip the keypoint
    cache entirely — the wire-sparse gate experiment compares ``uniform`` against
    ``obb`` and must not depend on E2's model being trained.
    """
    gaps = data_root / "gaps"
    xml = gaps / "musicxml" / f"{stem}.xml"
    vid = video_cache / f"{stem}.mp4"
    offset_pkl = cv_cache / f"{stem}.offset.pkl"
    rich = rawcv_cache_path(cv_cache, stem, conf, suffix=cache_suffix)
    kpt_path = kpt_cache_path(kpt_cache, stem, det_conf)
    needed = [
        ("musicxml", xml),
        ("video", vid),
        ("offset", offset_pkl),
        ("cv cache", rich),
    ]
    if "keypoint" in arms:
        needed.append(("keypoint cache", kpt_path))
    for label, path in needed:
        if not path.exists():
            print(f"  [skip] {stem}: missing {label} ({path.name})")
            return None

    gold = parse_gaps(xml)
    if not gold:
        print(f"  [skip] {stem}: empty gold")
        return None
    with open(offset_pkl, "rb") as fh:
        offset_s = float(pickle.load(fh).offset_s)
    _dur, fps = _probe_metadata(vid)
    with open(rich, "rb") as fh:
        raw: dict[int, RawFrameCV | None] = pickle.load(fh)
    kpts: dict[int, FretKeypointFrame | None] = {}
    if "keypoint" in arms:
        with open(kpt_path, "rb") as fh:
            kpts = pickle.load(fh)

    out: dict[str, ArmResult] = {}
    for arm in arms:
        per_frame = {}
        fire = 0
        usable = 0
        for fi, rec in raw.items():
            if rec is None:
                per_frame[fi] = None
                continue
            usable += 1
            if arm == "uniform":
                calibrate = None
            else:

                def calibrate(  # noqa: ANN001 - local CalibrateFn, closes over fi/arm
                    r: RawFrameCV, _fi: int = fi, _arm: str = arm
                ) -> tuple[Homography, np.ndarray | None]:
                    if _arm == "obb":
                        xs = calibrate_fret_xs(r.preds, r.homography, cfg)
                    else:
                        xs = calibrate_fret_xs_from_keypoints(kpts.get(_fi), r.homography, cfg)
                    return r.homography, xs

                if arm == "obb":
                    fired = calibrate_fret_xs(rec.preds, rec.homography, cfg) is not None
                else:
                    fired = (
                        calibrate_fret_xs_from_keypoints(kpts.get(fi), rec.homography, cfg)
                        is not None
                    )
                fire += int(fired)
            per_frame[fi] = fingering_from_raw(rec, cfg, t=fi / fps, calibrate=calibrate)

        diag = diagnose_clip_strings(
            gold,
            per_frame,
            offset_s,
            fps,
            cfg,
            window_s=window_s,
            max_frames=max_frames,
            stem=stem,
        )
        out[arm] = ArmResult(diag=diag, fire=fire, usable=usable)
    return out


def _agg(results: list[dict[str, ArmResult]], arm: str) -> tuple[int, int, float]:
    """(correct, have_cv, micro accuracy) pooled over clips for one arm."""
    correct = sum(r[arm].diag.correct for r in results)
    have = sum(r[arm].diag.have_cv for r in results)
    return correct, have, (correct / have if have else 0.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision" / "data")
    ap.add_argument("--video-cache", type=Path, default=DEFAULT_VIDEO_CACHE)
    ap.add_argument("--cv-cache", type=Path, default=DEFAULT_CV_CACHE)
    ap.add_argument("--kpt-cache", type=Path, default=DEFAULT_KPT_CACHE)
    ap.add_argument("--clips", default="clean12")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument(
        "--det-conf",
        type=float,
        default=0.10,
        help="keypoint cache floor to read; must match a built cache. 0.10 is "
        "the floor Phase A's OBB fret pass used, so both arms see the same one.",
    )
    ap.add_argument("--cache-suffix", default=".crop")
    ap.add_argument("--window-s", type=float, default=0.06)
    ap.add_argument("--vote-frames", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = GuitarConfig()
    stems = (
        CLEAN_12
        if args.clips == "clean12"
        else tuple(s.strip() for s in args.clips.split(",") if s.strip())
    )

    results: list[dict[str, ArmResult]] = []
    print(f"E2 fret-registration A/B — {len(stems)} clips, arms={ARMS}\n")
    for stem in stems:
        res = _run_clip(
            stem,
            data_root=args.data_root,
            video_cache=args.video_cache,
            cv_cache=args.cv_cache,
            kpt_cache=args.kpt_cache,
            cfg=cfg,
            conf=args.conf,
            det_conf=args.det_conf,
            cache_suffix=args.cache_suffix,
            window_s=args.window_s,
            max_frames=args.vote_frames,
        )
        if res is None:
            continue
        results.append(res)
        obb, kpt = res["obb"], res["keypoint"]
        obb_fire = obb.fire / obb.usable if obb.usable else 0.0
        kpt_fire = kpt.fire / kpt.usable if kpt.usable else 0.0
        print(
            f"  {stem}: amb={obb.diag.have_cv:4d} | "
            f"str_acc uniform={res['uniform'].diag.str_acc:.3f} "
            f"obb={obb.diag.str_acc:.3f} kpt={kpt.diag.str_acc:.3f} | "
            f"fire obb={obb_fire:.3f} kpt={kpt_fire:.3f}"
        )

    if not results:
        print("no clips ran — build the keypoint cache first")
        return 2

    print(f"\n{'=' * 74}\nPOOLED over {len(results)} clips (micro, ambiguous notes)")
    for arm in ARMS:
        c, h, acc = _agg(results, arm)
        print(f"  {arm:<9} {acc:.4f}  ({c}/{h})")

    sparse = [
        r
        for r in results
        if (r["obb"].fire / r["obb"].usable if r["obb"].usable else 0.0) < WIRE_SPARSE_FIRE_SHARE
    ]
    print(f"\nWIRE-SPARSE subset (obb calibration fires < {WIRE_SPARSE_FIRE_SHARE:.2f} of frames)")
    if sparse:
        print(f"  {len(sparse)} clips: {', '.join(r['obb'].diag.stem for r in sparse)}")
        for arm in ARMS:
            c, h, acc = _agg(sparse, arm)
            print(f"  {arm:<9} {acc:.4f}  ({c}/{h})")
        _, _, obb_acc = _agg(sparse, "obb")
        _, _, kpt_acc = _agg(sparse, "keypoint")
        _, _, all_obb = _agg(results, "obb")
        _, _, all_kpt = _agg(results, "keypoint")
        passed = kpt_acc > obb_acc and all_kpt >= all_obb
        print(
            f"\n  GO BAR: keypoint > obb on wire-sparse ({kpt_acc:.4f} vs {obb_acc:.4f}) "
            f"AND no overall regression ({all_kpt:.4f} vs {all_obb:.4f})"
            f"\n  VERDICT: {'PASS' if passed else 'FAIL'}"
        )
    else:
        print(
            f"  none — no clip's obb calibration fires below {WIRE_SPARSE_FIRE_SHARE:.2f}.\n"
            "  The pre-registered subset is empty, so the bar as written is not\n"
            "  testable on this cache. Showing the 3 lowest-firing clips as an\n"
            "  explicitly POST-HOC secondary view (not the pre-registered bar):"
        )
        ranked = sorted(results, key=lambda r: r["obb"].fire / max(1, r["obb"].usable))[:3]
        print(f"  {', '.join(r['obb'].diag.stem for r in ranked)}")
        for arm in ARMS:
            c, h, acc = _agg(ranked, arm)
            print(f"    {arm:<9} {acc:.4f}  ({c}/{h})")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
