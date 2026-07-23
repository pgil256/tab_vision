"""Accuracy-loop N3 — do the physics channel's signals help flag review notes?

The Phase 6 review ranker (`string_assignment_phase6`) surfaces likely
wrong-position notes for a human; it scored AUC 0.7127 (below its 0.75 gate)
and 38.76% wrong-reduction @60 s. It never had the two signals the Q6
inharmonicity channel produces per note it measures:

- **fit confidence** (``r2``), and
- **a physics posterior over the candidate strings**, from which the top1-top2
  margin and — most actionably — *whether physics disagrees with the decoder*
  fall out.

Probe-before-build: rather than retrain the MLP, this measures whether those
signals carry wrong-position information the decoder margin does not. Two
questions:

1. **P(decoder wrong | physics disagrees)** vs the 0.3452 base rate — is a
   physics-vs-decoder contradiction a strong "check this" flag?
2. **AUC** of decoder margin alone vs decoder margin + physics, as
   wrong-position detectors, on the notes physics fires on.

Labels and decoder features come from the banked Phase 0 lattice; physics
features are measured from GuitarSet audio (partial-aware isolation, the N1
higher-coverage mode). Reported as the assisted metric, separate from
automatic Tab F1. Nothing here changes the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.n2_muscriptor_merge import DEV_PLAYERS, _event_from_json
from tabvision.eval.guitarset_audio import load_mono_audio
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.inharmonicity import (
    _harmonic_frequencies,
    _isolated_flags,
    _overlapping,
    estimate_inharmonicity,
    inharmonicity_matrix,
)
from tabvision.fusion.string_physics import reference_stiffness_model
from tabvision.types import GuitarConfig

MIN_R2 = 0.50
SKIP_ATTACK_S = 0.030
MIN_WINDOW_S = 0.120
MAX_WINDOW_S = 0.400
SEPARATION_FACTOR = 3.0


def _decoder_margin(candidate_path: str) -> float:
    """Cost gap between the decoder's rank-1 and rank-2 candidate."""
    parts = [item for item in candidate_path.split(";") if item]
    if len(parts) < 2:
        return 10.0  # unambiguous-ish; a large margin
    return float(parts[1].split(":")[2])


def load_ambiguous(csv_path: Path) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    """Ambiguous dev-OOF notes keyed by track -> (onset_ms, pitch)."""
    out: dict[str, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "production_equivalent":
                continue
            if row["evaluation_split"] != "development_oof":
                continue
            if row["ambiguous_pitch_match"] != "1" or not row["reference_string"]:
                continue
            key = (int(round(float(row["onset_s"]) * 1000)), int(row["pitch_midi"]))
            out[row["track_id"]][key] = {
                "predicted_string": int(row["predicted_string"]),
                "reference_string": int(row["reference_string"]),
                "wrong": row["reference_rank"] != "1",
                "decoder_margin": _decoder_margin(row["candidate_path"]),
            }
    return out


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney rank statistic, tie-corrected.

    Uses *midranks* (average rank within a tied group) so that tied scores
    contribute 0.5 rather than an order-dependent 0 or 1 — decoder margins in
    particular have exact ties.
    """
    from scipy.stats import rankdata

    ranks = rankdata(scores, method="average")
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", type=Path, default=None)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--clips", type=int, default=0)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    lattice = args.lattice or (
        repo / "docs" / "EVAL_REPORTS" / "string_assignment_phase0_2026-07-15_notes.csv"
    )
    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    cache_dir = args.cache_dir or (data_root / "models" / "q6_full_dev_cache")

    cfg = GuitarConfig()
    model = reference_stiffness_model()
    ambiguous = load_ambiguous(lattice)
    tracks = sorted(t for t in ambiguous if t[:2] in DEV_PLAYERS)
    if args.clips:
        tracks = tracks[: args.clips]

    rows: list[dict[str, Any]] = []
    for ti, track_id in enumerate(tracks, 1):
        cache = cache_dir / f"{track_id}.ensemble.json"
        if not cache.is_file():
            continue
        events = sorted(
            (_event_from_json(x) for x in json.loads(cache.read_text("utf-8"))),
            key=lambda e: e.onset_s,
        )
        wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
        audio = np.asarray(wav, dtype=np.float64)
        isolated = _isolated_flags(events)
        notes = ambiguous[track_id]

        for index, (event, is_iso) in enumerate(zip(events, isolated, strict=True)):
            key = (int(round(event.onset_s * 1000)), event.pitch_midi)
            note = notes.get(key)
            if note is None:
                continue
            duration = event.offset_s - event.onset_s
            record = {**note, "physics_fired": False}
            if duration >= MIN_WINDOW_S + SKIP_ATTACK_S:
                start = int((event.onset_s + SKIP_ATTACK_S) * sr)
                stop = start + int(min(MAX_WINDOW_S, duration - SKIP_ATTACK_S) * sr)
                if 0 <= start and stop <= audio.size:
                    blocked: list[float] = []
                    separation = 0.0
                    if not is_iso:
                        separation = SEPARATION_FACTOR / max((stop - start) / sr, 1e-6)
                        for other in _overlapping(
                            events,
                            index,
                            event.onset_s + SKIP_ATTACK_S,
                            event.onset_s + duration,
                        ):
                            blocked.extend(_harmonic_frequencies(other.pitch_midi))
                    nominal = 440.0 * 2 ** ((event.pitch_midi - 69) / 12.0)
                    fit = estimate_inharmonicity(
                        audio[start:stop],
                        int(sr),
                        nominal,
                        blocked_hz=blocked,
                        min_separation_hz=separation,
                    )
                    if fit is not None and fit.r2 >= MIN_R2:
                        matrix = inharmonicity_matrix(event.pitch_midi, cfg, fit.log_b, model)
                        if matrix is not None:
                            cands = candidate_positions(event.pitch_midi, cfg)
                            probs = sorted(
                                ((matrix[c.string_idx, c.fret], c.string_idx) for c in cands),
                                reverse=True,
                            )
                            phys_top1 = probs[0][1]
                            margin = probs[0][0] - (probs[1][0] if len(probs) > 1 else 0.0)
                            prob_dec = 0.0
                            for c in cands:
                                if c.string_idx == note["predicted_string"]:
                                    prob_dec = float(matrix[c.string_idx, c.fret])
                                    break
                            record.update(
                                physics_fired=True,
                                r2=fit.r2,
                                isolated=bool(is_iso),
                                physics_top1=phys_top1,
                                physics_margin=margin,
                                physics_prob_decoder=prob_dec,
                                physics_disagrees=phys_top1 != note["predicted_string"],
                                physics_correct=phys_top1 == note["reference_string"],
                            )
            rows.append(record)
        if ti % 25 == 0:
            print(f"  [{ti}/{len(tracks)}] {len(rows)} notes", flush=True)

    n = len(rows)
    fired = [r for r in rows if r["physics_fired"]]
    base_wrong = sum(r["wrong"] for r in rows) / n
    disagree = [r for r in fired if r["physics_disagrees"]]
    agree = [r for r in fired if not r["physics_disagrees"]]

    summary: dict[str, Any] = {
        "notes": n,
        "physics_fired": len(fired),
        "fired_share": len(fired) / n,
        "base_wrong_rate": base_wrong,
        "P_wrong_given_physics_disagrees": (
            sum(r["wrong"] for r in disagree) / len(disagree) if disagree else float("nan")
        ),
        "P_wrong_given_physics_agrees": (
            sum(r["wrong"] for r in agree) / len(agree) if agree else float("nan")
        ),
        "disagree_count": len(disagree),
        "agree_count": len(agree),
        "physics_string_accuracy_on_fired": (
            sum(r["physics_correct"] for r in fired) / len(fired) if fired else float("nan")
        ),
    }

    iso = [r for r in fired if r["isolated"]]
    con = [r for r in fired if not r["isolated"]]
    summary["physics_acc_isolated"] = (
        sum(r["physics_correct"] for r in iso) / len(iso) if iso else float("nan")
    )
    summary["physics_acc_contaminated"] = (
        sum(r["physics_correct"] for r in con) / len(con) if con else float("nan")
    )
    summary["isolated_share_of_fired"] = len(iso) / len(fired) if fired else float("nan")

    # AUC comparison on the fired subset: decoder margin vs continuous physics.
    if fired:
        labels = np.asarray([1 if r["wrong"] else 0 for r in fired], dtype=np.float64)
        dec = np.asarray([-r["decoder_margin"] for r in fired])  # small margin -> wrong
        # Continuous: low physics support for the decoder's string -> wrong.
        phys_score = np.asarray([1.0 - r["physics_prob_decoder"] for r in fired])

        # Simple equal-weight blend of the two z-scored signals.
        def _z(x: np.ndarray) -> np.ndarray:
            sd = x.std()
            return (x - x.mean()) / sd if sd > 0 else x * 0.0

        combined = _z(dec) + _z(phys_score)
        summary["auc_fired"] = {
            "decoder_margin": _auc(dec, labels),
            "physics_prob_decoder": _auc(phys_score, labels),
            "combined": _auc(combined, labels),
        }
        # Same, restricted to genuinely isolated notes (strict), where the
        # physics measurement is trustworthy per Q6.
        if iso:
            ilab = np.asarray([1 if r["wrong"] else 0 for r in iso], dtype=np.float64)
            idec = np.asarray([-r["decoder_margin"] for r in iso])
            iph = np.asarray([1.0 - r["physics_prob_decoder"] for r in iso])
            summary["auc_isolated"] = {
                "decoder_margin": _auc(idec, ilab),
                "physics_prob_decoder": _auc(iph, ilab),
                "combined": _auc(_z(idec) + _z(iph), ilab),
                "n": len(iso),
            }
    # AUC of decoder margin over ALL ambiguous notes, for reference.
    all_labels = np.asarray([1 if r["wrong"] else 0 for r in rows], dtype=np.float64)
    all_dec = np.asarray([-r["decoder_margin"] for r in rows])
    summary["auc_all_decoder_margin"] = _auc(all_dec, all_labels)

    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nnotes={n} physics fired={len(fired)} ({len(fired) / n:.1%})")
    print(f"base wrong rate = {base_wrong:.4f}")
    print(
        f"P(wrong | physics disagrees) = {summary['P_wrong_given_physics_disagrees']:.4f} "
        f"(n={len(disagree)})"
    )
    print(
        f"P(wrong | physics agrees)    = {summary['P_wrong_given_physics_agrees']:.4f} "
        f"(n={len(agree)})"
    )
    print(f"physics string acc on fired  = {summary['physics_string_accuracy_on_fired']:.4f}")
    print(
        f"physics acc: isolated={summary['physics_acc_isolated']:.4f} "
        f"contaminated={summary['physics_acc_contaminated']:.4f} "
        f"(iso share {summary['isolated_share_of_fired']:.1%})"
    )
    if fired:
        a = summary["auc_fired"]
        print("\nAUC (fired subset, wrong-position detection):")
        print(f"  decoder margin only    : {a['decoder_margin']:.4f}")
        print(f"  physics prob (decoder) : {a['physics_prob_decoder']:.4f}")
        print(f"  combined               : {a['combined']:.4f}")
        if "auc_isolated" in summary:
            i = summary["auc_isolated"]
            print(
                f"\nAUC (isolated only, n={i['n']}): decoder={i['decoder_margin']:.4f} "
                f"physics={i['physics_prob_decoder']:.4f} combined={i['combined']:.4f}"
            )
    print(f"AUC decoder margin, all ambiguous: {summary['auc_all_decoder_margin']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
