"""Accuracy-loop Q2 (ROI deep-dive §3.2) — offline lattice rescoring gate.

Replays Phase 0's banked ambiguous-note lattice and re-ranks each note's
candidate positions with a pluggable scorer, without audio, a backend, or
the pipeline. The measured quantity is the program KPI for assignment work:
**ambiguous top-1**.

Gate (see `s1b_entry_substrate_2026-07-22.md` §2): the deep-dive's stated
0.6770 is the sealed `held_out_05` slice, so development runs against
`production_equivalent`/`development_oof` — baseline **0.6548**, n = 35,959,
target **0.7048** (+0.05).

Scoring mirrors the intended integration rather than replacing the decoder:

    combined_cost = cost_delta_from_best + lambda * (-log p_scorer(string))

``lambda = 0`` reproduces the decoder's own ranking exactly (the baseline
regression test below), and larger values hand more authority to the scorer.
That is the same shape as an emission-cost term in ``fuse()``, so a lambda
that wins here is directly portable to Q3's integration.

Scorers:

- ``decoder`` — uniform, i.e. the banked ranking. Reproduction check.
- ``marginal`` — P(string | pitch) counts from the SynthTab symbolic corpus.
  Context-free by construction: this is the S1a hypothesis (counts at scale)
  measured on the ambiguous slice specifically, and it is the control the
  contextual model has to beat to claim that *context* is what mattered.
- ``context`` — a trained sequence model (``s1b_train_context.py``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

BASELINE_CONDITION = "production_equivalent"
DEV_SPLIT = "development_oof"
HELD_OUT_SPLIT = "held_out_05"
DEV_BASELINE_TOP1 = 0.6548
GATE_DELTA = 0.05
OPEN_MIDI = (40, 45, 50, 55, 59, 64)
DEFAULT_LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9)


@dataclass
class LatticeNote:
    """One decoded note from the banked lattice."""

    track_id: str
    mode: str
    event_index: int
    cluster_index: int
    onset_s: float
    pitch_midi: int
    candidates: tuple[tuple[int, int, float], ...]
    """(string_idx, fret, cost_delta_from_best), in the decoder's rank order."""

    gold_string: int | None
    gold_fret: int | None
    ambiguous: bool


@dataclass
class Track:
    track_id: str
    mode: str
    notes: list[LatticeNote] = field(default_factory=list)


class Scorer(Protocol):
    """Returns per-note log-probabilities over the six strings."""

    def log_probs(self, track: Track) -> np.ndarray:
        """Shape ``(len(track.notes), 6)``, rows normalized in log space."""


class UniformScorer:
    """No opinion — combined cost reduces to the decoder's own ranking."""

    name = "decoder"

    def log_probs(self, track: Track) -> np.ndarray:
        return np.full((len(track.notes), 6), -math.log(6.0), dtype=np.float64)


class MarginalScorer:
    """P(string | pitch) from SynthTab counts — the context-free control."""

    name = "marginal"

    def __init__(self, corpus_path: Path, *, alpha: float = 1.0) -> None:
        with np.load(corpus_path) as payload:
            pitch = payload["pitch"].astype(np.int64)
            string = payload["string"].astype(np.int64)
        self.low = int(pitch.min())
        counts = np.zeros((int(pitch.max()) - self.low + 1, 6), dtype=np.float64)
        np.add.at(counts, (pitch - self.low, string), 1.0)
        counts += alpha
        self.table = np.log(counts / counts.sum(axis=1, keepdims=True))

    def log_probs(self, track: Track) -> np.ndarray:
        rows = np.empty((len(track.notes), 6), dtype=np.float64)
        for index, note in enumerate(track.notes):
            offset = note.pitch_midi - self.low
            if 0 <= offset < self.table.shape[0]:
                rows[index] = self.table[offset]
            else:
                rows[index] = -math.log(6.0)
        return rows


class OutOfFoldContextScorer:
    """Routes each track to the fine-tune fold that never saw its player.

    Stage 2 fine-tunes on GuitarSet players 00-04 — the same players the
    lattice is drawn from — so scoring a track with a model that trained on
    its player would be measuring memorization. This picks the fold keyed by
    the track's own player, which is exactly the one held out.
    """

    name = "context-oof"

    def __init__(self, checkpoint_dir: Path) -> None:
        from scripts.eval.s1b_train_context import load_context_scorer

        self.folds = {}
        for path in sorted(Path(checkpoint_dir).glob("context_v2_oof_*.pt")):
            player = path.stem.rsplit("_", 1)[1]
            self.folds[player] = load_context_scorer(path)
        if not self.folds:
            raise SystemExit(f"no context_v2_oof_*.pt checkpoints under {checkpoint_dir}")

    def log_probs(self, track: Track) -> np.ndarray:
        player = track.track_id[:2]
        fold = self.folds.get(player)
        if fold is None:
            raise SystemExit(f"no held-out fold for player {player!r} (track {track.track_id})")
        return fold.log_probs(track)


def load_lattice(
    csv_path: Path,
    *,
    condition: str = BASELINE_CONDITION,
    split: str = DEV_SPLIT,
) -> list[Track]:
    """Rebuild per-track decoded note sequences from the banked CSV.

    Every row for the condition is kept, not just the ambiguous ones: a
    contextual scorer needs the surrounding notes, and the note stream here
    is the *predicted* one, exactly what a model would see at inference.
    """
    by_track: dict[str, Track] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != condition or row["evaluation_split"] != split:
                continue
            track = by_track.get(row["track_id"])
            if track is None:
                track = Track(track_id=row["track_id"], mode=row["mode"])
                by_track[row["track_id"]] = track
            candidates = tuple(
                (int(parts[0]), int(parts[1]), float(parts[2]))
                for parts in (item.split(":") for item in row["candidate_path"].split(";") if item)
            )
            reference_string = row["reference_string"]
            reference_fret = row["reference_fret"]
            track.notes.append(
                LatticeNote(
                    track_id=row["track_id"],
                    mode=row["mode"],
                    event_index=int(row["event_index"]),
                    cluster_index=int(row["cluster_index"]),
                    onset_s=float(row["onset_s"]),
                    pitch_midi=int(row["pitch_midi"]),
                    candidates=candidates,
                    gold_string=int(reference_string) if reference_string else None,
                    gold_fret=int(reference_fret) if reference_fret else None,
                    ambiguous=row["ambiguous_pitch_match"] == "1",
                )
            )
    for track in by_track.values():
        track.notes.sort(key=lambda note: note.event_index)
    return [by_track[key] for key in sorted(by_track)]


def rescore(
    tracks: Sequence[Track],
    scorer: Scorer,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> dict[str, Any]:
    """Re-rank every ambiguous note at each lambda and score top-1."""
    totals = {value: {"n": 0, "top1": 0, "solo_n": 0, "solo_top1": 0} for value in lambdas}
    for value in lambdas:
        totals[value].update({"comp_n": 0, "comp_top1": 0, "rank2_n": 0, "rank2_flipped": 0})
    # Per-track hit counts, so the delta vs the decoder can be bootstrapped
    # over tracks (the resampling unit — notes within a track are not
    # independent, they share a player, tune and hand position).
    per_track: dict[float, list[tuple[int, int]]] = {value: [] for value in lambdas}

    for track in tracks:
        log_probs = scorer.log_probs(track)
        track_hits = {value: [0, 0] for value in lambdas}
        for index, note in enumerate(track.notes):
            if not note.ambiguous or note.gold_string is None:
                continue
            gold = (note.gold_string, note.gold_fret)
            # Rank of gold under the decoder, for the rank-2 flip diagnostic.
            baseline_rank = next(
                (
                    position
                    for position, candidate in enumerate(note.candidates, start=1)
                    if (candidate[0], candidate[1]) == gold
                ),
                None,
            )
            penalties = -log_probs[index]
            for value in lambdas:
                # Stable argmin: ties keep the decoder's original order, so
                # lambda = 0 reproduces the banked ranking exactly.
                best_index = 0
                best_cost = math.inf
                for position, (string_idx, _fret, cost_delta) in enumerate(note.candidates):
                    cost = (
                        cost_delta + value * penalties[string_idx]
                        if value < 1e8
                        else penalties[string_idx]
                    )
                    if cost < best_cost - 1e-12:
                        best_cost = cost
                        best_index = position
                chosen = note.candidates[best_index]
                hit = int((chosen[0], chosen[1]) == gold)
                bucket = totals[value]
                bucket["n"] += 1
                bucket["top1"] += hit
                if note.mode == "solo":
                    bucket["solo_n"] += 1
                    bucket["solo_top1"] += hit
                else:
                    bucket["comp_n"] += 1
                    bucket["comp_top1"] += hit
                if baseline_rank == 2:
                    bucket["rank2_n"] += 1
                    bucket["rank2_flipped"] += hit
                track_hits[value][0] += hit
                track_hits[value][1] += 1
        for value in lambdas:
            if track_hits[value][1]:
                per_track[value].append((track_hits[value][0], track_hits[value][1]))

    rows = []
    for value in lambdas:
        bucket = totals[value]
        rows.append(
            {
                "lambda": value,
                "ambiguous_notes": bucket["n"],
                "top1": bucket["top1"] / bucket["n"] if bucket["n"] else float("nan"),
                "solo_top1": (
                    bucket["solo_top1"] / bucket["solo_n"] if bucket["solo_n"] else float("nan")
                ),
                "comp_top1": (
                    bucket["comp_top1"] / bucket["comp_n"] if bucket["comp_n"] else float("nan")
                ),
                "rank2_flip_rate": (
                    bucket["rank2_flipped"] / bucket["rank2_n"]
                    if bucket["rank2_n"]
                    else float("nan")
                ),
                "rank2_notes": bucket["rank2_n"],
            }
        )
    return {
        "scorer": getattr(scorer, "name", type(scorer).__name__),
        "sweep": rows,
        "per_track": {str(value): per_track[value] for value in lambdas},
    }


def bootstrap_delta(
    per_track: Sequence[tuple[int, int]],
    baseline_per_track: Sequence[tuple[int, int]],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap over tracks of pooled ambiguous top-1 delta.

    Tracks are the resampling unit: notes inside one track share a player,
    tune and hand position, so treating them as independent would understate
    the interval.
    """
    hits = np.asarray([row[0] for row in per_track], dtype=np.float64)
    counts = np.asarray([row[1] for row in per_track], dtype=np.float64)
    base_hits = np.asarray([row[0] for row in baseline_per_track], dtype=np.float64)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(hits), size=(n_bootstrap, len(hits)))
    pooled = hits[index].sum(axis=1) / counts[index].sum(axis=1)
    pooled_base = base_hits[index].sum(axis=1) / counts[index].sum(axis=1)
    deltas = pooled - pooled_base
    return {
        "delta": float(hits.sum() / counts.sum() - base_hits.sum() / counts.sum()),
        "lo95": float(np.quantile(deltas, 0.025)),
        "hi95": float(np.quantile(deltas, 0.975)),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", type=Path, default=None)
    parser.add_argument(
        "--scorer",
        choices=("decoder", "marginal", "context", "context-oof"),
        default="decoder",
    )
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=(DEV_SPLIT, HELD_OUT_SPLIT), default=DEV_SPLIT)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    lattice = args.lattice or (
        repo_root / "docs" / "EVAL_REPORTS" / "string_assignment_phase0_2026-07-15_notes.csv"
    )
    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    corpus = args.corpus or (data_root / "models" / "s1b_symbolic" / "synthtab_all.npz")

    if args.split == HELD_OUT_SPLIT:
        raise SystemExit(
            "player-05 is sealed until config freeze + explicit user proceed "
            "(accuracy-loop STOP rule); develop against development_oof"
        )

    tracks = load_lattice(lattice, split=args.split)
    scorer: Scorer
    if args.scorer == "decoder":
        scorer = UniformScorer()
    elif args.scorer == "marginal":
        scorer = MarginalScorer(corpus)
    elif args.scorer == "context":
        from scripts.eval.s1b_train_context import load_context_scorer

        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --scorer context")
        scorer = load_context_scorer(args.checkpoint)
    else:
        checkpoint_dir = args.checkpoint or (data_root / "models" / "s1b_symbolic")
        scorer = OutOfFoldContextScorer(checkpoint_dir)

    summary = rescore(tracks, scorer)
    summary["split"] = args.split
    summary["tracks"] = len(tracks)
    summary["baseline_top1"] = DEV_BASELINE_TOP1
    summary["target_top1"] = round(DEV_BASELINE_TOP1 + GATE_DELTA, 4)
    best = max(summary["sweep"], key=lambda row: row["top1"])
    summary["best"] = best
    summary["best_delta"] = bootstrap_delta(
        summary["per_track"][str(best["lambda"])],
        summary["per_track"][str(0.0)],
    )
    summary["gate_pass"] = bool(best["top1"] >= DEV_BASELINE_TOP1 + GATE_DELTA)

    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"scorer={summary['scorer']} split={args.split} tracks={len(tracks)}")
    for row in summary["sweep"]:
        label = "inf" if row["lambda"] >= 1e8 else f"{row['lambda']:g}"
        print(
            f"  lambda={label:>5}: top1={row['top1']:.4f} "
            f"(solo {row['solo_top1']:.4f} / comp {row['comp_top1']:.4f}) "
            f"rank2_flip={row['rank2_flip_rate']:.4f}"
        )
    delta = summary["best_delta"]
    print(
        f"best top1={best['top1']:.4f} (lambda={best['lambda']:g}) "
        f"delta={delta['delta']:+.4f} [{delta['lo95']:+.4f}, {delta['hi95']:+.4f}] "
        f"vs target {summary['target_top1']:.4f} "
        f"→ {'PASS' if summary['gate_pass'] else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
