"""Learned fingering-sequence (transition) priors — roadmap item A15.

The hand-coded transition terms in :mod:`tabvision.fusion.playability`
(same-string bonus, position-shift cost) encode *assumptions* about how
players move between positions. This module learns those statistics from
tab-labelled corpora instead: counts over cluster-anchor-to-anchor
transitions, parameterized compactly so a few hundred tracks are enough.

Schemes (all return **negative log-probs** in nats, matching playability):

- ``delta`` — ``P(Δstring | Δpitch)``. The candidate set for a pitch
  differs only in which string carries it; conditioned on the pitch
  interval from the previous anchor, the string delta is the learnable
  convention ("small steps stay on the string, larger ascents cross").
- ``delta_fret`` — ``P(Δstring | Δpitch, prev-fret region)`` with
  count-based smoothing back to ``delta``. Captures position-dependent
  behaviour (open position favours string crossings that nut-area
  fingerings allow).

Transitions are extracted from gold ``TabEvent`` streams using the same
80 ms chord-cluster chain rule and the same anchor definition as the
decode (:func:`tabvision.fusion.chord.chord_anchor`), so the statistics
condition on exactly what the Viterbi transition sees at decode time.

Default **off** everywhere: :func:`tabvision.fusion.playability.transition_cost`
only consults a prior when one is installed (env vars
``TABVISION_TRANSITION_PRIOR`` / ``TABVISION_TRANSITION_PRIOR_WEIGHT``,
or :func:`tabvision.fusion.playability.set_transition_prior`).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tabvision.fusion.candidates import Candidate
from tabvision.types import GuitarConfig, TabEvent

_PRIORS_DIR = Path(__file__).with_name("priors")
_NAMED_PRIORS = {
    "guitarset-seq-v1": _PRIORS_DIR / "guitarset_seq_v1.json",
}

CLUSTER_GAP_S = 0.080
"""Chord-cluster chain gap — mirrors :data:`tabvision.fusion.chord.CHORD_MAX_GAP_S`
(not imported to keep this module free of the playability import cycle)."""

MAX_ABS_DELTA_PITCH = 24
"""Pitch intervals beyond ±2 octaves share one bucket per direction."""

FRET_REGION_EDGES = (1, 5, 10)
"""Prev-anchor fret regions: 0 (open), 1–4, 5–9, 10+."""

_N_STRINGS = 6
_DELTA_STRINGS = tuple(range(-(_N_STRINGS - 1), _N_STRINGS))  # -5..5


def _clip_dp(delta_pitch: int) -> int:
    return max(-MAX_ABS_DELTA_PITCH, min(MAX_ABS_DELTA_PITCH, delta_pitch))


def _fret_region(fret: int) -> int:
    for i, edge in enumerate(FRET_REGION_EDGES):
        if fret < edge:
            return i
    return len(FRET_REGION_EDGES)


@dataclass(frozen=True)
class TransitionPrior:
    """``-log P(Δstring | context)`` lookup for anchor-to-anchor transitions.

    ``delta_table`` maps clipped Δpitch → per-Δstring probabilities;
    ``delta_fret_table`` (scheme ``delta_fret`` only) maps
    ``(Δpitch, prev-fret region)`` → per-Δstring probabilities.
    Unseen contexts fall back to a uniform distribution, which adds the
    same cost to every candidate and therefore leaves rankings unchanged.
    """

    scheme: str
    delta_table: dict[int, tuple[float, ...]]
    delta_fret_table: dict[tuple[int, int], tuple[float, ...]] = field(default_factory=dict)

    def prob(self, prev: Candidate, curr: Candidate, cfg: GuitarConfig) -> float:
        dp = _clip_dp(
            (cfg.tuning_midi[curr.string_idx] + curr.fret)
            - (cfg.tuning_midi[prev.string_idx] + prev.fret)
        )
        ds = curr.string_idx - prev.string_idx
        ds_idx = ds + (_N_STRINGS - 1)
        if not 0 <= ds_idx < len(_DELTA_STRINGS):
            return 1.0 / len(_DELTA_STRINGS)

        row: tuple[float, ...] | None = None
        if self.scheme == "delta_fret":
            row = self.delta_fret_table.get((dp, _fret_region(prev.fret)))
        if row is None:
            row = self.delta_table.get(dp)
        if row is None:
            return 1.0 / len(_DELTA_STRINGS)
        return row[ds_idx]

    def cost(self, prev: Candidate, curr: Candidate, cfg: GuitarConfig) -> float:
        return -math.log(max(self.prob(prev, curr, cfg), 1e-12))


def anchor_of(cluster: Sequence[TabEvent]) -> TabEvent:
    """Anchor of a gold cluster — same rule as ``chord.chord_anchor``:
    lowest-fret pressed note, ties to the lowest string; all-open
    clusters anchor on their lowest (fret, string)."""
    pressed = [ev for ev in cluster if ev.fret > 0]
    pool = pressed if pressed else list(cluster)
    return min(pool, key=lambda ev: (ev.fret, ev.string_idx))


def extract_transitions(
    events: Sequence[TabEvent],
    *,
    cluster_gap_s: float = CLUSTER_GAP_S,
    singleton_only: bool = False,
) -> list[tuple[int, int, int]]:
    """``(Δpitch, Δstring, prev-anchor fret)`` samples from one track's gold.

    With ``singleton_only`` the sample set is restricted to
    singleton→singleton cluster moves — the exact transitions the decode
    applies the learned prior to (chord moves are gated out there).
    """
    ordered = sorted(events, key=lambda ev: (ev.onset_s, ev.string_idx, ev.fret))
    if not ordered:
        return []
    clusters: list[list[TabEvent]] = [[ordered[0]]]
    for ev in ordered[1:]:
        if ev.onset_s - clusters[-1][-1].onset_s <= cluster_gap_s:
            clusters[-1].append(ev)
        else:
            clusters.append([ev])

    out: list[tuple[int, int, int]] = []
    for prev_cluster, curr_cluster in zip(clusters, clusters[1:], strict=False):
        if singleton_only and (len(prev_cluster) > 1 or len(curr_cluster) > 1):
            continue
        prev = anchor_of(prev_cluster)
        curr = anchor_of(curr_cluster)
        out.append(
            (
                _clip_dp(curr.pitch_midi - prev.pitch_midi),
                curr.string_idx - prev.string_idx,
                prev.fret,
            )
        )
    return out


def learn_transition_prior(
    tracks: Iterable[Sequence[TabEvent]],
    *,
    scheme: str = "delta",
    alpha: float = 0.5,
    backoff_kappa: float = 8.0,
    singleton_only: bool = False,
) -> TransitionPrior:
    """Estimate a :class:`TransitionPrior` from gold tab tracks.

    ``alpha`` is the add-α smoothing mass per Δstring cell of the delta
    table; ``backoff_kappa`` is the pseudo-count mass the ``delta_fret``
    conditionals borrow from the delta backbone. ``singleton_only``
    restricts training to the singleton→singleton moves the decode
    actually applies the prior to.
    """
    samples: list[tuple[int, int, int]] = []
    for track in tracks:
        samples.extend(extract_transitions(track, singleton_only=singleton_only))
    return _learn_from_samples(
        samples,
        scheme=scheme,
        alpha=alpha,
        backoff_kappa=backoff_kappa,
    )


def load_transition_prior(name_or_path: str | Path) -> TransitionPrior:
    """Load a versioned transition-prior artifact (or a filesystem path)."""
    key = str(name_or_path)
    path = _NAMED_PRIORS.get(key)
    if path is None:
        candidate = Path(key)
        if candidate.is_file():
            path = candidate
        else:
            known = ", ".join(sorted(_NAMED_PRIORS))
            raise ValueError(f"unknown transition prior {key!r}; known: {known}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported transition-prior schema in {path}")
    counts = payload.get("counts")
    if not isinstance(counts, list):
        raise ValueError(f"transition-prior artifact missing counts: {path}")

    samples: list[tuple[int, int, int]] = []
    for row in counts:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"invalid transition-prior count row in {path}: {row!r}")
        dp, ds, prev_fret, count = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        if count < 0:
            raise ValueError(f"invalid negative transition-prior count in {path}: {row!r}")
        samples.extend((dp, ds, prev_fret) for _ in range(count))

    return _learn_from_samples(
        samples,
        scheme=str(payload.get("scheme", "delta")),
        alpha=float(payload.get("alpha", 0.5)),
        backoff_kappa=float(payload.get("backoff_kappa", 8.0)),
    )


def _learn_from_samples(
    samples: Sequence[tuple[int, int, int]],
    *,
    scheme: str,
    alpha: float,
    backoff_kappa: float,
) -> TransitionPrior:
    """Learner core shared by :func:`learn_transition_prior` and artifact loading."""
    if scheme not in ("delta", "delta_fret"):
        raise ValueError(f"unknown transition-prior scheme {scheme!r}")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    delta_counts: Counter[tuple[int, int]] = Counter()
    fret_counts: Counter[tuple[int, int, int]] = Counter()
    for dp, ds, prev_fret in samples:
        if abs(ds) >= _N_STRINGS:
            continue
        delta_counts[(_clip_dp(dp), ds)] += 1
        fret_counts[(_clip_dp(dp), _fret_region(prev_fret), ds)] += 1

    delta_table: dict[int, tuple[float, ...]] = {}
    for dp in {key[0] for key in delta_counts}:
        raw = [delta_counts.get((dp, ds), 0) + alpha for ds in _DELTA_STRINGS]
        total = sum(raw)
        delta_table[dp] = tuple(v / total for v in raw)

    delta_fret_table: dict[tuple[int, int], tuple[float, ...]] = {}
    if scheme == "delta_fret":
        for dp, region in {(key[0], key[1]) for key in fret_counts}:
            backbone = delta_table[dp]
            raw = [
                fret_counts.get((dp, region, ds), 0) + backoff_kappa * backbone[i]
                for i, ds in enumerate(_DELTA_STRINGS)
            ]
            total = sum(raw)
            delta_fret_table[(dp, region)] = tuple(v / total for v in raw)

    return TransitionPrior(
        scheme=scheme,
        delta_table=delta_table,
        delta_fret_table=delta_fret_table,
    )


__all__ = [
    "CLUSTER_GAP_S",
    "TransitionPrior",
    "anchor_of",
    "extract_transitions",
    "learn_transition_prior",
    "load_transition_prior",
]
