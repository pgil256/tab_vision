"""Pure TabCNN posterior complementarity and paired-evaluation helpers.

This module deliberately does not know how a TabCNN checkpoint is loaded or
how audio is converted to model features.  It consumes event-aligned posterior
matrices and provides the deterministic, dataset-independent part of the
experiment:

* pitch-correct one-to-one matching between the current decoder and gold;
* a full current-position vs TabCNN-position 2x2 table;
* fixed posterior-only and conservative prior-fusion policies;
* paired clip-level Tab F1 aggregation and the predeclared promotion gates.

TabCNN is never allowed to create, remove, retime, or repitch an
:class:`~tabvision.types.AudioEvent`.  :func:`attach_tabcnn_priors` uses
``dataclasses.replace`` and changes only ``fret_prior`` when the policy accepts
the posterior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from tabvision.eval.bootstrap import BootstrapResult, bootstrap_ci
from tabvision.eval.error_decomposition import (
    ErrorDecomposition,
    aggregate_decompositions,
    decompose_errors,
)
from tabvision.eval.metrics import TabF1Result, tab_f1
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.evidence import combine_candidate_evidence
from tabvision.types import AudioEvent, GuitarConfig, TabEvent

ONSET_TOLERANCE_S = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42

FusionVariant = Literal["posterior_only", "current_plus_tabcnn"]


@dataclass(frozen=True)
class TabCNNFusionPolicy:
    """A predeclared way to expose a TabCNN posterior to the current decoder.

    ``tabcnn_weight`` is the exponent used by the repository's weighted
    product-of-experts combiner.  ``include_existing`` distinguishes the
    diagnostic posterior-only arm from the promotion candidate.

    Confidence thresholds remain explicit so a future experiment cannot add
    one invisibly.  Both frozen policies set them to zero; only structural
    abstention is allowed.
    """

    name: FusionVariant
    tabcnn_weight: float
    include_existing: bool
    min_top_probability: float
    min_margin: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.tabcnn_weight) or self.tabcnn_weight <= 0.0:
            raise ValueError(f"tabcnn_weight must be finite and positive; got {self.tabcnn_weight}")
        for field_name, value in (
            ("min_top_probability", self.min_top_probability),
            ("min_margin", self.min_margin),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1]; got {value}")


POSTERIOR_ONLY_POLICY = TabCNNFusionPolicy(
    name="posterior_only",
    tabcnn_weight=1.0,
    include_existing=False,
    min_top_probability=0.0,
    min_margin=0.0,
)
"""Use every non-empty TabCNN posterior without blending."""

CONSERVATIVE_BLEND_POLICY = TabCNNFusionPolicy(
    name="current_plus_tabcnn",
    tabcnn_weight=0.35,
    include_existing=True,
    min_top_probability=0.0,
    min_margin=0.0,
)
"""Multiply current evidence by the frozen 0.35-exponent TabCNN likelihood."""


@dataclass(frozen=True)
class PositionProbability:
    """One pitch-compatible guitar position and its normalized probability."""

    string_idx: int
    fret: int
    probability: float


@dataclass(frozen=True)
class PositionChoice:
    """Top position selected deterministically from a compatible posterior."""

    string_idx: int
    fret: int
    probability: float
    margin: float


def map_prior_to_pitch_candidates(
    prior: np.ndarray,
    pitch_midi: int,
    cfg: GuitarConfig | None = None,
) -> tuple[PositionProbability, ...]:
    """Mask a ``[n_strings, max_fret+1]`` prior to playable positions.

    Returned candidates follow the canonical ``(fret, string_idx)`` order from
    :func:`tabvision.fusion.candidates.candidate_positions`.  Values are
    normalized across compatible positions.  A finite, non-negative prior with
    zero compatible mass returns an empty tuple, which means "abstain".
    """

    cfg = cfg or GuitarConfig()
    matrix = _validated_prior(prior, cfg, name="prior")
    candidates = candidate_positions(int(pitch_midi), cfg)
    if not candidates:
        return ()
    values = np.asarray(
        [matrix[candidate.string_idx, candidate.fret] for candidate in candidates],
        dtype=np.float64,
    )
    total = float(values.sum())
    if total <= 0.0:
        return ()
    values /= total
    return tuple(
        PositionProbability(
            string_idx=candidate.string_idx,
            fret=candidate.fret,
            probability=float(probability),
        )
        for candidate, probability in zip(candidates, values, strict=True)
    )


def select_tabcnn_position(
    prior: np.ndarray,
    pitch_midi: int,
    *,
    policy: TabCNNFusionPolicy = POSTERIOR_ONLY_POLICY,
    cfg: GuitarConfig | None = None,
) -> PositionChoice | None:
    """Return the accepted top compatible position, or ``None`` on abstention.

    Probability ties resolve to the first canonical candidate: lowest fret,
    then lowest string index.
    """

    compatible = map_prior_to_pitch_candidates(prior, pitch_midi, cfg)
    if not compatible:
        return None
    ranked = sorted(
        enumerate(compatible),
        key=lambda item: (-item[1].probability, item[0]),
    )
    top = ranked[0][1]
    second_probability = ranked[1][1].probability if len(ranked) > 1 else 0.0
    margin = top.probability - second_probability
    if top.probability < policy.min_top_probability or margin < policy.min_margin:
        return None
    return PositionChoice(
        string_idx=top.string_idx,
        fret=top.fret,
        probability=top.probability,
        margin=margin,
    )


def attach_tabcnn_prior(
    event: AudioEvent,
    tabcnn_prior: np.ndarray | None,
    *,
    policy: TabCNNFusionPolicy = POSTERIOR_ONLY_POLICY,
    cfg: GuitarConfig | None = None,
) -> AudioEvent:
    """Return a shallow event replacement with an accepted fused prior.

    On abstention, the returned replacement retains the exact existing
    ``fret_prior`` object.  On acceptance, all fields except ``fret_prior`` are
    shallow-preserved by :func:`dataclasses.replace`.
    """

    cfg = cfg or GuitarConfig()
    if tabcnn_prior is None:
        return replace(event)
    tabcnn_distribution = map_prior_to_pitch_candidates(
        tabcnn_prior,
        event.pitch_midi,
        cfg,
    )
    choice = select_tabcnn_position(
        tabcnn_prior,
        event.pitch_midi,
        policy=policy,
        cfg=cfg,
    )
    if choice is None:
        return replace(event)

    candidates = candidate_positions(event.pitch_midi, cfg)
    tabcnn_matrix = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
    for candidate, probability in zip(
        candidates,
        (item.probability for item in tabcnn_distribution),
        strict=True,
    ):
        tabcnn_matrix[candidate.string_idx, candidate.fret] = float(probability)

    if not policy.include_existing:
        return replace(event, fret_prior=tabcnn_matrix)

    fused = combine_candidate_evidence(
        event.pitch_midi,
        cfg,
        {
            "existing": (event.fret_prior, 1.0),
            "tabcnn": (tabcnn_matrix, policy.tabcnn_weight),
        },
    )
    if fused is None:
        return replace(event)
    return replace(event, fret_prior=fused)


def attach_tabcnn_priors(
    events: Sequence[AudioEvent],
    tabcnn_priors: Sequence[np.ndarray | None],
    *,
    policy: TabCNNFusionPolicy = POSTERIOR_ONLY_POLICY,
    cfg: GuitarConfig | None = None,
) -> tuple[AudioEvent, ...]:
    """Attach event-aligned posteriors without changing the event stream."""

    if len(events) != len(tabcnn_priors):
        raise ValueError("events and tabcnn_priors must be event-aligned and have equal length")
    return tuple(
        attach_tabcnn_prior(event, prior, policy=policy, cfg=cfg)
        for event, prior in zip(events, tabcnn_priors, strict=True)
    )


@dataclass(frozen=True)
class PitchMatch:
    """One pitch-correct current prediction paired one-to-one with gold."""

    prediction_index: int
    gold_index: int
    onset_delta_s: float


def match_pitch_correct_events(
    current: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    *,
    onset_tolerance_s: float = ONSET_TOLERANCE_S,
) -> tuple[PitchMatch, ...]:
    """Greedily pair pitch-correct events within tolerance, one-to-one.

    Predictions are visited in onset/index order.  Each takes the closest
    still-unclaimed same-pitch gold event, with gold onset/index as deterministic
    tie-breakers.  This mirrors the project's event-F1 greedy matching
    convention while retaining original indices for aligned posteriors.
    """

    if not np.isfinite(onset_tolerance_s) or onset_tolerance_s <= 0.0:
        raise ValueError(f"onset_tolerance_s must be finite and positive; got {onset_tolerance_s}")
    prediction_order = sorted(
        range(len(current)),
        key=lambda index: (current[index].onset_s, index),
    )
    gold_used: set[int] = set()
    matches: list[PitchMatch] = []
    for prediction_index in prediction_order:
        predicted = current[prediction_index]
        compatible = [
            (
                abs(predicted.onset_s - target.onset_s),
                target.onset_s,
                gold_index,
            )
            for gold_index, target in enumerate(gold)
            if gold_index not in gold_used
            and predicted.pitch_midi == target.pitch_midi
            and abs(predicted.onset_s - target.onset_s) <= onset_tolerance_s
        ]
        if not compatible:
            continue
        onset_delta_s, _gold_onset_s, gold_index = min(compatible)
        gold_used.add(gold_index)
        matches.append(
            PitchMatch(
                prediction_index=prediction_index,
                gold_index=gold_index,
                onset_delta_s=float(onset_delta_s),
            )
        )
    return tuple(matches)


@dataclass(frozen=True)
class ComplementarityResult:
    """Covered 2x2 table plus explicit abstention accounting.

    The four ``both/current_only/tabcnn_only/both_wrong`` cells are the full
    2x2 table on covered pitch matches.  Abstentions are split by current
    correctness so population-level probabilities and the fallback oracle do
    not silently discard them.
    """

    current_events: int
    gold_events: int
    eligible_pitch_matches: int
    both_correct: int
    current_only: int
    tabcnn_only: int
    both_wrong: int
    abstained_current_correct: int
    abstained_current_wrong: int

    @property
    def covered(self) -> int:
        return self.both_correct + self.current_only + self.tabcnn_only + self.both_wrong

    @property
    def abstained(self) -> int:
        return self.abstained_current_correct + self.abstained_current_wrong

    @property
    def unmatched_current(self) -> int:
        return self.current_events - self.eligible_pitch_matches

    @property
    def unmatched_gold(self) -> int:
        return self.gold_events - self.eligible_pitch_matches

    @property
    def coverage(self) -> float:
        return _ratio(self.covered, self.eligible_pitch_matches)

    @property
    def abstention_rate(self) -> float:
        return _ratio(self.abstained, self.eligible_pitch_matches)

    @property
    def current_wrong_position(self) -> int:
        return self.tabcnn_only + self.both_wrong + self.abstained_current_wrong

    @property
    def p_tabcnn_correct(self) -> float:
        """Marginal P(TabCNN correct) over all eligible pitch matches.

        Abstentions contribute no TabCNN correction instead of disappearing
        from the denominator.
        """

        return _ratio(
            self.both_correct + self.tabcnn_only,
            self.eligible_pitch_matches,
        )

    @property
    def p_tabcnn_correct_given_coverage(self) -> float:
        return _ratio(self.both_correct + self.tabcnn_only, self.covered)

    @property
    def p_tabcnn_correct_given_current_wrong(self) -> float:
        """P(TabCNN correct | current wrong-position), abstentions included."""

        return _ratio(self.tabcnn_only, self.current_wrong_position)

    @property
    def p_tabcnn_correct_given_current_wrong_and_coverage(self) -> float:
        return _ratio(self.tabcnn_only, self.tabcnn_only + self.both_wrong)

    @property
    def current_position_accuracy(self) -> float:
        return _ratio(
            self.both_correct + self.current_only + self.abstained_current_correct,
            self.eligible_pitch_matches,
        )

    @property
    def oracle_ceiling(self) -> float:
        """Accuracy if either correct source wins, with current on abstention."""

        return _ratio(
            self.both_correct
            + self.current_only
            + self.tabcnn_only
            + self.abstained_current_correct,
            self.eligible_pitch_matches,
        )

    @property
    def oracle_gain(self) -> float:
        return self.oracle_ceiling - self.current_position_accuracy


def evaluate_complementarity(
    current: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    tabcnn_priors: Sequence[np.ndarray | None],
    *,
    policy: TabCNNFusionPolicy = POSTERIOR_ONLY_POLICY,
    cfg: GuitarConfig | None = None,
    onset_tolerance_s: float = ONSET_TOLERANCE_S,
) -> ComplementarityResult:
    """Measure TabCNN position correctness on pitch-correct current events."""

    if len(current) != len(tabcnn_priors):
        raise ValueError(
            "current events and tabcnn_priors must be event-aligned and have equal length"
        )
    cfg = cfg or GuitarConfig()
    matches = match_pitch_correct_events(
        current,
        gold,
        onset_tolerance_s=onset_tolerance_s,
    )
    both_correct = 0
    current_only = 0
    tabcnn_only = 0
    both_wrong = 0
    abstained_current_correct = 0
    abstained_current_wrong = 0

    for match in matches:
        predicted = current[match.prediction_index]
        target = gold[match.gold_index]
        current_correct = (
            predicted.string_idx == target.string_idx and predicted.fret == target.fret
        )
        prior = tabcnn_priors[match.prediction_index]
        choice = (
            None
            if prior is None
            else select_tabcnn_position(
                prior,
                predicted.pitch_midi,
                policy=policy,
                cfg=cfg,
            )
        )
        if choice is None:
            if current_correct:
                abstained_current_correct += 1
            else:
                abstained_current_wrong += 1
            continue

        tabcnn_correct = choice.string_idx == target.string_idx and choice.fret == target.fret
        if current_correct and tabcnn_correct:
            both_correct += 1
        elif current_correct:
            current_only += 1
        elif tabcnn_correct:
            tabcnn_only += 1
        else:
            both_wrong += 1

    return ComplementarityResult(
        current_events=len(current),
        gold_events=len(gold),
        eligible_pitch_matches=len(matches),
        both_correct=both_correct,
        current_only=current_only,
        tabcnn_only=tabcnn_only,
        both_wrong=both_wrong,
        abstained_current_correct=abstained_current_correct,
        abstained_current_wrong=abstained_current_wrong,
    )


@dataclass(frozen=True)
class ClipEvaluation:
    """Paired current-vs-candidate results for one independently scored clip."""

    clip_id: str
    tier: str
    current_tab_f1: TabF1Result
    candidate_tab_f1: TabF1Result
    current_errors: ErrorDecomposition
    candidate_errors: ErrorDecomposition

    @property
    def delta_tab_f1(self) -> float:
        return self.candidate_tab_f1.f1 - self.current_tab_f1.f1


def score_clip(
    clip_id: str,
    tier: str,
    current: Sequence[TabEvent],
    candidate: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    *,
    onset_tolerance_s: float = ONSET_TOLERANCE_S,
) -> ClipEvaluation:
    """Score one clip without allowing matches to cross clip boundaries."""

    return ClipEvaluation(
        clip_id=clip_id,
        tier=tier,
        current_tab_f1=tab_f1(
            current,
            gold,
            onset_tolerance_s=onset_tolerance_s,
        ),
        candidate_tab_f1=tab_f1(
            candidate,
            gold,
            onset_tolerance_s=onset_tolerance_s,
        ),
        current_errors=decompose_errors(
            current,
            gold,
            onset_tolerance_s=onset_tolerance_s,
        ),
        candidate_errors=decompose_errors(
            candidate,
            gold,
            onset_tolerance_s=onset_tolerance_s,
        ),
    )


def micro_tab_f1(results: Sequence[TabF1Result]) -> TabF1Result:
    """Combine per-clip confusion counts without cross-clip onset collisions."""

    tp = sum(result.true_positives for result in results)
    fp = sum(result.false_positives for result in results)
    fn = sum(result.false_negatives for result in results)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2.0 * precision * recall, precision + recall)
    return TabF1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


@dataclass(frozen=True)
class AggregateEvaluation:
    """Macro, micro, paired-delta, and error results for a clip population."""

    clips: int
    current_macro: BootstrapResult
    candidate_macro: BootstrapResult
    paired_delta: BootstrapResult
    current_micro: TabF1Result
    candidate_micro: TabF1Result
    current_errors: ErrorDecomposition
    candidate_errors: ErrorDecomposition
    wrong_position_reduction: int
    wrong_position_relative_reduction: float
    improved_clips: tuple[str, ...]
    regressed_clips: tuple[str, ...]
    unchanged_clips: tuple[str, ...]


def aggregate_clip_evaluations(
    clips: Sequence[ClipEvaluation],
    *,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> AggregateEvaluation:
    """Aggregate paired per-clip scores with a deterministic clip bootstrap."""

    if not clips:
        raise ValueError("at least one clip evaluation is required")
    clip_ids = [clip.clip_id for clip in clips]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("clip_id values must be unique within an aggregate")

    current_values = [clip.current_tab_f1.f1 for clip in clips]
    candidate_values = [clip.candidate_tab_f1.f1 for clip in clips]
    deltas = [clip.delta_tab_f1 for clip in clips]
    current_errors = aggregate_decompositions(clip.current_errors for clip in clips)
    candidate_errors = aggregate_decompositions(clip.candidate_errors for clip in clips)
    baseline_wrong = current_errors.wrong_position_same_pitch
    wrong_reduction = baseline_wrong - candidate_errors.wrong_position_same_pitch

    return AggregateEvaluation(
        clips=len(clips),
        current_macro=bootstrap_ci(
            current_values,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        candidate_macro=bootstrap_ci(
            candidate_values,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        paired_delta=bootstrap_ci(
            deltas,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        current_micro=micro_tab_f1([clip.current_tab_f1 for clip in clips]),
        candidate_micro=micro_tab_f1([clip.candidate_tab_f1 for clip in clips]),
        current_errors=current_errors,
        candidate_errors=candidate_errors,
        wrong_position_reduction=wrong_reduction,
        wrong_position_relative_reduction=(
            wrong_reduction / baseline_wrong if baseline_wrong else 0.0
        ),
        improved_clips=tuple(clip.clip_id for clip in clips if clip.delta_tab_f1 > 0.0),
        regressed_clips=tuple(clip.clip_id for clip in clips if clip.delta_tab_f1 < 0.0),
        unchanged_clips=tuple(clip.clip_id for clip in clips if clip.delta_tab_f1 == 0.0),
    )


def aggregate_by_tier(
    clips: Sequence[ClipEvaluation],
    *,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, AggregateEvaluation]:
    """Return deterministic tier summaries, including visible regressions."""

    tiers = sorted({clip.tier for clip in clips})
    return {
        tier: aggregate_clip_evaluations(
            [clip for clip in clips if clip.tier == tier],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for tier in tiers
    }


@dataclass(frozen=True)
class PromotionThresholds:
    """Frozen automatic-promotion gates from the sequential accuracy plan."""

    aggregate_min_delta: float = 0.02
    solo_min_delta: float = 0.03
    wrong_position_min_relative_reduction: float = 0.10
    comp_mean_floor: float = -0.005


DEFAULT_PROMOTION_THRESHOLDS = PromotionThresholds()


@dataclass(frozen=True)
class PromotionGateResult:
    """Individual gate outcomes; ``passed`` requires every field to pass."""

    aggregate_effect: bool
    aggregate_positive_lower_95: bool
    solo_effect: bool
    wrong_position_reduction: bool
    comp_mean_noninferiority: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.aggregate_effect,
                self.aggregate_positive_lower_95,
                self.solo_effect,
                self.wrong_position_reduction,
                self.comp_mean_noninferiority,
            )
        )


def evaluate_promotion_gates(
    aggregate: AggregateEvaluation,
    by_tier: Mapping[str, AggregateEvaluation],
    *,
    thresholds: PromotionThresholds = DEFAULT_PROMOTION_THRESHOLDS,
    solo_tier: str = "solo",
    comp_tier: str = "comp",
) -> PromotionGateResult:
    """Evaluate the fixed aggregate, solo, and comp gates.

    Wrong-position reduction is evaluated on the same solo population.
    """

    if solo_tier not in by_tier:
        raise ValueError(f"missing required solo tier {solo_tier!r}")
    if comp_tier not in by_tier:
        raise ValueError(f"missing required comp tier {comp_tier!r}")
    solo = by_tier[solo_tier]
    comp = by_tier[comp_tier]
    return PromotionGateResult(
        aggregate_effect=(aggregate.paired_delta.statistic >= thresholds.aggregate_min_delta),
        aggregate_positive_lower_95=aggregate.paired_delta.lower > 0.0,
        solo_effect=solo.paired_delta.statistic >= thresholds.solo_min_delta,
        wrong_position_reduction=(
            solo.wrong_position_relative_reduction
            >= thresholds.wrong_position_min_relative_reduction
        ),
        comp_mean_noninferiority=(comp.paired_delta.statistic >= thresholds.comp_mean_floor),
    )


def _validated_prior(
    prior: np.ndarray,
    cfg: GuitarConfig,
    *,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(prior, dtype=np.float64)
    expected = (cfg.n_strings, cfg.max_fret + 1)
    if matrix.shape != expected:
        raise ValueError(f"{name} must have shape {expected}; got {matrix.shape}")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return matrix


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONSERVATIVE_BLEND_POLICY",
    "DEFAULT_PROMOTION_THRESHOLDS",
    "ONSET_TOLERANCE_S",
    "POSTERIOR_ONLY_POLICY",
    "AggregateEvaluation",
    "ClipEvaluation",
    "ComplementarityResult",
    "FusionVariant",
    "PitchMatch",
    "PositionChoice",
    "PositionProbability",
    "PromotionGateResult",
    "PromotionThresholds",
    "TabCNNFusionPolicy",
    "aggregate_by_tier",
    "aggregate_clip_evaluations",
    "attach_tabcnn_prior",
    "attach_tabcnn_priors",
    "evaluate_complementarity",
    "evaluate_promotion_gates",
    "map_prior_to_pitch_candidates",
    "match_pitch_correct_events",
    "micro_tab_f1",
    "score_clip",
    "select_tabcnn_position",
]
