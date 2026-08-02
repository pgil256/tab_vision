"""Track C — is there room for a player/session-adaptive position prior?

The shipped position prior is population statistics over five players. A player
whose habits differ from that population is systematically mis-assigned, and
nothing in the pipeline notices. The capo case showed how large prior/session
mismatch can get: routing capo>0 sessions through a capo-ignorant prior scored
0.2956 against a 0.6773 capo-0 control — a collapse, not a shortfall.

Phase 0 also measured a wide per-player spread under leave-one-out priors:
baseline aggregate Tab F1 ranges 0.5342 (player 01) to 0.6631 (player 03). Some
of that is difficulty; some may be prior mismatch. This track separates the two.

**Oracle before estimator.** N4 closed a multi-week build in one iteration by
computing the ceiling first and finding it was +0.0027 — no estimator can
rescue a ceiling that small. The same question here has two ceilings, and they
bracket every possible adaptation scheme:

``oracle_player``  the held-out player's prior built from *their own* clips
                   (in-sample). What perfect knowledge of this player's habits
                   is worth. The ceiling for player-level adaptation.
``oracle_clip``    a prior built from the clip's own gold. What perfect
                   knowledge of *this recording* is worth — a strictly looser
                   upper bound, and unreachable by construction, but it bounds
                   session-level adaptation too.

Neither is achievable: both use gold labels for the notes being scored. They are
here to answer "is there anything to chase", not to be shipped.

**Then the achievable version.** ``self_adapt`` uses no gold at all: decode once
with the shipped prior, harvest the decoder's own high-confidence assignments,
learn a session prior from them, blend it with the population prior, and
re-decode. This is the real proposal — a session teaching the prior about itself
from its own confident answers.

Gate: paired ΔTab F1 vs shipped on **dev only**, lo-95 > 0, and no regression on
the players who already do well (an adaptive prior that helps outliers by
hurting typical players is a loss). The sealed player is not opened.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.a_physics_coverage import load_banked
from scripts.eval.n2_muscriptor_merge import _event_from_json
from scripts.eval.phase0_rotation_baseline import (
    BURNED_PLAYER,
    DEV_PLAYERS,
    POSITION_ALPHA,
    POSITION_POWER,
    build_loo_priors,
    gold_by_player,
    score_leak_free,
)
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.guitarset_audio import score_audio_only
from tabvision.fusion.inharmonicity import apply_fits
from tabvision.fusion.playability import set_transition_prior
from tabvision.fusion.position_prior import (
    PitchPositionPrior,
    apply_pitch_position_prior,
    learn_pitch_position_prior,
)
from tabvision.fusion.string_physics import load_string_evidence, reference_stiffness_model
from tabvision.pipeline import SEQUENCE_PRIOR_WEIGHT
from tabvision.types import GuitarConfig, SessionConfig, TabEvent

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

CONFIDENCE_FLOOR = 0.5
"""Only assignments the decoder is sure of teach the session prior.

The confidence is the Viterbi string-flip margin (B4), i.e. how much more the
cheapest decode that moves this note to another string would cost. Harvesting
low-margin notes would feed the prior exactly the notes it was unsure about,
which is circular.
"""

BLEND_WEIGHTS = (0.15, 0.30, 0.50, 0.75, 1.00)
"""Includes the degenerate end deliberately.

At 1.00 the population prior is discarded entirely and the decode is re-run
against nothing but its own first-pass answers. Phase 0 measured the population
prior as worth a great deal, so if 1.00 were the best arm that would be evidence
of a self-confirmation artefact rather than of adaptation — the sweep has to be
able to show that.
"""

MISMATCH_WEIGHT = 0.50
"""Control arm: blend in *another clip's* session prior at the same weight.

The self-adaptive prior is learned from the decoder's own confident output, and
at a 90%+ harvest rate that prior is close to a restatement of the decode. Any
gain could therefore be generic sharpening — re-deciding in favour of what you
already decided — rather than anything about this session. If a *mismatched*
session prior helps as much, the effect is not session-specific and the arm is
measuring self-confirmation. This is the control that distinguishes them.
"""

FROZEN = {"baseline": 0.6083, "shipped": 0.6801}
FROZEN_TOLERANCE = 0.0015


def blend_priors(
    base: PitchPositionPrior, session: PitchPositionPrior, weight: float
) -> PitchPositionPrior:
    """Convex blend, per pitch, renormalised. Pitches the session never saw keep
    the population prior unchanged rather than being flattened toward it."""
    out: dict[int, np.ndarray] = {}
    for pitch, matrix in base.by_pitch.items():
        other = session.matrix_for_pitch(pitch)
        if other is None:
            out[pitch] = matrix
            continue
        mixed = (1.0 - weight) * matrix + weight * other
        total = float(mixed.sum())
        out[pitch] = mixed / total if total > 0 else matrix
    return PitchPositionPrior(by_pitch=out)


def decode(
    events: list[Any],
    clip_gold: list[TabEvent],
    position: PitchPositionPrior,
    sequence: Any,
    cfg: GuitarConfig,
    session: SessionConfig,
) -> list[TabEvent]:
    prepared = apply_pitch_position_prior(list(events), position)
    set_transition_prior(sequence, weight=SEQUENCE_PRIOR_WEIGHT)
    try:
        scored = score_audio_only(prepared, clip_gold, cfg=cfg, session=session)
    finally:
        set_transition_prior(None)
    return list(scored.decoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--fit-cache", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    fit_cache = args.fit_cache or (data_root / "models" / "a_partial_fit_cache")
    dev_cache = data_root / "models" / "q6_full_dev_cache"
    sealed_cache = data_root / "models" / "q6_player05_cache"

    cfg = GuitarConfig()
    session_cfg = SessionConfig()
    evidence = load_string_evidence()
    table = reference_stiffness_model()
    isolation = evidence.isolation

    gold = gold_by_player(data_home, cfg)
    clips = sorted(t for p in DEV_PLAYERS for t in gold[p])
    if args.limit:
        clips = clips[: args.limit]

    missing = [t for t in clips if not (fit_cache / f"{t}.{isolation}.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} clips unbanked; run track A's `bank` first")

    print("building leave-one-player-out priors...", flush=True)
    positions, sequences = build_loo_priors(gold, cfg)

    # In-sample per-player priors: the ceiling for player-level adaptation.
    print("building in-sample per-player priors (oracle)...", flush=True)
    own_player: dict[str, PitchPositionPrior] = {}
    for player in DEV_PLAYERS:
        events = [event for track in gold[player].values() for event in track]
        own_player[player] = learn_pitch_position_prior(
            events, cfg=cfg, alpha=POSITION_ALPHA, power=POSITION_POWER
        )

    arms = ["shipped", "oracle_player", "oracle_clip"]
    arms += [f"self_adapt_{w:.2f}" for w in BLEND_WEIGHTS]
    arms += [f"mismatched_{MISMATCH_WEIGHT:.2f}"]
    previous_session: PitchPositionPrior | None = None
    scores: dict[str, list[float]] = {arm: [] for arm in arms}
    per_player: dict[str, dict[str, list[float]]] = {arm: defaultdict(list) for arm in arms}
    harvested_total = 0
    decoded_total = 0

    def cache_for(track_id: str) -> Path:
        return sealed_cache if track_id[:2] == BURNED_PLAYER else dev_cache

    for index, track_id in enumerate(clips, start=1):
        player = track_id[:2]
        events = [
            _event_from_json(item)
            for item in json.loads(
                (cache_for(track_id) / f"{track_id}.ensemble.json").read_text("utf-8")
            )
        ]
        ordered = sorted(events, key=lambda event: event.onset_s)
        banked = load_banked(fit_cache / f"{track_id}.{isolation}.json")
        clip_gold = gold[player][track_id]

        # Shipped configuration: physics evidence on top of banked fits.
        physics_events, _ = apply_fits(
            ordered,
            banked,
            table,
            cfg,
            weight=evidence.weight,
            min_r2=evidence.min_r2,
            sigma=evidence.sigma,
            isolation=isolation,
        )

        clip_prior = learn_pitch_position_prior(
            clip_gold, cfg=cfg, alpha=POSITION_ALPHA, power=POSITION_POWER
        )

        # Pass 1 with the shipped prior, harvested for the self-adaptive arm.
        first = decode(
            physics_events, clip_gold, positions[player], sequences[player], cfg, session_cfg
        )
        confident = [event for event in first if event.confidence >= CONFIDENCE_FLOOR]
        harvested_total += len(confident)
        decoded_total += len(first)
        session_prior = (
            learn_pitch_position_prior(
                confident, cfg=cfg, alpha=POSITION_ALPHA, power=POSITION_POWER
            )
            if len(confident) >= 8
            else None
        )

        arm_priors: dict[str, PitchPositionPrior] = {
            "shipped": positions[player],
            "oracle_player": own_player[player],
            "oracle_clip": clip_prior,
        }
        for weight in BLEND_WEIGHTS:
            name = f"self_adapt_{weight:.2f}"
            arm_priors[name] = (
                blend_priors(positions[player], session_prior, weight)
                if session_prior is not None
                else positions[player]
            )
        # Control: the previous clip's session prior, at the same weight. If
        # this helps as much as the matched one, the gain is not about session
        # content.
        mismatch_name = f"mismatched_{MISMATCH_WEIGHT:.2f}"
        arm_priors[mismatch_name] = (
            blend_priors(positions[player], previous_session, MISMATCH_WEIGHT)
            if previous_session is not None
            else positions[player]
        )
        if session_prior is not None:
            previous_session = session_prior

        for arm, prior in arm_priors.items():
            metrics, _ = score_leak_free(
                physics_events,
                clip_gold,
                position=prior,
                sequence=sequences[player],
                cfg=cfg,
                session=session_cfg,
            )
            scores[arm].append(metrics["tab_f1"])
            per_player[arm][player].append(metrics["tab_f1"])

        if index % 50 == 0 or index == len(clips):
            print(f"  [{index}/{len(clips)}]", flush=True)

    ship = np.asarray(scores["shipped"], dtype=np.float64)
    results: dict[str, Any] = {
        "clips": len(clips),
        "confidence_floor": CONFIDENCE_FLOOR,
        "harvest_rate": harvested_total / max(decoded_total, 1),
        "arms": {},
        "per_player": {},
    }

    print(f"\n{'arm':<20}{'Tab F1':>9}{'vs shipped':>12}{'lo95':>9}{'hi95':>9}  verdict")
    for arm in arms:
        values = np.asarray(scores[arm], dtype=np.float64)
        deltas = values - ship
        ci = bootstrap_ci(deltas, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        if arm == "shipped":
            verdict = "-"
        elif ci.lower > 0:
            verdict = "PASS"
        elif ci.upper < 0:
            verdict = "regression"
        else:
            verdict = "inconclusive"
        results["arms"][arm] = {
            "tab_f1": float(values.mean()),
            "delta_vs_shipped": float(deltas.mean()),
            "lo95": ci.lower,
            "hi95": ci.upper,
            "verdict": verdict,
        }
        print(
            f"{arm:<20}{values.mean():>9.4f}{deltas.mean():>+12.4f}"
            f"{ci.lower:>+9.4f}{ci.upper:>+9.4f}  {verdict}"
        )

    print(f"\nharvest rate at confidence >= {CONFIDENCE_FLOOR}: {results['harvest_rate']:.1%}")

    print(f"\n{'player':<8}" + "".join(f"{arm.replace('self_adapt_', 'sa'):>16}" for arm in arms))
    for player in DEV_PLAYERS:
        row = f"{player:<8}"
        for arm in arms:
            values = per_player[arm].get(player, [])
            row += f"{float(np.mean(values)) if values else float('nan'):>16.4f}"
            results["per_player"].setdefault(player, {})[arm] = (
                float(np.mean(values)) if values else None
            )
        print(row)

    drift = results["arms"]["shipped"]["tab_f1"] - FROZEN["shipped"]
    results["frozen_drift"] = drift
    print(f"\nfrozen drift (shipped): {drift:+.4f}")
    if not args.limit and abs(drift) > FROZEN_TOLERANCE:
        raise SystemExit("drift from the Phase 0 frozen baseline; nothing above is trustworthy")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
