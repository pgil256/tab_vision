"""Step 3 of the learned-fusion plan
(docs/plans/2026-04-24-learned-fusion-design.md §4.4).

Trains a LightGBM lambdarank model that scores candidate (string, fret)
positions for one detected note, given the per-event and per-candidate
features captured by the fusion engine emitter.

Each event is a query; candidates are documents; label is 1 for the GT
position, 0 otherwise. Uses leave-one-video-out CV — train on 19 videos,
score the held-out video, repeat for all 20. Primary metric: per-event
top-1 accuracy.

If LOOCV beats heuristic baseline by ≥ 5pp and no per-video regression is
worse than 3pp, save:
  app/models/position_selector.lgb           — final booster (trained on
                                                all 20 videos)
  app/models/position_selector_features.json — feature schema for serving
And write tools/outputs/position_selector_report-YYYY-MM-DD.md.

Usage (from tabvision-server/):
  python tools/train_position_selector.py
  python tools/train_position_selector.py --input tools/outputs/position_dataset.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

THIS_FILE = os.path.abspath(__file__)
TOOLS_DIR = os.path.dirname(THIS_FILE)
SERVER_DIR = os.path.dirname(TOOLS_DIR)
OUTPUTS_DIR = os.path.join(TOOLS_DIR, 'outputs')
MODELS_DIR = os.path.join(SERVER_DIR, 'app', 'models')

FEATURES: list[str] = [
    # Per-event audio
    'midi_note', 'amplitude', 'basicpitch_confidence',
    # Per-event chord context
    'is_chord', 'chord_size', 'chord_string_span', 'num_candidates',
    # Per-event continuity
    'prev_position_string', 'prev_position_fret', 'seconds_since_prev',
    # Per-event anchor
    'hand_anchor_fret',
    # Per-candidate identity
    'cand_string', 'cand_fret',
    # Per-candidate distances
    'dist_anchor_fret', 'dist_prev_fret', 'dist_prev_string',
    # Per-candidate heuristic signal — explicitly included so the model can
    # *beat* the heuristic, not clone it (plan §6 "feature leakage" note).
    'heuristic_score', 'is_heuristic_pick',
]

LGB_RANKER_PARAMS = dict(
    objective='lambdarank',
    metric='ndcg',
    n_estimators=50,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.1,
    min_data_in_leaf=5,
    label_gain=[0, 1],
    verbose=-1,
    deterministic=True,
    random_state=42,
)

LGB_CLASSIFIER_PARAMS = dict(
    objective='binary',
    n_estimators=50,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.1,
    min_data_in_leaf=5,
    verbose=-1,
    deterministic=True,
    random_state=42,
)

SHIP_GATE_PP = 5.0   # +5 pp over heuristic LOOCV
WORST_REGRESSION_PP = 3.0  # no per-video regression > 3 pp


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_chord'] = df['is_chord'].astype(int)
    df['is_heuristic_pick'] = df['is_heuristic_pick'].astype(int)
    return df


def heuristic_per_event_topk(df: pd.DataFrame) -> tuple[int, int]:
    """Top-1 accuracy of `is_heuristic_pick` row vs label."""
    correct = 0
    total = 0
    for _, grp in df.groupby('event_id', sort=False):
        total += 1
        heur = grp[grp['is_heuristic_pick'] == 1]
        if len(heur) and int(heur['label'].iloc[0]) == 1:
            correct += 1
    return correct, total


def model_per_event_topk(scores_df: pd.DataFrame) -> tuple[int, int]:
    """Top-1 accuracy: argmax(score) per event matches label=1."""
    correct = 0
    total = 0
    for _, grp in scores_df.groupby('event_id', sort=False):
        total += 1
        idx = grp['score'].idxmax()
        if int(grp.loc[idx, 'label']) == 1:
            correct += 1
    return correct, total


def fit_ranker(train_df: pd.DataFrame) -> lgb.LGBMRanker:
    train_df = train_df.sort_values('event_id').reset_index(drop=True)
    X = train_df[FEATURES].values
    y = train_df['label'].astype(int).values
    groups = train_df.groupby('event_id', sort=False).size().values
    model = lgb.LGBMRanker(**LGB_RANKER_PARAMS)
    model.fit(X, y, group=groups)
    return model


def fit_classifier(train_df: pd.DataFrame) -> lgb.LGBMClassifier:
    X = train_df[FEATURES].values
    y = train_df['label'].astype(int).values
    model = lgb.LGBMClassifier(**LGB_CLASSIFIER_PARAMS)
    model.fit(X, y)
    return model


def predict_scores(model, df: pd.DataFrame) -> np.ndarray:
    """Return one score per row (higher = better candidate)."""
    if isinstance(model, lgb.LGBMClassifier):
        return model.predict_proba(df[FEATURES].values)[:, 1]
    return model.predict(df[FEATURES].values)


def model_with_margin_fallback(eval_scored: pd.DataFrame,
                               margin_threshold: float) -> tuple[int, int]:
    """Use model pick if its margin over the heuristic pick is high enough.

    Margin = score(model_argmax) - score(is_heuristic_pick row). When below
    threshold, fall back to the heuristic. Mirrors the plan §4.5 serving
    strategy.
    """
    correct = 0
    total = 0
    for _, grp in eval_scored.groupby('event_id', sort=False):
        total += 1
        model_idx = grp['score'].idxmax()
        heur_rows = grp[grp['is_heuristic_pick'] == 1]
        if len(heur_rows) == 0:
            chosen = model_idx
        else:
            heur_idx = heur_rows.index[0]
            margin = grp.loc[model_idx, 'score'] - grp.loc[heur_idx, 'score']
            chosen = model_idx if margin >= margin_threshold else heur_idx
        if int(grp.loc[chosen, 'label']) == 1:
            correct += 1
    return correct, total


def loocv(df: pd.DataFrame, fitter, margin_threshold: float = 0.0) -> dict:
    """Leave-one-video-out CV.

    fitter: callable(train_df) -> trained model that exposes predict-or-
            predict_proba via predict_scores().
    margin_threshold: when > 0, model only overrides heuristic when its
            argmax score exceeds the heuristic-pick row's score by this
            much.
    """
    videos = sorted(df['video_id'].unique())
    per_video: dict[str, dict] = {}
    total_correct = 0
    total_events = 0

    for v in videos:
        train = df[df['video_id'] != v]
        eval_ = df[df['video_id'] == v]
        if eval_.empty or train.empty:
            continue
        model = fitter(train)
        scores = predict_scores(model, eval_)
        eval_scored = eval_.assign(score=scores)
        if margin_threshold > 0:
            m_correct, m_total = model_with_margin_fallback(
                eval_scored, margin_threshold,
            )
        else:
            m_correct, m_total = model_per_event_topk(eval_scored)
        h_correct, h_total = heuristic_per_event_topk(eval_)
        per_video[v] = {
            'model_correct': m_correct,
            'heuristic_correct': h_correct,
            'events': m_total,
            'model_acc': m_correct / m_total if m_total else 0.0,
            'heuristic_acc': h_correct / h_total if h_total else 0.0,
        }
        total_correct += m_correct
        total_events += m_total

    return {
        'per_video': per_video,
        'total_correct': total_correct,
        'total_events': total_events,
        'overall_acc': total_correct / total_events if total_events else 0.0,
    }


def write_report(report_path: str, args, df: pd.DataFrame,
                 heur_correct: int, heur_total: int,
                 cv: dict, ship_decision: dict, params: dict) -> None:
    today = date.today().isoformat()
    lines: list[str] = []
    lines.append(f'# Position selector training — {today}\n')
    lines.append(f'Input: `{args.input}`')
    lines.append(f'Rows: {len(df)}, events: {df["event_id"].nunique()}, '
                 f'videos: {df["video_id"].nunique()}\n')

    heur_pct = 100.0 * heur_correct / heur_total if heur_total else 0.0
    model_pct = 100.0 * cv['overall_acc']
    delta = model_pct - heur_pct

    lines.append('## Headline\n')
    lines.append(f'- Heuristic baseline: {heur_correct}/{heur_total} = **{heur_pct:.1f}%**')
    lines.append(f'- Model LOOCV: {cv["total_correct"]}/{cv["total_events"]} = **{model_pct:.1f}%**')
    lines.append(f'- Δ: **{delta:+.1f}pp**')
    lines.append(f'- Ship gate: ≥ +{SHIP_GATE_PP:.0f}pp + no per-video regression > '
                 f'{WORST_REGRESSION_PP:.0f}pp')
    lines.append(f'- **Decision: {ship_decision["verdict"]}**')
    if ship_decision.get('reason'):
        lines.append(f'  - {ship_decision["reason"]}')
    lines.append('')

    lines.append('## Per-video accuracy\n')
    lines.append('| video | events | heuristic | model | Δpp |')
    lines.append('|---|---:|---:|---:|---:|')
    for v in sorted(cv['per_video']):
        r = cv['per_video'][v]
        h = 100.0 * r['heuristic_acc']
        m = 100.0 * r['model_acc']
        lines.append(
            f'| {v} | {r["events"]} | '
            f'{r["heuristic_correct"]}/{r["events"]} ({h:.1f}%) | '
            f'{r["model_correct"]}/{r["events"]} ({m:.1f}%) | '
            f'{m - h:+.1f} |'
        )
    lines.append('')

    lines.append(f'## LightGBM params ({args.model_type}, margin={args.margin})\n')
    lines.append('```')
    for k, vv in params.items():
        lines.append(f'{k}: {vv}')
    lines.append('```\n')

    lines.append('## Features\n')
    for f in FEATURES:
        lines.append(f'- {f}')

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',
                    default=os.path.join(OUTPUTS_DIR, 'position_dataset.parquet'),
                    help='input parquet from build_position_dataset.py')
    ap.add_argument('--model-path',
                    default=os.path.join(MODELS_DIR, 'position_selector.lgb'),
                    help='where to save the booster (only if ship gate passes)')
    ap.add_argument('--schema-path',
                    default=os.path.join(MODELS_DIR, 'position_selector_features.json'),
                    help='where to save the feature schema JSON')
    ap.add_argument('--report-path',
                    default=os.path.join(
                        OUTPUTS_DIR,
                        f'position_selector_report-{date.today().isoformat()}.md',
                    ),
                    help='where to save the markdown report')
    ap.add_argument('--save-anyway', action='store_true',
                    help='save the model even if the ship gate fails (for inspection)')
    ap.add_argument('--model-type', choices=('ranker', 'classifier'),
                    default='ranker',
                    help='lambdarank ranker (plan default) or binary classifier '
                         '(plan §4.4 fallback)')
    ap.add_argument('--margin', type=float, default=0.0,
                    help='Fallback to heuristic when model margin over heuristic '
                         'pick is below this. 0 = always trust model.')
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    df = prepare(df)
    print(f'rows: {len(df)} events: {df["event_id"].nunique()} '
          f'videos: {df["video_id"].nunique()}', file=sys.stderr)

    heur_correct, heur_total = heuristic_per_event_topk(df)
    heur_pct = 100.0 * heur_correct / heur_total if heur_total else 0.0
    print(f'heuristic baseline: {heur_correct}/{heur_total} = {heur_pct:.1f}%',
          file=sys.stderr)

    fitter = fit_classifier if args.model_type == 'classifier' else fit_ranker
    cv = loocv(df, fitter, margin_threshold=args.margin)
    model_pct = 100.0 * cv['overall_acc']
    print(f'model LOOCV: {cv["total_correct"]}/{cv["total_events"]} = '
          f'{model_pct:.1f}%', file=sys.stderr)
    print(f'Δ: {model_pct - heur_pct:+.1f}pp', file=sys.stderr)

    # Ship-gate decision.
    delta = model_pct - heur_pct
    worst_regress_pp = 0.0
    worst_video = None
    for v, r in cv['per_video'].items():
        diff = 100.0 * (r['model_acc'] - r['heuristic_acc'])
        if diff < worst_regress_pp:
            worst_regress_pp = diff
            worst_video = v

    if delta < SHIP_GATE_PP:
        verdict = f'NO SHIP — Δ {delta:+.1f}pp < gate +{SHIP_GATE_PP:.0f}pp'
        reason = ''
    elif worst_regress_pp < -WORST_REGRESSION_PP:
        verdict = (f'NO SHIP — {worst_video} regressed '
                   f'{worst_regress_pp:.1f}pp (gate: > -{WORST_REGRESSION_PP:.0f})')
        reason = ''
    else:
        verdict = (f'SHIP — Δ {delta:+.1f}pp, worst regression '
                   f'{worst_regress_pp:.1f}pp on {worst_video}')
        reason = ''

    ship_decision = {'verdict': verdict, 'reason': reason}
    print(f'\n{verdict}', file=sys.stderr)

    params_for_report = (LGB_CLASSIFIER_PARAMS if args.model_type == 'classifier'
                         else LGB_RANKER_PARAMS)
    write_report(args.report_path, args, df, heur_correct, heur_total, cv,
                 ship_decision, params_for_report)
    print(f'wrote {args.report_path}', file=sys.stderr)

    should_save = args.save_anyway or verdict.startswith('SHIP')
    if should_save:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        final_model = fitter(df)
        if isinstance(final_model, lgb.LGBMClassifier):
            final_model.booster_.save_model(args.model_path)
            params = LGB_CLASSIFIER_PARAMS
        else:
            final_model.booster_.save_model(args.model_path)
            params = LGB_RANKER_PARAMS
        with open(args.schema_path, 'w') as f:
            json.dump(
                {
                    'features': FEATURES,
                    'model_type': args.model_type,
                    'margin_threshold': args.margin,
                    'lightgbm_params': params,
                    'version': 1,
                },
                f, indent=2,
            )
        print(f'wrote {args.model_path}', file=sys.stderr)
        print(f'wrote {args.schema_path}', file=sys.stderr)
    else:
        print('not saving model — ship gate failed (use --save-anyway to override)',
              file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
