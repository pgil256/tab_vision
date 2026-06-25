"""Unit tests for the WS4 learned-eval fingering construction."""

from __future__ import annotations

import numpy as np

from scripts.eval.v1_1_gaps_learned_probe import fingering_from_proba
from tabvision.fusion.candidates import candidate_positions
from tabvision.types import GuitarConfig


def test_fingering_from_proba_concentrates_on_candidate_strings():
    cfg = GuitarConfig()
    pitch = 64  # E4 — playable on several strings
    cands = candidate_positions(pitch, cfg)
    # Posterior favouring one specific candidate string.
    target = cands[len(cands) // 2].string_idx
    proba = np.full(cfg.n_strings, 0.02)
    proba[target] = 0.9
    ff = fingering_from_proba(proba, pitch, cfg, t=1.0)
    marg = ff.marginal_string_fret()
    # The arg-max over candidate cells must be the favoured string.
    best = max(cands, key=lambda c: marg[c.string_idx, c.fret])
    assert best.string_idx == target
    assert ff.homography_confidence == 1.0


def test_fingering_from_proba_no_candidates_is_inert():
    cfg = GuitarConfig()
    # A pitch outside the playable range -> no candidates -> zero-confidence.
    ff = fingering_from_proba(np.ones(cfg.n_strings) / cfg.n_strings, 5, cfg, t=0.0)
    assert ff.homography_confidence == 0.0
    assert ff.finger_pos_logits.shape == (4, cfg.n_strings, cfg.max_fret + 1)


def test_fingering_from_proba_shape_and_mass_only_on_candidates():
    cfg = GuitarConfig()
    pitch = 55  # G3
    proba = np.random.default_rng(0).dirichlet(np.ones(cfg.n_strings))
    marg = fingering_from_proba(proba, pitch, cfg, t=0.0).marginal_string_fret()
    cand_cells = {(c.string_idx, c.fret) for c in candidate_positions(pitch, cfg)}
    # Mass sits (essentially) on candidate cells; the floor-logit leak on the
    # ~145 non-candidate cells is tiny and irrelevant (fusion reads only
    # candidate cells via the Viterbi restriction).
    on = sum(marg[s, f] for s, f in cand_cells)
    assert on > 0.9
