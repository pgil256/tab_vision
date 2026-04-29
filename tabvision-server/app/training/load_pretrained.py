"""Load Spotify's shipped Basic Pitch pretrained weights into the Keras model
returned by ``basic_pitch.models.model()``.

Without this, ``basic_pitch.train.main`` fits a fresh-weight model — that is
training-from-scratch, not fine-tuning. Plan
docs/plans/2026-04-24-audio-backbone-finetune-design.md §7 calls explicitly
for fine-tuning the pretrained backbone, so this module is the missing
prerequisite for Week 2.

Why a custom loader: the shipped model is a TF SavedModel
(`basic_pitch/saved_models/icassp_2022/nmp/`). The Keras 2.x install we use
doesn't expose ``tf.keras.layers.TFSMLayer`` and there's no Keras-native
checkpoint in the package. Fortunately the variable names + shapes match
1:1 (verified on 2026-04-29: 24 vars in both, all names+shapes equal), so
a simple by-name copy works cleanly.

Usage:
    from basic_pitch import models
    from app.training.load_pretrained import load_pretrained_basic_pitch_weights

    model = models.model(no_contours=False)
    load_pretrained_basic_pitch_weights(model)
    # model is now initialized with pretrained weights and ready to fine-tune.
"""
from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


def _iter_pretrained_variables(saved_model_path: str | None = None):
    """Yield (name, tf.Variable) for the shipped Basic Pitch SavedModel."""
    import tensorflow as tf
    from basic_pitch import ICASSP_2022_MODEL_PATH

    path = saved_model_path or str(ICASSP_2022_MODEL_PATH)
    sm = tf.saved_model.load(path)
    for v in sm.variables:
        yield v.name, v


def load_pretrained_basic_pitch_weights(
    keras_model,
    saved_model_path: str | None = None,
    strict: bool = True,
) -> dict:
    """Copy pretrained weights into ``keras_model`` by variable name.

    Args:
        keras_model: a Keras Model from ``basic_pitch.models.model()``.
        saved_model_path: override path to the SavedModel
            (defaults to ``basic_pitch.ICASSP_2022_MODEL_PATH``).
        strict: if True, raise when a Keras variable is not covered by the
            SavedModel. Set False to allow architecture extensions
            (added heads etc.) — uncovered vars stay at their init values.

    Returns:
        ``{"matched": int, "mismatched_shape": [name...], "missing": [name...]}``.
    """
    import tensorflow as tf

    pre = {name: v for name, v in _iter_pretrained_variables(saved_model_path)}
    matched = 0
    mismatched_shape: list[str] = []
    missing: list[str] = []

    for kvar in keras_model.variables:
        name = kvar.name
        pv = pre.get(name)
        if pv is None:
            missing.append(name)
            continue
        if tuple(pv.shape) != tuple(kvar.shape):
            mismatched_shape.append(name)
            continue
        kvar.assign(pv.numpy())
        matched += 1

    summary = {
        'matched': matched,
        'mismatched_shape': mismatched_shape,
        'missing': missing,
        'pretrained_total': len(pre),
        'keras_total': len(keras_model.variables),
    }
    logger.info('pretrained-weight load: %s', summary)
    if strict and (mismatched_shape or missing):
        raise RuntimeError(
            f'pretrained weight load incomplete: '
            f'mismatched_shape={mismatched_shape} missing={missing}'
        )
    return summary


def verify_forward_equivalence(
    keras_model,
    saved_model_path: str | None = None,
    *,
    n_samples: int = 43844,
    seed: int = 0,
    atol: float = 1e-5,
) -> dict:
    """Sanity-check: random audio frame should produce equal outputs in both
    the SavedModel and the Keras model after weight transfer.

    Returns max abs diff per output head.
    """
    import numpy as np
    import tensorflow as tf
    from basic_pitch import ICASSP_2022_MODEL_PATH

    path = saved_model_path or str(ICASSP_2022_MODEL_PATH)
    sm = tf.saved_model.load(path)
    sig = sm.signatures['serving_default']

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, n_samples, 1)).astype(np.float32)

    sm_out = sig(input_2=tf.constant(x))
    kr_out = keras_model(tf.constant(x), training=False)

    max_abs = {}
    for head in ('onset', 'note', 'contour'):
        a = sm_out[head].numpy()
        b = kr_out[head].numpy() if isinstance(kr_out, dict) else kr_out[head]
        if hasattr(b, 'numpy'):
            b = b.numpy()
        max_abs[head] = float(abs(a - b).max())

    summary = {
        'max_abs_diff': max_abs,
        'within_atol': all(v <= atol for v in max_abs.values()),
        'atol': atol,
    }
    logger.info('forward equivalence check: %s', summary)
    return summary
