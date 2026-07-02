"""Unit tests for the WS4 learned string-resolution model (CPU, random weights).

Exercises the preprocessing + forward + posterior + checkpoint round-trip without
training or a dataset. ``torch``/``torchvision`` are required (already deps via
ultralytics); skipped if absent.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tabvision.video.hand.learned_string import (  # noqa: E402
    StringResolverNet,
    load_string_resolver,
    predict_string_proba,
    preprocess_crops,
)


def test_preprocess_shapes_single_and_batch():
    single = preprocess_crops(np.zeros((224, 224, 3), dtype=np.uint8))
    assert single.shape == (1, 3, 224, 224)
    batch = preprocess_crops(np.zeros((4, 224, 224, 3), dtype=np.uint8))
    assert batch.shape == (4, 3, 224, 224)


def test_preprocess_rejects_bad_shape():
    with pytest.raises(ValueError, match="N, H, W, 3"):
        preprocess_crops(np.zeros((224, 224), dtype=np.uint8))


def test_forward_and_proba_shapes():
    model = StringResolverNet(n_strings=6, pretrained=False)
    # single crop -> (6,) distribution
    p1 = predict_string_proba(model, np.zeros((224, 224, 3), dtype=np.uint8))
    assert p1.shape == (6,)
    assert p1.sum() == pytest.approx(1.0, abs=1e-5)
    assert (p1 >= 0).all()
    # batch -> (N, 6)
    pn = predict_string_proba(model, np.zeros((3, 224, 224, 3), dtype=np.uint8))
    assert pn.shape == (3, 6)
    np.testing.assert_allclose(pn.sum(axis=1), 1.0, atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    model = StringResolverNet(n_strings=6, pretrained=False)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "n_strings": 6}, ckpt)
    loaded = load_string_resolver(ckpt, n_strings=6)
    x = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    np.testing.assert_allclose(
        predict_string_proba(model, x), predict_string_proba(loaded, x), atol=1e-6
    )
