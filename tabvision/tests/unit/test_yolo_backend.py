"""Unit tests for tabvision.video.guitar.yolo_backend.

Inference is exercised in a separate (gated) integration test once a
trained checkpoint is available; here we just verify the construction
and missing-checkpoint error path so AGPL-tainted ultralytics doesn't
need to be installed for the core test suite.
"""

import pytest

from tabvision.errors import BackendError
from tabvision.video.guitar.yolo_backend import (
    DEFAULT_CHECKPOINT_ENV,
    YoloOBBBackend,
    _default_checkpoint_path,
)


def test_default_checkpoint_path_uses_data_root(monkeypatch):
    monkeypatch.setenv("TABVISION_DATA_ROOT", "/tmp/tv-test-root")
    monkeypatch.delenv(DEFAULT_CHECKPOINT_ENV, raising=False)
    p = _default_checkpoint_path()
    assert str(p).endswith("/tmp/tv-test-root/models/guitar-yolo-obb-finetuned.pt")


def test_env_overrides_default_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv(DEFAULT_CHECKPOINT_ENV, str(tmp_path / "custom.pt"))
    p = _default_checkpoint_path()
    assert p == tmp_path / "custom.pt"


def test_missing_checkpoint_raises_helpful_backend_error(tmp_path):
    """Trying to detect without a trained checkpoint should raise BackendError
    with instructions, not a cryptic FileNotFoundError."""
    backend = YoloOBBBackend(checkpoint_path=tmp_path / "does-not-exist.pt")
    import numpy as np

    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(BackendError, match="train"):
        backend.detect(fake_frame)


def test_constructor_does_not_load_model(tmp_path):
    """Construction must be cheap; model load is lazy."""
    backend = YoloOBBBackend(checkpoint_path=tmp_path / "x.pt")
    assert backend._model is None
