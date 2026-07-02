"""``--audio-filters`` wiring for the composite-eval harness.

Mirrors the ``tabvision`` CLI's ``--audio-filters`` flag (commit bf61d4e) so
``scripts/eval/composite_eval.py`` can measure whether forcing post-detection
filtering on for the ``highres`` backend (built-in default: off) reduces the
``extra_detection`` loss bucket without regressing Tab F1 — see
``docs/EVAL_REPORTS/tab_f1_error_decomposition_2026-05-13.md``.
"""

from __future__ import annotations

import pytest

from tabvision.eval.composite import _resolve_audio_filters, make_run_pipeline_predictor


@pytest.mark.parametrize(
    "choice,expected",
    [("auto", None), ("on", True), ("off", False)],
)
def test_resolve_audio_filters(choice: str, expected: bool | None) -> None:
    assert _resolve_audio_filters(choice) is expected


def test_predictor_threads_audio_filters_into_shared_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named (non-"auto") backend is built once, up front — audio_filters
    must reach that single ``make_audio_backend`` call, not just per-clip
    ``run_pipeline`` calls."""
    captured: dict[str, object] = {}

    def fake_make_audio_backend(name: str, **kwargs: object) -> object:
        captured["name"] = name
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        "tabvision.audio.backend.make",
        fake_make_audio_backend,
    )

    make_run_pipeline_predictor(
        audio_backend_name="highres",
        position_prior=None,
        audio_filters=True,
    )

    assert captured["name"] == "highres"
    assert captured["kwargs"] == {"filter_config": True}


def test_predictor_omits_filter_kwarg_when_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_make_audio_backend(name: str, **kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        "tabvision.audio.backend.make",
        fake_make_audio_backend,
    )

    make_run_pipeline_predictor(
        audio_backend_name="highres",
        position_prior=None,
        audio_filters=None,
    )

    assert captured["kwargs"] == {}


def test_predictor_passes_audio_filters_to_run_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The "auto" backend name defers backend construction to run_pipeline
    per clip, so audio_filters must also be forwarded as a run_pipeline kwarg."""
    captured_kwargs: dict[str, object] = {}

    def fake_run_pipeline(media_path: str, **kwargs: object) -> list[object]:
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("tabvision.pipeline.run_pipeline", fake_run_pipeline)

    predictor = make_run_pipeline_predictor(
        audio_backend_name="auto",
        position_prior=None,
        audio_filters=False,
    )
    from tabvision.types import SessionConfig

    predictor("clip.wav", SessionConfig())

    assert captured_kwargs["audio_filters"] is False
    assert captured_kwargs["audio_backend"] is None
