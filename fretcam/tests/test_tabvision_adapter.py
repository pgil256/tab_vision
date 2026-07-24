from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from fretcam.position import EstimatorConfig
from fretcam.tabvision_adapter import FretCamPositionAnalyzer
from tabvision.types import GuitarConfig


@dataclass(frozen=True)
class _EstimateSpec:
    state: str = "locked"
    position: int | None = 5
    confidence: float = 0.8
    timestamp_s: float | None = None


class _FakeEstimator:
    def __init__(self, specs: list[_EstimateSpec] | None = None) -> None:
        self.specs = list(specs or [])
        self.calls: list[tuple[float | None, float, float]] = []

    def update(
        self,
        *,
        index_fret: float | None,
        vision_confidence: float,
        timestamp_s: float,
    ) -> SimpleNamespace:
        self.calls.append((index_fret, vision_confidence, timestamp_s))
        spec = self.specs.pop(0) if self.specs else _EstimateSpec()
        estimate_timestamp = (
            timestamp_s if spec.timestamp_s is None else spec.timestamp_s
        )
        return SimpleNamespace(
            timestamp_s=estimate_timestamp,
            state=spec.state,
            position=spec.position,
            window_frets=(0, 4, 5, 6, 7, 8, 9),
            confidence=spec.confidence,
            observation_confidence=vision_confidence,
            reason="locked" if spec.state == "locked" else "fixture",
        )


class _EstimatorFactory:
    def __init__(self, estimator: _FakeEstimator) -> None:
        self.estimator = estimator
        self.config: EstimatorConfig | None = None

    def __call__(self, config: EstimatorConfig) -> _FakeEstimator:
        self.config = config
        return self.estimator


def _detection(
    timestamp_s: float,
    *,
    composite_available: bool = True,
    position_fret: float | None = 5.0,
    observation_confidence: float = 0.8,
    index_fret: float | None = 7.0,
    anchor_confidence: float = 0.9,
    neck_locked: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp_s=timestamp_s,
        composite_available=composite_available,
        position_fret=position_fret,
        observation_confidence=observation_confidence,
        index_fret=index_fret,
        anchor=SimpleNamespace(confidence=anchor_confidence),
        neck_locked=neck_locked,
        confidence_factors=SimpleNamespace(
            landmark_quality=0.73,
            blockers=("fixture_blocker",),
        ),
        hand_points=((10.0, 20.0),),
        hand_pose_quality=0.61,
    )


class _FakeChain:
    def __init__(
        self,
        *,
        detections: list[SimpleNamespace] | None = None,
        fail_on_call: int | None = None,
    ) -> None:
        self.detections = list(detections or [])
        self.fail_on_call = fail_on_call
        self.timestamps: list[float] = []
        self.frame_shapes: list[tuple[int, ...]] = []
        self.hints: list[object] = []
        self.closed = False

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        timestamp_s: float,
    ) -> SimpleNamespace:
        call_number = len(self.timestamps) + 1
        self.timestamps.append(timestamp_s)
        self.frame_shapes.append(frame.shape)
        if self.fail_on_call == call_number:
            raise RuntimeError("fixture inference failure")
        if self.detections:
            detection = self.detections.pop(0)
            detection.timestamp_s = timestamp_s
            return detection
        return _detection(timestamp_s)

    def set_hand_search_hint(self, hint: object) -> None:
        self.hints.append(hint)

    def close(self) -> None:
        self.closed = True


class _ChainFactory:
    def __init__(self, chain: _FakeChain) -> None:
        self.chain = chain
        self.kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> _FakeChain:
        self.kwargs = kwargs
        return self.chain


def _frames(
    timestamps: tuple[float, ...],
    *,
    width: int = 100,
    height: int = 50,
) -> list[tuple[float, np.ndarray]]:
    return [
        (timestamp, np.zeros((height, width, 3), dtype=np.uint8))
        for timestamp in timestamps
    ]


def _analyzer(
    chain: _FakeChain,
    estimator: _FakeEstimator | None = None,
    *,
    config: GuitarConfig | None = None,
) -> tuple[FretCamPositionAnalyzer, _ChainFactory, _EstimatorFactory]:
    chain_factory = _ChainFactory(chain)
    estimator_factory = _EstimatorFactory(estimator or _FakeEstimator())
    analyzer = FretCamPositionAnalyzer(
        config,
        chain_factory=chain_factory,  # type: ignore[arg-type]
        estimator_factory=estimator_factory,  # type: ignore[arg-type]
    )
    return analyzer, chain_factory, estimator_factory


def test_analyzer_samples_by_stride_on_the_original_media_clock() -> None:
    config = GuitarConfig(max_fret=19)
    chain = _FakeChain()
    estimator = _FakeEstimator()
    analyzer, chain_factory, estimator_factory = _analyzer(
        chain,
        estimator,
        config=config,
    )

    observations = analyzer.analyze(
        _frames((0.0, 0.04, 0.08, 0.12, 0.16)),
        stride=2,
    )

    assert chain.timestamps == pytest.approx([0.0, 0.08, 0.16])
    assert [call[2] for call in estimator.calls] == pytest.approx([0.0, 0.08, 0.16])
    assert [observation.timestamp_s for observation in observations] == pytest.approx(
        [0.0, 0.08, 0.16]
    )
    assert chain_factory.kwargs == {
        "guitar_config": config,
        "detector_hz": 2.0,
        "background_detector": False,
        "crop_hand": True,
    }
    assert estimator_factory.config is not None
    assert estimator_factory.config.max_fret == 19
    assert chain.closed


def test_analyzer_gates_state_confidence_and_forwards_feedback() -> None:
    specs = [
        _EstimateSpec(state="locked", confidence=0.20),
        _EstimateSpec(state="holding", confidence=0.75),
        _EstimateSpec(state="acquiring", position=None, confidence=0.9),
        _EstimateSpec(state="locked", confidence=0.199),
        _EstimateSpec(state="locked", confidence=float("nan")),
        _EstimateSpec(state="locked", position=None, confidence=0.8),
        _EstimateSpec(state="locked", confidence=0.8, timestamp_s=float("nan")),
    ]
    chain = _FakeChain()
    analyzer, _, _ = _analyzer(chain, _FakeEstimator(specs))

    observations = analyzer.analyze(_frames(tuple(index * 0.1 for index in range(7))))

    assert [(item.state, item.confidence) for item in observations] == [
        ("locked", pytest.approx(0.20)),
        ("holding", pytest.approx(0.75)),
    ]
    assert len(chain.hints) == 7
    first_hint = chain.hints[0]
    assert first_hint.position_state == "locked"  # type: ignore[attr-defined]
    assert first_hint.landmark_quality == pytest.approx(0.73)  # type: ignore[attr-defined]
    assert first_hint.blockers == ("fixture_blocker",)  # type: ignore[attr-defined]
    assert first_hint.hand_visible is True  # type: ignore[attr-defined]
    assert first_hint.pose_quality == pytest.approx(0.61)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("width", "height", "expected_shape"),
    [
        (1920, 1080, (360, 640, 3)),
        (1080, 1920, (480, 270, 3)),
    ],
)
def test_analyzer_fits_frames_within_640_by_480(
    width: int,
    height: int,
    expected_shape: tuple[int, ...],
) -> None:
    chain = _FakeChain()
    analyzer, _, _ = _analyzer(chain)

    analyzer.analyze(_frames((0.0,), width=width, height=height))

    assert chain.frame_shapes == [expected_shape]


def test_analyzer_matches_hud_composite_abstention_and_legacy_fallback() -> None:
    chain = _FakeChain(
        detections=[
            _detection(
                0.0,
                composite_available=True,
                position_fret=None,
                observation_confidence=0.0,
            ),
            _detection(
                0.1,
                composite_available=False,
                position_fret=None,
                observation_confidence=0.0,
            ),
        ]
    )
    estimator = _FakeEstimator()
    analyzer, _, _ = _analyzer(chain, estimator)

    analyzer.analyze(_frames((0.0, 0.1)))

    assert estimator.calls == [
        (None, 0.0, pytest.approx(0.0)),
        (7.0, 0.9, pytest.approx(0.1)),
    ]


def test_analyzer_closes_chain_when_inference_raises() -> None:
    chain = _FakeChain(fail_on_call=2)
    analyzer, _, _ = _analyzer(chain)

    with pytest.raises(RuntimeError, match="fixture inference failure"):
        analyzer.analyze(_frames((0.0, 0.1)))

    assert chain.closed
