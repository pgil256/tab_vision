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


@dataclass(frozen=True)
class _Contact:
    """Minimal stand-in for :class:`fretcam.detection.FingerContact`."""

    string: int | None
    fret: int
    visible: bool = True


def _detection(
    timestamp_s: float,
    *,
    composite_available: bool = True,
    position_fret: float | None = 5.0,
    observation_confidence: float = 0.8,
    index_fret: float | None = 7.0,
    anchor_confidence: float = 0.9,
    neck_locked: bool = True,
    finger_contacts: tuple[_Contact, ...] = (),
    fret_ticks: tuple[object, ...] = (),
    neck_quad: tuple[tuple[float, float], ...] = (),
    body_joint_fret: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp_s=timestamp_s,
        composite_available=composite_available,
        position_fret=position_fret,
        observation_confidence=observation_confidence,
        index_fret=index_fret,
        anchor=SimpleNamespace(confidence=anchor_confidence),
        neck_locked=neck_locked,
        finger_contacts=finger_contacts,
        fret_ticks=fret_ticks,
        neck_quad=neck_quad,
        body_joint_fret=body_joint_fret,
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


def test_contacts_use_tab_string_numbering_not_low_e_first() -> None:
    """FretCam numbers strings 1 = high E; TabVision uses 0 = low E.

    Getting this backwards does not crash — it silently produces evidence with
    a likelihood ratio of exactly 1.00 (see
    ``docs/EVAL_REPORTS/fretcam_contact_evidence_2026-07-25.md``), so it is
    pinned here rather than left to a reviewer to notice.
    """
    config = GuitarConfig()
    chain = _FakeChain(
        detections=[
            _detection(
                0.0,
                finger_contacts=(
                    _Contact(string=1, fret=5),  # high E  -> string_idx 5
                    _Contact(string=6, fret=3),  # low E   -> string_idx 0
                ),
            )
        ]
    )
    analyzer, _, _ = _analyzer(chain, config=config)

    bundle = analyzer.analyze_all(_frames((0.0,)))

    assert len(bundle.contacts) == 1
    assert bundle.contacts[0].positions == ((0, 3), (5, 5))


def test_contacts_are_emitted_without_a_position_lock() -> None:
    """The estimator's lock gate is a display requirement, not an evidence one.

    The shipped window reached only 2.6% of target notes because it inherits
    that gate. Contacts must not.
    """
    chain = _FakeChain(
        detections=[
            _detection(0.0, finger_contacts=(_Contact(string=3, fret=7),)),
            _detection(0.1, finger_contacts=(_Contact(string=3, fret=7),)),
        ]
    )
    estimator = _FakeEstimator(
        [
            _EstimateSpec(state="lost", position=None, confidence=0.0),
            _EstimateSpec(state="acquiring", position=None, confidence=0.0),
        ]
    )
    analyzer, _, _ = _analyzer(chain, estimator)

    bundle = analyzer.analyze_all(_frames((0.0, 0.1)))

    assert bundle.windows == ()
    assert len(bundle.contacts) == 2


def test_invisible_and_unassigned_contacts_are_dropped() -> None:
    chain = _FakeChain(
        detections=[
            _detection(
                0.0,
                finger_contacts=(
                    _Contact(string=None, fret=4),
                    _Contact(string=2, fret=6, visible=False),
                    _Contact(string=2, fret=9),
                ),
            )
        ]
    )
    analyzer, _, _ = _analyzer(chain)

    bundle = analyzer.analyze_all(_frames((0.0,)))

    assert bundle.contacts[0].positions == ((4, 9),)


def test_out_of_range_contacts_are_dropped_not_clamped() -> None:
    """A clamp would invent a plausible-looking position out of a bad one."""
    config = GuitarConfig(max_fret=12)
    chain = _FakeChain(
        detections=[
            _detection(
                0.0,
                finger_contacts=(
                    _Contact(string=99, fret=4),
                    _Contact(string=2, fret=40),
                    _Contact(string=2, fret=8),
                ),
            )
        ]
    )
    analyzer, _, _ = _analyzer(chain, config=config)

    bundle = analyzer.analyze_all(_frames((0.0,)))

    assert bundle.contacts[0].positions == ((4, 8),)


def test_analyze_returns_only_windows_and_matches_analyze_all() -> None:
    """``analyze`` stays the narrow contract the PositionAnalyzer protocol needs."""
    chain = _FakeChain(
        detections=[_detection(0.0, finger_contacts=(_Contact(string=3, fret=7),))]
    )
    analyzer, _, _ = _analyzer(chain)
    windows = analyzer.analyze(_frames((0.0,)))

    chain2 = _FakeChain(
        detections=[_detection(0.0, finger_contacts=(_Contact(string=3, fret=7),))]
    )
    analyzer2, _, _ = _analyzer(chain2)
    bundle = analyzer2.analyze_all(_frames((0.0,)))

    assert windows == list(bundle.windows)
    assert bundle.contacts


@dataclass(frozen=True)
class _Tick:
    fret: int
    start: tuple[float, float]
    end: tuple[float, float]


def _straight_ticks(count: int = 10) -> tuple[_Tick, ...]:
    return tuple(
        _Tick(
            fret=index, start=(4.0 + 6.0 * index, 10.0), end=(4.0 + 6.0 * index, 40.0)
        )
        for index in range(count)
    )


def test_capo_estimate_is_surfaced_and_abstains_without_evidence() -> None:
    """A capo-free session must report an abstention, never a fret."""
    chain = _FakeChain(
        detections=[
            _detection(index * 0.1, fret_ticks=_straight_ticks()) for index in range(30)
        ]
    )
    analyzer, _, _ = _analyzer(chain)

    bundle = analyzer.analyze_all(_frames(tuple(index * 0.1 for index in range(30))))

    assert bundle.capo is not None
    assert bundle.capo.fret is None
    assert bundle.capo.confidence == 0.0


def test_capo_estimate_is_present_even_with_no_fret_ticks() -> None:
    """The record is always emitted so callers can distinguish 'no capo' from
    'never looked'."""
    chain = _FakeChain(detections=[_detection(index * 0.1) for index in range(5)])
    analyzer, _, _ = _analyzer(chain)

    bundle = analyzer.analyze_all(_frames(tuple(index * 0.1 for index in range(5))))

    assert bundle.capo is not None
    assert bundle.capo.frames_observed == 0
    assert bundle.capo.reason == "insufficient_frames"
