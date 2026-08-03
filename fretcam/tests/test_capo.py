"""Tests for session-level capo detection.

The load-bearing test is `test_a_barre_is_not_reported_as_a_capo`: in any single
frame a barre chord and a capo look nearly identical, and persistence is the
only thing separating them. A detector that fails that test would mislabel
every barre-heavy song.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from fretcam.capo import MAX_CAPO_FRET, MIN_FRAMES, CapoDetector, CapoObservation

WIDTH, HEIGHT = 640, 360
NECK_X0, NECK_X1 = 40.0, 600.0
NECK_Y0, NECK_Y1 = 140.0, 220.0


@dataclass(frozen=True)
class _Tick:
    """Stand-in for :class:`fretcam.detection.FretTick`."""

    fret: int
    start: tuple[float, float]
    end: tuple[float, float]


def _wire_fractions(n: int = MAX_CAPO_FRET + 3) -> np.ndarray:
    """Rule-of-18 wire positions as fractions along the neck."""
    ratio = 2.0 ** (-1.0 / 12.0)
    frets = np.arange(n, dtype=np.float64)
    raw = 1.0 - np.power(ratio, frets)
    return raw / raw[-1]


def _wire_x(index: int, fractions: np.ndarray) -> float:
    return NECK_X0 + float(fractions[index]) * (NECK_X1 - NECK_X0)


def _ticks(fractions: np.ndarray) -> list[_Tick]:
    return [
        _Tick(
            fret=index,
            start=(_wire_x(index, fractions), NECK_Y0),
            end=(_wire_x(index, fractions), NECK_Y1),
        )
        for index in range(len(fractions))
    ]


def _frame(
    bar_at_fret: int | None,
    fractions: np.ndarray,
    *,
    noise_seed: int = 0,
) -> np.ndarray:
    """A synthetic fretboard, optionally with a dark bar behind one wire.

    Synthetic on purpose: it proves the detector responds to the right
    *geometry*. It is not evidence about real capos, which have shadows,
    varied colour, and imperfect alignment.
    """
    rng = np.random.default_rng(noise_seed)
    frame = np.full((HEIGHT, WIDTH, 3), 150, dtype=np.uint8)
    frame[int(NECK_Y0) : int(NECK_Y1), int(NECK_X0) : int(NECK_X1)] = 170
    noise = rng.integers(-6, 7, size=frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    for index in range(len(fractions)):
        col = int(round(_wire_x(index, fractions)))
        if 0 <= col < WIDTH:
            frame[int(NECK_Y0) : int(NECK_Y1), col] = 210

    if bar_at_fret is not None:
        # A capo at fret N clamps in cell N — between wires N-1 and N, pressed
        # against wire N. It sits *behind* its wire, not in front of it.
        end = int(round(_wire_x(bar_at_fret, fractions)))
        cell = end - int(round(_wire_x(bar_at_fret - 1, fractions)))
        start = end - max(2, int(round(0.35 * cell)))
        frame[int(NECK_Y0) : int(NECK_Y1), start:end] = 25
    return frame


def _run(bars: list[int | None], *, seed_base: int = 0) -> CapoObservation:
    fractions = _wire_fractions()
    ticks = _ticks(fractions)
    detector = CapoDetector()
    for index, bar in enumerate(bars):
        detector.observe(_frame(bar, fractions, noise_seed=seed_base + index), ticks)
    return detector.estimate()


def test_a_persistent_bar_is_reported_at_its_fret() -> None:
    result = _run([3] * 40)
    assert result.detected
    assert result.fret == 3
    assert result.confidence > 0.5
    assert result.reason == "detected"


@pytest.mark.parametrize("fret", [1, 2, 4, 5, 7])
def test_the_reported_fret_is_the_bar_fret(fret: int) -> None:
    assert _run([fret] * 30).fret == fret


def test_a_bare_neck_reports_no_capo() -> None:
    """The false-positive case. A detector that fires here is unusable."""
    result = _run([None] * 40)
    assert not result.detected
    assert result.fret is None


def test_a_barre_is_not_reported_as_a_capo() -> None:
    """A barre looks like a capo in one frame; only persistence separates them.

    Thirty percent of frames barred at fret 5, the rest open — a plausible
    barre-chord song. This must abstain.
    """
    bars: list[int | None] = [5 if index % 10 < 3 else None for index in range(40)]
    result = _run(bars)
    assert not result.detected
    assert result.reason in {"not_persistent", "no_dark_band"}


def test_a_wandering_barre_is_not_reported_as_a_capo() -> None:
    """Barres at different frets across the song must not average into a capo."""
    result = _run([(index % 5) + 1 for index in range(40)])
    assert not result.detected


def test_too_few_frames_abstains() -> None:
    result = _run([3] * (MIN_FRAMES - 1))
    assert not result.detected
    assert result.reason == "insufficient_frames"
    assert result.frames_observed == MIN_FRAMES - 1


def test_missing_ticks_are_skipped_not_counted() -> None:
    fractions = _wire_fractions()
    detector = CapoDetector()
    for _ in range(30):
        assert detector.observe(_frame(3, fractions), []) is None
    result = detector.estimate()
    assert result.frames_observed == 0
    assert result.reason == "insufficient_frames"


def test_partial_tick_coverage_is_rejected() -> None:
    """A capo at fret N needs wires N and N+1; short coverage must abstain."""
    fractions = _wire_fractions()
    detector = CapoDetector()
    assert detector.observe(_frame(None, fractions), _ticks(fractions)[:3]) is None


def test_non_finite_tick_coordinates_do_not_raise() -> None:
    fractions = _wire_fractions()
    ticks = _ticks(fractions)
    ticks[2] = _Tick(fret=2, start=(float("nan"), 0.0), end=(0.0, float("inf")))
    detector = CapoDetector()
    assert detector.observe(_frame(3, fractions), ticks) is None


def test_detector_never_reports_above_its_configured_ceiling() -> None:
    fractions = _wire_fractions()
    ticks = _ticks(fractions)
    detector = CapoDetector(max_capo_fret=3)
    for index in range(30):
        detector.observe(_frame(2, fractions, noise_seed=index), ticks)
    result = detector.estimate()
    assert result.fret is None or 1 <= result.fret <= 3


def test_rejects_a_nonsense_ceiling() -> None:
    with pytest.raises(ValueError, match="max_capo_fret"):
        CapoDetector(max_capo_fret=0)


def test_detection_survives_an_oblique_view() -> None:
    """Real footage is never a head-on rectangle; ticks slant and foreshorten."""
    fractions = _wire_fractions()
    ticks = [
        _Tick(
            fret=index,
            start=(_wire_x(index, fractions), NECK_Y0 + 6.0 * index),
            end=(_wire_x(index, fractions) + 14.0, NECK_Y1 + 6.0 * index),
        )
        for index in range(len(fractions))
    ]
    detector = CapoDetector()
    rng = np.random.default_rng(7)
    for _ in range(30):
        frame = np.clip(
            np.full((HEIGHT, WIDTH, 3), 170, dtype=np.int16)
            + rng.integers(-6, 7, size=(HEIGHT, WIDTH, 3), dtype=np.int16),
            0,
            255,
        ).astype(np.uint8)
        _fill_capo_band(frame, ticks[3], ticks[2], depth=0.35)
        detector.observe(frame, ticks)
    assert detector.estimate().fret == 3


def _fill_capo_band(
    frame: np.ndarray, wire: _Tick, behind: _Tick, *, depth: float, value: int = 25
) -> None:
    """Fill the solid quad a real capo occupies between two (possibly slanted) wires."""

    def toward(a: tuple[float, float], b: tuple[float, float]) -> tuple[int, int]:
        return (
            int(round(a[0] + depth * (b[0] - a[0]))),
            int(round(a[1] + depth * (b[1] - a[1]))),
        )

    polygon = np.array(
        [
            [int(round(wire.start[0])), int(round(wire.start[1]))],
            [int(round(wire.end[0])), int(round(wire.end[1]))],
            toward(wire.end, behind.end),
            toward(wire.start, behind.start),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [polygon], (value, value, value))


def test_ticks_at_the_exact_frame_edge_do_not_index_out_of_bounds() -> None:
    """Real footage hit this: 639.6 passes `< 640` and then rounds to 640.

    Found by the GAPS negative control, not by the synthetic tests, so it is
    pinned here.
    """
    fractions = _wire_fractions()
    ticks = [
        _Tick(
            fret=index,
            start=(float(WIDTH) - 0.4 + index * 1e-3, float(HEIGHT) - 0.4),
            end=(float(WIDTH) - 0.4 + index * 1e-3, float(HEIGHT) - 0.2),
        )
        for index in range(len(fractions))
    ]
    detector = CapoDetector()
    # Must not raise; abstaining is the correct outcome for degenerate geometry.
    assert detector.observe(_frame(None, fractions), ticks) is None


# --- neck-quad fallback ---------------------------------------------------

_QUAD = ((NECK_X0, NECK_Y0), (NECK_X1, NECK_Y0), (NECK_X1, NECK_Y1), (NECK_X0, NECK_Y1))


def test_neck_quad_fallback_places_wires_by_rule_of_18() -> None:
    from fretcam.capo import fret_ticks_from_neck_quad

    ticks = fret_ticks_from_neck_quad(_QUAD, body_joint_fret=14, count=8)
    assert [t.fret for t in ticks] == list(range(8))
    # Fret 0 is the nut; wires march toward the body and bunch up as they go.
    assert ticks[0].start[0] == pytest.approx(NECK_X0)
    gaps = [ticks[i + 1].start[0] - ticks[i].start[0] for i in range(7)]
    assert all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1))


@pytest.mark.parametrize(
    "quad, joint",
    [((), 14), (_QUAD[:3], 14), (_QUAD, 0), (_QUAD, -1)],
)
def test_neck_quad_fallback_rejects_bad_input(quad, joint) -> None:
    from fretcam.capo import fret_ticks_from_neck_quad

    assert fret_ticks_from_neck_quad(quad, joint, count=8) == ()


def test_fallback_recovers_a_capo_when_the_fret_map_never_locks() -> None:
    """5 of 12 GAPS clips emitted ticks; the rest abstained blind. This is why."""
    from fretcam.capo import fret_ticks_from_neck_quad

    derived = fret_ticks_from_neck_quad(_QUAD, body_joint_fret=14, count=10)
    fractions = np.array(
        [(t.start[0] - NECK_X0) / (NECK_X1 - NECK_X0) for t in derived],
        dtype=np.float64,
    )
    detector = CapoDetector()
    for index in range(30):
        frame = _frame(3, fractions, noise_seed=index)
        # No calibrated ticks at all — only the quad.
        assert (
            detector.observe(frame, [], neck_quad=_QUAD, body_joint_fret=14) is not None
        )
    assert detector.estimate().fret == 3


def test_calibrated_ticks_take_precedence_over_the_fallback() -> None:
    fractions = _wire_fractions()
    ticks = _ticks(fractions)
    detector = CapoDetector()
    for index in range(30):
        detector.observe(
            _frame(2, fractions, noise_seed=index),
            ticks,
            neck_quad=_QUAD,
            body_joint_fret=14,
        )
    assert detector.estimate().fret == 2


def test_fallback_is_not_used_when_it_is_not_supplied() -> None:
    fractions = _wire_fractions()
    detector = CapoDetector()
    assert detector.observe(_frame(3, fractions), []) is None


def test_the_band_sampled_is_the_cell_behind_the_wire() -> None:
    """Pins the physical convention: a capo at fret N lives in cell N.

    Cell N runs from wire N-1 to wire N, and the capo presses against wire N —
    that is what makes the string speak at fret N. Sampling the cell *after*
    wire N instead would report every capo one fret low, which is worse than
    abstaining because it looks plausible.
    """
    fractions = _wire_fractions()
    ticks = _ticks(fractions)
    detector = CapoDetector()

    # Bar drawn strictly between wires 2 and 3 -> that is a capo at fret 3.
    left = int(round(_wire_x(2, fractions)))
    right = int(round(_wire_x(3, fractions)))
    for index in range(30):
        frame = _frame(None, fractions, noise_seed=index)
        frame[int(NECK_Y0) : int(NECK_Y1), left + 1 : right] = 25
        detector.observe(frame, ticks)

    assert detector.estimate().fret == 3


def test_a_stationary_hand_is_not_reported_as_a_capo() -> None:
    """The failure the GAPS control actually found.

    A player who stays in one position darkens that cell in most frames, which
    is exactly a capo's *temporal* signature — persistence alone cannot tell
    them apart. On the capo-free control, fret 4 led 26/90 frames on
    `027_Zpswc` and 44/90 on `179_pM1wc`, and `212_y41wc` false-positived.

    What separates them is width: a capo clamps every string, fingers cover two
    or three. This pins that.
    """
    fractions = _wire_fractions()
    ticks = _ticks(fractions)
    detector = CapoDetector()
    left = int(round(_wire_x(3, fractions)))
    right = int(round(_wire_x(4, fractions)))
    span = int(NECK_Y1 - NECK_Y0)
    for index in range(40):
        frame = _frame(None, fractions, noise_seed=index)
        # Two adjacent strings darkened, every frame, always the same cell.
        top = int(NECK_Y0) + span // 3
        frame[top : top + max(2, span // 6), left + 1 : right] = 25
        detector.observe(frame, ticks)
    result = detector.estimate()
    assert not result.detected, f"a stationary hand read as capo {result.fret}"


def test_a_full_width_bar_still_passes_the_coverage_gate() -> None:
    """The coverage gate must not reject genuine capos along with hands."""
    result = _run([4] * 40)
    assert result.detected
    assert result.fret == 4
