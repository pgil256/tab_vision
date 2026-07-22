"""F3 temporal estimator for the live FretCam position readout."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Literal

PositionState = Literal["acquiring", "locked", "shifting", "holding", "lost"]


@dataclass(frozen=True)
class EstimatorConfig:
    """Fixed temporal constants from the FretCam design."""

    ema_alpha: float = 0.35
    hysteresis_frames: int = 5
    agreement_window: int = 10
    dropout_hold_frames: int = 5
    boundary_slack_fret: float = 0.15
    min_vision_confidence: float = 0.05
    max_fret: int = 24

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.hysteresis_frames < 1 or self.agreement_window < 1:
            raise ValueError("frame counts must be positive")
        if self.dropout_hold_frames < 0:
            raise ValueError("dropout_hold_frames must be non-negative")
        if not 0.0 <= self.boundary_slack_fret < 0.5:
            raise ValueError("boundary_slack_fret must be in [0, 0.5)")
        if not 0.0 <= self.min_vision_confidence <= 1.0:
            raise ValueError("min_vision_confidence must be in [0, 1]")
        if self.max_fret < 1:
            raise ValueError("max_fret must be positive")


@dataclass(frozen=True)
class PositionEstimate:
    """One stable, transitional, held, or missing FretCam readout."""

    timestamp_s: float
    state: PositionState
    label: str
    raw_index_fret: float | None
    smoothed_index_fret: float | None
    position: int | None
    previous_position: int | None
    window_frets: tuple[int, ...]
    confidence: float
    temporal_agreement: float
    open_strings_possible: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def roman_position(position: int) -> str:
    """Render a positive classical guitar position as a Roman numeral."""
    if position <= 0:
        raise ValueError("position must be positive")
    values = (
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )
    remaining = position
    parts: list[str] = []
    for value, numeral in values:
        while remaining >= value:
            parts.append(numeral)
            remaining -= value
    return "".join(parts)


def position_window(position: int, *, max_fret: int = 24) -> tuple[int, ...]:
    """Return ``[N-1, N+4] union {0}``, clipped to the configured neck."""
    if position <= 0 or max_fret <= 0:
        raise ValueError("position and max_fret must be positive")
    low = max(1, position - 1)
    high = min(max_fret, position + 4)
    return (0, *range(low, high + 1))


class PositionEstimator:
    """EMA readout with consecutive-frame hysteresis and dropout holding."""

    def __init__(self, config: EstimatorConfig | None = None) -> None:
        self.config = config or EstimatorConfig()
        self.reset()

    def reset(self) -> None:
        self._smoothed_fret: float | None = None
        self._stable_position: int | None = None
        self._pending_position: int | None = None
        self._pending_count = 0
        self._missing_frames = 0
        self._history: deque[int | None] = deque(maxlen=self.config.agreement_window)
        self._last_locked_confidence = 0.0
        self._last_timestamp_s: float | None = None

    def update(
        self,
        *,
        index_fret: float | None,
        vision_confidence: float,
        timestamp_s: float,
    ) -> PositionEstimate:
        """Consume one frame's index coordinate and return the HUD state."""
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if self._last_timestamp_s is not None and timestamp < self._last_timestamp_s:
            raise ValueError(
                "timestamps must be monotonic; call reset for a new source"
            )
        self._last_timestamp_s = timestamp

        confidence = float(vision_confidence)
        valid = (
            index_fret is not None
            and math.isfinite(float(index_fret))
            and math.isfinite(confidence)
            and confidence >= self.config.min_vision_confidence
        )
        if not valid:
            return self._update_dropout(timestamp)

        raw_fret = min(self.config.max_fret, max(0.0, float(index_fret)))
        confidence = min(1.0, max(0.0, confidence))
        self._missing_frames = 0
        if self._smoothed_fret is None:
            self._smoothed_fret = raw_fret
        else:
            alpha = self.config.ema_alpha
            self._smoothed_fret = alpha * raw_fret + (1.0 - alpha) * self._smoothed_fret

        candidate = self._candidate_position(raw_fret)
        self._history.append(candidate)
        previous: int | None = None
        if self._stable_position is None:
            self._advance_pending(candidate)
            if self._pending_count >= self.config.hysteresis_frames:
                self._stable_position = candidate
                self._clear_pending()
                state: PositionState = "locked"
            else:
                state = "acquiring"
        elif candidate == self._stable_position:
            self._clear_pending()
            state = "locked"
        else:
            previous = self._stable_position
            self._advance_pending(candidate)
            if self._pending_count >= self.config.hysteresis_frames:
                self._stable_position = candidate
                self._clear_pending()
                state = "locked"
            else:
                state = "shifting"

        agreement = self._temporal_agreement(self._stable_position)
        if state == "locked" and self._stable_position is not None:
            output_confidence = confidence * agreement
            self._last_locked_confidence = output_confidence
            return self._estimate(
                timestamp,
                state,
                raw_fret=raw_fret,
                position=self._stable_position,
                previous_position=previous,
                confidence=output_confidence,
                agreement=agreement,
            )
        return self._estimate(
            timestamp,
            state,
            raw_fret=raw_fret,
            position=None,
            previous_position=previous,
            confidence=0.0,
            agreement=agreement,
        )

    def _candidate_position(self, raw_fret: float) -> int:
        raw_position = min(self.config.max_fret, max(1, math.floor(raw_fret)))
        stable = self._stable_position
        if stable is None:
            return raw_position
        slack = self.config.boundary_slack_fret
        if stable - slack <= raw_fret < stable + 1 + slack:
            return stable
        return raw_position

    def _advance_pending(self, candidate: int) -> None:
        if candidate == self._pending_position:
            self._pending_count += 1
        else:
            self._pending_position = candidate
            self._pending_count = 1

    def _clear_pending(self) -> None:
        self._pending_position = None
        self._pending_count = 0

    def _temporal_agreement(self, position: int | None) -> float:
        valid = [value for value in self._history if value is not None]
        if not valid:
            return 0.0
        if position is None:
            return max(Counter(valid).values()) / len(valid)
        return valid.count(position) / len(valid)

    def _update_dropout(self, timestamp_s: float) -> PositionEstimate:
        self._missing_frames += 1
        self._history.append(None)
        self._clear_pending()
        agreement = self._temporal_agreement(self._stable_position)
        if (
            self._stable_position is not None
            and self._missing_frames <= self.config.dropout_hold_frames
        ):
            decay = 1.0 - self._missing_frames / (self.config.dropout_hold_frames + 1)
            return self._estimate(
                timestamp_s,
                "holding",
                raw_fret=None,
                position=self._stable_position,
                previous_position=None,
                confidence=self._last_locked_confidence * decay,
                agreement=agreement,
            )

        self._stable_position = None
        self._smoothed_fret = None
        self._history.clear()
        self._last_locked_confidence = 0.0
        return self._estimate(
            timestamp_s,
            "lost",
            raw_fret=None,
            position=None,
            previous_position=None,
            confidence=0.0,
            agreement=0.0,
        )

    def _estimate(
        self,
        timestamp_s: float,
        state: PositionState,
        *,
        raw_fret: float | None,
        position: int | None,
        previous_position: int | None,
        confidence: float,
        agreement: float,
    ) -> PositionEstimate:
        if state == "shifting":
            label = "Shifting…"
        elif state == "acquiring":
            label = "Acquiring…"
        elif state == "lost":
            label = "No hand"
        elif position is not None:
            label = f"Position {roman_position(position)}"
        else:  # pragma: no cover - guarded by the state machine
            label = "No position"
        return PositionEstimate(
            timestamp_s=timestamp_s,
            state=state,
            label=label,
            raw_index_fret=raw_fret,
            smoothed_index_fret=self._smoothed_fret,
            position=position,
            previous_position=previous_position,
            window_frets=(
                position_window(position, max_fret=self.config.max_fret)
                if position is not None
                else (0,)
            ),
            confidence=min(1.0, max(0.0, confidence)),
            temporal_agreement=min(1.0, max(0.0, agreement)),
        )


__all__ = [
    "EstimatorConfig",
    "PositionEstimate",
    "PositionEstimator",
    "PositionState",
    "position_window",
    "roman_position",
]
