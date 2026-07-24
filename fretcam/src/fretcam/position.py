"""Time-based temporal estimator for the live FretCam position readout."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Literal

PositionState = Literal["acquiring", "locked", "shifting", "holding", "lost"]


@dataclass(frozen=True)
class EstimatorConfig:
    """Temporal constants expressed in seconds, independent of camera FPS."""

    ema_alpha: float = 0.35
    acquisition_duration_s: float = 0.25
    shift_duration_s: float = 0.25
    agreement_window_s: float = 1.0
    agreement_sample_hold_s: float = 0.50
    dropout_hold_s: float = 0.50
    evidence_gap_tolerance_s: float = 0.12
    reacquisition_gap_s: float = 0.25
    boundary_slack_fret: float = 0.4
    max_single_frame_jump_fret: float = 10.0
    jump_confirmation_s: float = 0.08
    jump_confirmation_tolerance_fret: float = 2.0
    motion_threshold_frets_per_s: float = 7.0
    min_vision_confidence: float = 0.20
    max_fret: int = 24

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.acquisition_duration_s < 0.0 or self.shift_duration_s < 0.0:
            raise ValueError("lock durations must be non-negative")
        if self.agreement_window_s <= 0.0:
            raise ValueError("agreement_window_s must be positive")
        if self.agreement_sample_hold_s <= 0.0:
            raise ValueError("agreement_sample_hold_s must be positive")
        if self.dropout_hold_s < 0.0:
            raise ValueError("dropout_hold_s must be non-negative")
        if not 0.0 <= self.evidence_gap_tolerance_s <= self.dropout_hold_s:
            raise ValueError("evidence_gap_tolerance_s must be within the dropout hold")
        if not (
            self.evidence_gap_tolerance_s
            <= self.reacquisition_gap_s
            <= self.dropout_hold_s
        ):
            raise ValueError(
                "reacquisition_gap_s must be between evidence tolerance and "
                "the dropout hold"
            )
        if not 0.0 <= self.boundary_slack_fret < 0.5:
            raise ValueError("boundary_slack_fret must be in [0, 0.5)")
        if self.max_single_frame_jump_fret <= 0.0:
            raise ValueError("max_single_frame_jump_fret must be positive")
        if self.jump_confirmation_s < 0.0:
            raise ValueError("jump_confirmation_s must be non-negative")
        if self.jump_confirmation_tolerance_fret < 0.0:
            raise ValueError("jump_confirmation_tolerance_fret must be non-negative")
        if self.motion_threshold_frets_per_s <= 0.0:
            raise ValueError("motion_threshold_frets_per_s must be positive")
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
    observation_confidence: float = 0.0
    reason: str = "no_observation"
    stable_for_ms: float = 0.0
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
    """EMA readout with elapsed-time hysteresis and dropout holding."""

    def __init__(self, config: EstimatorConfig | None = None) -> None:
        self.config = config or EstimatorConfig()
        self.reset()

    def reset(self) -> None:
        self._smoothed_fret: float | None = None
        self._stable_position: int | None = None
        self._pending_position: int | None = None
        self._pending_since_s: float | None = None
        self._last_valid_s: float | None = None
        self._history: deque[tuple[float, int | None]] = deque()
        self._last_locked_confidence = 0.0
        self._last_timestamp_s: float | None = None
        self._last_accepted_fret: float | None = None
        self._last_accepted_s: float | None = None
        self._suspected_jump_fret: float | None = None
        self._suspected_jump_since_s: float | None = None
        self._transition_active = False
        self._reacquiring_after_gap = False

    def update(
        self,
        *,
        index_fret: float | None,
        vision_confidence: float,
        timestamp_s: float,
    ) -> PositionEstimate:
        """Consume one composite position observation and return the HUD state."""
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if self._last_timestamp_s is not None and timestamp < self._last_timestamp_s:
            raise ValueError(
                "timestamps must be monotonic; call reset for a new source"
            )
        self._last_timestamp_s = timestamp

        confidence = float(vision_confidence)
        finite_fret = index_fret is not None and math.isfinite(float(index_fret))
        finite_confidence = math.isfinite(confidence)
        valid = (
            finite_fret
            and finite_confidence
            and confidence >= self.config.min_vision_confidence
        )
        if not valid:
            hinted_fret = (
                min(self.config.max_fret, max(0.0, float(index_fret)))
                if finite_fret
                else None
            )
            if self._stable_position is not None and hinted_fret is not None:
                if self._candidate_position(hinted_fret) != self._stable_position:
                    # Weak geometry may not assert a destination, but a large
                    # displacement is still useful evidence that the previous
                    # position is no longer safe to display.
                    self._transition_active = True
            reason = (
                "low_confidence"
                if hinted_fret is not None and finite_confidence
                else "no_observation"
            )
            return self._update_dropout(
                timestamp,
                reason=reason,
                raw_fret=hinted_fret,
                observation_confidence=(
                    min(1.0, max(0.0, confidence)) if finite_confidence else 0.0
                ),
            )

        raw_fret = min(self.config.max_fret, max(0.0, float(index_fret)))
        confidence = min(1.0, max(0.0, confidence))
        previous_motion_fret = self._last_accepted_fret
        previous_motion_s = self._last_accepted_s
        filtered_fret, jump_accepted = self._filter_single_frame_jump(
            raw_fret, timestamp
        )
        if not jump_accepted:
            return self._update_rejected_jump(
                timestamp,
                raw_fret=raw_fret,
                observation_confidence=confidence,
            )

        self._last_valid_s = timestamp
        candidate = self._candidate_position(filtered_fret)
        moving = self._is_change_point(
            filtered_fret,
            timestamp,
            previous_fret=previous_motion_fret,
            previous_s=previous_motion_s,
        )
        if self._smoothed_fret is None or moving:
            # A real change point should move the diagnostic readout promptly;
            # the position label is still protected by elapsed-time evidence.
            self._smoothed_fret = filtered_fret
        else:
            alpha = self.config.ema_alpha
            self._smoothed_fret = (
                alpha * filtered_fret + (1.0 - alpha) * self._smoothed_fret
            )

        self._append_history(timestamp, candidate)
        previous: int | None = None
        stable_for_s = self._advance_pending(candidate, timestamp)
        if (
            self._stable_position is None
            and stable_for_s == 0.0
            and len(self._history) > 1
        ):
            # Before the first lock, a replacement candidate starts its own
            # acquisition interval. Retaining votes for the abandoned
            # candidate can otherwise suppress a new candidate even after it
            # has satisfied the full elapsed-time stability requirement.
            latest = self._history[-1]
            self._history.clear()
            self._history.append(latest)
        candidate_agreement = self._temporal_agreement(candidate)
        candidate_confidence = confidence * candidate_agreement
        evidence_gap_s = (
            math.inf
            if previous_motion_s is None
            else max(0.0, timestamp - previous_motion_s)
        )
        if (
            self._stable_position is not None
            and candidate == self._stable_position
            and evidence_gap_s > self.config.reacquisition_gap_s
        ):
            self._reacquiring_after_gap = True
        elif candidate != self._stable_position:
            self._reacquiring_after_gap = False
        reacquiring_after_gap = bool(
            self._reacquiring_after_gap
            and self._stable_position is not None
            and candidate == self._stable_position
        )
        reason = "insufficient_stability"
        if self._stable_position is None:
            stable_enough = stable_for_s >= self.config.acquisition_duration_s
            confident_enough = candidate_confidence >= self.config.min_vision_confidence
            if stable_enough and confident_enough:
                self._stable_position = candidate
                self._clear_pending()
                self._transition_active = False
                self._reacquiring_after_gap = False
                state: PositionState = "locked"
                stable_for_s = self.config.acquisition_duration_s
            else:
                state = "acquiring"
                if stable_enough:
                    reason = "temporal_disagreement"
        elif candidate == self._stable_position and not moving:
            stable_enough = (
                stable_for_s >= self.config.acquisition_duration_s
                if reacquiring_after_gap
                else True
            )
            if (
                stable_enough
                and candidate_confidence >= self.config.min_vision_confidence
            ):
                self._clear_pending()
                self._reacquiring_after_gap = False
                stable_for_s = (
                    self.config.acquisition_duration_s
                    if reacquiring_after_gap
                    else self.config.shift_duration_s
                )
                self._transition_active = False
                state = "locked"
            else:
                self._transition_active = True
                previous = self._stable_position
                state = "acquiring"
                reason = (
                    "evidence_gap"
                    if reacquiring_after_gap and not stable_enough
                    else "temporal_disagreement"
                )
        else:
            previous = self._stable_position
            self._transition_active = True
            stable_enough = stable_for_s >= self.config.shift_duration_s
            confident_enough = candidate_confidence >= self.config.min_vision_confidence
            if stable_enough and confident_enough:
                self._stable_position = candidate
                self._clear_pending()
                self._transition_active = False
                self._reacquiring_after_gap = False
                state = "locked"
                stable_for_s = self.config.shift_duration_s
            else:
                state = "shifting"
                if moving:
                    reason = "motion_detected"
                elif stable_enough:
                    reason = "temporal_disagreement"

        agreement = self._temporal_agreement(self._stable_position)
        if state == "locked" and self._stable_position is not None:
            output_confidence = confidence * agreement
            if output_confidence < self.config.min_vision_confidence:
                # This guard keeps the public contract honest even if future
                # state-machine changes alter the candidate/commit ordering.
                self._transition_active = True
                return self._estimate(
                    timestamp,
                    "acquiring",
                    raw_fret=raw_fret,
                    position=None,
                    previous_position=self._stable_position,
                    confidence=0.0,
                    observation_confidence=confidence,
                    agreement=agreement,
                    reason="temporal_disagreement",
                    stable_for_s=stable_for_s,
                )
            self._last_locked_confidence = output_confidence
            return self._estimate(
                timestamp,
                state,
                raw_fret=raw_fret,
                position=self._stable_position,
                previous_position=previous,
                confidence=output_confidence,
                observation_confidence=confidence,
                agreement=agreement,
                reason="locked",
                stable_for_s=stable_for_s,
            )
        return self._estimate(
            timestamp,
            state,
            raw_fret=raw_fret,
            position=None,
            previous_position=previous,
            confidence=0.0,
            observation_confidence=confidence,
            agreement=agreement,
            reason=reason,
            stable_for_s=stable_for_s,
        )

    def _candidate_position(self, raw_fret: float) -> int:
        raw_position = min(
            self.config.max_fret,
            max(1, math.floor(raw_fret + 0.5)),
        )
        stable = self._stable_position
        if stable is None:
            return raw_position
        slack = self.config.boundary_slack_fret
        if stable - 0.5 - slack <= raw_fret < stable + 0.5 + slack:
            return stable
        return raw_position

    def _filter_single_frame_jump(
        self, raw_fret: float, timestamp_s: float
    ) -> tuple[float, bool]:
        """Hold an implausible jump until it persists for a minimum duration."""
        previous = self._last_accepted_fret
        if previous is None:
            self._accept_fret(raw_fret, timestamp_s)
            return raw_fret, True

        if abs(raw_fret - previous) <= self.config.max_single_frame_jump_fret:
            self._accept_fret(raw_fret, timestamp_s)
            self._clear_suspected_jump()
            return raw_fret, True

        suspected = self._suspected_jump_fret
        if (
            suspected is not None
            and abs(raw_fret - suspected)
            <= self.config.jump_confirmation_tolerance_fret
        ):
            since = self._suspected_jump_since_s
            if since is not None and (
                timestamp_s - since >= self.config.jump_confirmation_s
            ):
                self._accept_fret(raw_fret, timestamp_s)
                self._clear_suspected_jump()
                return raw_fret, True
        else:
            self._suspected_jump_fret = raw_fret
            self._suspected_jump_since_s = timestamp_s
        return previous, False

    def _accept_fret(self, fret: float, timestamp_s: float) -> None:
        self._last_accepted_fret = fret
        self._last_accepted_s = timestamp_s

    def _clear_suspected_jump(self) -> None:
        self._suspected_jump_fret = None
        self._suspected_jump_since_s = None

    def _is_change_point(
        self,
        fret: float,
        timestamp_s: float,
        *,
        previous_fret: float | None,
        previous_s: float | None,
    ) -> bool:
        if previous_fret is None or previous_s is None or timestamp_s <= previous_s:
            return False
        velocity = abs(fret - previous_fret) / (timestamp_s - previous_s)
        return (
            velocity >= self.config.motion_threshold_frets_per_s
            and self._stable_position is not None
            and self._candidate_position(fret) != self._stable_position
        )

    def _advance_pending(self, candidate: int, timestamp_s: float) -> float:
        if candidate != self._pending_position:
            self._pending_position = candidate
            self._pending_since_s = timestamp_s
            return 0.0
        assert self._pending_since_s is not None
        return max(0.0, timestamp_s - self._pending_since_s)

    def _clear_pending(self) -> None:
        self._pending_position = None
        self._pending_since_s = None

    def _append_history(self, timestamp_s: float, candidate: int | None) -> None:
        self._history.append((timestamp_s, candidate))
        cutoff = timestamp_s - self.config.agreement_window_s
        # Keep one predecessor so a segment that crosses the window boundary
        # contributes only its in-window duration.
        while len(self._history) > 1 and self._history[1][0] <= cutoff:
            self._history.popleft()

    def _temporal_agreement(self, position: int | None) -> float:
        if not self._history:
            return 0.0
        if len(self._history) == 1:
            only = self._history[0][1]
            if only is None:
                return 0.0
            return 1.0 if position is None or only == position else 0.0

        current_s = (
            self._last_timestamp_s
            if self._last_timestamp_s is not None
            else self._history[-1][0]
        )
        cutoff = current_s - self.config.agreement_window_s
        weights: dict[int, float] = {}
        covered_s = 0.0
        events = tuple(self._history)
        for index, (start_s, value) in enumerate(events):
            next_s = events[index + 1][0] if index + 1 < len(events) else current_s
            segment_start = max(start_s, cutoff)
            segment_end = min(
                next_s,
                start_s + self.config.agreement_sample_hold_s,
                current_s,
            )
            duration_s = max(0.0, segment_end - segment_start)
            if duration_s <= 0.0:
                continue
            covered_s += duration_s
            if value is not None:
                weights[value] = weights.get(value, 0.0) + duration_s
        if covered_s <= 1e-9 or not weights:
            latest = self._history[-1][1]
            if latest is None:
                return 0.0
            return 1.0 if position is None or latest == position else 0.0
        if position is None:
            return max(weights.values()) / covered_s
        return weights.get(position, 0.0) / covered_s

    def _update_rejected_jump(
        self,
        timestamp_s: float,
        *,
        raw_fret: float,
        observation_confidence: float,
    ) -> PositionEstimate:
        """Abstain on an unconfirmed jump without refreshing old evidence."""

        self._append_history(timestamp_s, None)
        self._clear_pending()
        previous = self._stable_position
        elapsed = (
            math.inf
            if self._last_valid_s is None
            else max(0.0, timestamp_s - self._last_valid_s)
        )
        if elapsed > self.config.dropout_hold_s:
            self._stable_position = None
            self._smoothed_fret = None
            self._history.clear()
            self._last_locked_confidence = 0.0
            self._last_accepted_fret = None
            self._last_accepted_s = None
            self._last_valid_s = None
            self._transition_active = False
            self._reacquiring_after_gap = False
            previous = None
            state: PositionState = "acquiring"
            agreement = 0.0
        else:
            self._transition_active = previous is not None
            state = "shifting" if previous is not None else "acquiring"
            agreement = self._temporal_agreement(previous)
        return self._estimate(
            timestamp_s,
            state,
            raw_fret=raw_fret,
            position=None,
            previous_position=previous,
            confidence=0.0,
            observation_confidence=observation_confidence,
            agreement=agreement,
            reason="implausible_jump",
            stable_for_s=0.0,
        )

    def _update_dropout(
        self,
        timestamp_s: float,
        *,
        reason: str,
        raw_fret: float | None,
        observation_confidence: float,
    ) -> PositionEstimate:
        self._clear_suspected_jump()
        self._append_history(timestamp_s, None)
        elapsed = (
            math.inf
            if self._last_valid_s is None
            else max(0.0, timestamp_s - self._last_valid_s)
        )
        preserve_evidence = elapsed <= self.config.evidence_gap_tolerance_s
        if not preserve_evidence:
            self._clear_pending()
        agreement = self._temporal_agreement(self._stable_position)
        if (
            self._stable_position is None
            and self._pending_position is not None
            and preserve_evidence
        ):
            return self._estimate(
                timestamp_s,
                "acquiring",
                raw_fret=raw_fret,
                position=None,
                previous_position=None,
                confidence=0.0,
                observation_confidence=observation_confidence,
                agreement=agreement,
                reason=reason,
                stable_for_s=0.0,
            )
        if (
            self._stable_position is not None
            and self._transition_active
            and elapsed <= self.config.dropout_hold_s
        ):
            return self._estimate(
                timestamp_s,
                "shifting",
                raw_fret=raw_fret,
                position=None,
                previous_position=self._stable_position,
                confidence=0.0,
                observation_confidence=observation_confidence,
                agreement=agreement,
                reason=reason,
                stable_for_s=0.0,
            )
        if self._stable_position is not None and elapsed <= self.config.dropout_hold_s:
            decay = 1.0 - elapsed / max(self.config.dropout_hold_s, 1e-9)
            return self._estimate(
                timestamp_s,
                "holding",
                raw_fret=raw_fret,
                position=self._stable_position,
                previous_position=None,
                confidence=self._last_locked_confidence * max(0.0, decay),
                observation_confidence=observation_confidence,
                agreement=agreement,
                reason=reason,
                stable_for_s=0.0,
            )

        self._stable_position = None
        self._smoothed_fret = None
        self._history.clear()
        self._last_locked_confidence = 0.0
        self._last_accepted_fret = None
        self._last_accepted_s = None
        self._last_valid_s = None
        self._transition_active = False
        self._reacquiring_after_gap = False
        self._clear_pending()
        return self._estimate(
            timestamp_s,
            "acquiring" if reason == "low_confidence" else "lost",
            raw_fret=raw_fret,
            position=None,
            previous_position=None,
            confidence=0.0,
            observation_confidence=observation_confidence,
            agreement=0.0,
            reason=reason,
            stable_for_s=0.0,
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
        observation_confidence: float,
        agreement: float,
        reason: str,
        stable_for_s: float,
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
            observation_confidence=min(1.0, max(0.0, observation_confidence)),
            reason=reason,
            stable_for_ms=round(max(0.0, stable_for_s) * 1000.0, 3),
        )


__all__ = [
    "EstimatorConfig",
    "PositionEstimate",
    "PositionEstimator",
    "PositionState",
    "position_window",
    "roman_position",
]
