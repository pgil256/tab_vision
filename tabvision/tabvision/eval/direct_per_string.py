"""Original CPU-feasible six-string second-opinion model for Phase 5."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from tabvision.fusion.candidates import candidate_positions
from tabvision.types import GuitarConfig

SAMPLE_RATE = 16_000
WINDOW_START_S = -0.064
WINDOW_END_S = 0.448
WINDOW_SAMPLES = round((WINDOW_END_S - WINDOW_START_S) * SAMPLE_RATE)
N_FFT = 512
HOP_LENGTH = 128
MEL_BANDS = 64
MIDI_BEGIN = 21
PITCH_CLASSES = 88
STRINGS = 6


@dataclass(frozen=True)
class DirectOutputs:
    onset_logits: torch.Tensor
    frame_logits: torch.Tensor
    global_pitch_logits: torch.Tensor
    occupancy_logits: torch.Tensor


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    onset: torch.Tensor
    frame: torch.Tensor
    global_pitch: torch.Tensor
    occupancy: torch.Tensor
    duplicate_inhibition: torch.Tensor


class DirectPerStringNet(nn.Module):
    """Shared convolutional encoder with six onset/frame-pitch heads."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            self._block(1, 12, groups=3),
            self._block(12, 24, groups=6),
            self._block(24, 48, groups=8),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.onset_head = nn.Linear(48, STRINGS * PITCH_CLASSES)
        self.frame_head = nn.Linear(48, STRINGS * PITCH_CLASSES)
        self.global_pitch_head = nn.Linear(48, PITCH_CLASSES)
        self.occupancy_head = nn.Linear(48, STRINGS)

    @staticmethod
    def _block(input_channels: int, output_channels: int, *, groups: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, log_mel: torch.Tensor) -> DirectOutputs:
        if log_mel.ndim != 3 or log_mel.shape[1:] != (MEL_BANDS, 65):
            raise ValueError(
                f"expected log-mel shape (batch, {MEL_BANDS}, 65), got {tuple(log_mel.shape)}"
            )
        embedding = self.encoder(log_mel.unsqueeze(1)).flatten(1)
        batch = len(log_mel)
        return DirectOutputs(
            onset_logits=self.onset_head(embedding).view(batch, STRINGS, PITCH_CLASSES),
            frame_logits=self.frame_head(embedding).view(batch, STRINGS, PITCH_CLASSES),
            global_pitch_logits=self.global_pitch_head(embedding),
            occupancy_logits=self.occupancy_head(embedding),
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def multitask_loss(
    outputs: DirectOutputs,
    onset_targets: torch.Tensor,
    frame_targets: torch.Tensor,
    global_pitch_targets: torch.Tensor,
    occupancy_targets: torch.Tensor,
) -> LossBreakdown:
    """Compute the fixed multi-task objective including duplicate inhibition."""

    _validate_targets(
        outputs,
        onset_targets,
        frame_targets,
        global_pitch_targets,
        occupancy_targets,
    )
    onset = functional.binary_cross_entropy_with_logits(
        outputs.onset_logits,
        onset_targets,
        pos_weight=torch.as_tensor(32.0, device=outputs.onset_logits.device),
    )
    frame = functional.binary_cross_entropy_with_logits(
        outputs.frame_logits,
        frame_targets,
        pos_weight=torch.as_tensor(16.0, device=outputs.frame_logits.device),
    )
    global_pitch = functional.binary_cross_entropy_with_logits(
        outputs.global_pitch_logits,
        global_pitch_targets,
        pos_weight=torch.as_tensor(16.0, device=outputs.global_pitch_logits.device),
    )
    occupancy = functional.binary_cross_entropy_with_logits(
        outputs.occupancy_logits,
        occupancy_targets,
        pos_weight=torch.as_tensor(2.0, device=outputs.occupancy_logits.device),
    )
    predicted_counts = torch.sigmoid(outputs.frame_logits).sum(dim=1)
    annotated_counts = frame_targets.sum(dim=1)
    duplicate_inhibition = torch.mean(torch.relu(predicted_counts - annotated_counts) ** 2)
    total = (
        onset + 0.5 * frame + 0.25 * global_pitch + 0.25 * occupancy + 0.05 * duplicate_inhibition
    )
    return LossBreakdown(total, onset, frame, global_pitch, occupancy, duplicate_inhibition)


def gold_pitch_string_scores(
    outputs: DirectOutputs,
    pitch_midi: torch.Tensor,
) -> torch.Tensor:
    """Return six string scores for each known MIDI pitch."""

    pitch_indices = pitch_midi.to(dtype=torch.long) - MIDI_BEGIN
    if pitch_indices.ndim != 1 or len(pitch_indices) != len(outputs.onset_logits):
        raise ValueError("pitch_midi must have one value per batch row")
    if torch.any((pitch_indices < 0) | (pitch_indices >= PITCH_CLASSES)):
        raise ValueError("pitch_midi lies outside the model range")
    gather_indices = pitch_indices[:, None, None].expand(-1, STRINGS, 1)
    onset = torch.gather(outputs.onset_logits, 2, gather_indices).squeeze(2)
    frame = torch.gather(outputs.frame_logits, 2, gather_indices).squeeze(2)
    return onset + 0.5 * frame


def string_scores_to_fret_prior(
    pitch_midi: int,
    string_scores: np.ndarray,
    cfg: GuitarConfig | None = None,
) -> np.ndarray:
    """Mask and normalize six model scores into the existing fret-prior channel."""

    cfg = cfg or GuitarConfig()
    scores = np.asarray(string_scores, dtype=np.float64)
    if scores.shape != (cfg.n_strings,) or np.any(~np.isfinite(scores)):
        raise ValueError("string_scores must be one finite score per string")
    candidates = candidate_positions(int(pitch_midi), cfg)
    if not candidates:
        raise ValueError(f"pitch {pitch_midi} has no playable guitar position")
    selected = np.asarray([scores[candidate.string_idx] for candidate in candidates])
    selected -= float(np.max(selected))
    probabilities = np.exp(selected)
    probabilities /= float(np.sum(probabilities))
    matrix = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
    for candidate, probability in zip(candidates, probabilities, strict=True):
        matrix[candidate.string_idx, candidate.fret] = float(probability)
    return matrix


def extract_window(wav_16k: np.ndarray, onset_s: float) -> np.ndarray:
    """Extract the fixed event-centered 16 kHz waveform with zero padding."""

    signal = np.asarray(wav_16k, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError(f"expected mono waveform, got {signal.shape}")
    if not math.isfinite(onset_s):
        raise ValueError("onset_s must be finite")
    start = int(round((onset_s + WINDOW_START_S) * SAMPLE_RATE))
    stop = start + WINDOW_SAMPLES
    output = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    source_start = max(0, start)
    source_stop = min(len(signal), stop)
    if source_stop > source_start:
        destination_start = source_start - start
        output[destination_start : destination_start + source_stop - source_start] = signal[
            source_start:source_stop
        ]
    return output


def log_mel_batch(windows: torch.Tensor) -> torch.Tensor:
    """Convert a batch of fixed 16 kHz windows to deterministic 64-band log-mel."""

    if windows.ndim != 2 or windows.shape[1] != WINDOW_SAMPLES:
        raise ValueError(
            f"expected waveform shape (batch, {WINDOW_SAMPLES}), got {tuple(windows.shape)}"
        )
    taper = torch.hann_window(N_FFT, dtype=windows.dtype, device=windows.device)
    spectrum = torch.stft(
        windows,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=taper,
        center=True,
        return_complex=True,
    )
    power = torch.abs(spectrum) ** 2
    filters = torch.as_tensor(
        mel_filterbank(),
        dtype=power.dtype,
        device=power.device,
    )
    mel = torch.einsum("mf,bft->bmt", filters, power)
    return torch.log1p(10.0 * mel)


def mel_filterbank() -> np.ndarray:
    """Return the fixed Slaney-style triangular filter bank without librosa."""

    minimum_hz = 55.0
    maximum_hz = SAMPLE_RATE / 2.0

    def hz_to_mel(value: float) -> float:
        return 2595.0 * math.log10(1.0 + value / 700.0)

    def mel_to_hz(value: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (value / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(minimum_hz), hz_to_mel(maximum_hz), MEL_BANDS + 2)
    hz_points = mel_to_hz(mel_points)
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    filters = np.zeros((MEL_BANDS, len(frequencies)), dtype=np.float32)
    for index in range(MEL_BANDS):
        lower, center, upper = hz_points[index : index + 3]
        rising = (frequencies - lower) / max(center - lower, 1.0e-12)
        falling = (upper - frequencies) / max(upper - center, 1.0e-12)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
        total = float(np.sum(filters[index]))
        if total:
            filters[index] /= total
    return filters


def _validate_targets(
    outputs: DirectOutputs,
    onset_targets: torch.Tensor,
    frame_targets: torch.Tensor,
    global_pitch_targets: torch.Tensor,
    occupancy_targets: torch.Tensor,
) -> None:
    expected = outputs.onset_logits.shape
    if onset_targets.shape != expected or frame_targets.shape != expected:
        raise ValueError("onset/frame target shape does not match model output")
    if global_pitch_targets.shape != outputs.global_pitch_logits.shape:
        raise ValueError("global-pitch target shape does not match model output")
    if occupancy_targets.shape != outputs.occupancy_logits.shape:
        raise ValueError("occupancy target shape does not match model output")


__all__ = [
    "DirectOutputs",
    "DirectPerStringNet",
    "HOP_LENGTH",
    "LossBreakdown",
    "MEL_BANDS",
    "MIDI_BEGIN",
    "N_FFT",
    "PITCH_CLASSES",
    "SAMPLE_RATE",
    "STRINGS",
    "WINDOW_END_S",
    "WINDOW_SAMPLES",
    "WINDOW_START_S",
    "extract_window",
    "gold_pitch_string_scores",
    "log_mel_batch",
    "mel_filterbank",
    "multitask_loss",
    "parameter_count",
    "string_scores_to_fret_prior",
]
