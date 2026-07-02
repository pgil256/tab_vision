"""Learned string-resolution model — v1.1 chunk-6 WS4 (impl-side of SPEC §8).

A pretrained image backbone over the YOLO **neck-crop** predicts a 6-way string
posterior, bypassing the fret-detection bottleneck that caps the geometric chain
(``docs/plans/2026-06-25-v1.1-ws4-learned-string-model-design.md``). At fusion the
known pitch restricts the posterior to its candidate strings; the evidence rides
the existing ``marginal_string_fret`` → ``AudioEvent.fret_prior`` channel, so no
§8 dataclass / Protocol / entrypoint signature changes.

``torch`` / ``torchvision`` are imported at module load, so this module is
imported *lazily* by its consumers (the eval probe / fusion hook) — the rest of
the package stays importable without torch, exactly like the lazy
``ultralytics`` import in ``video.guitar.yolo_backend``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models

# ImageNet normalisation for the pretrained backbone (RGB).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
DEFAULT_CROP_SIZE = 224


class StringResolverNet(nn.Module):
    """ResNet-18 backbone → ``n_strings``-way string-classification head.

    Unconditional over strings; pitch-conditioning happens at fusion (mask to the
    pitch's candidate strings + renormalise). Pretrained ImageNet weights give a
    strong visual prior on a small in-domain dataset.
    """

    def __init__(self, n_strings: int = 6, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, n_strings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.backbone(x)


def preprocess_crops(crops_bgr: np.ndarray) -> torch.Tensor:
    """``(N, H, W, 3)`` BGR uint8 → normalised ``(N, 3, H, W)`` float tensor.

    Accepts a single ``(H, W, 3)`` crop too. BGR→RGB, scale to [0,1], ImageNet
    normalise — matching the training transform.
    """
    arr = np.asarray(crops_bgr)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3) BGR crops, got {arr.shape}")
    rgb = arr[..., ::-1].astype(np.float32) / 255.0  # BGR→RGB, [0,1]
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2)  # (N,3,H,W)
    mean = torch.tensor(_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_STD).view(1, 3, 1, 1)
    return (t - mean) / std


@torch.no_grad()
def predict_string_proba(
    model: StringResolverNet,
    crops_bgr: np.ndarray,
    *,
    device: str | None = None,
) -> np.ndarray:
    """String posterior(s) for crop(s): ``(N, n_strings)`` softmax (or ``(n_strings,)``).

    A single ``(H, W, 3)`` crop returns a 1-D ``(n_strings,)`` vector.
    """
    single = np.asarray(crops_bgr).ndim == 3
    x = preprocess_crops(crops_bgr)
    if device is not None:
        x = x.to(device)
        model = model.to(device)
    model.eval()
    logits = model(x)
    proba = torch.softmax(logits, dim=1).cpu().numpy()
    return proba[0] if single else proba


def load_string_resolver(
    checkpoint_path: str | Path,
    *,
    n_strings: int = 6,
    device: str | None = None,
) -> StringResolverNet:
    """Load a trained :class:`StringResolverNet` from a checkpoint (state_dict)."""
    model = StringResolverNet(n_strings=n_strings, pretrained=False)
    state = torch.load(str(checkpoint_path), map_location=device or "cpu")
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


__all__ = [
    "DEFAULT_CROP_SIZE",
    "StringResolverNet",
    "preprocess_crops",
    "predict_string_proba",
    "load_string_resolver",
]
