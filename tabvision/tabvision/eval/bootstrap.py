"""Bootstrap confidence intervals for per-tier acceptance gates.

The 2026-05-12 design plan (§5) requires every per-tier Tab F1 number
to be reported with a 95% bootstrap CI, and the acceptance gate is
``lower_95_CI >= target`` — not just ``mean >= target``. This module
provides that primitive.

Resamples observations (typically per-clip Tab F1 values) with
replacement, applies a user-supplied statistic to each resample, and
returns the original-sample statistic plus the symmetric percentile
interval over the bootstrap distribution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap statistic + symmetric confidence interval.

    ``lower`` and ``upper`` are the ``(1-confidence)/2`` and
    ``(1+confidence)/2`` quantiles of the bootstrap distribution.
    For a single observation, ``statistic == lower == upper`` and
    ``n_bootstrap`` is ``0`` (no resampling performed).
    """

    statistic: float
    lower: float
    upper: float
    n_observations: int
    n_bootstrap: int
    confidence: float


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap a confidence interval over ``values``.

    ``statistic`` defaults to ``numpy.mean``. Pass a different callable
    (e.g. ``numpy.median``) for other functionals. The callable receives
    a 1-D ``numpy.ndarray`` of float64 values.

    ``seed`` is the integer seed for ``numpy.random.default_rng``;
    calling with the same seed + values produces identical output.
    """
    if len(values) == 0:
        raise ValueError("bootstrap_ci requires at least one observation")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1; got {n_bootstrap}")

    stat_fn: Callable[[np.ndarray], float] = statistic if statistic is not None else np.mean
    arr = np.asarray(values, dtype=np.float64).ravel()
    n_obs = arr.shape[0]
    point = float(stat_fn(arr))

    if n_obs == 1:
        return BootstrapResult(
            statistic=point,
            lower=point,
            upper=point,
            n_observations=1,
            n_bootstrap=0,
            confidence=confidence,
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_obs, size=(n_bootstrap, n_obs))
    resamples = arr[indices]  # shape (n_bootstrap, n_obs)

    if statistic is None or statistic is np.mean:
        # Fast path: vectorized mean over rows.
        dist = resamples.mean(axis=1)
    else:
        # General path: apply user statistic per resample.
        dist = np.fromiter(
            (float(stat_fn(resamples[i])) for i in range(n_bootstrap)),
            dtype=np.float64,
            count=n_bootstrap,
        )

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(dist, alpha))
    upper = float(np.quantile(dist, 1.0 - alpha))

    return BootstrapResult(
        statistic=point,
        lower=lower,
        upper=upper,
        n_observations=n_obs,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )


__all__ = ["BootstrapResult", "bootstrap_ci"]
