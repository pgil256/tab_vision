"""Validated, content-addressed performance receipts for TabCNN caching."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RECEIPT_FORMAT_VERSION = 1


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(raw_temp, path)
    finally:
        if os.path.exists(raw_temp):
            os.unlink(raw_temp)


def peak_rss_bytes() -> int | None:
    """Return this process's high-water RSS in bytes on supported platforms."""

    try:
        import resource
    except ImportError:  # pragma: no cover - Windows Python
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0


def _validate_row(row: Mapping[str, Any], *, model_name: str) -> None:
    digest = row.get("posterior_sha256")
    timing = row.get("timing_seconds")
    determinism = row.get("determinism")
    if (
        row.get("model") != model_name
        or not isinstance(row.get("clip_id"), str)
        or not row["clip_id"]
        or not isinstance(row.get("cache_key"), str)
        or not row["cache_key"]
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(row.get("size_bytes"), int)
        or row["size_bytes"] <= 0
        or not _finite_nonnegative(row.get("duration_s"))
        or float(row["duration_s"]) <= 0.0
        or not isinstance(timing, Mapping)
        or any(
            not _finite_nonnegative(timing.get(field)) for field in ("resample", "cqt", "inference")
        )
        or not isinstance(determinism, Mapping)
        or determinism.get("verified") is not True
    ):
        raise RuntimeError(f"invalid cache performance row for {row.get('clip_id')!r}")


def _validate_payload_shape(payload: Mapping[str, Any], *, model_name: str) -> None:
    rows = payload.get("posteriors")
    if (
        payload.get("format_version") != RECEIPT_FORMAT_VERSION
        or payload.get("model") != model_name
        or payload.get("fresh_process_per_model") is not True
        or not isinstance(payload.get("code_revision"), Mapping)
        or not isinstance(payload.get("runtime"), Mapping)
        or not isinstance(payload.get("inference_chunk_size"), int)
        or payload["inference_chunk_size"] <= 0
        or not _finite_nonnegative(payload.get("model_load_seconds"))
        or float(payload["model_load_seconds"]) <= 0.0
        or not isinstance(payload.get("peak_rss_bytes"), int)
        or payload["peak_rss_bytes"] <= 0
        or not isinstance(rows, list)
        or not rows
    ):
        raise RuntimeError("invalid cache performance receipt header")
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError("cache performance receipt rows must be objects")
        _validate_row(raw, model_name=model_name)
    clip_ids = [str(row["clip_id"]) for row in rows]
    if len(set(clip_ids)) != len(clip_ids):
        raise RuntimeError("cache performance receipt contains duplicate clip IDs")
    observed_duration = sum(float(row["duration_s"]) for row in rows)
    if not _finite_nonnegative(payload.get("duration_s")) or not math.isclose(
        float(payload["duration_s"]),
        observed_duration,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError("cache performance receipt duration is inconsistent")


def write_cache_performance_receipt(
    cache_root: str | Path,
    payload: Mapping[str, Any],
    *,
    model_name: str,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Write one immutable receipt and update its per-model evidence pointer."""

    _validate_payload_shape(payload, model_name=model_name)
    text = _canonical_json(payload)
    raw = text.encode("utf-8")
    digest = _sha256_bytes(raw)
    root = Path(cache_root)
    immutable = root / f"posterior-cache-summary-{model_name}-{digest[:16]}.json"
    pointer = root / f"posterior-cache-evidence-{model_name}.json"
    if immutable.is_file() and immutable.read_bytes() != raw:
        raise RuntimeError(f"content-address collision at {immutable}")
    if not immutable.is_file():
        _atomic_text(immutable, text)
    _atomic_text(pointer, text)
    if destination is not None:
        requested = Path(destination)
        if requested not in {immutable, pointer}:
            _atomic_text(requested, text)
    return {
        "verified": True,
        "model": model_name,
        "path": str(immutable.resolve()),
        "sha256": digest,
        "pointer_path": str(pointer.resolve()),
    }


def load_cache_performance_receipt(
    cache_root: str | Path,
    *,
    model_name: str,
    expected_clip_ids: Sequence[str],
    expected_code_revision: Mapping[str, Any],
    expected_runtime: Mapping[str, Any],
    expected_chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    """Load one exact per-model cache receipt and reject stale/tampered evidence."""

    root = Path(cache_root)
    pointer = root / f"posterior-cache-evidence-{model_name}.json"
    try:
        raw = pointer.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"missing or invalid cache performance receipt for {model_name}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("cache performance receipt root must be an object")
    digest = _sha256_bytes(raw)
    immutable = root / f"posterior-cache-summary-{model_name}-{digest[:16]}.json"
    if not immutable.is_file() or immutable.read_bytes() != raw:
        raise RuntimeError("cache performance pointer lacks matching immutable bytes")
    _validate_payload_shape(payload, model_name=model_name)
    expected_ids = set(expected_clip_ids)
    if len(expected_ids) != len(expected_clip_ids):
        raise ValueError("expected clip IDs must be unique")
    rows = {str(row["clip_id"]): row for row in payload["posteriors"]}
    if (
        set(rows) != expected_ids
        or payload.get("code_revision") != expected_code_revision
        or payload.get("runtime") != expected_runtime
        or payload.get("inference_chunk_size") != expected_chunk_size
    ):
        raise RuntimeError("cache performance receipt selection/runtime is stale")
    evidence = {
        "verified": True,
        "model": model_name,
        "path": str(immutable.resolve()),
        "sha256": digest,
        "pointer_path": str(pointer.resolve()),
        "clips": len(rows),
        "duration_s": float(payload["duration_s"]),
        "model_load_seconds": float(payload["model_load_seconds"]),
        "peak_rss_bytes": int(payload["peak_rss_bytes"]),
        "fresh_process_per_model": True,
    }
    return evidence, rows


__all__ = [
    "RECEIPT_FORMAT_VERSION",
    "load_cache_performance_receipt",
    "peak_rss_bytes",
    "write_cache_performance_receipt",
]
