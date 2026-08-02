from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from tabvision.eval import tabcnn_runtime_evidence as evidence


def _payload() -> dict[str, Any]:
    rows = [
        {
            "model": "synthtab",
            "clip_id": clip_id,
            "cache_key": f"key-{clip_id}",
            "posterior_sha256": character * 64,
            "size_bytes": 100,
            "duration_s": duration,
            "timing_seconds": {
                "resample": 0.1,
                "cqt": 0.2,
                "inference": 0.3,
            },
            "determinism": {"verified": True},
        }
        for clip_id, character, duration in (("a", "a", 1.0), ("b", "b", 2.0))
    ]
    return {
        "format_version": evidence.RECEIPT_FORMAT_VERSION,
        "model": "synthtab",
        "fresh_process_per_model": True,
        "code_revision": {"evaluation_sha256": "code"},
        "runtime": {"python": "3.12.3"},
        "inference_chunk_size": 256,
        "model_load_seconds": 0.5,
        "peak_rss_bytes": 123_456,
        "duration_s": 3.0,
        "posteriors": rows,
    }


def _write(tmp_path: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return evidence.write_cache_performance_receipt(
        tmp_path,
        payload or _payload(),
        model_name="synthtab",
    )


def test_receipt_round_trip_is_content_addressed_and_exact(tmp_path: Path) -> None:
    written = _write(tmp_path)

    receipt, rows = evidence.load_cache_performance_receipt(
        tmp_path,
        model_name="synthtab",
        expected_clip_ids=["a", "b"],
        expected_code_revision={"evaluation_sha256": "code"},
        expected_runtime={"python": "3.12.3"},
        expected_chunk_size=256,
    )

    assert receipt["sha256"] == written["sha256"]
    assert receipt["clips"] == 2
    assert receipt["duration_s"] == 3.0
    assert receipt["model_load_seconds"] == 0.5
    assert receipt["peak_rss_bytes"] == 123_456
    assert set(rows) == {"a", "b"}
    assert Path(receipt["path"]).read_bytes() == Path(written["pointer_path"]).read_bytes()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload.update(model_load_seconds=0.0),
            "header",
        ),
        (
            lambda payload: payload.update(peak_rss_bytes=0),
            "header",
        ),
        (
            lambda payload: payload["posteriors"][1].update(model="dafx"),
            "row",
        ),
        (
            lambda payload: payload["posteriors"][1].update(clip_id="a"),
            "duplicate",
        ),
        (
            lambda payload: payload["posteriors"][1]["determinism"].update(verified=False),
            "row",
        ),
        (
            lambda payload: payload.update(duration_s=99.0),
            "duration",
        ),
    ],
)
def test_writer_rejects_incomplete_or_mixed_receipts(
    tmp_path: Path,
    mutator: Any,
    message: str,
) -> None:
    payload = copy.deepcopy(_payload())
    mutator(payload)

    with pytest.raises(RuntimeError, match=message):
        _write(tmp_path, payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_clip_ids": ["a"]},
        {"expected_code_revision": {"evaluation_sha256": "changed"}},
        {"expected_runtime": {"python": "3.11"}},
        {"expected_chunk_size": 128},
    ],
)
def test_loader_rejects_stale_selection_or_runtime(
    tmp_path: Path,
    overrides: dict[str, Any],
) -> None:
    _write(tmp_path)
    arguments: dict[str, Any] = {
        "model_name": "synthtab",
        "expected_clip_ids": ["a", "b"],
        "expected_code_revision": {"evaluation_sha256": "code"},
        "expected_runtime": {"python": "3.12.3"},
        "expected_chunk_size": 256,
    }
    arguments.update(overrides)

    with pytest.raises(RuntimeError, match="stale"):
        evidence.load_cache_performance_receipt(tmp_path, **arguments)


def test_loader_rejects_pointer_without_matching_immutable_bytes(tmp_path: Path) -> None:
    written = _write(tmp_path)
    pointer = Path(written["pointer_path"])
    pointer.write_bytes(pointer.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="immutable"):
        evidence.load_cache_performance_receipt(
            tmp_path,
            model_name="synthtab",
            expected_clip_ids=["a", "b"],
            expected_code_revision={"evaluation_sha256": "code"},
            expected_runtime={"python": "3.12.3"},
            expected_chunk_size=256,
        )


def test_peak_rss_is_positive_on_linux() -> None:
    observed = evidence.peak_rss_bytes()

    assert observed is None or observed > 0
