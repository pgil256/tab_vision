"""Focused tests for resumable, verified EGSet12 acquisition."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from scripts.acquire.egset12 import (
    EGSet12AcquisitionError,
    download_dataset,
    download_file,
    verify_dataset,
    verify_file,
)
from tabvision.eval.egset12 import PUBLISHED_FILES, PublishedFile


def _published(name: str, payload: bytes) -> PublishedFile:
    return PublishedFile(
        name=name,
        size_bytes=len(payload),
        md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    )


class _Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        content_range: str = "",
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_published_table_is_exactly_twelve_wav_jams_pairs() -> None:
    assert len(PUBLISHED_FILES) == 24
    assert [published.name for published in PUBLISHED_FILES] == [
        name
        for track_id in range(1, 13)
        for name in (f"{track_id:02d}.jams", f"{track_id:02d}.wav")
    ]
    assert PUBLISHED_FILES[0] == PublishedFile(
        "01.jams",
        52_172,
        "083c7dae8e6556c20b9a2d762e2c977f",
    )
    assert PUBLISHED_FILES[-1] == PublishedFile(
        "12.wav",
        6_702_584,
        "e1ee73508f37d5c28c69877a588665d2",
    )


def test_verification_rejects_missing_size_and_hash_mismatches(tmp_path: Path) -> None:
    payload = b"correct"
    published = _published("01.jams", payload)

    missing = verify_file(tmp_path / published.name, published)
    assert missing is not None and missing.reason == "missing"

    path = tmp_path / published.name
    path.write_bytes(b"x")
    wrong_size = verify_file(path, published)
    assert wrong_size is not None and wrong_size.reason.startswith("size mismatch")

    path.write_bytes(b"wrong!!")
    wrong_hash = verify_file(path, published)
    assert wrong_hash is not None and wrong_hash.reason.startswith("MD5 mismatch")

    path.write_bytes(payload)
    assert verify_file(path, published) is None
    assert verify_dataset(tmp_path, files=(published,)) == ()


def test_download_resumes_part_and_promotes_only_after_verification(tmp_path: Path) -> None:
    payload = b"abcdefghij"
    published = _published("01.jams", payload)
    (tmp_path / "01.jams.part").write_bytes(payload[:4])
    seen_ranges: list[str | None] = []

    def opener(request, timeout):
        del timeout
        seen_ranges.append(request.get_header("Range"))
        return _Response(
            payload[4:],
            status=206,
            content_range=f"bytes 4-{len(payload) - 1}/{len(payload)}",
        )

    status = download_file(published, tmp_path, opener=opener)

    assert status == "downloaded"
    assert seen_ranges == ["bytes=4-"]
    assert (tmp_path / "01.jams").read_bytes() == payload
    assert not (tmp_path / "01.jams.part").exists()


def test_server_ignoring_range_restarts_owned_part(tmp_path: Path) -> None:
    payload = b"complete"
    published = _published("02.jams", payload)
    (tmp_path / "02.jams.part").write_bytes(b"old")

    def opener(request, timeout):
        del request, timeout
        return _Response(payload, status=200)

    assert download_file(published, tmp_path, opener=opener) == "downloaded"
    assert (tmp_path / "02.jams").read_bytes() == payload


def test_valid_existing_file_is_idempotent_and_never_opens_network(tmp_path: Path) -> None:
    payload = b"already here"
    published = _published("03.jams", payload)
    (tmp_path / published.name).write_bytes(payload)

    def opener(request, timeout):
        del request, timeout
        raise AssertionError("network should not be called for a verified file")

    assert download_file(published, tmp_path, opener=opener) == "verified"


def test_invalid_existing_final_is_never_overwritten(tmp_path: Path) -> None:
    published = _published("04.jams", b"expected")
    final = tmp_path / published.name
    final.write_bytes(b"corrupt!")

    with pytest.raises(EGSet12AcquisitionError, match="refusing to overwrite"):
        download_file(published, tmp_path)
    assert final.read_bytes() == b"corrupt!"


def test_dataset_download_verifies_every_requested_file(tmp_path: Path) -> None:
    payloads = {"01.jams": b"one", "01.wav": b"two"}
    files = tuple(_published(name, payload) for name, payload in payloads.items())

    def opener(request, timeout):
        del timeout
        name = request.full_url.rsplit("/", 2)[-2]
        return _Response(payloads[name], status=200)

    statuses = download_dataset(tmp_path, files=files, opener=opener)

    assert statuses == {"01.jams": "downloaded", "01.wav": "downloaded"}
    assert verify_dataset(tmp_path, files=files) == ()
