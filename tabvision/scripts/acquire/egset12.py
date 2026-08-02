"""Acquire the 12 WAV/JAMS pairs from official EGSet12 Zenodo record 11406378.

The default destination is ``$TABVISION_DATA_ROOT/egset12`` (falling back to
``~/.tabvision/data/egset12``).  Downloads are individual, resumable ``.part``
files and are only promoted after exact size and MD5 verification.

Examples::

    python -m scripts.acquire.egset12
    python -m scripts.acquire.egset12 --verify-only
    python -m scripts.acquire.egset12 --target /external/data/egset12
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal
from urllib.parse import quote

from tabvision.eval.egset12 import (
    PUBLISHED_FILES,
    ZENODO_RECORD_ID,
    PublishedFile,
    default_egset12_root,
)

ZENODO_CONTENT_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"
DEFAULT_RETRIES = 4
DEFAULT_TIMEOUT_S = 60.0
CHUNK_SIZE = 1024 * 1024

OpenUrl = Callable[
    [urllib.request.Request, float],
    AbstractContextManager[BinaryIO],
]


class EGSet12AcquisitionError(RuntimeError):
    """Raised when an artifact cannot be safely acquired or verified."""


@dataclass(frozen=True)
class VerificationIssue:
    file_name: str
    reason: str


DownloadStatus = Literal["downloaded", "verified"]


def md5_file(path: Path) -> str:
    """Return a file's MD5 digest (integrity matching, not security use)."""

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, published: PublishedFile) -> VerificationIssue | None:
    """Validate one path against its immutable published size and digest."""

    if not path.is_file():
        return VerificationIssue(published.name, "missing")
    size = path.stat().st_size
    if size != published.size_bytes:
        return VerificationIssue(
            published.name,
            f"size mismatch: expected {published.size_bytes}, found {size}",
        )
    digest = md5_file(path)
    if digest != published.md5:
        return VerificationIssue(
            published.name,
            f"MD5 mismatch: expected {published.md5}, found {digest}",
        )
    return None


def verify_dataset(
    root: str | Path,
    *,
    files: Iterable[PublishedFile] = PUBLISHED_FILES,
) -> tuple[VerificationIssue, ...]:
    """Return every missing, partial, or hash-mismatched published file."""

    dataset_root = Path(root)
    issues = [
        issue
        for published in files
        if (issue := verify_file(dataset_root / published.name, published)) is not None
    ]
    return tuple(issues)


def _content_url(published: PublishedFile) -> str:
    return f"{ZENODO_CONTENT_BASE}/{quote(published.name, safe='')}/content"


def _response_status(response: BinaryIO) -> int | None:
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    getcode = getattr(response, "getcode", None)
    value = getcode() if callable(getcode) else None
    return value if isinstance(value, int) else None


def _open(
    request: urllib.request.Request,
    timeout_s: float,
) -> AbstractContextManager[BinaryIO]:
    return urllib.request.urlopen(request, timeout=timeout_s)  # noqa: S310


def _promote_verified_part(part_path: Path, final_path: Path, published: PublishedFile) -> bool:
    """Promote a complete valid part; reject a complete corrupt part."""

    if not part_path.is_file():
        return False
    size = part_path.stat().st_size
    if size > published.size_bytes:
        raise EGSet12AcquisitionError(
            f"{part_path} exceeds published size {published.size_bytes}; remove it and retry"
        )
    if size < published.size_bytes:
        return False
    issue = verify_file(part_path, published)
    if issue is not None:
        raise EGSet12AcquisitionError(f"{part_path}: {issue.reason}")
    os.replace(part_path, final_path)
    return True


def download_file(
    published: PublishedFile,
    root: str | Path,
    *,
    retries: int = DEFAULT_RETRIES,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    opener: OpenUrl | None = None,
) -> DownloadStatus:
    """Download one file with byte-range resume and atomic verified promotion."""

    if retries < 1:
        raise ValueError("retries must be at least 1")
    dataset_root = Path(root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    final_path = dataset_root / published.name
    part_path = dataset_root / f"{published.name}.part"

    if final_path.exists():
        issue = verify_file(final_path, published)
        if issue is None:
            return "verified"
        raise EGSet12AcquisitionError(
            f"refusing to overwrite invalid existing file {final_path}: {issue.reason}"
        )
    if _promote_verified_part(part_path, final_path, published):
        return "downloaded"

    open_url = opener or _open
    last_error: BaseException | None = None
    for attempt in range(retries):
        offset = part_path.stat().st_size if part_path.is_file() else 0
        headers = {"User-Agent": "TabVision-EGSet12-Acquirer/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(_content_url(published), headers=headers)

        try:
            with open_url(request, timeout_s) as response:
                status = _response_status(response)
                if status not in (None, 200, 206):
                    raise EGSet12AcquisitionError(
                        f"{published.name}: unexpected HTTP status {status}"
                    )

                # If a server ignores Range and returns 200, safely restart the
                # owned .part file rather than append duplicate bytes.
                append = bool(offset and status == 206)
                if append:
                    content_range = getattr(response, "headers", {}).get("Content-Range", "")
                    if content_range and not content_range.startswith(f"bytes {offset}-"):
                        raise EGSet12AcquisitionError(
                            f"{published.name}: bad Content-Range {content_range!r}"
                        )
                mode = "ab" if append else "wb"
                with part_path.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=CHUNK_SIZE)
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
                continue
            break

        size = part_path.stat().st_size
        if size > published.size_bytes:
            raise EGSet12AcquisitionError(
                f"{part_path} exceeds published size {published.size_bytes}; remove it and retry"
            )
        if size < published.size_bytes:
            last_error = EGSet12AcquisitionError(
                f"{published.name}: partial download {size}/{published.size_bytes} bytes"
            )
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
                continue
            break
        if _promote_verified_part(part_path, final_path, published):
            return "downloaded"

    detail = f": {last_error}" if last_error is not None else ""
    raise EGSet12AcquisitionError(
        f"failed to acquire {published.name} after {retries} attempts{detail}"
    )


def download_dataset(
    root: str | Path,
    *,
    files: Iterable[PublishedFile] = PUBLISHED_FILES,
    retries: int = DEFAULT_RETRIES,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    opener: OpenUrl | None = None,
) -> dict[str, DownloadStatus]:
    """Acquire and verify every requested EGSet12 file."""

    requested = tuple(files)
    statuses = {
        published.name: download_file(
            published,
            root,
            retries=retries,
            timeout_s=timeout_s,
            opener=opener,
        )
        for published in requested
    }
    issues = verify_dataset(root, files=requested)
    if issues:
        details = "; ".join(f"{issue.file_name}: {issue.reason}" for issue in issues)
        raise EGSet12AcquisitionError(f"post-download verification failed: {details}")
    return statuses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="destination (default: $TABVISION_DATA_ROOT/egset12)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify all 24 published WAV/JAMS files without downloading",
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)

    target = (args.target or default_egset12_root()).expanduser()
    if args.verify_only:
        issues = verify_dataset(target)
        if issues:
            print(f"EGSet12 verification failed at {target}:")
            for issue in issues:
                print(f"  {issue.file_name}: {issue.reason}")
            return 1
        print(f"EGSet12 verified: 24 files at {target}")
        return 0

    try:
        statuses = download_dataset(
            target,
            retries=args.retries,
            timeout_s=args.timeout_s,
        )
    except (EGSet12AcquisitionError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    downloaded = sum(status == "downloaded" for status in statuses.values())
    print(f"EGSet12 verified: 24 files at {target} ({downloaded} downloaded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT_S",
    "EGSet12AcquisitionError",
    "VerificationIssue",
    "download_dataset",
    "download_file",
    "main",
    "md5_file",
    "verify_dataset",
    "verify_file",
]
