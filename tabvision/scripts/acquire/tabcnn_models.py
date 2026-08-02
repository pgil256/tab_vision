"""Explicitly acquire the frozen TabCNN evaluation artifacts.

Downloads are resumable per-file ``.part`` files. A file is atomically
promoted only after exact size, SHA-256, and Git-LFS-pointer checks pass.
Invalid existing final files are never overwritten.

Examples::

    python -m scripts.acquire.tabcnn_models
    python -m scripts.acquire.tabcnn_models --verify-only
    python -m scripts.acquire.tabcnn_models --artifact synthtab_pretrained_x4
    python -m scripts.acquire.tabcnn_models \
        --source synthtab_pretrained_x4=/path/to/SynthTab-Pretrained.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from tabvision.eval.tabcnn_artifacts import (
    FROZEN_ARTIFACTS,
    TabCNNArtifact,
    artifact_by_id,
    artifact_manifest_json_bytes,
    default_models_root,
)

DEFAULT_RETRIES = 4
DEFAULT_TIMEOUT_S = 60.0
CHUNK_SIZE = 1024 * 1024
GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

OpenUrl = Callable[
    [urllib.request.Request, float],
    AbstractContextManager[BinaryIO],
]
DownloadStatus = Literal["downloaded", "verified"]


class TabCNNArtifactError(RuntimeError):
    """Raised when an artifact cannot be safely acquired or verified."""


@dataclass(frozen=True)
class VerificationIssue:
    artifact_id: str
    path: Path
    reason: str


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def is_git_lfs_pointer(path: str | Path) -> bool:
    """Return whether a file contains a Git-LFS pointer instead of payload bytes."""

    candidate = Path(path)
    if not candidate.is_file():
        return False
    with candidate.open("rb") as handle:
        return handle.read(len(GIT_LFS_POINTER_PREFIX)) == GIT_LFS_POINTER_PREFIX


def verify_artifact(path: str | Path, artifact: TabCNNArtifact) -> VerificationIssue | None:
    """Validate one path against the frozen size and SHA-256."""

    candidate = Path(path)
    if not candidate.exists():
        return VerificationIssue(artifact.artifact_id, candidate, "missing")
    if not candidate.is_file():
        return VerificationIssue(artifact.artifact_id, candidate, "not a regular file")
    if is_git_lfs_pointer(candidate):
        return VerificationIssue(
            artifact.artifact_id,
            candidate,
            "Git-LFS pointer found instead of artifact bytes",
        )
    size = candidate.stat().st_size
    if size != artifact.size_bytes:
        return VerificationIssue(
            artifact.artifact_id,
            candidate,
            f"size mismatch: expected {artifact.size_bytes}, found {size}",
        )
    digest = sha256_file(candidate)
    if digest != artifact.sha256:
        return VerificationIssue(
            artifact.artifact_id,
            candidate,
            f"SHA-256 mismatch: expected {artifact.sha256}, found {digest}",
        )
    return None


def verify_artifacts(
    models_root: str | Path,
    *,
    artifacts: Iterable[TabCNNArtifact] = FROZEN_ARTIFACTS,
) -> tuple[VerificationIssue, ...]:
    """Return all missing or invalid artifacts under a models root."""

    root = Path(models_root)
    return tuple(
        issue
        for artifact in artifacts
        if (issue := verify_artifact(artifact.path_below(root), artifact)) is not None
    )


def _open_url(
    request: urllib.request.Request,
    timeout_s: float,
) -> AbstractContextManager[BinaryIO]:
    return urllib.request.urlopen(request, timeout=timeout_s)  # noqa: S310


def _response_status(response: BinaryIO) -> int | None:
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    getcode = getattr(response, "getcode", None)
    value = getcode() if callable(getcode) else None
    return value if isinstance(value, int) else None


def _reject_pointer(path: Path, artifact: TabCNNArtifact) -> None:
    if is_git_lfs_pointer(path):
        raise TabCNNArtifactError(
            f"{artifact.artifact_id}: {path} is a Git-LFS pointer, not artifact bytes"
        )


def _promote_verified_part(
    part_path: Path,
    final_path: Path,
    artifact: TabCNNArtifact,
) -> bool:
    if not part_path.is_file():
        return False
    _reject_pointer(part_path, artifact)
    size = part_path.stat().st_size
    if size > artifact.size_bytes:
        raise TabCNNArtifactError(
            f"{artifact.artifact_id}: {part_path} exceeds frozen size "
            f"{artifact.size_bytes}; remove the owned .part file and retry"
        )
    if size < artifact.size_bytes:
        return False
    issue = verify_artifact(part_path, artifact)
    if issue is not None:
        raise TabCNNArtifactError(f"{artifact.artifact_id}: {issue.reason}")
    os.replace(part_path, final_path)
    return True


def _local_prefix_matches(part_path: Path, source_path: Path) -> bool:
    remaining = part_path.stat().st_size
    with part_path.open("rb") as part, source_path.open("rb") as source:
        while remaining:
            count = min(CHUNK_SIZE, remaining)
            if part.read(count) != source.read(count):
                return False
            remaining -= count
    return True


def _copy_local_source(
    source_path: Path,
    part_path: Path,
    final_path: Path,
    artifact: TabCNNArtifact,
) -> DownloadStatus:
    issue = verify_artifact(source_path, artifact)
    if issue is not None:
        raise TabCNNArtifactError(
            f"{artifact.artifact_id}: invalid explicit source {source_path}: {issue.reason}"
        )
    offset = part_path.stat().st_size if part_path.is_file() else 0
    if offset > artifact.size_bytes:
        raise TabCNNArtifactError(
            f"{artifact.artifact_id}: {part_path} exceeds frozen size "
            f"{artifact.size_bytes}; remove the owned .part file and retry"
        )
    if offset and not _local_prefix_matches(part_path, source_path):
        raise TabCNNArtifactError(
            f"{artifact.artifact_id}: existing .part file is not a prefix of "
            f"the explicit source; remove {part_path} and retry"
        )
    with source_path.open("rb") as source:
        source.seek(offset)
        with part_path.open("ab" if offset else "wb") as destination:
            shutil.copyfileobj(source, destination, length=CHUNK_SIZE)
    if not _promote_verified_part(part_path, final_path, artifact):
        raise TabCNNArtifactError(f"{artifact.artifact_id}: explicit source copy was partial")
    return "downloaded"


def download_artifact(
    artifact: TabCNNArtifact,
    models_root: str | Path,
    *,
    source_path: str | Path | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    opener: OpenUrl | None = None,
) -> DownloadStatus:
    """Acquire one artifact from an explicit source or its frozen URL."""

    if retries < 1:
        raise ValueError("retries must be at least 1")
    root = Path(models_root)
    final_path = artifact.path_below(root)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = final_path.with_name(f"{final_path.name}.part")

    if final_path.exists():
        issue = verify_artifact(final_path, artifact)
        if issue is None:
            return "verified"
        raise TabCNNArtifactError(
            f"refusing to overwrite invalid existing file {final_path}: {issue.reason}"
        )
    if _promote_verified_part(part_path, final_path, artifact):
        return "downloaded"

    if source_path is not None:
        source = Path(source_path).expanduser()
        return _copy_local_source(source, part_path, final_path, artifact)
    if artifact.download_url is None:
        raise TabCNNArtifactError(
            f"{artifact.artifact_id} has no verified direct download URL; "
            f"provide --source {artifact.artifact_id}=PATH"
        )

    open_url = opener or _open_url
    last_error: BaseException | None = None
    for attempt in range(retries):
        offset = part_path.stat().st_size if part_path.is_file() else 0
        headers = {"User-Agent": "TabVision-TabCNN-Acquirer/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(artifact.download_url, headers=headers)

        try:
            with open_url(request, timeout_s) as response:
                status = _response_status(response)
                if status not in (None, 200, 206):
                    raise TabCNNArtifactError(
                        f"{artifact.artifact_id}: unexpected HTTP status {status}"
                    )
                append = bool(offset and status == 206)
                if append:
                    content_range = getattr(response, "headers", {}).get("Content-Range", "")
                    if content_range and not content_range.startswith(f"bytes {offset}-"):
                        raise TabCNNArtifactError(
                            f"{artifact.artifact_id}: bad Content-Range {content_range!r}"
                        )
                with part_path.open("ab" if append else "wb") as destination:
                    shutil.copyfileobj(response, destination, length=CHUNK_SIZE)
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
                continue
            break

        _reject_pointer(part_path, artifact)
        size = part_path.stat().st_size
        if size > artifact.size_bytes:
            raise TabCNNArtifactError(
                f"{artifact.artifact_id}: {part_path} exceeds frozen size "
                f"{artifact.size_bytes}; remove the owned .part file and retry"
            )
        if size < artifact.size_bytes:
            last_error = TabCNNArtifactError(
                f"{artifact.artifact_id}: partial download {size}/{artifact.size_bytes} bytes"
            )
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
                continue
            break
        if _promote_verified_part(part_path, final_path, artifact):
            return "downloaded"

    detail = f": {last_error}" if last_error is not None else ""
    raise TabCNNArtifactError(
        f"failed to acquire {artifact.artifact_id} after {retries} attempts{detail}"
    )


def download_artifacts(
    models_root: str | Path,
    *,
    artifacts: Iterable[TabCNNArtifact] = FROZEN_ARTIFACTS,
    source_paths: Mapping[str, str | Path] | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    opener: OpenUrl | None = None,
) -> dict[str, DownloadStatus]:
    """Acquire and post-verify every requested frozen artifact."""

    requested = tuple(artifacts)
    sources = source_paths or {}
    statuses = {
        artifact.artifact_id: download_artifact(
            artifact,
            models_root,
            source_path=sources.get(artifact.artifact_id),
            retries=retries,
            timeout_s=timeout_s,
            opener=opener,
        )
        for artifact in requested
    }
    issues = verify_artifacts(models_root, artifacts=requested)
    if issues:
        details = "; ".join(f"{issue.artifact_id}: {issue.reason}" for issue in issues)
        raise TabCNNArtifactError(f"post-download verification failed: {details}")
    return statuses


def _parse_sources(values: Iterable[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        artifact_id, separator, raw_path = value.partition("=")
        if not separator or not artifact_id or not raw_path:
            raise ValueError(f"invalid --source {value!r}; expected ARTIFACT_ID=PATH")
        artifact_by_id(artifact_id)
        if artifact_id in sources:
            raise ValueError(f"duplicate --source for {artifact_id}")
        sources[artifact_id] = Path(raw_path).expanduser()
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        choices=tuple(artifact.artifact_id for artifact in FROZEN_ARTIFACTS),
        help="artifact to process (repeatable; default: all frozen artifacts)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="models root (default: $TABVISION_DATA_ROOT/models)",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="ARTIFACT_ID=PATH",
        help="copy a verified local source instead of using its frozen URL",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify selected artifacts without downloading",
    )
    parser.add_argument(
        "--print-manifest",
        action="store_true",
        help="print the selected frozen provenance manifest and exit",
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)

    root = (args.target or default_models_root()).expanduser()
    selected = (
        tuple(artifact_by_id(artifact_id) for artifact_id in args.artifact)
        if args.artifact
        else FROZEN_ARTIFACTS
    )

    if args.print_manifest:
        sys.stdout.buffer.write(artifact_manifest_json_bytes(artifacts=selected, models_root=root))
        return 0

    if args.verify_only:
        issues = verify_artifacts(root, artifacts=selected)
        if issues:
            print(f"TabCNN artifact verification failed at {root}:")
            for issue in issues:
                print(f"  {issue.artifact_id}: {issue.reason} ({issue.path})")
            return 1
        print(f"Verified {len(selected)} TabCNN artifacts at {root}")
        return 0

    try:
        sources = _parse_sources(args.source)
        unselected = sorted(set(sources) - {artifact.artifact_id for artifact in selected})
        if unselected:
            raise ValueError(
                "--source supplied for unselected artifact(s): " + ", ".join(unselected)
            )
        statuses = download_artifacts(
            root,
            artifacts=selected,
            source_paths=sources,
            retries=args.retries,
            timeout_s=args.timeout_s,
        )
    except (TabCNNArtifactError, ValueError) as exc:
        print(f"TabCNN artifact acquisition failed: {exc}", file=sys.stderr)
        return 1

    for artifact in selected:
        print(
            f"{statuses[artifact.artifact_id]:10} "
            f"{artifact.artifact_id}: {artifact.path_below(root)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
