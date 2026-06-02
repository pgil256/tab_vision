"""Dataset acquisition — see SPEC.md §6.2.

Each subcommand fetches one dataset, verifies a checksum where possible,
and places it under ``$TABVISION_DATA_ROOT`` (defaults to
``~/.tabvision/data``). Idempotent — skips if already present.

Credentials are read from a ``.env`` at the repo root (gitignored). See
``.env.example`` for the expected variable names.

Usage::

    # Set up credentials once:
    cp .env.example .env  # then edit .env to fill in ROBOFLOW_API_KEY

    # Download the YOLO-OBB guitar detector training set (Phase 3).
    python -m scripts.acquire.datasets roboflow-guitar

    # Download EGDB (author-granted access URL; Phase 0 distorted-electric eval).
    python -m scripts.acquire.datasets egdb --url '<grant-url>'

    # List supported datasets.
    python -m scripts.acquire.datasets list
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / ".tabvision" / "data"


def _load_dotenv() -> None:
    """Load .env from the repo root. Best-effort; missing dotenv is fine."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _data_root() -> Path:
    return Path(os.environ.get("TABVISION_DATA_ROOT", DEFAULT_DATA_ROOT))


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="acquire-datasets")
    sub = parser.add_subparsers(dest="dataset", required=True)

    sub.add_parser("list", help="list supported datasets")

    eg = sub.add_parser(
        "egdb",
        help="EGDB electric-guitar dataset (Phase 0 distorted-electric eval). "
        "Author-granted use 2026-06-01; eval-only, not redistributed.",
    )
    eg.add_argument(
        "--url",
        default=None,
        help="EGDB source URL; defaults to the public project Drive folder. "
        "Falls back to $EGDB_DOWNLOAD_URL. Override only for a mirror.",
    )
    eg.add_argument(
        "--sha256",
        default=None,
        help="optional expected SHA-256 of the downloaded archive; verified "
        "before extraction. Falls back to $EGDB_SHA256.",
    )

    rb = sub.add_parser(
        "roboflow-guitar",
        help="Roboflow b101/guitar-3 (YOLO-OBB training, Phase 3)",
    )
    rb.add_argument("--workspace", default="b101")
    rb.add_argument("--project", default="guitar-3")
    rb.add_argument(
        "--version",
        type=int,
        default=None,
        help="dataset version. Defaults to the latest available; pass an "
        "integer to pin to a specific version.",
    )
    rb.add_argument(
        "--format",
        default="yolov8-obb",
        help="export format; yolov8-obb is what we train on (oriented bboxes)",
    )
    rb.add_argument(
        "--list-versions",
        action="store_true",
        help="just print available versions for this project and exit",
    )

    args = parser.parse_args(argv)

    if args.dataset == "list":
        print("Supported datasets:")
        print("  roboflow-guitar — Roboflow b101/guitar-3 (Phase 3, YOLO-OBB)")
        print("  egdb           — EGDB electric guitar (Phase 0 distorted-electric eval)")
        return 0

    if args.dataset == "egdb":
        return _acquire_egdb(
            url=args.url or os.environ.get("EGDB_DOWNLOAD_URL"),
            sha256=args.sha256 or os.environ.get("EGDB_SHA256"),
        )

    if args.dataset == "roboflow-guitar":
        return _acquire_roboflow_guitar(
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            export_format=args.format,
            list_versions=args.list_versions,
        )

    parser.error(f"unknown dataset: {args.dataset}")
    return 2


def _acquire_roboflow_guitar(
    *,
    workspace: str,
    project: str,
    version: int | None,
    export_format: str,
    list_versions: bool = False,
) -> int:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print(
            "error: ROBOFLOW_API_KEY missing.\n\n"
            "How to provide it:\n"
            "  cp .env.example .env\n"
            "  # then edit .env and set ROBOFLOW_API_KEY=...\n"
            "  # (.env is gitignored; never commit it)\n\n"
            "Get a key at https://roboflow.com → Settings → API.\n",
            file=sys.stderr,
        )
        return 2

    try:
        from roboflow import Roboflow
    except ImportError:
        print(
            "error: roboflow package not installed. "
            "Install with: pip install roboflow (or the full vision extras).",
            file=sys.stderr,
        )
        return 2

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)

    versions = _list_project_versions(proj)
    if list_versions:
        print(f"versions for {workspace}/{project}:")
        for v_num, v_name in versions:
            print(f"  v{v_num}  {v_name}")
        return 0
    if not versions:
        print(f"error: no versions found for {workspace}/{project}", file=sys.stderr)
        return 2

    if version is None:
        version = max(v for v, _ in versions)
        print(f"defaulting to latest version: v{version}")

    if version not in {v for v, _ in versions}:
        print(
            f"error: version {version} not found. Available: "
            f"{', '.join(f'v{v}' for v, _ in versions)}",
            file=sys.stderr,
        )
        return 2

    target = _data_root() / "datasets" / f"roboflow-{workspace}-{project}-v{version}"
    if target.exists() and any(target.iterdir()):
        print(f"already present: {target}")
        print("(delete the directory to force re-download)")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"downloading roboflow {workspace}/{project} v{version} → {target}")
    ver = proj.version(version)
    dataset = ver.download(export_format, location=str(target))

    license_info = getattr(ver, "license", None) or "unknown (check Roboflow page)"
    citation = (
        f"Roboflow Universe project {workspace}/{project} v{version}, accessed {dataset.location}"
    )
    print(f"\nattribution required:\n  {citation}\n  license: {license_info}")
    print("Add the above to docs/HISTORY.md and to the repo README before merging Phase 3.")
    return 0


# Public Google Drive folder linked from the EGDB project page
# (https://ss12f32v.github.io/Guitar-Transcription/, verified 2026-06-01).
# Access is open; the *license* is the gate (see LICENSES.md), cleared by the
# author's written grant. Override with --url / $EGDB_DOWNLOAD_URL if mirrored.
EGDB_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1h9DrB4dk4QstgjNaHh7lL7IMeKdYw82_"


def _acquire_egdb(*, url: str | None, sha256: str | None) -> int:
    """Fetch EGDB for the Phase-0 distorted-electric eval tier.

    EGDB ships as a *public* Google Drive folder (link above); access is open.
    The gate is the *license*, not the download: the EGDB repo has no LICENSE
    file, so portfolio use needs the author's written grant (on record
    2026-06-01 — see LICENSES.md). Eval-only: not redistributed here, not a
    shipped-weight substrate.
    """
    url = url or EGDB_DRIVE_FOLDER
    target = _data_root() / "datasets" / "egdb"
    if target.exists() and any(target.iterdir()):
        print(f"already present: {target}")
        print("(delete the directory to force re-download)")
        return 0
    target.mkdir(parents=True, exist_ok=True)

    if "drive.google.com" in url and "/folders/" in url:
        return _download_drive_folder(url, target)
    return _download_archive(url, target, sha256)


def _download_drive_folder(url: str, target: Path) -> int:
    try:
        import gdown
    except ImportError:
        print(
            "EGDB is a Google Drive folder; this needs `gdown`. Either:\n"
            "  1) pip install gdown   (then re-run this command), or\n"
            "  2) download the folder manually from:\n"
            f"       {url}\n"
            "     and unzip its contents into:\n"
            f"       {target}\n",
            file=sys.stderr,
        )
        return 2
    print(f"downloading EGDB Drive folder → {target}")
    gdown.download_folder(url=url, output=str(target), quiet=False, use_cookies=False)
    _egdb_done_message()
    return 0


def _download_archive(url: str, target: Path, sha256: str | None) -> int:
    archive = target.parent / "egdb.download"
    print(f"downloading EGDB archive → {archive}")
    try:
        urllib.request.urlretrieve(url, archive)  # noqa: S310 (trusted, user-supplied)
    except OSError as exc:
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 1

    if sha256:
        digest = _sha256_file(archive)
        if digest.lower() != sha256.lower():
            print(
                f"error: SHA-256 mismatch.\n  expected {sha256}\n  got      {digest}",
                file=sys.stderr,
            )
            archive.unlink(missing_ok=True)
            return 1
        print(f"sha256 OK: {digest}")

    print(f"extracting → {target}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(target)  # noqa: S202 (trusted archive)
    else:
        print(
            "error: downloaded file is neither a zip nor a tar archive. "
            f"Left in place at {archive} for manual inspection.",
            file=sys.stderr,
        )
        return 1
    archive.unlink(missing_ok=True)
    _egdb_done_message()
    return 0


def _egdb_done_message() -> None:
    print(
        "\nEGDB acquired (eval-only).\n"
        "  - Confirm the author's license-grant email is saved under docs/ and "
        "logged in docs/DECISIONS.md.\n"
        "  - Add an `egdb_gp` parser under tabvision/tabvision/eval/parsers/ to "
        "fold the distorted-electric tier into the composite manifest.\n"
        "  - Do NOT commit the extracted audio."
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_project_versions(proj) -> list[tuple[int, str]]:  # type: ignore[no-untyped-def]
    """Return [(version_number, name), ...] sorted by number ascending."""
    out: list[tuple[int, str]] = []
    for v in getattr(proj, "versions", lambda: [])():
        # roboflow's Version objects expose a `.id` like "workspace/project/3"
        # and a `.name`. Number is the trailing integer.
        vid = str(getattr(v, "id", ""))
        try:
            num = int(vid.rsplit("/", 1)[-1])
        except ValueError:
            continue
        out.append((num, getattr(v, "name", f"v{num}") or f"v{num}"))
    out.sort()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
