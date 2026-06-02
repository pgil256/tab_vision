"""Dataset acquisition - see SPEC.md §6.2.

Each subcommand fetches one dataset, verifies a checksum where possible,
and places it under ``$TABVISION_DATA_ROOT`` (defaults to
``~/.tabvision/data``). Idempotent - skips if already present.

Credentials are read from a ``.env`` at the repo root (gitignored). See
``.env.example`` for the expected variable names.

Usage::

    # Set up credentials once:
    cp .env.example .env  # then edit .env to fill in ROBOFLOW_API_KEY

    # Download GuitarSet (mirdata) + Guitar-TECHS (Zenodo) for the #2 eval.
    python -m scripts.acquire.datasets guitarset
    python -m scripts.acquire.datasets guitar-techs

    # Download the YOLO-OBB guitar detector training set (Phase 3).
    python -m scripts.acquire.datasets roboflow-guitar

    # Download EGDB (public Drive folder; Phase 0 distorted-electric eval).
    python -m scripts.acquire.datasets egdb

    # List supported datasets.
    python -m scripts.acquire.datasets list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

GUITAR_TECHS_ZENODO_RECORD = "14963133"  # https://zenodo.org/records/14963133 (CC-BY-4.0)

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

    gs = sub.add_parser(
        "guitarset",
        help="GuitarSet via mirdata (clean-acoustic eval tiers + guitarset-v1 "
        "prior source). CC-BY-4.0.",
    )
    gs.add_argument(
        "--data-home",
        type=Path,
        default=None,
        help="GuitarSet root; defaults to $TABVISION_DATA_ROOT/guitarset "
        "(the layout the composite-eval GuitarSet scanner expects).",
    )

    gt = sub.add_parser(
        "guitar-techs",
        help="Guitar-TECHS from Zenodo (clean_electric eval tier; cross-dataset "
        "prior-generalization target). CC-BY-4.0.",
    )
    gt.add_argument(
        "--data-home",
        type=Path,
        default=None,
        help="target dir; defaults to $TABVISION_DATA_ROOT/guitar-techs.",
    )
    gt.add_argument(
        "--record",
        default=GUITAR_TECHS_ZENODO_RECORD,
        help=f"Zenodo record id (default {GUITAR_TECHS_ZENODO_RECORD}).",
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
        print("  guitarset      - GuitarSet via mirdata (clean-acoustic tiers + prior)")
        print("  guitar-techs   - Guitar-TECHS via Zenodo (clean_electric tier)")
        print("  egdb           - EGDB electric guitar (Phase 0 distorted-electric eval)")
        print("  roboflow-guitar - Roboflow b101/guitar-3 (Phase 3, YOLO-OBB)")
        return 0

    if args.dataset == "guitarset":
        return _acquire_guitarset(data_home=args.data_home)

    if args.dataset == "guitar-techs":
        return _acquire_guitar_techs(record=args.record, target=args.data_home)

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
            "Get a key at https://roboflow.com -> Settings -> API.\n",
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

    print(f"downloading roboflow {workspace}/{project} v{version} -> {target}")
    ver = proj.version(version)
    dataset = ver.download(export_format, location=str(target))

    license_info = getattr(ver, "license", None) or "unknown (check Roboflow page)"
    citation = (
        f"Roboflow Universe project {workspace}/{project} v{version}, accessed {dataset.location}"
    )
    print(f"\nattribution required:\n  {citation}\n  license: {license_info}")
    print("Add the above to docs/HISTORY.md and to the repo README before merging Phase 3.")
    return 0


def _acquire_guitarset(*, data_home: Path | None) -> int:
    """Download GuitarSet via mirdata into the layout the eval expects.

    mirdata lays GuitarSet out as ``<data_home>/annotation/*.jams`` and
    ``<data_home>/audio_mono-mic/*_mic.wav`` - exactly what
    ``tabvision.eval.manifest_builder.scan_guitarset`` and the checked-in
    ``data/eval/composite.toml`` reference. Default data_home =
    ``$TABVISION_DATA_ROOT/guitarset``. CC-BY-4.0; not redistributed here.
    """
    home = data_home or (_data_root() / "guitarset")
    annotation_dir = home / "annotation"
    audio_dir = home / "audio_mono-mic"
    if (
        annotation_dir.is_dir()
        and any(annotation_dir.glob("*.jams"))
        and audio_dir.is_dir()
        and any(audio_dir.glob("*.wav"))
    ):
        print(f"already present: {home}")
        print("(delete the directory to force re-download)")
        return 0

    try:
        import mirdata
    except ImportError:
        print(
            "error: mirdata not installed. Install with:\n"
            "  pip install mirdata        # or: pip install -e '.[train]'\n",
            file=sys.stderr,
        )
        return 2

    home.mkdir(parents=True, exist_ok=True)
    print(f"downloading GuitarSet (annotations + mono-mic only) via mirdata -> {home}")
    dataset = mirdata.initialize("guitarset", data_home=str(home))
    # The composite eval reads only annotation/*.jams + audio_mono-mic/*_mic.wav
    # (see scan_guitarset). Skip the multi-GB hex-pickup + mix partitions.
    dataset.download(partial_download=["annotations", "audio_mic"])
    print(
        "\nGuitarSet acquired (CC-BY-4.0; not redistributed).\n"
        f"  annotation/ + audio_mono-mic/ under {home}\n"
        "  Attribution: Xi et al., 'GuitarSet' (ISMIR 2018)."
    )
    return 0


def _acquire_guitar_techs(*, record: str, target: Path | None) -> int:
    """Download Guitar-TECHS from Zenodo via the public API.

    Enumerates the record's files through the Zenodo REST API (so no archive
    filenames are hard-coded), downloads each into ``<target>``, and extracts
    any zips. Default target = ``$TABVISION_DATA_ROOT/guitar-techs``.
    Electric-guitar, per-string MIDI (Fishman Triple Play) -> clean_electric
    tier. CC-BY-4.0; not redistributed here.
    """
    dest = target or (_data_root() / "guitar-techs")
    if dest.exists() and any(dest.iterdir()):
        print(f"already present: {dest}")
        print("(delete the directory to force re-download)")
        return 0
    dest.mkdir(parents=True, exist_ok=True)

    api = f"https://zenodo.org/api/records/{record}"
    print(f"querying Zenodo record {record} ...")
    try:
        with urllib.request.urlopen(api) as resp:  # noqa: S310 (trusted Zenodo API)
            meta = json.load(resp)
    except OSError as exc:
        print(f"error: Zenodo API request failed: {exc}", file=sys.stderr)
        return 1

    files = meta.get("files", [])
    if not files:
        print("error: no files listed on the Zenodo record.", file=sys.stderr)
        return 1

    for entry in files:
        key = entry.get("key", "file")
        links = entry.get("links", {})
        link = links.get("self") or links.get("download")
        if not link:
            print(f"  skip {key}: no download link", file=sys.stderr)
            continue
        out = dest / key
        print(f"  downloading {key} ...")
        try:
            urllib.request.urlretrieve(link, out)  # noqa: S310 (trusted Zenodo file)
        except OSError as exc:
            print(f"error: download of {key} failed: {exc}", file=sys.stderr)
            return 1
        if zipfile.is_zipfile(out):
            print(f"  extracting {key} ...")
            with zipfile.ZipFile(out) as zf:
                zf.extractall(dest)
            out.unlink(missing_ok=True)

    print(f"\nGuitar-TECHS acquired -> {dest} (CC-BY-4.0; not redistributed).")
    print("  Top-level entries (use these to verify the scanner's layout):")
    for child in sorted(dest.iterdir())[:25]:
        print(f"    {child.name}{'/' if child.is_dir() else ''}")
    print(
        "  Next: build the composite manifest with `--guitar-techs "
        f"{dest}` (see docs/plans/2026-06-02-tab-f1-phase-0-local-run.md).\n"
        "  If the manifest shows 0 GuitarTECHS clips, the on-disk layout "
        "differs from the assumed one - adjust globs in "
        "manifest_builder.scan_guitar_techs."
    )
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
    2026-06-01 - see LICENSES.md). Eval-only: not redistributed here, not a
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
    print(f"downloading EGDB Drive folder -> {target}")
    gdown.download_folder(url=url, output=str(target), quiet=False, use_cookies=False)
    _egdb_done_message()
    return 0


def _download_archive(url: str, target: Path, sha256: str | None) -> int:
    archive = target.parent / "egdb.download"
    print(f"downloading EGDB archive -> {archive}")
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

    print(f"extracting -> {target}")
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
