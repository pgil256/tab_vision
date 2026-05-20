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

    # Download Guitar-TECHS (Phase 0, no credentials required).
    python -m scripts.acquire.datasets guitar-techs

    # List supported datasets.
    python -m scripts.acquire.datasets list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / ".tabvision" / "data"

# Guitar-TECHS (ICASSP 2025 — Pedroza, Taheri, Abreu, Corey, Roman).
# CC-BY-4.0; ~5 GB; Phase 0 acquisition.
ZENODO_GUITAR_TECHS_RECORD_ID = "14963133"
GUITAR_TECHS_DEFAULT_SUBDIR = f"datasets/zenodo-{ZENODO_GUITAR_TECHS_RECORD_ID}-guitar-techs-v1"
GUITAR_TECHS_CITATION = (
    "Pedroza, H. E. V., Taheri, T., Abreu, W., Corey, R., & Roman, I. R. (2025). "
    "Guitar-TECHS: An Electric Guitar Dataset Covering Techniques, Musical "
    "Excerpts, Chords and Scales Using a Diverse Array of Hardware. "
    "ICASSP 2025. Zenodo: https://doi.org/10.5281/zenodo.14963133 (CC-BY-4.0)."
)


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

    gt = sub.add_parser(
        "guitar-techs",
        help="Guitar-TECHS (Zenodo, CC-BY-4.0, ~5 GB; Phase 0)",
    )
    gt.add_argument(
        "--record-id",
        default=ZENODO_GUITAR_TECHS_RECORD_ID,
        help=f"Zenodo record id (default: {ZENODO_GUITAR_TECHS_RECORD_ID})",
    )
    gt.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            f"destination directory (default: $TABVISION_DATA_ROOT/{GUITAR_TECHS_DEFAULT_SUBDIR})"
        ),
    )
    gt.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch the Zenodo manifest and print files + checksums, do not download",
    )
    gt.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "skip downloads and just MD5-verify whatever is already in --output "
            "against the Zenodo manifest. Useful when files were transferred "
            "out-of-band (e.g. browser download on another machine) and you "
            "need to confirm integrity."
        ),
    )

    args = parser.parse_args(argv)

    if args.dataset == "list":
        print("Supported datasets:")
        print("  roboflow-guitar — Roboflow b101/guitar-3 (Phase 3, YOLO-OBB)")
        print("  guitar-techs    — Guitar-TECHS (Phase 0, CC-BY-4.0, Zenodo)")
        return 0

    if args.dataset == "roboflow-guitar":
        return _acquire_roboflow_guitar(
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            export_format=args.format,
            list_versions=args.list_versions,
        )

    if args.dataset == "guitar-techs":
        return _acquire_guitar_techs(
            record_id=args.record_id,
            output_dir=args.output,
            dry_run=args.dry_run,
            verify_only=args.verify_only,
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


def _acquire_guitar_techs(
    *,
    record_id: str,
    output_dir: Path | None,
    dry_run: bool = False,
    verify_only: bool = False,
) -> int:
    """Acquire Guitar-TECHS from Zenodo.

    Source: https://zenodo.org/records/{record_id}
    License: CC-BY-4.0 (verified 2026-05-19 against LICENSES.md).
    No credentials required.

    The function pulls the record JSON, then for each file:
      1. skips if a same-size file with matching MD5 already exists at the target,
      2. otherwise downloads to a ``.part`` sibling and renames on success,
      3. verifies MD5 against the Zenodo manifest. Mismatch is a hard error.

    ``--dry-run`` prints the manifest and exits.
    ``--verify-only`` skips downloads and only checks existing files.
    """
    target = output_dir if output_dir is not None else (_data_root() / GUITAR_TECHS_DEFAULT_SUBDIR)

    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"fetching Zenodo record metadata: {api_url}")
    try:
        with urllib.request.urlopen(api_url, timeout=60) as resp:
            record = json.load(resp)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(
            f"error: failed to reach Zenodo: {exc}\n\n"
            "Diagnostics:\n"
            "  curl -v --max-time 30 https://zenodo.org/api/records/14963133\n\n"
            "If TCP connect times out on CERN IP ranges (137.138.0.0/16,\n"
            "188.184.0.0/15) this network is blocking Zenodo upstream. Options:\n"
            "  - retry on a different network (VPN, mobile hotspot, other machine)\n"
            "  - download the archive via browser elsewhere, place each file under\n"
            f"      {target}\n"
            "    preserving the original filenames, then re-run with --verify-only.\n",
            file=sys.stderr,
        )
        return 2

    files = record.get("files") or []
    if not files:
        print(f"error: Zenodo record {record_id} contains no files", file=sys.stderr)
        return 2

    total_bytes = sum(int(f.get("size", 0)) for f in files)
    license_id = ""
    metadata = record.get("metadata") or {}
    lic = metadata.get("license")
    if isinstance(lic, dict):
        license_id = lic.get("id", "") or ""
    elif isinstance(lic, str):
        license_id = lic
    print(
        f"record {record_id}: {len(files)} file(s), "
        f"~{total_bytes / 1024**3:.2f} GB total, license={license_id or '?'}"
    )
    print(f"target: {target}")

    if dry_run:
        for f in files:
            size_mb = int(f.get("size", 0)) / 1024**2
            print(f"  {f.get('key', '?')}  {size_mb:.1f} MB  {f.get('checksum', '?')}")
        return 0

    target.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_dl = 0
    for f in files:
        key = f.get("key")
        if not key:
            continue
        expected_size = int(f.get("size", 0))
        checksum = f.get("checksum", "")  # e.g. "md5:abcdef..."
        expected_md5 = checksum[4:] if checksum.startswith("md5:") else ""
        download_url = (f.get("links") or {}).get("self")

        dest = target / key
        if dest.exists() and dest.stat().st_size == expected_size:
            if expected_md5:
                actual_md5 = _md5_file(dest)
                if actual_md5 == expected_md5:
                    print(f"  ok    {key} (cached, md5 verified)")
                    n_ok += 1
                    continue
                print(
                    f"  WARN  {key} present but md5 mismatch "
                    f"(got {actual_md5}, want {expected_md5}); "
                    "will redownload"
                    if not verify_only
                    else "; verify-only: aborting"
                )
                if verify_only:
                    return 2
                dest.unlink()
            else:
                print(f"  ok    {key} (cached, size matches; no checksum to verify)")
                n_ok += 1
                continue

        if verify_only:
            print(f"  MISSING  {key} (verify-only mode)", file=sys.stderr)
            return 2

        if not download_url:
            print(f"  error: no download URL for {key} in Zenodo manifest", file=sys.stderr)
            return 2

        print(f"  get   {key} ({expected_size / 1024**2:.1f} MB)")
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            urllib.request.urlretrieve(download_url, part)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  ERROR {key}: download failed: {exc}", file=sys.stderr)
            if part.exists():
                part.unlink()
            return 2

        if expected_md5:
            actual_md5 = _md5_file(part)
            if actual_md5 != expected_md5:
                print(
                    f"  ERROR {key}: md5 mismatch after download "
                    f"(got {actual_md5}, want {expected_md5})",
                    file=sys.stderr,
                )
                part.unlink()
                return 2
        part.rename(dest)
        print("        md5 verified")
        n_dl += 1

    print(f"\nguitar-techs ready at {target}")
    print(f"  cached: {n_ok}  downloaded: {n_dl}  total: {len(files)}")
    print("\nattribution required (add to LICENSES.md / README / portfolio surfaces):")
    print(f"  {GUITAR_TECHS_CITATION}")
    return 0


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute MD5 hex digest of a file, streaming."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
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
