"""Download and convert ``guitar_kroma.safetensors`` to a loadable ``.pth``.

Program N (NC second-opinion), Phase N1 — see
``docs/plans/2026-07-20-nc-second-opinion-and-synthtab-program.md``.

The pinned ``hf_midi_transcription`` loader expects
``torch.load(path)["model"]`` (a ``piano_transcription_inference``
convention), so the safetensors state dict is wrapped accordingly. Source
checkpoint: ``xavriley/midi-transcription-models`` (MIT). Output lands under
``$TABVISION_DATA_ROOT/models`` with a provenance manifest (source repo/file,
SHA-256 of both files, tensor/parameter counts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_REPO = "xavriley/midi-transcription-models"
DEFAULT_FILE = "guitar_kroma.safetensors"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--filename", default=DEFAULT_FILE)
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Output directory (default: $TABVISION_DATA_ROOT/models).",
    )
    args = parser.parse_args()

    if args.models_dir is not None:
        models_dir = Path(args.models_dir)
    else:
        data_root = os.environ.get("TABVISION_DATA_ROOT")
        if not data_root:
            raise SystemExit("set TABVISION_DATA_ROOT or pass --models-dir")
        models_dir = Path(data_root) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    source = Path(hf_hub_download(repo_id=args.repo, filename=args.filename))
    state = load_file(str(source))
    params = int(sum(value.numel() for value in state.values()))
    out_path = models_dir / "guitar-kroma.pth"
    torch.save({"model": state}, out_path)

    manifest = {
        "source_repo": args.repo,
        "source_file": args.filename,
        "source_sha256": _sha256(source),
        "output_file": str(out_path),
        "output_sha256": _sha256(out_path),
        "parameters": params,
        "tensors": len(state),
        "license": "MIT (repo-level, xavriley/midi-transcription-models)",
        "converted_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "wrapper": "{'model': state_dict} for piano_transcription_inference",
    }
    manifest_path = models_dir / "guitar-kroma.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
