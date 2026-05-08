#!/usr/bin/env bash
set -euo pipefail

# Fresh-clone smoke for Phase 9. Defaults to the current repository root, but
# accepts an explicit repo path or URL as the first argument.
REPO_SOURCE="${1:-$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)}"
WORKDIR="$(mktemp -d)"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

git clone "$REPO_SOURCE" "$WORKDIR/tabvision"
cd "$WORKDIR/tabvision/tabvision"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

tabvision --version
python scripts/check_default_licenses.py --pyproject pyproject.toml
pytest -m render
