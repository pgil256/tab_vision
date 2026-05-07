#!/usr/bin/env bash
set -euo pipefail

# Fresh-clone smoke for Phase 9. Defaults to the current repository root, but
# accepts an explicit repo path or URL as the first argument.
REPO_SOURCE="${1:-$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)}"
WORKDIR="$(mktemp -d)"
PYTHON_BIN="${PYTHON:-}"

if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "python3 or python is required to run the fresh-install smoke" >&2
    exit 127
  fi
fi

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

git clone "$REPO_SOURCE" "$WORKDIR/tabvision"
cd "$WORKDIR/tabvision/tabvision"

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

tabvision --version
python scripts/check_default_licenses.py --pyproject pyproject.toml
pytest -m render
