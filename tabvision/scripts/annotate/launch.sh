#!/usr/bin/env bash
# Launch the labeling tool with the right venv + working dir already
# set up.  Run from anywhere::
#
#   bash tabvision/scripts/annotate/launch.sh /path/to/clips
#   bash tabvision/scripts/annotate/launch.sh /path/to/clips 5005
#
# Picks whichever venv exists first: $TABVISION_VENV, ./venv,
# ./tabvision-server/venv.  Falls through to system python3 with a
# helpful pip-install hint if none has flask installed.
set -euo pipefail

# Resolve the repo root (= the directory two levels up from this script).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PKG_ROOT="$REPO_ROOT/tabvision"

if [ "${1:-}" = "" ]; then
  cat >&2 <<EOF
[label] missing clip directory.
        Usage: bash tabvision/scripts/annotate/launch.sh /path/to/clips [port]
        Personal training videos were removed from this repo; pass an explicit
        public/offline validation corpus directory when using this optional tool.
EOF
  exit 2
fi

CLIPS="$1"
PORT="${2:-5005}"

# Pick the first venv that has flask + cv2.
choose_venv() {
  for candidate in "${TABVISION_VENV:-}" "$REPO_ROOT/venv" "$REPO_ROOT/tabvision-server/venv"; do
    [ -n "$candidate" ] || continue
    [ -x "$candidate/bin/python" ] || continue
    if "$candidate/bin/python" -c 'import flask, cv2' >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if VENV="$(choose_venv)"; then
  echo "[label] using venv: $VENV" >&2
  PYTHON="$VENV/bin/python"
else
  cat >&2 <<EOF
[label] no venv with flask+cv2 found.
        Install in an existing venv (recommended):
            source $REPO_ROOT/venv/bin/activate
            pip install flask opencv-python
        Or create a fresh one:
            python3 -m venv $REPO_ROOT/venv && source $REPO_ROOT/venv/bin/activate
            pip install -e $PKG_ROOT && pip install flask opencv-python
EOF
  exit 1
fi

cd "$PKG_ROOT"
echo "[label] clips=$CLIPS port=$PORT" >&2
exec "$PYTHON" -m scripts.annotate.label_clips --clips "$CLIPS" --port "$PORT"
