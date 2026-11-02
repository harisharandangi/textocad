#!/usr/bin/env sh
# Sets up TextoCAD on first run, then starts it. Extra arguments go to "textocad serve".
set -e
cd "$(dirname "$0")"

if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] not in ((3, 11), (3, 12)))' 2>/dev/null; then
  echo "The .venv folder uses an unsupported Python version. Recreating it."
  rm -rf .venv
fi

if [ ! -x .venv/bin/python ]; then
  echo "Setting up TextoCAD for the first time. This takes a few minutes."
  if command -v uv >/dev/null 2>&1; then
    uv venv --seed --python 3.12 .venv
  else
    PY=""
    for c in python3.12 python3.11 python3 python; do
      if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info[:2] not in ((3, 11), (3, 12)))' 2>/dev/null; then
        PY="$c"
        break
      fi
    done
    if [ -z "$PY" ]; then
      echo "TextoCAD needs Python 3.11 or 3.12. Install one from https://www.python.org/downloads/"
      echo "or install uv (https://docs.astral.sh/uv/) and run this again."
      exit 1
    fi
    "$PY" -m venv .venv
  fi
fi

if ! cmp -s pyproject.toml .venv/installed.toml; then
  EXTRAS=server
  command -v git >/dev/null 2>&1 && EXTRAS=server,gears
  echo "Installing TextoCAD..."
  .venv/bin/python -m pip install --disable-pip-version-check -q -e ".[$EXTRAS]"
  cp pyproject.toml .venv/installed.toml
fi

echo "Starting TextoCAD. Press Ctrl+C to stop it."
exec .venv/bin/python -m textocad.cli serve "$@"
