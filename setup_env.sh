#!/usr/bin/env bash

set -euo pipefail

MIN_PYTHON_VERSION='(3, 6, 2)'

if ! python3 -c "import sys; sys.version_info > $MIN_PYTHON_VERSION or sys.exit(1)"; then
  echo "The minimum Python version is $MIN_PYTHON_VERSION"
  exit 1
fi

if [ -z "${VIRTUAL_ENV-}" ]; then
  python3 -c "import virtualenv" 2>/dev/null || python3 -m easy-install virtualenv
  test -f .venv/bin/python || python3 -m virtualenv .venv
  PS1= source .venv/bin/activate
  ln -sf .venv/bin/activate activate
fi

pip install -r requirements.txt

echo "Setup finished. To start working in the virtual environment, run:"
echo "    . activate"
