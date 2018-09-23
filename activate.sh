# To run this script:
# source activate
# . activate

function check_python() {
  local MIN_PYTHON_VERSION='(3, 6, 2)'
  if ! python3 -c "import sys; sys.version_info > $MIN_PYTHON_VERSION or sys.exit(1)"; then
    echo "The minimum Python version is $MIN_PYTHON_VERSION"
    return 1
  fi
}


function set_up_env() {
  if [ ! -z "${VIRTUAL_ENV-}" ]; then
    deactivate
  fi
  if ! python3 -c "import virtualenv" 2> /dev/null; then
    python3 -m easy-install virtualenv || return
  fi
  if [ ! -f .venv/bin/python ]; then
    python3 -m virtualenv .venv || return
  fi
  source .venv/bin/activate || return
}


function install_deps() {
  if ! md5sum requirements.txt | diff -q .venv/requirements.txt.md5 -; then
    pip install -r requirements.txt || return
    md5sum requirements.txt > .venv/requirements.txt.md5 || return
  fi
  if ! md5sum requirements-test.txt | diff -q .venv/requirements-test.txt.md5 -; then
    pip install -r requirements-test.txt || return
    md5sum requirements-test.txt > .venv/requirements-test.txt.md5 || return
  fi
  if ! md5sum requirements-dev.txt | diff -q .venv/requirements-dev.txt.md5 -; then
    pip install -r requirements-dev.txt || return
    md5sum requirements-dev.txt > .venv/requirements-dev.txt.md5 || return
  fi
}


check_python || return 1
set_up_env || return 1
install_deps || return 1

unset -f check_python
unset -f set_up_env
unset -f install_deps
