

set -x
PY_ARGS=${@:1}

python3 tools/test.py --enable-dist ${PY_ARGS}
