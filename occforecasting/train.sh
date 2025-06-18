

set -x
PY_ARGS=${@:1}

python3 tools/train.py --enable-dist ${PY_ARGS}
