#!/usr/bin/env bash
# 在本机（Python 3.11 + CentOS7）创建 .venv 并安装依赖（清华源）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export TMPDIR="${TMPDIR:-$ROOT/.pip_tmp}"
mkdir -p "$TMPDIR"

echo "==> Python: $($PY --version)"
echo "==> 项目目录: $ROOT"
echo "==> pip 临时目录: $TMPDIR"
echo "==> 镜像: $PIP_INDEX"

if [[ ! -d .venv ]]; then
  echo "==> 创建虚拟环境 .venv ..."
  "$PY" -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel -i "$PIP_INDEX"
pip install -r requirements.txt -i "$PIP_INDEX"

python -c "
import numpy, pandas, sklearn, scipy, numba, tqdm
import xgboost, lightgbm, catboost, matplotlib
from attribute_reduction import prepare_dataframe
from config import get_model_factories
print('依赖检查通过')
print('可用模型:', ', '.join(sorted(get_model_factories().keys())))
"

echo ""
echo "安装完成。激活环境:"
echo "  source $ROOT/.venv/bin/activate"
echo "试运行:"
echo "  python run_all_datas_batch.py --dry-run"