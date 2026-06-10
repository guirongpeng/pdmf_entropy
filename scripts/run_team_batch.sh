#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Step 1/4: 安装/更新虚拟环境依赖"
bash scripts/install_venv.sh

echo "==> Step 2/4: 激活虚拟环境"
# shellcheck source=/dev/null
source .venv/bin/activate

echo "==> Step 3/4: 预检查（dry-run）"
python run_all_datas_batch.py --dry-run

echo "==> Step 4/4: 正式运行批处理"
python run_all_datas_batch.py

echo "==> 完成"
